"""Golden tests for Anthropic cost math.

Expected values are hand-computed from the bundled price book's
claude-haiku-4-5 rates (per 1M tokens): input 1.00, output 5.00,
cache_read 0.10, cache_write_5m 1.25, cache_write_1h 2.00 — mirroring
Anthropic's published 0.1x cache-read / 1.25x 5m-write / 2x 1h-write
multipliers.
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from llmcost import DictPriceBook, cost

MODEL = "claude-haiku-4-5"


def test_input_output_only():
    usage = {"input_tokens": 1000, "output_tokens": 2000}
    b = cost("anthropic", MODEL, usage)
    # 1000*1.00/1M + 2000*5.00/1M = 0.001 + 0.01
    assert b.total == Decimal("0.011")
    assert b.unpriced == []


def test_cache_read_discount():
    usage = {"input_tokens": 1000, "output_tokens": 0, "cache_read_input_tokens": 30000}
    b = cost("anthropic", MODEL, usage)
    # 1000*1.00/1M + 30000*0.10/1M = 0.001 + 0.003
    assert b.total == Decimal("0.004")


def test_cache_write_ttl_split():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 2000,
        "cache_read_input_tokens": 30000,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 1000,
            "ephemeral_1h_input_tokens": 2000,
        },
    }
    b = cost("anthropic", MODEL, usage)
    # (1000*1 + 2000*5 + 30000*0.1 + 1000*1.25 + 2000*2) / 1M = 19250/1M
    assert b.total == Decimal("0.01925")
    dims = {line.dimension: line.amount for line in b.lines}
    assert dims["cache_write_5m"] == Decimal("0.00125")
    assert dims["cache_write_1h"] == Decimal("0.004")


def test_flat_cache_creation_falls_back_to_5m_rate():
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 4000,
    }
    b = cost("anthropic", MODEL, usage)
    # 4000 * 1.25 / 1M
    assert b.total == Decimal("0.005")
    assert [line.dimension for line in b.lines] == ["cache_write_5m"]


def test_split_takes_precedence_over_flat_total():
    usage = {
        "cache_creation_input_tokens": 9999,  # stale aggregate, must be ignored
        "cache_creation": {"ephemeral_1h_input_tokens": 1000},
    }
    b = cost("anthropic", MODEL, usage)
    assert b.total == Decimal("0.002")  # 1000 * 2.00 / 1M


def test_missing_cache_rates_bill_at_input_rate():
    """No-discount convention: absent cache rates fall back to `input`,
    never to free."""
    book = DictPriceBook({"anthropic": {"m": {"input": 2.0, "output": 10.0}}})
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 1000,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 1000,
            "ephemeral_1h_input_tokens": 1000,
        },
    }
    b = cost("anthropic", "m", usage, prices=book)
    # All three cache dims at the 2.00 input rate.
    assert b.total == Decimal("0.006")
    assert b.unpriced == []


def test_batch_tier_halves_token_cost():
    usage = {"input_tokens": 1000, "output_tokens": 2000}
    b = cost("anthropic", MODEL, usage, tier="batch")
    assert b.total == Decimal("0.0055")
    assert b.lines[0].rate == Decimal("0.50")  # effective, multiplier applied


def test_per_call_server_tools():
    # DictPriceBook has no per_call channel; build a book object inline.
    class Book:
        def get_entry(self, provider, model, at=None):
            from llmcost import PriceEntry

            return PriceEntry(
                rates={"input": Decimal("1.0"), "output": Decimal("5.0")},
                per_call={"web_search": Decimal("0.01")},
            )

    usage = {
        "input_tokens": 1000,
        "output_tokens": 0,
        "server_tool_use": {"web_search_requests": 3},
    }
    b = cost("anthropic", "m", usage, prices=Book())
    tool_lines = [line for line in b.lines if line.unit == "per_call"]
    assert len(tool_lines) == 1
    assert tool_lines[0].dimension == "web_search"
    assert tool_lines[0].amount == Decimal("0.03")
    assert b.total == Decimal("0.031")


def test_unpriced_server_tool_is_surfaced():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 0,
        "server_tool_use": {"web_search_requests": 2},
    }
    b = cost("anthropic", MODEL, usage)
    assert "web_search" in b.unpriced
    assert b.total == Decimal("0.001")  # tokens still costed


def test_batch_does_not_discount_per_call():
    class Book:
        def get_entry(self, provider, model, at=None):
            from llmcost import PriceEntry

            return PriceEntry(
                rates={"input": Decimal("1.0"), "output": Decimal("5.0")},
                batch_multiplier=Decimal("0.5"),
                per_call={"web_search": Decimal("0.01")},
            )

    usage = {
        "input_tokens": 1000,
        "output_tokens": 0,
        "server_tool_use": {"web_search_requests": 2},
    }
    b = cost("anthropic", "m", usage, tier="batch", prices=Book())
    # tokens halved (0.0005), tool calls full price (0.02)
    assert b.total == Decimal("0.0205")


def test_usage_object_with_attributes():
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=2000,
        cache_read_input_tokens=0,
        cache_creation=SimpleNamespace(
            ephemeral_5m_input_tokens=500, ephemeral_1h_input_tokens=0
        ),
    )
    b = cost("anthropic", MODEL, usage)
    # 1000*1 + 2000*5 + 500*1.25 = 11625 / 1M
    assert b.total == Decimal("0.011625")


def test_dated_model_alias():
    usage = {"input_tokens": 1000, "output_tokens": 0}
    assert (
        cost("anthropic", "claude-haiku-4-5-20251001", usage).total
        == cost("anthropic", "claude-haiku-4-5", usage).total
    )


@pytest.mark.parametrize("missing", [None, {}])
def test_zero_and_missing_fields(missing):
    b = cost("anthropic", MODEL, missing or {})
    assert b.total == Decimal(0)
    assert b.lines == []
