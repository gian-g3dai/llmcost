"""Golden tests for OpenAI cost math.

Expected values are hand-computed from the bundled price book:
gpt-5.1 (input 1.25 / cached 0.13 / output 10.00), gpt-5-pro (input 15 /
cached null / output 120), gpt-5.4 (input 2.50 / cached 0.25 / output 15.00,
long-context tier over 272k: 5.00 / 0.50 / 22.50).
"""

from decimal import Decimal
from types import SimpleNamespace

from llmcost import DictPriceBook, cost


def test_uncached_cached_split():
    usage = {
        "input_tokens": 10000,
        "output_tokens": 500,
        "input_tokens_details": {"cached_tokens": 8000},
    }
    b = cost("openai", "gpt-5.1", usage)
    # 2000*1.25 + 8000*0.13 + 500*10 = 2500 + 1040 + 5000 = 8540 / 1M
    assert b.total == Decimal("0.00854")
    dims = {line.dimension: line.quantity for line in b.lines}
    assert dims == {"input": 2000, "cached_input": 8000, "output": 500}


def test_null_cached_rate_means_no_discount():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 0,
        "input_tokens_details": {"cached_tokens": 1000},
    }
    b = cost("openai", "gpt-5-pro", usage)
    # cached billed at the 15.00 input rate, never free
    assert b.total == Decimal("0.015")
    assert b.unpriced == []


def test_cached_exceeding_input_clamps_uncached_to_zero():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 0,
        "input_tokens_details": {"cached_tokens": 1500},
    }
    b = cost("openai", "gpt-5.1", usage)
    # uncached = max(1000-1500, 0) = 0; cached 1500*0.13
    assert b.total == Decimal("0.000195")


def test_long_context_tier_boundary():
    at_threshold = {"input_tokens": 272000, "output_tokens": 100}
    over_threshold = {"input_tokens": 272001, "output_tokens": 100}
    b_short = cost("openai", "gpt-5.4", at_threshold)
    b_long = cost("openai", "gpt-5.4", over_threshold)
    # Exactly at the threshold stays on short rates (strictly-greater).
    # 272000*2.50 + 100*15 = 680000 + 1500 = 681500 / 1M
    assert b_short.total == Decimal("0.6815")
    # Over: whole request repriced. 272001*5.00 + 100*22.50 = 1360005+2250 /1M
    assert b_long.total == Decimal("1.362255")


def test_long_context_tier_applies_to_cached_and_output():
    usage = {
        "input_tokens": 300000,
        "output_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 100000},
    }
    b = cost("openai", "gpt-5.4", usage)
    # 200000*5.00 + 100000*0.50 + 1000*22.50 = 1000000+50000+22500 / 1M
    assert b.total == Decimal("1.0725")


def test_legacy_flat_long_keys():
    """Legacy flat `*_long` rate keys behave like a 272k tier."""
    book = DictPriceBook(
        {
            "openai": {
                "m": {
                    "input": 2.50,
                    "cached": 0.25,
                    "output": 15.00,
                    "input_long": 5.00,
                    "cached_long": 0.50,
                    "output_long": 22.50,
                }
            }
        }
    )
    short = cost("openai", "m", {"input_tokens": 1000, "output_tokens": 0}, prices=book)
    long = cost(
        "openai", "m", {"input_tokens": 272001, "output_tokens": 0}, prices=book
    )
    assert short.total == Decimal("0.0025")
    assert long.total == Decimal("1.360005")


def test_no_long_rates_means_no_repricing():
    usage = {"input_tokens": 500000, "output_tokens": 0}
    b = cost("openai", "gpt-5.1", usage)  # gpt-5.1 has no tiers
    assert b.total == Decimal("0.625")  # 500000 * 1.25 / 1M


def test_reasoning_itemized_not_double_billed():
    usage = {
        "input_tokens": 0,
        "output_tokens": 1000,
        "output_tokens_details": {"reasoning_tokens": 400},
    }
    b = cost("openai", "gpt-5.1", usage)
    dims = {line.dimension: line for line in b.lines}
    assert dims["output"].quantity == 600
    assert dims["reasoning"].quantity == 400
    assert dims["reasoning"].rate == Decimal("10.00")  # billed as output
    # Total identical to billing all 1000 as output.
    assert b.total == Decimal("0.01")


def test_batch_tier():
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    b = cost("openai", "gpt-5.1", usage, tier="batch")
    assert b.total == Decimal("0.625")  # half of 1.25


def test_usage_object_with_attributes():
    usage = SimpleNamespace(
        input_tokens=10000,
        output_tokens=500,
        input_tokens_details=SimpleNamespace(cached_tokens=8000),
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    b = cost("openai", "gpt-5.1", usage)
    assert b.total == Decimal("0.00854")


def test_dated_model_alias():
    usage = {"input_tokens": 1000, "output_tokens": 0}
    assert (
        cost("openai", "gpt-5.1-2025-11-13", usage).total
        == cost("openai", "gpt-5.1", usage).total
    )
