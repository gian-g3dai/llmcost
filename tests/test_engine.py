"""Engine behavior: Decimal exactness, unknown models (warn once, zero
cost), unpriced surfacing, tier validation, legacy book compat, ledger."""

import warnings
from decimal import Decimal

import pytest

from llmcost import CostLedger, DictPriceBook, cost


def test_decimal_exactness_no_float_drift():
    # 0.1 $/1M over 3 tokens: exact 3E-7, impossible with float arithmetic.
    book = DictPriceBook({"anthropic": {"m": {"input": 0.1, "output": 0.1}}})
    b = cost("anthropic", "m", {"input_tokens": 3, "output_tokens": 0}, prices=book)
    assert b.total == Decimal("3E-7")
    assert isinstance(b.total, Decimal)


def test_unknown_model_warns_once_and_costs_zero():
    usage = {"input_tokens": 1000, "output_tokens": 500}
    with pytest.warns(RuntimeWarning, match="No price entry"):
        b = cost("anthropic", "claude-imaginary-9", usage)
    assert b.total == Decimal(0)
    assert b.priced is False
    # Every observed dimension is surfaced, never silently dropped.
    assert set(b.unpriced) == {"input", "output"}
    # Second call: no repeated warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cost("anthropic", "claude-imaginary-9", usage)


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unsupported provider"):
        cost("mistral", "mistral-large", {"input_tokens": 1})


def test_unknown_tier_raises():
    with pytest.raises(ValueError, match="Unknown tier"):
        cost("openai", "gpt-5.1", {"input_tokens": 1}, tier="priority")


def test_batch_without_multiplier_bills_standard():
    """No-discount convention extends to the batch multiplier."""
    book = DictPriceBook({"openai": {"m": {"input": 2.0, "output": 4.0}}})
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert cost("openai", "m", usage, tier="batch", prices=book).total == Decimal(
        "2.0"
    )


def test_partial_pricing_surfaces_unpriced_dimension():
    book = DictPriceBook({"openai": {"m": {"input": 1.0}}})  # no output rate
    usage = {"input_tokens": 1000, "output_tokens": 500}
    b = cost("openai", "m", usage, prices=book)
    assert b.total == Decimal("0.001")  # input still costed
    assert b.unpriced == ["output"]


def test_legacy_get_rates_book_is_accepted():
    """A book exposing only the legacy `get_rates` protocol works directly."""

    class LegacyBook:
        def get_rates(self, provider, model):
            if (provider, model) == ("openai", "m"):
                return {"input": 1.25, "cached": 0.13, "output": 10.0}
            return None

    usage = {
        "input_tokens": 10000,
        "output_tokens": 500,
        "input_tokens_details": {"cached_tokens": 8000},
    }
    b = cost("openai", "m", usage, prices=LegacyBook())
    assert b.total == Decimal("0.00854")
    with pytest.warns(RuntimeWarning):
        assert cost("openai", "other", usage, prices=LegacyBook()).total == Decimal(0)


def test_breakdown_lines_reconcile_with_total():
    usage = {
        "input_tokens": 1234,
        "output_tokens": 567,
        "cache_read_input_tokens": 89012,
        "cache_creation": {"ephemeral_5m_input_tokens": 345},
    }
    b = cost("anthropic", "claude-sonnet-5", usage)
    assert b.total == sum((line.amount for line in b.lines), Decimal(0))


def test_ledger_totals_and_grouping():
    ledger = CostLedger()
    u = {"input_tokens": 1_000_000, "output_tokens": 0}
    ledger.add(cost("openai", "gpt-5.1", u), run="train")
    ledger.add(cost("openai", "gpt-5.1", u), run="eval")
    ledger.add(cost("anthropic", "claude-haiku-4-5", u), run="train")
    assert len(ledger) == 3
    assert ledger.total == Decimal("3.50")  # 1.25 + 1.25 + 1.00
    assert ledger.total_by("provider") == {
        "openai": Decimal("2.50"),
        "anthropic": Decimal("1.00"),
    }
    assert ledger.total_by("model")[("openai", "gpt-5.1")] == Decimal("2.50")
    assert ledger.total_by("run") == {
        "train": Decimal("2.25"),
        "eval": Decimal("1.25"),
    }
    assert ledger.unpriced == set()
    ledger.clear()
    assert ledger.total == Decimal(0)
    assert len(ledger) == 0


def test_ledger_collects_unpriced():
    ledger = CostLedger()
    with pytest.warns(RuntimeWarning):
        ledger.add(cost("openai", "gpt-unknown", {"input_tokens": 5}))
    assert ledger.unpriced == {("openai", "gpt-unknown", "input")}
