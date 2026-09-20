"""`estimate()`: cost from token counts alone, with uniform semantics
across providers (`input_tokens` includes the cached portion)."""

from decimal import Decimal

import pytest

from llmcost import cost, estimate


def test_openai_estimate_matches_cost_on_equivalent_usage():
    est = estimate(
        "openai",
        "gpt-5.1",
        input_tokens=50_000,
        cached_input_tokens=40_000,
        output_tokens=2_000,
    )
    exact = cost(
        "openai",
        "gpt-5.1",
        {
            "input_tokens": 50_000,
            "output_tokens": 2_000,
            "input_tokens_details": {"cached_tokens": 40_000},
        },
    )
    # 10000*1.25 + 40000*0.125 + 2000*10 = 37500 / 1M
    assert est.total == exact.total == Decimal("0.0375")


def test_anthropic_estimate_splits_input_cached_and_writes():
    est = estimate(
        "anthropic",
        "claude-sonnet-5",
        input_tokens=100_000,
        cached_input_tokens=60_000,
        cache_write_tokens=30_000,
        output_tokens=1_000,
    )
    dims = {line.dimension: line.quantity for line in est.lines}
    # Anthropic bills plain input, cache reads, and cache writes separately;
    # the uniform `input_tokens` total is decomposed for the caller.
    assert dims == {
        "input": 10_000,
        "cache_read": 60_000,
        "cache_write_5m": 30_000,
        "output": 1_000,
    }
    # 10000*2 + 60000*0.2 + 30000*2.5 + 1000*10 = 117000 / 1M
    assert est.total == Decimal("0.117")


def test_openai_cache_writes_cost_nothing_extra():
    plain = estimate("openai", "gpt-5.1", input_tokens=10_000)
    with_writes = estimate(
        "openai", "gpt-5.1", input_tokens=10_000, cache_write_tokens=8_000
    )
    assert plain.total == with_writes.total


def test_cached_clamped_to_input():
    est = estimate(
        "anthropic",
        "claude-haiku-4-5",
        input_tokens=1_000,
        cached_input_tokens=5_000,
    )
    # All 1000 tokens read from cache; nothing goes negative.
    assert est.total == Decimal("0.0001")  # 1000 * 0.10 / 1M


def test_tier_and_at_pass_through():
    standard = estimate("openai", "gpt-5.1", input_tokens=1_000_000)
    batch = estimate("openai", "gpt-5.1", input_tokens=1_000_000, tier="batch")
    assert batch.total == standard.total / 2


def test_long_context_estimate():
    est = estimate("openai", "gpt-5.4", input_tokens=300_000)
    assert est.total == Decimal("1.5")  # 300000 * 5.00 (long tier) / 1M


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unsupported provider"):
        estimate("gemini", "gemini-3-pro", input_tokens=1)


def test_openai_estimate_bills_cache_writes_on_models_that_charge():
    """gpt-5.6-terra: writes at $2.50 replace the $2.00 input rate."""
    est = estimate(
        "openai",
        "gpt-5.6-terra",
        input_tokens=10_000,
        cache_write_tokens=10_000,
    )
    assert [(l.dimension, l.quantity) for l in est.lines] == [("cache_write", 10_000)]
    assert est.total == Decimal("0.025")
