"""Golden tests for `estimate_conversation()`.

Expected values hand-computed from the bundled book:
claude-haiku-4-5 — input 1.00 / output 5.00 / cache_read 0.10 /
cache_write_5m 1.25 / cache_write_1h 2.00;
gpt-5.1 — input 1.25 / cached 0.125 / output 10.00.

Shared scenario: 3 turns, 10k system prefix, 1k user message and 2k reply
per turn. Prompts are therefore 11k, 14k, 17k tokens.
"""

from decimal import Decimal

import pytest

from llmcost import estimate_conversation

SCENARIO = dict(
    turns=3,
    system_tokens=10_000,
    user_tokens_per_turn=1_000,
    output_tokens_per_turn=2_000,
)


def test_anthropic_cached_conversation():
    est = estimate_conversation("anthropic", "claude-haiku-4-5", **SCENARIO)
    # Turn 0: write 11k @1.25 + out 2k @5           = 13750 + 10000 = 23750
    # Turn 1: read 11k @0.1 + write 3k @1.25 + out  =  1100 +  3750 + 10000
    # Turn 2: read 14k @0.1 + write 3k @1.25 + out  =  1400 +  3750 + 10000
    assert [t.total for t in est.turns] == [
        Decimal("0.02375"),
        Decimal("0.01485"),
        Decimal("0.01515"),
    ]
    assert est.total == Decimal("0.05375")
    # No plain-input line ever appears: every prompt token is a read or a
    # write under the always-cache assumption.
    for turn in est.turns:
        assert "input" not in {line.dimension for line in turn.lines}


def test_anthropic_no_cache_conversation():
    est = estimate_conversation(
        "anthropic", "claude-haiku-4-5", cache_ttl=None, **SCENARIO
    )
    # (11k + 14k + 17k) @1.00 + 3 * 2k @5.00 = 42000 + 30000 = 72000
    assert est.total == Decimal("0.072")
    # Caching the same conversation is ~25% cheaper.
    cached = estimate_conversation("anthropic", "claude-haiku-4-5", **SCENARIO)
    assert cached.total < est.total


def test_anthropic_expired_cache_costs_more_than_no_cache():
    """A cache outlived by its conversation rewrites every prompt at the
    write premium — honestly worse than not caching."""
    expired = estimate_conversation(
        "anthropic",
        "claude-haiku-4-5",
        turn_interval_seconds=600,  # > 5m TTL
        **SCENARIO,
    )
    # (11k + 14k + 17k) @1.25 + 30000 = 52500 + 30000
    assert expired.total == Decimal("0.0825")
    no_cache = estimate_conversation(
        "anthropic", "claude-haiku-4-5", cache_ttl=None, **SCENARIO
    )
    assert expired.total > no_cache.total


def test_anthropic_1h_ttl_survives_slow_turns_at_higher_write_rate():
    est = estimate_conversation(
        "anthropic",
        "claude-haiku-4-5",
        cache_ttl="1h",
        turn_interval_seconds=600,  # within 1h TTL
        **SCENARIO,
    )
    # Turn 0: 11k @2.00 + 10000 = 32000; turns 1-2: reads + 3k @2.00 + out
    # = (1100 + 6000 + 10000) + (1400 + 6000 + 10000)
    assert est.total == Decimal("0.0665")


def test_openai_cached_conversation():
    est = estimate_conversation("openai", "gpt-5.1", **SCENARIO)
    # Turn 0: 11k @1.25 + 2k @10                       = 13750 + 20000
    # Turn 1: 11k @0.125 + 3k @1.25 + 20000            =  1375 +  3750 + 20000
    # Turn 2: 14k @0.125 + 3k @1.25 + 20000            =  1750 +  3750 + 20000
    assert est.total == Decimal("0.084375")
    # OpenAI writes carry no premium, so caching can only help: expiry just
    # degrades to the no-cache total.
    expired = estimate_conversation(
        "openai", "gpt-5.1", turn_interval_seconds=600, **SCENARIO
    )
    no_cache = estimate_conversation("openai", "gpt-5.1", cache_ttl=None, **SCENARIO)
    assert expired.total == no_cache.total


def test_per_turn_sequences():
    est = estimate_conversation(
        "anthropic",
        "claude-haiku-4-5",
        turns=2,
        system_tokens=0,
        user_tokens_per_turn=[1_000, 3_000],
        output_tokens_per_turn=[500, 0],
        cache_ttl=None,
    )
    # Prompts: 1000, then 1000+500+3000=4500. (1000+4500)@1 + 500@5 = 8000
    assert est.total == Decimal("0.008")
    assert len(est.turns) == 2


def test_per_turn_length_mismatch_raises():
    with pytest.raises(ValueError, match="user_tokens_per_turn has 2 values"):
        estimate_conversation(
            "anthropic", "claude-haiku-4-5", turns=3, user_tokens_per_turn=[1, 2]
        )


def test_bad_ttl_and_negative_turns_raise():
    with pytest.raises(ValueError, match="cache_ttl"):
        estimate_conversation("openai", "gpt-5.1", turns=1, cache_ttl="10m")
    with pytest.raises(ValueError, match="turns"):
        estimate_conversation("openai", "gpt-5.1", turns=-1)


def test_zero_turns():
    est = estimate_conversation("openai", "gpt-5.1", turns=0)
    assert est.turns == []
    assert est.total == Decimal(0)


def test_batch_tier_passes_through():
    standard = estimate_conversation("openai", "gpt-5.1", **SCENARIO)
    batch = estimate_conversation("openai", "gpt-5.1", tier="batch", **SCENARIO)
    assert batch.total == standard.total / 2


def test_total_reconciles_with_turns():
    est = estimate_conversation("anthropic", "claude-sonnet-5", **SCENARIO)
    assert est.total == sum((t.total for t in est.turns), Decimal(0))
