"""Golden tests for OpenAI cost math.

Expected values are hand-computed from the bundled price book:
gpt-5.1 (input 1.25 / cached 0.125 / output 10.00), gpt-5-pro (input 15 /
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
    # 2000*1.25 + 8000*0.125 + 500*10 = 2500 + 1000 + 5000 = 8500 / 1M
    assert b.total == Decimal("0.0085")
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
    # uncached = max(1000-1500, 0) = 0; cached 1500*0.125
    assert b.total == Decimal("0.0001875")


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
    assert b.total == Decimal("0.0085")


def test_dated_model_alias():
    usage = {"input_tokens": 1000, "output_tokens": 0}
    assert (
        cost("openai", "gpt-5.1-2025-11-13", usage).total
        == cost("openai", "gpt-5.1", usage).total
    )


def test_chat_completions_usage_shape():
    """Chat Completions reports `prompt_tokens` / `completion_tokens` with
    `prompt_tokens_details` / `completion_tokens_details`; same math as the
    Responses shape."""
    usage = {
        "prompt_tokens": 10000,
        "completion_tokens": 500,
        "total_tokens": 10500,
        "prompt_tokens_details": {"cached_tokens": 8000, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 200,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }
    b = cost("openai", "gpt-5.1", usage)
    # 2000*1.25 + 8000*0.125 + 300*10 + 200*10 = 2500 + 1000 + 3000 + 2000
    assert b.total == Decimal("0.0085")
    dims = {line.dimension: line.quantity for line in b.lines}
    assert dims == {
        "input": 2000,
        "cached_input": 8000,
        "output": 300,
        "reasoning": 200,
    }
    assert b.unpriced == []


def test_chat_completions_matches_responses_shape():
    chat = {
        "prompt_tokens": 10000,
        "completion_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": 8000},
        "completion_tokens_details": {"reasoning_tokens": 200},
    }
    responses = {
        "input_tokens": 10000,
        "output_tokens": 500,
        "input_tokens_details": {"cached_tokens": 8000},
        "output_tokens_details": {"reasoning_tokens": 200},
    }
    assert (
        cost("openai", "gpt-5.1", chat).lines
        == cost("openai", "gpt-5.1", responses).lines
    )


def test_chat_completions_usage_object_with_attributes():
    usage = SimpleNamespace(
        prompt_tokens=10000,
        completion_tokens=500,
        total_tokens=10500,
        prompt_tokens_details=SimpleNamespace(cached_tokens=8000),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    assert cost("openai", "gpt-5.1", usage).total == Decimal("0.0085")


def test_chat_completions_without_details():
    """Older Chat Completions payloads carry no `*_details` blocks."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 100}
    b = cost("openai", "gpt-5.1", usage)
    assert b.total == Decimal("0.00225")  # 1000*1.25 + 100*10 = 2250 / 1M
    dims = {line.dimension: line.quantity for line in b.lines}
    assert dims == {"input": 1000, "output": 100}


def test_responses_shape_takes_precedence_over_chat_fields():
    """A payload carrying both shapes (e.g. a proxy that mirrors fields)
    is read via the Responses fields; explicit zero is not treated as
    absent."""
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "prompt_tokens": 5000,
        "completion_tokens": 5000,
    }
    b = cost("openai", "gpt-5.1", usage)
    assert b.total == Decimal("0")
    assert b.lines == []


def test_batch_keeps_cached_discount_where_provider_does():
    """gpt-5.1's Batch API row lists cached input at $0.0625 (half of
    $0.125), so the plain multiplier applies."""
    usage = {
        "input_tokens": 10000,
        "output_tokens": 0,
        "input_tokens_details": {"cached_tokens": 8000},
    }
    b = cost("openai", "gpt-5.1", usage, tier="batch")
    dims = {line.dimension: line for line in b.lines}
    assert dims["input"].rate == Decimal("0.625")
    assert dims["cached_input"].rate == Decimal("0.0625")
    assert b.total == Decimal("0.00175")  # 2000*0.625 + 8000*0.0625


def test_batch_drops_cached_discount_where_provider_does():
    """gpt-4.1's Batch API row shows no cached-input price ($1.00 input,
    $4.00 output), so cached tokens bill at the batch input rate — not at
    half the standard cached rate."""
    usage = {
        "input_tokens": 10000,
        "output_tokens": 0,
        "input_tokens_details": {"cached_tokens": 8000},
    }
    standard = cost("openai", "gpt-4.1", usage)
    assert {l.dimension: l.rate for l in standard.lines} == {
        "input": Decimal("2.00"),
        "cached_input": Decimal("0.50"),
    }
    batch = cost("openai", "gpt-4.1", usage, tier="batch")
    assert {l.dimension: l.rate for l in batch.lines} == {
        "input": Decimal("1.00"),
        "cached_input": Decimal("1.00"),
    }
    assert batch.total == Decimal("0.01")  # all 10000 tokens at 1.00


def test_explicit_batch_rate_overrides_multiplier():
    from llmcost.book import PriceEntry

    class Book:
        def get_entry(self, provider, model, at=None):
            return PriceEntry(
                rates={"input": Decimal("2.0"), "output": Decimal("4.0")},
                batch_multiplier=Decimal("0.5"),
                batch_rates={"output": Decimal("3.0")},
            )

    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    b = cost("openai", "m", usage, tier="batch", prices=Book())
    assert {l.dimension: l.rate for l in b.lines} == {
        "input": Decimal("1.0"),  # multiplier
        "output": Decimal("3.0"),  # explicit batch rate, not 2.0
    }


def test_cache_write_billed_as_replacement_rate():
    """gpt-5.6-sol: $4 input, $0.40 cached, $5 cache write, $20 output.
    Written tokens are a slice of `input_tokens` billed at the write rate
    instead of the input rate (OpenAI: 'not an additive fee')."""
    usage = {
        "input_tokens": 10000,
        "output_tokens": 100,
        "input_tokens_details": {"cached_tokens": 6000, "cache_write_tokens": 3000},
    }
    b = cost("openai", "gpt-5.6-sol", usage)
    assert {l.dimension: (l.quantity, l.rate) for l in b.lines} == {
        "input": (1000, Decimal("4.00")),
        "cached_input": (6000, Decimal("0.40")),
        "cache_write": (3000, Decimal("5.00")),
        "output": (100, Decimal("20.00")),
    }
    # 1000*4 + 6000*0.40 + 3000*5 + 100*20 = 4000 + 2400 + 15000 + 2000
    assert b.total == Decimal("0.0234")


def test_cache_write_no_premium_on_models_without_write_rate():
    """Before GPT-5.6 cache writes carry no charge: written tokens bill at
    the input rate, so reporting them changes nothing."""
    with_writes = {
        "input_tokens": 10000,
        "output_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4000},
    }
    without = {"input_tokens": 10000, "output_tokens": 0}
    assert (
        cost("openai", "gpt-5.1", with_writes).total
        == cost("openai", "gpt-5.1", without).total
        == Decimal("0.0125")
    )


def test_cache_write_chat_completions_shape_and_long_context():
    usage = {
        "prompt_tokens": 300000,
        "completion_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 300000},
    }
    b = cost("openai", "gpt-6-astra", usage)
    assert [(l.dimension, l.rate) for l in b.lines] == [
        ("cache_write", Decimal("25.00"))  # long-context write rate
    ]
    batch = cost("openai", "gpt-6-astra", usage, tier="batch")
    assert batch.lines[0].rate == Decimal("12.500")
