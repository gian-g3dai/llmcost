"""OpenAI (Responses / Chat Completions) cost math.

Rate-key vocabulary (provider-defined): `input`, `cached`, `cache_write`,
`output`, with
long-context repricing expressed either as schema-v2 `tiers` entries or as
the legacy flat keys `input_long` / `cached_long` / `output_long`
(threshold 272k tokens).

Usage-object mapping. Both OpenAI usage shapes are accepted and
normalized to the same lines:

- **Responses API**: `input_tokens`, `input_tokens_details.cached_tokens`,
  `input_tokens_details.cache_write_tokens`, `output_tokens`,
  `output_tokens_details.reasoning_tokens`.
- **Chat Completions API**: `prompt_tokens`,
  `prompt_tokens_details.cached_tokens`,
  `prompt_tokens_details.cache_write_tokens`, `completion_tokens`,
  `completion_tokens_details.reasoning_tokens`.

Field semantics:

- `input_tokens` (`prompt_tokens`) **includes** cached and cache-written
  tokens; the cached portion is `input_tokens_details.cached_tokens`
  (`prompt_tokens_details.cached_tokens`), the written portion is
  `input_tokens_details.cache_write_tokens`
  (`prompt_tokens_details.cache_write_tokens`), and the uncached remainder
  (`max(input - cached - written, 0)`) is billed at `input`.
- Cached tokens are billed at `cached`; a `null`/missing `cached` rate means
  "no cache discount" — billed at the `input` rate, never free.
- Cache-written tokens are billed at `cache_write` *instead of* `input`
  (OpenAI's write pricing is a replacement rate, not an additive fee:
  1.25x input on GPT-5.6 and later). A `null`/missing `cache_write` rate
  means "no write premium" — billed at the `input` rate, as on models
  before GPT-5.6.
- Long-context tiers reprice the **whole request** (input, cached, output)
  once `input_tokens` exceeds the threshold.
- Batch requests multiply every token rate by the entry's
  `batch_multiplier`, except dimensions listed in the entry's `batch`
  exceptions: OpenAI's Batch API offers no cached-input discount on several
  models (`"batch": {"cached": null}`), so their cached tokens bill at the
  batch `input` rate.
- Reasoning tokens (`output_tokens_details.reasoning_tokens` /
  `completion_tokens_details.reasoning_tokens`) are billed as output; they
  are itemized on their own line for visibility but never double-billed (the
  `output` line covers only the non-reasoning remainder).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .._common import CostLine, LineBuilder, get_field
from ..book import PriceEntry

# Default long-context threshold when the price entry uses the legacy flat
# `*_long` rate keys instead of explicit `tiers`.
LONG_CONTEXT_THRESHOLD = 272_000

_MISSING = object()


def _first_field(usage: Any, *names: str) -> Any:
    """Return the first of `names` present on `usage` (Responses-API name
    first, then its Chat Completions synonym). A present-but-`None` field
    counts as present, so a Responses payload never falls through to a
    stale Chat Completions field and vice versa."""
    for name in names:
        value = get_field(usage, name, _MISSING)
        if value is not _MISSING:
            return value
    return None


def _effective_rates(
    entry: PriceEntry, input_tok: int
) -> Dict[str, Optional[Decimal]]:
    """Resolve the rate set for this request, applying long-context
    repricing when the prompt crosses a threshold."""
    rates = dict(entry.rates)
    if entry.tiers:
        # Cumulative overlay, ascending by threshold: the highest tier the
        # prompt exceeds wins.
        for tier in entry.tiers:
            if input_tok > tier.over_input_tokens:
                rates.update(tier.rates)
        return rates
    if input_tok > LONG_CONTEXT_THRESHOLD and "input_long" in rates:
        return {
            "input": rates.get("input_long"),
            "cached": rates.get("cached_long"),
            "output": rates.get("output_long"),
        }
    return rates


def _final_rates(
    rates: Dict[str, Optional[Decimal]],
    entry: PriceEntry,
    multiplier: Decimal,
    tier: str,
) -> Dict[str, Optional[Decimal]]:
    """Apply the tier discount, honoring per-dimension `batch_rates`
    exceptions: an explicit batch rate replaces `rate * multiplier`, and a
    `None` batch rate means "no batch discount for this dimension" (the
    caller's fallback chain then bills it at its base dimension's rate)."""
    out: Dict[str, Optional[Decimal]] = {}
    for key, rate in rates.items():
        if tier == "batch" and key in entry.batch_rates:
            out[key] = entry.batch_rates[key]
        else:
            out[key] = rate * multiplier if rate is not None else None
    return out


def compute(
    usage: Any, entry: PriceEntry, multiplier: Decimal, tier: str = "standard"
) -> Tuple[List[CostLine], List[str]]:
    """Cost one OpenAI usage payload against a price entry.

    Args:
        usage: OpenAI usage object or plain dict, in either the Responses
            API or the Chat Completions API shape.
        entry: Resolved price entry for the model.
        multiplier: Token-rate multiplier (1 for standard, the entry's
            `batch_multiplier` for batch requests).
        tier: Request tier; selects the entry's `batch_rates` exceptions
            (OpenAI drops the cached-input discount on the Batch API for
            some models, which a plain multiplier cannot express).

    Returns:
        `(lines, unpriced)` — itemized charges and dimensions billed zero
        for lack of a rate.
    """
    input_tok = _first_field(usage, "input_tokens", "prompt_tokens") or 0
    output_tok = _first_field(usage, "output_tokens", "completion_tokens") or 0

    input_details = _first_field(
        usage, "input_tokens_details", "prompt_tokens_details"
    )
    cached = get_field(input_details, "cached_tokens") or 0
    written = get_field(input_details, "cache_write_tokens") or 0
    written = min(written, max(input_tok - cached, 0))
    uncached = max(input_tok - cached - written, 0)

    output_details = _first_field(
        usage, "output_tokens_details", "completion_tokens_details"
    )
    reasoning = get_field(output_details, "reasoning_tokens") or 0
    reasoning = min(reasoning, output_tok)

    rates = _final_rates(_effective_rates(entry, input_tok), entry, multiplier, tier)
    input_rate = rates.get("input")
    cached_rate = rates.get("cached")
    if cached_rate is None:  # null/missing cached rate: no cache discount
        cached_rate = input_rate
    write_rate = rates.get("cache_write")
    if write_rate is None:  # null/missing write rate: no write premium
        write_rate = input_rate
    output_rate = rates.get("output")

    # Rates are already final (tier discount applied above), so the builder
    # must not multiply again.
    b = LineBuilder(Decimal(1))
    b.token_line("input", uncached, input_rate)
    b.token_line("cached_input", cached, cached_rate)
    b.token_line("cache_write", written, write_rate)
    b.token_line("output", output_tok - reasoning, output_rate)
    # Reasoning is billed as output; itemized separately, not double-billed.
    b.token_line("reasoning", reasoning, output_rate)
    return b.result()


__all__ = ["LONG_CONTEXT_THRESHOLD", "compute"]
