"""OpenAI (Responses / Chat Completions) cost math.

Rate-key vocabulary (provider-defined): `input`, `cached`, `output`, with
long-context repricing expressed either as schema-v2 `tiers` entries or as
the legacy flat keys `input_long` / `cached_long` / `output_long`
(threshold 272k tokens).

Usage-object mapping (OpenAI `usage` fields):

- `input_tokens` **includes** cached tokens; the cached portion is
  `input_tokens_details.cached_tokens`, and the uncached remainder
  (`max(input - cached, 0)`) is billed at `input`.
- Cached tokens are billed at `cached`; a `null`/missing `cached` rate means
  "no cache discount" — billed at the `input` rate, never free.
- Long-context tiers reprice the **whole request** (input, cached, output)
  once `input_tokens` exceeds the threshold.
- Reasoning tokens (`output_tokens_details.reasoning_tokens`) are billed as
  output; they are itemized on their own line for visibility but never
  double-billed (the `output` line covers only the non-reasoning remainder).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .._common import CostLine, LineBuilder, get_field
from ..book import PriceEntry

# Default long-context threshold when the price entry uses the legacy flat
# `*_long` rate keys instead of explicit `tiers`.
LONG_CONTEXT_THRESHOLD = 272_000


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


def compute(
    usage: Any, entry: PriceEntry, multiplier: Decimal
) -> Tuple[List[CostLine], List[str]]:
    """Cost one OpenAI usage payload against a price entry.

    Args:
        usage: OpenAI usage object or plain dict.
        entry: Resolved price entry for the model.
        multiplier: Token-rate multiplier (1 for standard, the entry's
            `batch_multiplier` for batch requests).

    Returns:
        `(lines, unpriced)` — itemized charges and dimensions billed zero
        for lack of a rate.
    """
    input_tok = get_field(usage, "input_tokens") or 0
    output_tok = get_field(usage, "output_tokens") or 0

    input_details = get_field(usage, "input_tokens_details")
    cached = get_field(input_details, "cached_tokens") or 0
    uncached = max(input_tok - cached, 0)

    output_details = get_field(usage, "output_tokens_details")
    reasoning = get_field(output_details, "reasoning_tokens") or 0
    reasoning = min(reasoning, output_tok)

    rates = _effective_rates(entry, input_tok)
    input_rate = rates.get("input")
    cached_rate = rates.get("cached")
    if cached_rate is None:  # null/missing cached rate: no cache discount
        cached_rate = input_rate
    output_rate = rates.get("output")

    b = LineBuilder(multiplier)
    b.token_line("input", uncached, input_rate)
    b.token_line("cached_input", cached, cached_rate)
    b.token_line("output", output_tok - reasoning, output_rate)
    # Reasoning is billed as output; itemized separately, not double-billed.
    b.token_line("reasoning", reasoning, output_rate)
    return b.result()


__all__ = ["LONG_CONTEXT_THRESHOLD", "compute"]
