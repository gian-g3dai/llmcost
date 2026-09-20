"""Anthropic Messages API cost math.

Rate-key vocabulary (provider-defined): `input`, `output`, `cache_read`,
`cache_write_5m`, `cache_write_1h`; server-side tools under `per_call` (e.g.
`web_search`).

Usage-object mapping (Anthropic Messages API `usage` fields):

- `input_tokens` already **excludes** cached tokens — billed at `input`.
- `cache_read_input_tokens` billed at `cache_read`, defaulting to the
  `input` rate when absent (no discount, never free).
- Cache **writes** are billed per TTL bucket: the precise `cache_creation`
  `{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}` split is
  preferred when present; otherwise the flat `cache_creation_input_tokens`
  total is billed at the 5-minute rate. Both write rates default to the
  `input` rate when absent.
- `server_tool_use` counts (e.g. `web_search_requests: 3`) are billed per
  invocation from the entry's `per_call` table; a tool with no per-call
  price surfaces in `unpriced`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, List, Tuple

from .._common import CostLine, LineBuilder, get_field, items_of
from ..book import PriceEntry


def compute(
    usage: Any, entry: PriceEntry, multiplier: Decimal, tier: str = "standard"
) -> Tuple[List[CostLine], List[str]]:
    """Cost one Anthropic usage payload against a price entry.

    Args:
        usage: Anthropic usage object or plain dict.
        entry: Resolved price entry for the model.
        multiplier: Token-rate multiplier (1 for standard, the entry's
            `batch_multiplier` for batch requests).
        tier: Request tier; Anthropic's batch discount applies uniformly to
            every token dimension, so only `multiplier` is consulted.

    Returns:
        `(lines, unpriced)` — itemized charges and dimensions billed zero
        for lack of a rate.
    """
    input_tok = get_field(usage, "input_tokens") or 0
    output_tok = get_field(usage, "output_tokens") or 0
    cache_read = get_field(usage, "cache_read_input_tokens") or 0

    cache_creation = get_field(usage, "cache_creation")
    if cache_creation:
        write_5m = get_field(cache_creation, "ephemeral_5m_input_tokens") or 0
        write_1h = get_field(cache_creation, "ephemeral_1h_input_tokens") or 0
    else:
        # No TTL split reported: bill the flat total at the 5-minute rate.
        write_5m = get_field(usage, "cache_creation_input_tokens") or 0
        write_1h = 0

    rates = entry.rates
    input_rate = rates.get("input")

    def rate_or_input(key: str):
        rate = rates.get(key)
        return rate if rate is not None else input_rate

    b = LineBuilder(multiplier)
    b.token_line("input", input_tok, input_rate)
    b.token_line("output", output_tok, rates.get("output"))
    b.token_line("cache_read", cache_read, rate_or_input("cache_read"))
    b.token_line("cache_write_5m", write_5m, rate_or_input("cache_write_5m"))
    b.token_line("cache_write_1h", write_1h, rate_or_input("cache_write_1h"))

    for key, count in items_of(get_field(usage, "server_tool_use")):
        if not count:
            continue
        # `web_search_requests` -> per_call key `web_search`.
        tool = key[: -len("_requests")] if key.endswith("_requests") else key
        b.call_line(tool, count, entry.per_call.get(tool))

    return b.result()


__all__ = ["compute"]
