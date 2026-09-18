"""The cost engine: usage + prices -> itemized `CostBreakdown`.

Design invariants (see the project README for the full rationale):

- **Cost from provider-returned usage, never client-side estimation.** The
  usage object is authoritative and itemized.
- **`Decimal` end to end.** Invoice reconciliation to the cent is impossible
  with floats, so rates are `Decimal` from the moment JSON is parsed and
  amounts are never rounded by the engine.
- **`unpriced` is surfaced, never swallowed.** A usage dimension with tokens
  (or calls) but no applicable rate contributes zero dollars and is listed in
  `CostBreakdown.unpriced`. Silence is how trackers become unrealistic
  counters.
- **Best-effort on unknown models.** An unknown `(provider, model)` warns
  once per process (`RuntimeWarning`), contributes zero cost, and reports
  every observed dimension as unpriced. Token accounting elsewhere is never
  affected. An unknown *provider* raises `ValueError` instead — that is a
  caller bug, not a price-data gap.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from ._common import PER_MILLION, CostLine, get_field, items_of
from .book import (
    DateLike,
    JSONPriceBook,
    PriceBook,
    PriceEntry,
    entry_from_rates,
)
from .providers import anthropic, openai

#: Request tiers understood in v0.1. `"batch"` applies the price entry's
#: `batch_multiplier` to token rates (per-call tool prices are not
#: discounted); an entry without a `batch_multiplier` bills batch requests at
#: standard rates, per the no-discount convention.
_KNOWN_TIERS = ("standard", "batch")

_COMPUTE = {
    "anthropic": anthropic.compute,
    "openai": openai.compute,
}


@dataclass
class CostBreakdown:
    """Itemized cost of one request.

    Attributes:
        provider: Provider string the usage was costed under.
        model: Model identifier as passed by the caller (pre-aliasing).
        tier: Request tier the cost was computed for.
        lines: Per-dimension charges; `total` is their exact sum.
        unpriced: Dimensions present in the usage with nonzero quantity but
            no applicable rate — billed zero, never silently.
        priced: False when the `(provider, model)` pair was missing from the
            price book entirely (everything observed lands in `unpriced`).
    """

    provider: str
    model: str
    tier: str
    lines: List[CostLine] = field(default_factory=list)
    unpriced: List[str] = field(default_factory=list)
    priced: bool = True

    @property
    def total(self) -> Decimal:
        """Exact total in dollars (sum of `lines`)."""
        return sum((line.amount for line in self.lines), Decimal(0))


# Private process-wide default book, parsed lazily on first use.
_default_book: Optional[JSONPriceBook] = None

# (provider, model) pairs already warned about, so a missing price warns once
# rather than on every call in a loop.
_WARNED_PAIRS: Set[Tuple[str, str]] = set()


def _get_default_book() -> JSONPriceBook:
    global _default_book
    if _default_book is None:
        _default_book = JSONPriceBook()
    return _default_book


def _warn_once(provider: str, model: str) -> None:
    key = (provider, model)
    if key in _WARNED_PAIRS:
        return
    _WARNED_PAIRS.add(key)
    warnings.warn(
        f"No price entry for {provider!r} model {model!r}; its cost is "
        f"reported as zero with every dimension listed in "
        f"`CostBreakdown.unpriced`. Pass a custom `prices=` book to supply "
        f"rates.",
        RuntimeWarning,
        stacklevel=3,
    )


def _lookup_entry(
    book: Any, provider: str, model: str, at: DateLike
) -> Optional[PriceEntry]:
    """Resolve a price entry from a book, accepting both the llmcost
    `get_entry` protocol and the legacy `get_rates(provider, model)`
    protocol."""
    if hasattr(book, "get_entry"):
        return book.get_entry(provider, model, at)
    rates = book.get_rates(provider, model)
    return entry_from_rates(rates) if rates is not None else None


def cost(
    provider: str,
    model: str,
    usage: Any,
    *,
    tier: str = "standard",
    at: DateLike = None,
    prices: Optional[PriceBook] = None,
) -> CostBreakdown:
    """Compute the exact list-price cost of one request.

    Args:
        provider: Provider string, e.g. `"anthropic"` or `"openai"`.
            Unknown providers raise `ValueError`.
        model: Model identifier as returned by the API (dated/aliased IDs
            resolve through the book's alias table).
        usage: The provider-returned usage object — a plain dict or an SDK
            object (attributes are read directly; nested detail objects
            work either way).
        tier: `"standard"` (default) or `"batch"`. Batch applies the price
            entry's `batch_multiplier` to token rates.
        at: Request timestamp (ISO string, `date`, or `datetime`) used for
            effective-date resolution; `None` uses the latest prices.
        prices: Optional `PriceBook` override; defaults to the bundled book.
            Objects exposing only a legacy `get_rates(provider, model)`
            method are accepted.

    Returns:
        An itemized [`CostBreakdown`][llmcost.engine.CostBreakdown].
    """
    if tier not in _KNOWN_TIERS:
        raise ValueError(
            f"Unknown tier {tier!r}; expected one of {_KNOWN_TIERS}."
        )
    compute = _COMPUTE.get(provider)
    if compute is None:
        raise ValueError(
            f"Unsupported provider {provider!r}; supported: "
            f"{sorted(_COMPUTE)}."
        )

    book = prices if prices is not None else _get_default_book()
    entry = _lookup_entry(book, provider, model, at)

    if entry is None:
        _warn_once(provider, model)
        # Run the provider math against an empty entry: every observed
        # dimension has no rate, so all of them surface in `unpriced`.
        _, unpriced = compute(usage, PriceEntry(rates={}), Decimal(1))
        return CostBreakdown(
            provider=provider,
            model=model,
            tier=tier,
            unpriced=unpriced,
            priced=False,
        )

    multiplier = Decimal(1)
    if tier == "batch" and entry.batch_multiplier is not None:
        multiplier = entry.batch_multiplier

    lines, unpriced = compute(usage, entry, multiplier)
    return CostBreakdown(
        provider=provider, model=model, tier=tier, lines=lines, unpriced=unpriced
    )


def estimate(
    provider: str,
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_ttl: str = "5m",
    tier: str = "standard",
    at: DateLike = None,
    prices: Optional[PriceBook] = None,
) -> CostBreakdown:
    """Estimate a request's cost from token counts alone — no API call, no
    usage object. For planning, budgeting, and comparing models.

    The token semantics are uniform across providers so you can compare
    like for like:

    Args:
        input_tokens: Total prompt size, **including** any cached portion.
        output_tokens: Expected completion size (reasoning tokens included).
        cached_input_tokens: Portion of the prompt served from the prompt
            cache (billed at the provider's cache-read/cached-input rate).
        cache_write_tokens: Portion of the prompt newly written to the
            cache. Billed at the cache-write premium on Anthropic; on
            OpenAI cache writes cost nothing extra, so they simply remain
            regular input.
        cache_ttl: Cache TTL for the written tokens, `"5m"` (default) or
            `"1h"` — selects Anthropic's write rate; ignored for OpenAI.
        tier: `"standard"` or `"batch"`.
        at: Timestamp for effective-date price resolution.
        prices: Optional `PriceBook` override.

    Returns:
        A [`CostBreakdown`][llmcost.engine.CostBreakdown], same shape as
        [`cost()`][llmcost.engine.cost].

    This is an *estimate*: real bills depend on the token counts the
    provider reports. For accounting, use `cost()` with the returned usage
    object.
    """
    if cache_ttl not in ("5m", "1h"):
        raise ValueError(f"Unknown cache_ttl {cache_ttl!r}; expected '5m' or '1h'.")
    cached = min(cached_input_tokens, input_tokens)
    if provider == "anthropic":
        # Anthropic's `input_tokens` excludes cached reads and writes.
        writes = min(cache_write_tokens, input_tokens - cached)
        ttl_key = f"ephemeral_{cache_ttl}_input_tokens"
        usage: Dict[str, Any] = {
            "input_tokens": input_tokens - cached - writes,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cached,
            "cache_creation": {ttl_key: writes},
        }
    else:
        # OpenAI's `input_tokens` includes the cached portion; writes are
        # ordinary input.
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_tokens_details": {"cached_tokens": cached},
        }
    return cost(provider, model, usage, tier=tier, at=at, prices=prices)


class CostLedger:
    """Sums breakdowns across calls, with grouping by provider/model/tag.

    The generalization of per-adapter usage accumulation, decoupled from any
    framework: `add()` breakdowns as requests complete, read `total` or
    `total_by(...)` at any point, `clear()` between phases.
    """

    def __init__(self) -> None:
        self._records: List[Tuple[CostBreakdown, Dict[str, Any]]] = []

    def add(self, breakdown: CostBreakdown, **tags: Any) -> None:
        """Record one breakdown, optionally tagged (e.g. `run="train"`)."""
        self._records.append((breakdown, tags))

    def __len__(self) -> int:
        return len(self._records)

    @property
    def total(self) -> Decimal:
        """Exact ledger total in dollars."""
        return sum((b.total for b, _ in self._records), Decimal(0))

    def total_by(self, key: str = "model") -> Dict[Any, Decimal]:
        """Group totals by `"provider"`, `"model"` (-> `(provider, model)`
        keys), or any tag name passed to `add()` (untagged records group
        under `None`)."""
        out: Dict[Any, Decimal] = {}
        for breakdown, tags in self._records:
            if key == "provider":
                group: Any = breakdown.provider
            elif key == "model":
                group = (breakdown.provider, breakdown.model)
            else:
                group = tags.get(key)
            out[group] = out.get(group, Decimal(0)) + breakdown.total
        return out

    @property
    def unpriced(self) -> Set[Tuple[str, str, str]]:
        """Every `(provider, model, dimension)` that went unpriced — empty
        means the ledger total is complete at list price."""
        out: Set[Tuple[str, str, str]] = set()
        for breakdown, _ in self._records:
            for dim in breakdown.unpriced:
                out.add((breakdown.provider, breakdown.model, dim))
        return out

    def clear(self) -> None:
        """Drop all recorded breakdowns."""
        self._records.clear()


__all__ = sorted(
    [
        "CostBreakdown",
        "CostLedger",
        "CostLine",
        "PER_MILLION",
        "cost",
        "estimate",
        "get_field",
        "items_of",
    ]
)
assert __all__ == sorted(__all__)
