"""Price books: effective-dated `(provider, model)` -> rates lookup.

A price book does **lookup only** — it never turns tokens into dollars. The
token->dollar math is provider-specific and lives in `llmcost.providers.*`;
the book's job is to answer "what were the rates for this model at this
time?".

Two concrete books are provided:

- [`JSONPriceBook`][llmcost.book.JSONPriceBook] — backed by a JSON file, by
  default the bundled `prices/prices.json`. Understands both the v2
  effective-dated schema and a legacy flat schema.
- [`DictPriceBook`][llmcost.book.DictPriceBook] — backed by an in-memory
  `{provider: {model: rates}}` dict, for tests and overrides.

The rate-key vocabulary is **provider-defined**: Anthropic exposes
`cache_write_5m` / `cache_read`, OpenAI exposes `cached` / `cached_long`;
only the provider module that owns those keys reads them. A `None`-valued or
missing discount rate means "no discount" (billed at the base rate), never
"free".

All prices are parsed into `Decimal` so downstream cost math is exact.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

try:  # Python >= 3.8
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable

# Path to the bundled default price book (package data; see pyproject.toml
# `[tool.setuptools.package-data]`).
DEFAULT_PRICES_PATH: str = os.path.join(
    os.path.dirname(__file__), "prices", "prices.json"
)

#: A rate value: dollars per 1M tokens (or per call), or None for "no
#: discount defined" (bill at the base rate).
Rate = Optional[Decimal]

#: Date-like inputs accepted wherever a request timestamp is expected.
DateLike = Union[str, date, datetime, None]


def _to_decimal(value: Any) -> Rate:
    """Convert a JSON/Python number to `Decimal` (via `str` to avoid float
    artifacts); `None` passes through."""
    if value is None or isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _parse_date(value: DateLike) -> Optional[date]:
    """Normalize a date-like value (ISO string, `date`, `datetime`) to `date`."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


@dataclass(frozen=True)
class PriceTier:
    """A long-context repricing tier: `rates` overrides apply to the whole
    request once input tokens exceed `over_input_tokens`."""

    over_input_tokens: int
    rates: Mapping[str, Rate]


@dataclass(frozen=True)
class PriceEntry:
    """One effective-dated price record for a pricing key.

    Attributes:
        rates: Rate-key -> dollars per 1M tokens (provider-defined keys).
            A `None` value means "no discount" for that dimension.
        effective_from: First day these rates apply; `None` means "always"
            (undated books). `effective_to` is implied by the next entry.
        unit: Denomination of `rates`; only `per_1M_tokens` in v0.1.
        batch_multiplier: Multiplier applied to token rates for `tier="batch"`
            requests; `None` means no batch discount is defined (billed at
            standard rates, per the no-discount convention).
        batch_rates: Per-dimension exceptions to the batch multiplier, keyed
            like `rates`. A numeric value is the *final* batch rate for that
            dimension (the multiplier is not applied on top); `None` means
            the provider offers no batch discount for that dimension, so it
            falls back to the batch rate of its base dimension (e.g. OpenAI's
            `cached: null` bills cached tokens at the batch `input` rate).
            Ignored for standard-tier requests.
        per_call: Server-side tool name -> dollars per invocation.
        tiers: Long-context repricing tiers, ascending by threshold.
        fast_rates: Rate overrides applied when the request ran in fast
            mode (`speed="fast"`), e.g. Anthropic's premium Opus pricing.
            Empty means fast mode is not priced for this model.
        geo_multipliers: `inference_geo` value -> multiplier applied to
            every token rate (e.g. `{"us": 1.1}` for US-only inference on
            Claude 4.6+). Regions absent from the map are unpriced.
    """

    rates: Mapping[str, Rate]
    effective_from: Optional[date] = None
    unit: str = "per_1M_tokens"
    batch_multiplier: Optional[Decimal] = None
    batch_rates: Mapping[str, Rate] = field(default_factory=dict)
    per_call: Mapping[str, Decimal] = field(default_factory=dict)
    tiers: Tuple[PriceTier, ...] = ()
    fast_rates: Mapping[str, Rate] = field(default_factory=dict)
    geo_multipliers: Mapping[str, Decimal] = field(default_factory=dict)


@runtime_checkable
class PriceBook(Protocol):
    """Effective-dated lookup of a `(provider, model)` price entry.

    Structural (duck-typed) protocol — any object with a matching `get_entry`
    satisfies it. Objects exposing only the legacy-style
    `get_rates(provider, model)` are also accepted by
    [`cost()`][llmcost.engine.cost] directly.
    """

    def get_entry(
        self, provider: str, model: str, at: DateLike = None
    ) -> Optional[PriceEntry]:
        """Return the entry for `(provider, model)` effective at `at`
        (latest when `at` is None), or `None` if the pair is unknown."""
        ...


def _rates_from_raw(raw: Mapping[str, Any]) -> Dict[str, Rate]:
    return {k: _to_decimal(v) for k, v in raw.items()}


def entry_from_rates(raw: Mapping[str, Any]) -> PriceEntry:
    """Wrap a flat rate dict (legacy shape) into an undated
    `PriceEntry`."""
    return PriceEntry(rates=_rates_from_raw(raw))


def _entry_from_raw(raw: Mapping[str, Any]) -> PriceEntry:
    """Build a `PriceEntry` from a v2 schema entry dict."""
    tiers = tuple(
        PriceTier(
            over_input_tokens=int(t["over_input_tokens"]),
            rates=_rates_from_raw(t.get("rates", {})),
        )
        for t in raw.get("tiers", [])
    )
    per_call = {
        k: _to_decimal(v)
        for k, v in raw.get("per_call", {}).items()
        if v is not None
    }
    return PriceEntry(
        rates=_rates_from_raw(raw.get("rates", {})),
        effective_from=_parse_date(raw.get("effective_from")),
        unit=raw.get("unit", "per_1M_tokens"),
        batch_multiplier=_to_decimal(raw.get("batch_multiplier")),
        batch_rates=_rates_from_raw(raw.get("batch", {})),
        per_call=per_call,
        tiers=tuple(sorted(tiers, key=lambda t: t.over_input_tokens)),
        fast_rates=_rates_from_raw(raw.get("fast", {})),
        geo_multipliers={
            k: _to_decimal(v)
            for k, v in raw.get("inference_geo", {}).items()
            if v is not None
        },
    )


def _resolve(entries: List[PriceEntry], at: DateLike) -> PriceEntry:
    """Pick the entry effective at `at`.

    Entries are sorted ascending by `effective_from` (undated first). `at=None`
    means "latest". A timestamp earlier than every entry clamps to the oldest
    entry — best-effort by design; the book's history starts somewhere.
    """
    if at is None:
        return entries[-1]
    at_date = _parse_date(at)
    chosen = entries[0]
    for entry in entries:
        if entry.effective_from is None or entry.effective_from <= at_date:
            chosen = entry
    return chosen


class _BaseBook:
    """Shared alias resolution + effective-date lookup over normalized data.

    `self._providers` maps provider -> (aliases, {pricing_key: [entries]}).
    """

    _providers: Dict[str, Tuple[Dict[str, str], Dict[str, List[PriceEntry]]]]

    def get_entry(
        self, provider: str, model: str, at: DateLike = None
    ) -> Optional[PriceEntry]:
        section = self._providers.get(provider)
        if not section:
            return None
        aliases, pricing = section
        # A model absent from the alias table is its own pricing key.
        pricing_key = aliases.get(model, model)
        entries = pricing.get(pricing_key)
        if not entries:
            return None
        return _resolve(entries, at)

    def get_rates(self, provider: str, model: str) -> Optional[Dict[str, float]]:
        """Legacy-protocol lookup: latest rates as plain floats
        (`None` values preserved), or `None` if the pair is unknown."""
        entry = self.get_entry(provider, model)
        if entry is None:
            return None
        return {
            k: (float(v) if v is not None else None) for k, v in entry.rates.items()
        }


class JSONPriceBook(_BaseBook):
    """[`PriceBook`][llmcost.book.PriceBook] backed by a JSON file.

    Accepts two schemas, detected automatically:

    - **v2** (`schema_version: 2`): `{"providers": {provider: {"models":
      {alias: key}, "pricing": {key: [entry, ...]}}}}` where each entry is
      effective-dated and may carry `batch_multiplier`, `batch`, `per_call`,
      `tiers`, `fast`, `inference_geo`.
    - **legacy flat**: `{provider: {"models": ...,
      "pricing": {key: {rate_key: price}}}}` — each flat rate dict becomes a
      single undated entry.

    The file is read and parsed once at construction, with all prices as
    `Decimal` (floats never enter the pipeline: JSON numbers are parsed
    straight to `Decimal`).
    """

    def __init__(self, path: str = DEFAULT_PRICES_PATH):
        with open(path, "r") as f:
            data = json.load(f, parse_float=Decimal)
        raw_providers = (
            data["providers"] if data.get("schema_version", 1) >= 2 else data
        )
        self._providers = {}
        for provider, section in raw_providers.items():
            aliases = dict(section.get("models", {}))
            pricing: Dict[str, List[PriceEntry]] = {}
            for key, value in section.get("pricing", {}).items():
                if isinstance(value, Mapping):  # legacy flat rate dict
                    entries = [entry_from_rates(value)]
                else:  # v2 list of dated entries
                    entries = sorted(
                        (_entry_from_raw(e) for e in value),
                        key=lambda e: e.effective_from or date.min,
                    )
                pricing[key] = entries
            self._providers[provider] = (aliases, pricing)


class DictPriceBook(_BaseBook):
    """[`PriceBook`][llmcost.book.PriceBook] backed by an in-memory dict, for
    tests and overrides.

    The data shape is the direct `{provider: {model: rates}}` mapping — no
    `models`/`pricing` alias indirection and no dates, so it's terse inline:

        DictPriceBook({"anthropic": {"claude-opus-5": {"input": 5.0, ...}}})
    """

    def __init__(self, data: Mapping[str, Mapping[str, Mapping[str, Any]]]):
        self._providers = {
            provider: (
                {},
                {model: [entry_from_rates(rates)] for model, rates in models.items()},
            )
            for provider, models in data.items()
        }


__all__ = sorted(
    [
        "DEFAULT_PRICES_PATH",
        "DictPriceBook",
        "JSONPriceBook",
        "PriceBook",
        "PriceEntry",
        "PriceTier",
        "entry_from_rates",
    ]
)
assert __all__ == sorted(__all__)
