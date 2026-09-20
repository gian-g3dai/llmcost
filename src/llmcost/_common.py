"""Shared primitives for the engine and provider modules.

Lives in its own module so `llmcost.providers.*` can use `CostLine` and the
usage-field helpers without importing `llmcost.engine` (which imports the
providers for dispatch).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterator, List, Mapping, Optional, Tuple

#: Denominator for `per_1M_tokens` rates.
PER_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class CostLine:
    """One itemized charge: `quantity` units of `dimension` at `rate`.

    `rate` is the *effective* rate (batch multiplier already applied), in
    dollars per 1M tokens for `unit="per_1M_tokens"` lines and dollars per
    invocation for `unit="per_call"` lines. `amount` is exact `Decimal`.
    """

    dimension: str
    quantity: int
    rate: Decimal
    unit: str
    amount: Decimal


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read `name` from a mapping or an attribute-style object (SDK usage
    objects and their nested `*_details` work unmodified)."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def items_of(obj: Any) -> Iterator[Tuple[str, Any]]:
    """Iterate `(key, value)` pairs of a mapping or a simple object."""
    if obj is None:
        return iter(())
    if isinstance(obj, Mapping):
        return iter(obj.items())
    if hasattr(obj, "model_dump"):  # pydantic v2 SDK objects
        return iter(obj.model_dump().items())
    return iter(vars(obj).items())


class LineBuilder:
    """Accumulates `CostLine`s and `unpriced` dimensions under the shared
    conventions: zero-quantity dimensions are skipped; a dimension whose rate
    (after fallback) is unknown is billed zero and recorded as unpriced.
    Quantities are coerced to `int` (token and call counts are integral, but
    some usage payloads carry them as floats, e.g. after JSON round-trips)."""

    def __init__(self, multiplier: Decimal):
        self.multiplier = multiplier
        self.lines: List[CostLine] = []
        self.unpriced: List[str] = []

    def token_line(
        self, dimension: str, quantity: int, rate: Optional[Decimal]
    ) -> None:
        """Add a per-1M-tokens charge; `rate=None` means no applicable rate
        (the caller already exhausted its fallback chain)."""
        quantity = int(quantity or 0)
        if not quantity:
            return
        if rate is None:
            self.unpriced.append(dimension)
            return
        effective = rate * self.multiplier
        self.lines.append(
            CostLine(
                dimension=dimension,
                quantity=quantity,
                rate=effective,
                unit="per_1M_tokens",
                amount=effective * quantity / PER_MILLION,
            )
        )

    def call_line(
        self, dimension: str, quantity: int, rate: Optional[Decimal]
    ) -> None:
        """Add a per-invocation charge (server-side tools). The batch
        multiplier does not apply to per-call prices."""
        quantity = int(quantity or 0)
        if not quantity:
            return
        if rate is None:
            self.unpriced.append(dimension)
            return
        self.lines.append(
            CostLine(
                dimension=dimension,
                quantity=quantity,
                rate=rate,
                unit="per_call",
                amount=rate * quantity,
            )
        )

    def result(self) -> Tuple[List[CostLine], List[str]]:
        return self.lines, self.unpriced


__all__ = sorted(
    [
        "CostLine",
        "LineBuilder",
        "PER_MILLION",
        "get_field",
        "items_of",
    ]
)
assert __all__ == sorted(__all__)
