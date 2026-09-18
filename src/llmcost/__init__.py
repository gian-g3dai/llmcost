"""llmcost — price estimator and exact cost tracker for OpenAI and
Anthropic API usage.

Two entry points, one itemized `CostBreakdown`:

- [`estimate()`][llmcost.engine.estimate] — before you spend: cost from
  token counts alone, for planning, budgeting, and comparing models.
- [`cost()`][llmcost.engine.cost] — after the call: exact list-price cost
  from the provider-returned usage object.

Both understand the dimensions that dominate real bills — prompt-cache
reads and TTL-priced cache writes, batch discounts, long-context tiers,
reasoning tokens, server-side tools, and repricing over time — with
`Decimal` arithmetic throughout, and anything unpriceable surfaced, never
swallowed.

Quickstart:

    from llmcost import cost, estimate

    # What would this run cost?
    estimate("openai", "gpt-5.1", input_tokens=50_000,
             cached_input_tokens=40_000, output_tokens=2_000).total

    # What did this request cost, exactly?
    breakdown = cost("anthropic", response.model, response.usage)
    breakdown.total     # Decimal — exact dollars
    breakdown.lines     # itemized per billing dimension
    breakdown.unpriced  # dimensions with no rate (billed zero, surfaced)
"""

from ._common import CostLine
from .book import (
    DEFAULT_PRICES_PATH,
    DictPriceBook,
    JSONPriceBook,
    PriceBook,
    PriceEntry,
    PriceTier,
)
from .conversation import ConversationEstimate, estimate_conversation
from .engine import CostBreakdown, CostLedger, cost, estimate

__version__ = "0.1.0"

__all__ = sorted(
    [
        "ConversationEstimate",
        "CostBreakdown",
        "CostLedger",
        "CostLine",
        "DEFAULT_PRICES_PATH",
        "DictPriceBook",
        "JSONPriceBook",
        "PriceBook",
        "PriceEntry",
        "PriceTier",
        "cost",
        "estimate",
        "estimate_conversation",
    ]
)
assert __all__ == sorted(__all__)
