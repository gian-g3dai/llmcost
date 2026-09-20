"""Multi-turn conversation cost estimation.

Cache-heavy conversational workloads — agents, chatbots, RAG loops — are
exactly where naive per-token estimators go ~10x wrong, because the cost of
turn *i* depends on what the prompt cache did on turns 0..i-1.
[`estimate_conversation()`][llmcost.conversation.estimate_conversation]
models those dynamics with a deliberately simple, documented set of
assumptions:

- The prompt at turn *i* is the system/tool prefix, plus every earlier
  user message and assistant reply, plus turn *i*'s user message.
- With caching enabled, each turn's full prompt is written to the cache
  (breakpoint at the end of the prompt), so turn *i* reads everything
  turn *i-1* wrote and writes only the new tail (the previous reply plus
  the new user message). Turn 0 writes the whole prompt.
- The cache survives between consecutive turns unless `turn_interval`
  exceeds the TTL, in which case **every** turn misses and rewrites its
  full prompt — on Anthropic that is *more* expensive than not caching at
  all (writes carry a premium), and the estimate will honestly say so.
- Provider billing differences are inherited from
  [`estimate()`][llmcost.engine.estimate]: Anthropic bills writes at the
  TTL's premium rate and reads at the discounted rate; OpenAI reads bill
  at the `cached` rate and writes at `cache_write` where the model defines
  one (GPT-5.6 and later), otherwise at the plain input rate.

This is an estimate: real conversations have variable message sizes and
providers apply their own cache-eligibility rules (e.g. minimum prefix
sizes, block-size rounding). For accounting, cost each real response with
[`cost()`][llmcost.engine.cost].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional, Sequence, Set, Tuple, Union

from .book import DateLike, PriceBook
from .engine import CostBreakdown, estimate

_TTL_SECONDS = {"5m": 300, "1h": 3600}

#: A per-turn token count: one int for every turn, or a sequence with one
#: value per turn.
PerTurn = Union[int, Sequence[int]]


def _per_turn(value: PerTurn, turns: int, name: str) -> List[int]:
    if isinstance(value, int):
        return [value] * turns
    values = list(value)
    if len(values) != turns:
        raise ValueError(
            f"{name} has {len(values)} values for {turns} turns."
        )
    return values


@dataclass
class ConversationEstimate:
    """Per-turn breakdowns for an estimated conversation.

    `turns[i]` is the [`CostBreakdown`][llmcost.engine.CostBreakdown] of
    turn *i*; `total` is their exact sum.
    """

    provider: str
    model: str
    tier: str
    turns: List[CostBreakdown] = field(default_factory=list)

    @property
    def total(self) -> Decimal:
        """Exact conversation total in dollars."""
        return sum((t.total for t in self.turns), Decimal(0))

    @property
    def unpriced(self) -> Set[str]:
        """Union of unpriced dimensions across all turns."""
        out: Set[str] = set()
        for t in self.turns:
            out.update(t.unpriced)
        return out


def estimate_conversation(
    provider: str,
    model: str,
    *,
    turns: int,
    system_tokens: int = 0,
    user_tokens_per_turn: PerTurn = 0,
    output_tokens_per_turn: PerTurn = 0,
    cache_ttl: Optional[str] = "5m",
    turn_interval_seconds: Optional[float] = None,
    tier: str = "standard",
    at: DateLike = None,
    prices: Optional[PriceBook] = None,
) -> ConversationEstimate:
    """Estimate the cost of a multi-turn conversation, cache dynamics
    included.

    Args:
        turns: Number of request/response rounds.
        system_tokens: Fixed prefix present on every turn (system prompt,
            tool definitions, RAG corpus...).
        user_tokens_per_turn: New user-message tokens per turn — one int
            for all turns, or a sequence with one value per turn.
        output_tokens_per_turn: Assistant-reply tokens per turn (same
            shapes). Replies are appended to the conversation, so they
            become input on every later turn.
        cache_ttl: `"5m"` (default) or `"1h"` to model prompt caching at
            that TTL; `None` disables caching (every prompt billed as
            plain input).
        turn_interval_seconds: Expected time between turns. When it
            exceeds the TTL, every turn misses the cache and rewrites its
            full prompt — the honest cost of a cache outlived by its
            conversation.
        tier: `"standard"` or `"batch"`.
        at: Timestamp for effective-date price resolution.
        prices: Optional `PriceBook` override.

    Returns:
        A [`ConversationEstimate`][llmcost.conversation.ConversationEstimate]
        with one breakdown per turn.
    """
    if turns < 0:
        raise ValueError(f"turns must be >= 0, got {turns}.")
    if cache_ttl is not None and cache_ttl not in _TTL_SECONDS:
        raise ValueError(
            f"Unknown cache_ttl {cache_ttl!r}; expected '5m', '1h', or None."
        )
    users = _per_turn(user_tokens_per_turn, turns, "user_tokens_per_turn")
    outputs = _per_turn(output_tokens_per_turn, turns, "output_tokens_per_turn")

    caching = cache_ttl is not None
    expires = (
        caching
        and turn_interval_seconds is not None
        and turn_interval_seconds > _TTL_SECONDS[cache_ttl]
    )

    result = ConversationEstimate(provider=provider, model=model, tier=tier)
    history = 0  # user+output tokens accumulated from earlier turns
    prev_prompt = 0  # what the previous turn wrote to the cache
    for i in range(turns):
        prompt = system_tokens + history + users[i]
        if not caching:
            cached, writes = 0, 0
        elif expires:
            cached, writes = 0, prompt  # full rewrite every turn
        else:
            cached = prev_prompt  # everything the previous turn wrote
            writes = prompt - cached  # previous reply + new user message
        result.turns.append(
            estimate(
                provider,
                model,
                input_tokens=prompt,
                output_tokens=outputs[i],
                cached_input_tokens=cached,
                cache_write_tokens=writes,
                cache_ttl=cache_ttl or "5m",
                tier=tier,
                at=at,
                prices=prices,
            )
        )
        history += users[i] + outputs[i]
        prev_prompt = prompt
    return result


__all__ = sorted(["ConversationEstimate", "estimate_conversation"])
assert __all__ == sorted(__all__)
