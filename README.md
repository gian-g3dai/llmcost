# llmcost

**Price estimator and exact cost tracker for OpenAI and Anthropic API
usage.** Know what a run will cost before you send it, and what it actually
cost after — down to the cent, at list price.

Most cost calculators do `input_tokens × rate + output_tokens × rate` and
get real bills badly wrong, because they ignore what actually dominates
them: prompt-cache reads (up to 10× cheaper) and cache writes (billed at a
premium on Anthropic), batch discounts, long-context repricing, reasoning
tokens, server-side tools, and prices that change over time. `llmcost`
handles all of it, with `Decimal` arithmetic end to end, and **surfaces
anything it can't price instead of silently dropping it**.

## Install

```bash
pip install llmcost
```

Zero required dependencies. Python ≥ 3.9.

## Estimate before you spend

No API call needed — just token counts. Great for budgeting, capacity
planning, and "which model is cheaper for *my* workload?":

```python
from llmcost import estimate

# A cache-heavy agent turn: 50k prompt, 40k of it cached, ~2k out
estimate("openai", "gpt-5.1", input_tokens=50_000,
         cached_input_tokens=40_000, output_tokens=2_000).total
# Decimal('0.0377')

estimate("anthropic", "claude-sonnet-5", input_tokens=50_000,
         cached_input_tokens=40_000, output_tokens=2_000).total
# Decimal('0.048')

# Same workload on the Batch API
estimate("openai", "gpt-5.1", input_tokens=50_000,
         cached_input_tokens=40_000, output_tokens=2_000, tier="batch").total
```

The result is an itemized breakdown, not a bare number — you see exactly
where the money goes:

```python
for line in estimate("anthropic", "claude-sonnet-5", input_tokens=100_000,
                     cache_write_tokens=90_000, output_tokens=500).lines:
    print(line.dimension, line.quantity, line.rate, line.amount)
# input          10000   2.00   0.02
# output           500  10.00   0.005
# cache_write_5m 90000   2.50   0.225
```

## Estimate a whole conversation

Cache-heavy multi-turn workloads are where naive estimators go ~10× wrong,
because turn *i*'s cost depends on what the cache did on turns 0..*i-1*.
`estimate_conversation()` models it: turn 1 writes the prefix (at
Anthropic's write premium), later turns read the growing conversation at
the cache-read discount and write only the new tail:

```python
from llmcost import estimate_conversation

est = estimate_conversation(
    "anthropic", "claude-haiku-4-5",
    turns=3,
    system_tokens=10_000,          # system prompt + tools, on every turn
    user_tokens_per_turn=1_000,    # or a per-turn list: [1_000, 500, ...]
    output_tokens_per_turn=2_000,  # replies feed back into later prompts
)
est.total                          # Decimal('0.05375')
[t.total for t in est.turns]       # per-turn breakdowns

# The same conversation without caching costs 34% more:
estimate_conversation("anthropic", "claude-haiku-4-5", turns=3,
                      system_tokens=10_000, user_tokens_per_turn=1_000,
                      output_tokens_per_turn=2_000, cache_ttl=None).total
# Decimal('0.072')
```

It's honest about failure modes too: pass `turn_interval_seconds` and if
your turns arrive slower than the cache TTL (`"5m"` or `"1h"`), every turn
misses and rewrites its prompt — which on Anthropic is *more* expensive
than not caching at all, and the estimate will say so.

## Exact cost from a response

After the call, pass the provider's usage object as-is (SDK object or
dict) — this is authoritative, not an estimate:

```python
from llmcost import cost

response = client.messages.create(...)          # or OpenAI responses.create
breakdown = cost("anthropic", response.model, response.usage)

breakdown.total      # Decimal — exact dollars, no float drift
breakdown.lines      # itemized: dimension, quantity, effective rate, amount
breakdown.unpriced   # anything with no rate (billed zero — never silently)
```

Precision needs request context the usage object doesn't carry:

```python
cost("openai", model, usage,
     tier="batch",        # Batch API discount
     at="2026-03-01")     # cost at the rates in force on that date
```

## Track spend across calls

```python
from llmcost import CostLedger

ledger = CostLedger()
ledger.add(cost("openai", model, usage), user="alice")
ledger.total                  # exact Decimal total
ledger.total_by("model")      # {(provider, model): Decimal}
ledger.total_by("user")       # group by any tag you passed to add()
ledger.unpriced               # every (provider, model, dimension) billed zero
```

## What's handled

| Dimension | Behavior |
|---|---|
| Prompt-cache **reads** | Billed at the provider's discounted rate (e.g. 0.1× on Anthropic); no rate defined → billed at the input rate, never free |
| Prompt-cache **writes** (Anthropic) | Billed per TTL bucket from the `cache_creation` 5m/1h split; flat totals fall back to the 5m rate |
| Cached input (OpenAI) | `input_tokens_details.cached_tokens` at the `cached` rate; a `null` rate means no discount |
| **Batch API** | `tier="batch"` applies the model's batch multiplier to token rates (per-call tool prices are not discounted) |
| **Long-context tiers** | The whole request repriced when input tokens cross the threshold (e.g. 272k on OpenAI) |
| **Reasoning tokens** | Billed as output, itemized on their own line, never double-billed |
| **Server-side tools** | Per-invocation pricing (e.g. Anthropic `server_tool_use`) |
| **Repricing over time** | Effective-dated price entries; pass `at=` to cost a run at last month's rates |

## Scope: exact at list price

Costs from `cost()` are exact **at list price**. Negotiated discounts,
credits, free tiers, and provisioned throughput are unknowable from the
outside — if you have custom pricing, supply your own price book:

```python
from llmcost import DictPriceBook

book = DictPriceBook({"anthropic": {"claude-opus-5": {"input": 4.0, "output": 20.0}}})
cost("anthropic", "claude-opus-5", usage, prices=book)
```

`estimate()` is additionally approximate in the obvious way: it trusts your
token counts. Use `cost()` with the real usage object for accounting.

## The price book

Prices ship as effective-dated JSON package data
(`src/llmcost/prices/prices.json`), also consumable directly by non-Python
tools. Price changes land as dated new entries — historical entries are
never edited, so old runs cost correctly forever. Timestamps before a
model's first entry clamp to its oldest entry (best-effort, by design).
Dated model IDs (`gpt-5.1-2025-11-13`, `claude-haiku-4-5-20251001`) resolve
to their pricing automatically.

## Using llmcost from a framework

`llmcost` is framework-agnostic and easy to embed: the core is pure
functions plus bundled JSON, with no required dependencies. Price books are
duck-typed — any object with a `get_entry(provider, model, at)` method, or
even just a flat `get_rates(provider, model)` method, works as a `prices=`
override — and simple flat price files (`{provider: {"pricing": {model:
rates}}}`) load unchanged, so an existing in-house price table can usually
be dropped in as-is.

## Roadmap

- **v0.2** — Gemini provider; first documented invoice-reconciliation run
- **v0.3** — streaming usage helpers; price-drift CI
- **v0.4** — published raw price JSON endpoint; community price-update workflow
- **Later** — estimation from message text (tokenizer-based, optional
  extras); more providers

## Contributing

Price updates are a 5-minute task: a dated new entry plus a link to the
provider's pricing page. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0
