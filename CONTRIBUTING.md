# Contributing to llmcost

## Price updates (the most valuable contribution)

Prices going stale is the failure mode that kills trust in this project.
A price update is a small, dated PR:

1. Edit `src/llmcost/prices/prices.json`.
2. **Never edit an existing entry** — append a new entry to the model's list
   with `effective_from` set to the day the provider's new pricing took
   effect. Old entries stay so historical runs keep costing correctly.
3. For a **new model**, add a pricing key (and `models` aliases for dated
   IDs). Include every billing dimension the provider documents; set a
   discount rate to `null` only when the provider genuinely offers no
   discount for that dimension (null means "billed at the base rate", not
   "free").
4. In the PR description, **link the provider's pricing page** and quote the
   numbers you transcribed.
5. Run the tests: `pip install -e .[testing] && pytest`.

Price-only changes are released liberally as patch versions.

### Entry template

```jsonc
"model-pricing-key": [
  {
    "effective_from": "YYYY-MM-DD",
    "unit": "per_1M_tokens",
    "rates": { "input": 0.0, "output": 0.0 },
    "batch_multiplier": 0.5,          // omit if no batch discount
    "batch": { "cached": null },      // per-dimension batch exceptions: null =
                                      //   no batch discount for that dimension
    "per_call": { "web_search": 0.01 }, // server-side tools, $/invocation (0 = free)
    "fast": { "input": 10.0, ... },     // fast-mode rate overrides, if offered
    "inference_geo": { "us": 1.1 },     // region multipliers on all token rates
    "tiers": [                          // long-context repricing, if any
      { "over_input_tokens": 272000, "rates": { "input": 0.0, "output": 0.0 } }
    ]
  }
]
```

## Code contributions

- Python ≥ 3.9, no required runtime dependencies in the core.
- `black` formatting, 88-char lines; keep `__all__` lists sorted (asserted
  at import time).
- Every billing-dimension behavior needs a golden test with the expected
  value computed by hand and the provider's pricing page cited in the test
  docstring.
- Provider modules own their rate-key vocabulary; don't force a common
  shape across providers.

## Reconciliation runs

The claim "precise" must be falsifiable. If you can, re-run a scripted
workload with known composition against a real API after a price-book
update and compare the ledger total with the provider's billing dashboard
export. Open an issue with the procedure and results either way.
