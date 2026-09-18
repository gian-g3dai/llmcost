"""Provider-specific cost math.

Each module owns its provider's rate-key vocabulary and usage-object shape,
and exposes `compute(usage, entry, multiplier) -> (lines, unpriced)`. Only
the module that owns a provider reads that provider's rate keys — there is
deliberately no forced common shape across providers.
"""

from . import anthropic, openai

__all__ = sorted(["anthropic", "openai"])
assert __all__ == sorted(__all__)
