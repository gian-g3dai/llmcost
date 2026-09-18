"""Price book: aliasing, effective-date resolution, schema handling, and the
missing-rate ("no discount, never free") conventions."""

import json
from datetime import date, datetime
from decimal import Decimal

from llmcost import (
    DEFAULT_PRICES_PATH,
    DictPriceBook,
    JSONPriceBook,
    PriceBook,
    cost,
)

DATED_BOOK = {
    "schema_version": 2,
    "providers": {
        "openai": {
            "models": {"gpt-x-2026-01-15": "gpt-x"},
            "pricing": {
                "gpt-x": [
                    {
                        "effective_from": "2026-01-15",
                        "rates": {"input": 1.25, "cached": 0.125, "output": 10.00},
                    },
                    {
                        "effective_from": "2026-06-01",
                        "rates": {"input": 1.00, "cached": 0.10, "output": 8.00},
                    },
                ]
            },
        }
    },
}


def _write_book(tmp_path, data):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(data))
    return JSONPriceBook(str(path))


def test_alias_resolution(tmp_path):
    book = _write_book(tmp_path, DATED_BOOK)
    dated = book.get_entry("openai", "gpt-x-2026-01-15")
    base = book.get_entry("openai", "gpt-x")
    assert dated == base
    assert dated.rates["input"] == Decimal("1.00")


def test_unknown_pairs_return_none(tmp_path):
    book = _write_book(tmp_path, DATED_BOOK)
    assert book.get_entry("openai", "gpt-unknown") is None
    assert book.get_entry("mistral", "gpt-x") is None


def test_effective_date_resolution(tmp_path):
    book = _write_book(tmp_path, DATED_BOOK)
    # No timestamp -> latest entry.
    assert book.get_entry("openai", "gpt-x").rates["input"] == Decimal("1.00")
    # Between the two entries -> the older one.
    entry = book.get_entry("openai", "gpt-x", at="2026-03-01")
    assert entry.rates["input"] == Decimal("1.25")
    # Exactly on the boundary -> the new entry applies from that day.
    entry = book.get_entry("openai", "gpt-x", at=date(2026, 6, 1))
    assert entry.rates["input"] == Decimal("1.00")
    # datetime inputs work too.
    entry = book.get_entry("openai", "gpt-x", at=datetime(2026, 7, 4, 12, 30))
    assert entry.rates["input"] == Decimal("1.00")
    # Before the book's history starts -> clamp to the oldest entry.
    entry = book.get_entry("openai", "gpt-x", at="2020-01-01")
    assert entry.rates["input"] == Decimal("1.25")


def test_repriced_model_costed_at_two_dates(tmp_path):
    """The same usage costs differently under `at=` on either side of a
    repricing — the structural guarantee naive trackers can't give."""
    book = _write_book(tmp_path, DATED_BOOK)
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    before = cost("openai", "gpt-x", usage, at="2026-03-01", prices=book)
    after = cost("openai", "gpt-x", usage, at="2026-06-02", prices=book)
    assert before.total == Decimal("1.25")
    assert after.total == Decimal("1.00")


def test_legacy_flat_schema_loads(tmp_path):
    legacy = {
        "anthropic": {
            "models": {"claude-haiku-4-5-20251001": "claude-haiku-4-5"},
            "pricing": {
                "claude-haiku-4-5": {"input": 1.00, "output": 5.00}
            },
        }
    }
    book = _write_book(tmp_path, legacy)
    entry = book.get_entry("anthropic", "claude-haiku-4-5-20251001")
    assert entry.rates == {"input": Decimal("1.00"), "output": Decimal("5.00")}
    assert entry.effective_from is None


def test_rates_are_decimal_not_float(tmp_path):
    book = _write_book(tmp_path, DATED_BOOK)
    entry = book.get_entry("openai", "gpt-x", at="2026-03-01")
    assert isinstance(entry.rates["cached"], Decimal)
    assert entry.rates["cached"] == Decimal("0.125")


def test_get_rates_legacy_compat(tmp_path):
    """`get_rates` keeps the legacy protocol shape: plain floats,
    explicit None preserved."""
    data = {
        "schema_version": 2,
        "providers": {
            "openai": {
                "pricing": {
                    "gpt-pro": [
                        {
                            "effective_from": "2026-01-01",
                            "rates": {"input": 15.0, "cached": None, "output": 120.0},
                        }
                    ]
                }
            }
        },
    }
    book = _write_book(tmp_path, data)
    rates = book.get_rates("openai", "gpt-pro")
    assert rates == {"input": 15.0, "cached": None, "output": 120.0}
    assert isinstance(rates["input"], float)
    assert book.get_rates("openai", "nope") is None


def test_dict_price_book():
    book = DictPriceBook(
        {"anthropic": {"claude-opus-5": {"input": 5.0, "output": 25.0}}}
    )
    entry = book.get_entry("anthropic", "claude-opus-5")
    assert entry.rates["input"] == Decimal("5.0")
    assert book.get_entry("anthropic", "other") is None
    assert isinstance(book, PriceBook)


def test_bundled_book_loads_and_aliases():
    book = JSONPriceBook(DEFAULT_PRICES_PATH)
    assert isinstance(book, PriceBook)
    dated = book.get_entry("anthropic", "claude-haiku-4-5-20251001")
    base = book.get_entry("anthropic", "claude-haiku-4-5")
    assert dated == base
    assert base.rates["cache_write_1h"] == Decimal("2.00")
    # Long-context models carry explicit tiers.
    entry = book.get_entry("openai", "gpt-5.4")
    assert entry.tiers[0].over_input_tokens == 272000
