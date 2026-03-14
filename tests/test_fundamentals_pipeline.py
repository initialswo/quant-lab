from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_lab.data.fundamentals import (
    align_fundamentals_to_daily_panel,
    load_fundamentals_file,
    normalize_ticker_symbol,
)
from quant_lab.factors.registry import compute_factor, list_factors


def _synthetic_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "fundamentals_synthetic.csv"


def test_ticker_normalization_respected() -> None:
    assert normalize_ticker_symbol("aaa.us") == "AAA"
    assert normalize_ticker_symbol(" brk.b ") == "BRK-B"


def test_loader_dedup_and_fallback_available_date() -> None:
    df = load_fundamentals_file(_synthetic_path(), fallback_lag_days=45)
    # AAA duplicate should dedupe to one row.
    assert int((df["ticker"] == "AAA").sum()) == 1
    brk = df.loc[df["ticker"] == "BRK-B"].iloc[0]
    assert pd.Timestamp(brk["available_date"]) == pd.Timestamp("2020-05-15")


def test_loader_supports_parquet_when_available(tmp_path) -> None:
    try:
        import pyarrow  # noqa: F401

        _ = pyarrow
    except Exception:
        try:
            import fastparquet  # noqa: F401

            _ = fastparquet
        except Exception:
            return
    src = pd.read_csv(_synthetic_path())
    p = tmp_path / "fundamentals.parquet"
    src.to_parquet(p, index=False)
    out = load_fundamentals_file(p, fallback_lag_days=45)
    assert not out.empty
    assert {"AAA", "BRK-B", "BAD"}.issubset(set(out["ticker"]))


def test_no_value_before_available_date() -> None:
    fundamentals = load_fundamentals_file(_synthetic_path(), fallback_lag_days=45)
    idx = pd.date_range("2020-05-14", "2020-05-18", freq="D")
    cols = ["AAA", "BRK-B", "BAD"]
    aligned = align_fundamentals_to_daily_panel(fundamentals, idx, cols, value_columns=["gross_profit"])
    gp = aligned["gross_profit"]
    assert pd.isna(gp.loc[pd.Timestamp("2020-05-14"), "AAA"])
    assert gp.loc[pd.Timestamp("2020-05-15"), "AAA"] == 100.0


def test_gross_profitability_and_invalid_denominator() -> None:
    fundamentals = load_fundamentals_file(_synthetic_path(), fallback_lag_days=45)
    idx = pd.date_range("2020-05-15", "2020-05-22", freq="D")
    cols = ["AAA", "BRK-B", "BAD"]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    aligned = align_fundamentals_to_daily_panel(fundamentals, idx, cols)
    out = compute_factor("gross_profitability", close, fundamentals_aligned=aligned)
    # preferred numerator path
    assert out.loc[pd.Timestamp("2020-05-16"), "AAA"] == 0.1
    # fallback numerator path: (revenue-cogs)/assets = (800-500)/1500 = 0.2
    assert abs(float(out.loc[pd.Timestamp("2020-05-16"), "BRK-B"]) - 0.2) < 1e-12
    # invalid denominator <=0 -> NaN
    assert pd.isna(out.loc[pd.Timestamp("2020-05-22"), "BAD"])


def test_factor_registered() -> None:
    assert "gross_profitability" in set(list_factors())
