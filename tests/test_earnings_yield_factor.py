from __future__ import annotations

import pandas as pd

from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel
from quant_lab.factors.registry import compute_factor, list_factors


def test_earnings_yield_formula_and_shape() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    cols = ["AAA", "BBB"]
    close = pd.DataFrame([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]], index=idx, columns=cols)

    fundamentals = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "period_end": "2023-12-31",
                "available_date": "2024-01-01",
                "net_income": 100.0,
                "shares_outstanding": 10.0,
            },
            {
                "ticker": "BBB",
                "period_end": "2023-12-31",
                "available_date": "2024-01-01",
                "net_income": 200.0,
                "shares_outstanding": 20.0,
            },
        ]
    )
    aligned = align_fundamentals_to_daily_panel(fundamentals, idx, cols)
    out = compute_factor("earnings_yield", close, fundamentals_aligned=aligned)

    assert out.shape == close.shape
    # AAA: 100 / (10*10) = 1.0 ; BBB: 200 / (20*20) = 0.5
    assert abs(float(out.loc[idx[1], "AAA"]) - 1.0) < 1e-12
    assert abs(float(out.loc[idx[1], "BBB"]) - 0.5) < 1e-12


def test_earnings_yield_no_values_before_available_date() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    close = pd.DataFrame(10.0, index=idx, columns=["AAA"])
    fundamentals = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "period_end": "2023-12-31",
                "available_date": "2024-01-02",
                "net_income": 100.0,
                "shares_outstanding": 10.0,
            }
        ]
    )
    aligned = align_fundamentals_to_daily_panel(fundamentals, idx, close.columns)
    out = compute_factor("earnings_yield", close, fundamentals_aligned=aligned)

    assert pd.isna(out.loc[pd.Timestamp("2024-01-01"), "AAA"])
    assert abs(float(out.loc[pd.Timestamp("2024-01-02"), "AAA"]) - 1.0) < 1e-12


def test_earnings_yield_invalid_denominator_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    close = pd.DataFrame([10.0, 0.0], index=idx, columns=["AAA"])
    fundamentals = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "period_end": "2023-12-31",
                "available_date": "2024-01-01",
                "net_income": 100.0,
                "shares_outstanding": 0.0,
            }
        ]
    )
    aligned = align_fundamentals_to_daily_panel(fundamentals, idx, close.columns)
    out = compute_factor("earnings_yield", close, fundamentals_aligned=aligned)

    assert pd.isna(out.loc[idx[0], "AAA"])
    assert pd.isna(out.loc[idx[1], "AAA"])


def test_earnings_yield_registry_discovery() -> None:
    assert "earnings_yield" in set(list_factors())
