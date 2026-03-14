from __future__ import annotations

import pandas as pd

from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel
from quant_lab.factors.registry import compute_factor, list_factors


def test_roa_formula_and_shape() -> None:
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
                "total_assets": 500.0,
            },
            {
                "ticker": "BBB",
                "period_end": "2023-12-31",
                "available_date": "2024-01-01",
                "net_income": 200.0,
                "total_assets": 1000.0,
            },
        ]
    )
    aligned = align_fundamentals_to_daily_panel(fundamentals, idx, cols)
    out = compute_factor("roa", close, fundamentals_aligned=aligned)

    assert out.shape == close.shape
    assert abs(float(out.loc[idx[1], "AAA"]) - 0.2) < 1e-12
    assert abs(float(out.loc[idx[1], "BBB"]) - 0.2) < 1e-12


def test_roa_no_values_before_available_date() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    close = pd.DataFrame(10.0, index=idx, columns=["AAA"])
    fundamentals = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "period_end": "2023-12-31",
                "available_date": "2024-01-02",
                "net_income": 100.0,
                "total_assets": 500.0,
            }
        ]
    )
    aligned = align_fundamentals_to_daily_panel(fundamentals, idx, close.columns)
    out = compute_factor("roa", close, fundamentals_aligned=aligned)

    assert pd.isna(out.loc[pd.Timestamp("2024-01-01"), "AAA"])
    assert abs(float(out.loc[pd.Timestamp("2024-01-02"), "AAA"]) - 0.2) < 1e-12


def test_roa_invalid_or_missing_denominator_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    close = pd.DataFrame(10.0, index=idx, columns=["AAA", "BBB", "CCC"])
    fundamentals = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "period_end": "2023-12-31",
                "available_date": "2024-01-01",
                "net_income": 100.0,
                "total_assets": 0.0,
            },
            {
                "ticker": "BBB",
                "period_end": "2023-12-31",
                "available_date": "2024-01-01",
                "net_income": 100.0,
                "total_assets": -1.0,
            },
            {
                "ticker": "CCC",
                "period_end": "2023-12-31",
                "available_date": "2024-01-01",
                "net_income": 100.0,
            },
        ]
    )
    aligned = align_fundamentals_to_daily_panel(fundamentals, idx, close.columns)
    out = compute_factor("roa", close, fundamentals_aligned=aligned)

    assert pd.isna(out.loc[idx[0], "AAA"])
    assert pd.isna(out.loc[idx[0], "BBB"])
    assert pd.isna(out.loc[idx[0], "CCC"])


def test_roa_missing_numerator_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    close = pd.DataFrame(10.0, index=idx, columns=["AAA"])
    fundamentals = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "period_end": "2023-12-31",
                "available_date": "2024-01-01",
                "total_assets": 500.0,
            }
        ]
    )
    aligned = align_fundamentals_to_daily_panel(fundamentals, idx, close.columns)
    out = compute_factor("roa", close, fundamentals_aligned=aligned)

    assert pd.isna(out.loc[idx[0], "AAA"])


def test_roa_registry_discovery() -> None:
    assert "roa" in set(list_factors())
