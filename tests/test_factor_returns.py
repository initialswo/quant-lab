from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.research.factor_returns import (
    plot_factor_seasonality,
    run_factor_return_analysis,
    run_factor_return_correlation,
    run_factor_seasonality,
)


def _toy_scores_and_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2021-01-01", periods=10, freq="B")
    cols = ["A", "B", "C", "D", "E", "F"]
    scores = pd.DataFrame(
        np.tile(np.arange(1, 7, dtype=float), (len(idx), 1)),
        index=idx,
        columns=cols,
    )
    fwd = pd.DataFrame(
        np.tile(np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006], dtype=float), (len(idx), 1)),
        index=idx,
        columns=cols,
    )
    return scores, fwd


def test_monotonic_quantile_returns_and_positive_spread() -> None:
    scores, fwd = _toy_scores_and_returns()
    rep = run_factor_return_analysis(scores, fwd, quantiles=3, rolling_window=3)
    q = rep["quantile_mean_returns"]
    assert q["Q1"] < q["Q2"] < q["Q3"]
    assert rep["top_minus_bottom_spread_mean"] > 0.0


def test_spread_summary_fields_present_and_sensible() -> None:
    scores, fwd = _toy_scores_and_returns()
    rep = run_factor_return_analysis(scores, fwd, quantiles=3, rolling_window=3)
    s = rep["spread_summary"]
    for key in [
        "mean_period_return",
        "annualized_return",
        "annualized_vol",
        "sharpe",
        "cumulative_return",
        "cumulative_pnl",
        "max_drawdown",
        "hit_rate",
    ]:
        assert key in s
    assert s["annualized_return"] > 0.0
    assert s["hit_rate"] >= 0.9
    assert s["return_convention"] == "long_short_spread_pnl"


def test_max_drawdown_computation() -> None:
    idx = pd.date_range("2022-01-01", periods=3, freq="B")
    cols = ["TOP", "BOT"]
    scores = pd.DataFrame([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]], index=idx, columns=cols)
    fwd = pd.DataFrame([[0.10, 0.00], [-0.20, 0.00], [0.05, 0.00]], index=idx, columns=cols)
    rep = run_factor_return_analysis(scores, fwd, quantiles=2, rolling_window=2)
    mdd = float(rep["spread_summary"]["max_drawdown"])
    # PnL-style drawdown: spreads are [0.1, -0.2, 0.05] -> pnl [0.1,-0.1,-0.05], maxDD=-0.2
    assert np.isclose(mdd, -0.2, atol=1e-12)


def test_extreme_negative_spread_path_is_finite_and_drawdown_not_percentage_broken() -> None:
    idx = pd.date_range("2022-01-01", periods=5, freq="B")
    cols = ["TOP", "BOT"]
    scores = pd.DataFrame(np.tile([2.0, 1.0], (len(idx), 1)), index=idx, columns=cols)
    # Spread path: [-1.8, 0.4, -0.2, 0.1, 0.0]
    fwd = pd.DataFrame(
        [[-0.9, 0.9], [0.2, -0.2], [-0.1, 0.1], [0.05, -0.05], [0.0, 0.0]],
        index=idx,
        columns=cols,
    )
    rep = run_factor_return_analysis(scores, fwd, quantiles=2, rolling_window=2)
    s = rep["spread_summary"]
    assert np.isfinite(float(s["annualized_return"]))
    assert np.isfinite(float(s["max_drawdown"]))
    assert float(s["max_drawdown"]) <= 0.0


def test_missing_values_and_sparse_universe_do_not_crash() -> None:
    scores, fwd = _toy_scores_and_returns()
    scores.iloc[:2, :] = np.nan
    scores.iloc[3, 0:4] = np.nan
    fwd.iloc[4, 2:] = np.nan
    rep = run_factor_return_analysis(scores, fwd, quantiles=5, rolling_window=3)
    assert "coverage_summary" in rep
    assert rep["coverage_summary"]["max_assets_used"] >= rep["coverage_summary"]["min_assets_used"]


def test_factor_return_correlation_helper() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    a = pd.Series([0.01, 0.02, -0.01, 0.00, 0.03, 0.01], index=idx)
    b = pd.Series([0.02, 0.04, -0.02, 0.00, 0.06, 0.02], index=idx)
    c = pd.Series([-0.01, -0.02, 0.01, 0.00, -0.03, -0.01], index=idx)
    corr = run_factor_return_correlation({"a": a, "b": b, "c": c})
    assert corr.shape == (3, 3)
    assert corr.loc["a", "b"] > 0.99
    assert corr.loc["a", "c"] < -0.99


def test_factor_seasonality_monthly_aggregation() -> None:
    idx = pd.to_datetime(
        [
            "2022-01-03",
            "2022-01-10",
            "2022-02-07",
            "2022-02-14",
            "2022-02-21",
        ]
    )
    spread = pd.Series([0.01, -0.01, 0.02, 0.00, -0.01], index=idx)
    seas = run_factor_seasonality(spread_returns=spread, periods_per_year=252)
    assert list(seas.columns) == ["mean_return", "volatility", "sharpe", "hit_rate", "n_obs"]
    assert int(seas.loc[1, "n_obs"]) == 2
    assert int(seas.loc[2, "n_obs"]) == 3
    assert np.isclose(float(seas.loc[1, "mean_return"]), 0.0, atol=1e-12)
    assert np.isclose(float(seas.loc[2, "mean_return"]), (0.02 + 0.0 - 0.01) / 3.0, atol=1e-12)
    assert np.isnan(float(seas.loc[3, "mean_return"]))


def test_factor_seasonality_plot_writes_png(tmp_path) -> None:
    idx = pd.date_range("2022-01-01", periods=30, freq="B")
    spread = pd.Series(np.linspace(-0.01, 0.01, num=len(idx)), index=idx)
    seas = run_factor_seasonality(spread_returns=spread)
    out = tmp_path / "seasonality.png"
    plot_factor_seasonality(seasonality=seas, outpath=str(out), title="Test")
    assert out.exists()
