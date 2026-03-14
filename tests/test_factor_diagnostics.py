from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.research.factor_diagnostics import run_factor_diagnostics


def _toy_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2020-01-01", periods=8, freq="B")
    cols = ["A", "B", "C", "D", "E", "F"]
    base = np.arange(1, len(cols) + 1, dtype=float)
    scores = pd.DataFrame(
        np.vstack([base + i * 0.1 for i in range(len(idx))]),
        index=idx,
        columns=cols,
    )
    fwd = pd.DataFrame(
        np.vstack([(base * 0.01) + i * 0.0001 for i in range(len(idx))]),
        index=idx,
        columns=cols,
    )
    return scores, fwd


def test_ic_positive_in_toy_example() -> None:
    scores, fwd = _toy_panels()
    report = run_factor_diagnostics(scores, fwd, method="pearson", quantiles=3, horizons=[1, 5])
    assert report["ic_summary"]["mean_ic"] > 0.95
    assert report["ic_summary"]["ic_hit_rate"] >= 0.99


def test_rank_ic_works() -> None:
    scores, fwd = _toy_panels()
    report = run_factor_diagnostics(scores, fwd, method="spearman", quantiles=3, horizons=[1])
    assert report["ic_summary"]["mean_ic"] > 0.99


def test_quantile_returns_monotonic_and_spread_positive() -> None:
    scores, fwd = _toy_panels()
    report = run_factor_diagnostics(scores, fwd, method="spearman", quantiles=3, horizons=[1])
    q = report["quantile_return_summary"]
    assert q["Q1"] < q["Q2"] < q["Q3"]
    assert report["top_minus_bottom_spread"] > 0.0


def test_decay_contains_requested_horizons() -> None:
    scores, fwd = _toy_panels()
    report = run_factor_diagnostics(scores, fwd, method="spearman", quantiles=3, horizons=[1, 5, 21])
    assert set(report["decay_summary"].keys()) == {1, 5, 21}


def test_missing_values_and_sparse_dates_do_not_crash() -> None:
    scores, fwd = _toy_panels()
    scores.iloc[0, :] = np.nan
    scores.iloc[1, :4] = np.nan
    fwd.iloc[2, 2:] = np.nan
    fwd.iloc[-1, :] = np.nan

    report = run_factor_diagnostics(scores, fwd, method="spearman", quantiles=5, horizons=[1, 5])
    assert "ic_summary" in report
    assert "coverage_summary" in report
    assert isinstance(report["quantile_returns_by_date"], pd.DataFrame)
    assert report["coverage_summary"]["used_in_ic_median"] >= 0.0
