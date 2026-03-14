from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.research.cs_factor_diagnostics import (
    compute_factor_correlation_summary,
    compute_forward_returns,
    compute_ic_by_date,
    compute_quantile_returns_by_date,
    summarize_coverage,
    summarize_ic,
)


def test_forward_returns_alignment_is_future_only() -> None:
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    close = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0],
            "B": [100.0, 100.0, 110.0],
        },
        index=idx,
    )
    fwd = compute_forward_returns(close, horizon=1)
    assert abs(float(fwd.loc[idx[0], "A"]) - 0.10) < 1e-12
    assert abs(float(fwd.loc[idx[0], "B"]) - 0.00) < 1e-12
    assert abs(float(fwd.loc[idx[1], "B"]) - 0.10) < 1e-12
    assert pd.isna(fwd.loc[idx[-1], "A"])


def test_ic_uses_same_date_signal_vs_future_return_only() -> None:
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    factor = pd.DataFrame(
        {
            "A": [10.0, 10.0, np.nan],
            "B": [0.0, 0.0, np.nan],
        },
        index=idx,
    )
    close = pd.DataFrame(
        {
            "A": [1.0, 2.0, 2.0],
            "B": [1.0, 1.0, 2.0],
        },
        index=idx,
    )
    fwd = compute_forward_returns(close, horizon=1)
    ic = compute_ic_by_date(factor, fwd, min_obs=2)
    assert float(ic.loc[idx[0], "IC"]) > 0.99
    assert float(ic.loc[idx[1], "IC"]) < -0.99


def test_quantile_assignment_sparse_dates_does_not_crash() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    cols = ["A", "B", "C", "D", "E"]
    factor = pd.DataFrame(np.nan, index=idx, columns=cols)
    fwd = pd.DataFrame(np.nan, index=idx, columns=cols)

    factor.loc[idx[0], cols[:3]] = [1.0, 2.0, 3.0]
    fwd.loc[idx[0], cols[:3]] = [0.01, 0.02, 0.03]
    factor.loc[idx[1], cols] = [1, 2, 3, 4, 5]
    fwd.loc[idx[1], cols] = [0.05, 0.04, 0.03, 0.02, 0.01]

    out = compute_quantile_returns_by_date(factor, fwd, quantiles=5)
    assert isinstance(out, pd.DataFrame)
    assert set(["Q1", "Q5", "Spread_QTop_Q1", "N"]).issubset(set(out.columns))


def test_subperiod_coverage_summary_has_all_windows() -> None:
    idx = pd.to_datetime(["2006-01-03", "2011-01-03", "2016-01-04", "2021-01-04"])
    coverage = pd.DataFrame(
        {
            "valid_names": [10, 20, 30, 40],
            "eligible_names": [20, 20, 40, 50],
            "coverage_fraction": [0.5, 1.0, 0.75, 0.8],
        },
        index=idx,
    )
    summary = summarize_coverage(coverage)
    keys = set(summary["by_subperiod"].keys())
    assert keys == {"2005-2009", "2010-2014", "2015-2019", "2020-2024"}
    assert abs(float(summary["overall"]["valid_names"]["mean"]) - 25.0) < 1e-12


def test_subperiod_ic_summary_has_expected_stats() -> None:
    idx = pd.to_datetime(["2006-01-03", "2011-01-03", "2016-01-04", "2021-01-04"])
    ic_by_date = pd.DataFrame({"IC": [0.1, -0.2, 0.3, 0.4], "N": [10, 10, 10, 10]}, index=idx)
    out = summarize_ic(ic_by_date)
    assert abs(float(out["overall"]["mean_ic"]) - 0.15) < 1e-12
    assert "2015-2019" in out["by_subperiod"]
    assert abs(float(out["by_subperiod"]["2015-2019"]["mean_ic"]) - 0.3) < 1e-12


def test_factor_correlation_handles_missing_overlap_safely() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    target = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [2.0, 3.0, 4.0, 5.0],
            "C": [3.0, 4.0, 5.0, 6.0],
        },
        index=idx,
    )
    peer = pd.DataFrame(
        {
            "A": [1.0, np.nan, np.nan, np.nan],
            "B": [np.nan, 2.0, np.nan, np.nan],
            "C": [np.nan, np.nan, 3.0, np.nan],
        },
        index=idx,
    )
    by_date, summary = compute_factor_correlation_summary(
        target_factor=target,
        peer_factors={"peer": peer},
        min_overlap=2,
    )
    assert not by_date.empty
    assert int(summary.loc[summary["Peer"] == "peer", "Count"].iloc[0]) == 0
    assert pd.isna(summary.loc[summary["Peer"] == "peer", "AverageCorrelation"].iloc[0])
