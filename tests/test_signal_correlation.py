from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.research.signal_correlation import print_signal_correlation, run_signal_correlation


def _panel(values: list[list[float]]) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=len(values), freq="B")
    cols = ["A", "B", "C", "D"]
    return pd.DataFrame(values, index=idx, columns=cols, dtype=float)


def test_perfect_positive_correlation() -> None:
    s1 = _panel([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    s2 = 2.0 * s1 + 5.0
    report = run_signal_correlation({"s1": s1, "s2": s2}, method="pearson")
    corr = float(report["average_correlation_matrix"].loc["s1", "s2"])
    assert corr > 0.999


def test_perfect_negative_correlation() -> None:
    s1 = _panel([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    s2 = -s1
    report = run_signal_correlation({"s1": s1, "s2": s2}, method="spearman")
    corr = float(report["average_correlation_matrix"].loc["s1", "s2"])
    assert corr < -0.999


def test_low_correlation_case() -> None:
    idx = pd.date_range("2022-01-01", periods=6, freq="B")
    cols = ["A", "B", "C", "D"]
    s1 = pd.DataFrame(np.tile([1.0, 2.0, 3.0, 4.0], (len(idx), 1)), index=idx, columns=cols)
    s2 = pd.DataFrame(
        [[4.0, 1.0, 3.0, 2.0], [2.0, 4.0, 1.0, 3.0], [3.0, 1.0, 4.0, 2.0], [1.0, 3.0, 2.0, 4.0], [2.0, 3.0, 4.0, 1.0], [4.0, 2.0, 1.0, 3.0]],
        index=idx,
        columns=cols,
    )
    report = run_signal_correlation({"s1": s1, "s2": s2}, method="spearman")
    corr = float(report["average_correlation_matrix"].loc["s1", "s2"])
    assert abs(corr) < 0.6


def test_missing_values_and_overlap_counts() -> None:
    s1 = _panel([[1, 2, 3, 4], [2, np.nan, 4, 5], [3, 4, np.nan, 6], [4, 5, 6, 7]])
    s2 = _panel([[1, 2, 3, 4], [2, 3, np.nan, 5], [np.nan, 4, 5, 6], [4, 5, 6, 7]])
    report = run_signal_correlation({"s1": s1, "s2": s2}, method="pearson")
    cov = report["coverage_summary"]
    assert int(cov.loc[0, "valid_dates"]) >= 1
    assert float(cov.loc[0, "median_overlap"]) >= 2.0


def test_sparse_helper_path_no_crash() -> None:
    idx = pd.date_range("2022-01-01", periods=5, freq="B")
    cols = ["A", "B", "C"]
    s1 = pd.DataFrame(np.nan, index=idx, columns=cols)
    s2 = pd.DataFrame(np.nan, index=idx, columns=cols)
    s1.loc[idx[-1], :] = [1.0, 2.0, 3.0]
    s2.loc[idx[-1], :] = [3.0, 2.0, 1.0]
    report = run_signal_correlation({"s1": s1, "s2": s2}, method="spearman")
    assert "average_correlation_matrix" in report
    print_signal_correlation(report)

