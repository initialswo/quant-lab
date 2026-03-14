from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.research.rank_decay import (
    assign_quantile_buckets,
    monotonicity_score_from_cagr,
    run_rank_decay_backtest,
)


def test_assign_quantile_buckets_is_deterministic_on_ties() -> None:
    row = pd.Series({"B": 1.0, "A": 1.0, "D": 2.0, "C": 2.0})
    buckets = assign_quantile_buckets(row, quantiles=2)
    assert buckets == {"Q1": ["A", "B"], "Q2": ["C", "D"]}


def test_rank_decay_skips_rebalance_when_too_few_names() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    close = pd.DataFrame(
        {
            "A": [100, 101, 102, 103, 104, 105],
            "B": [100, 100, 100, 100, 100, 100],
            "C": [100, 99, 98, 97, 96, 95],
        },
        index=idx,
        dtype=float,
    )
    scores = pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "B": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "C": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        },
        index=idx,
        dtype=float,
    )
    report = run_rank_decay_backtest(scores=scores, close=close, quantiles=5, rebalance="daily")
    assert int(report["skipped_rebalance_dates"]) == len(idx)
    spread = pd.Series(report["Spread"], dtype=float)
    assert np.isclose(float(spread.abs().sum()), 0.0, atol=1e-12)


def test_rank_decay_backtest_produces_positive_top_minus_bottom_spread() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    close = pd.DataFrame(
        {
            "A": np.linspace(100.0, 130.0, len(idx)),
            "B": np.linspace(100.0, 124.0, len(idx)),
            "C": np.linspace(100.0, 118.0, len(idx)),
            "D": np.linspace(100.0, 112.0, len(idx)),
            "E": np.linspace(100.0, 106.0, len(idx)),
            "F": np.linspace(100.0, 101.0, len(idx)),
        },
        index=idx,
    )
    scores = pd.DataFrame(
        np.tile(np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=float), (len(idx), 1)),
        index=idx,
        columns=close.columns,
    )
    report = run_rank_decay_backtest(scores=scores, close=close, quantiles=3, rebalance="weekly")
    q1 = report["Q1"]
    q3 = report["Q3"]
    assert float(q3.summary["CAGR"]) > float(q1.summary["CAGR"])
    assert float(pd.Series(report["Spread"], dtype=float).mean()) > 0.0


def test_monotonicity_score_bounds_and_direction() -> None:
    assert np.isclose(monotonicity_score_from_cagr([0.01, 0.02, 0.03]), 1.0, atol=1e-12)
    assert np.isclose(monotonicity_score_from_cagr([0.03, 0.02, 0.01]), 0.0, atol=1e-12)
