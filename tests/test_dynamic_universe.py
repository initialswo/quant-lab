from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.data.universe_dynamic import (
    apply_universe_filter_to_scores,
    compute_eligibility_matrix,
)
from quant_lab.factors import momentum
from quant_lab.strategies.topn import build_topn_weights


def test_compute_eligibility_matrix_history_and_valid_frac() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    close = pd.DataFrame(index=idx)
    close["A"] = 10.0 + np.arange(len(idx), dtype=float)
    close["B"] = np.nan
    close.loc[idx[5]:, "B"] = 20.0 + np.arange(len(idx) - 5, dtype=float)
    close["C"] = 30.0 + np.arange(len(idx), dtype=float)
    close.loc[idx[[2, 4, 6, 8, 10, 12, 14, 16, 18]], "C"] = np.nan

    elig = compute_eligibility_matrix(
        close=close,
        min_history_days=5,
        valid_lookback=5,
        min_valid_frac=0.8,
        min_price=1.0,
    )

    assert bool(elig.loc[idx[8], "A"])
    assert not bool(elig.loc[idx[8], "B"])  # not enough history yet
    assert not bool(elig.loc[idx[12], "C"])  # low valid fraction in lookback


def test_selection_masks_ineligible() -> None:
    idx = pd.date_range("2024-01-01", periods=1, freq="B")
    scores = pd.DataFrame([[3.0, 2.0, 1.0]], index=idx, columns=["A", "B", "C"])
    elig = pd.DataFrame([[False, True, True]], index=idx, columns=["A", "B", "C"])

    masked = apply_universe_filter_to_scores(scores, elig, exempt=set())
    assert pd.isna(masked.loc[idx[0], "A"])
    assert np.isfinite(float(masked.loc[idx[0], "B"]))


def test_topn_capped_when_few_eligible() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="B")
    cols = [f"T{i}" for i in range(10)]
    close = pd.DataFrame(
        {c: 100.0 + np.arange(len(idx), dtype=float) for c in cols},
        index=idx,
    )
    scores = pd.DataFrame(1.0, index=idx, columns=cols)
    # Keep only first 5 eligible on each date.
    elig = pd.DataFrame(False, index=idx, columns=cols)
    elig.loc[:, cols[:5]] = True
    masked_scores = apply_universe_filter_to_scores(scores, elig, exempt=set())

    w = build_topn_weights(
        scores=masked_scores,
        close=close,
        top_n=20,
        rebalance="daily",
        weighting="equal",
        max_weight=1.0,
    )
    holdings = (w.abs() > 0).sum(axis=1)
    assert int(holdings.max()) <= 5


def test_eligibility_uses_full_history_not_window_slice() -> None:
    idx = pd.date_range("2010-01-01", periods=800, freq="B")
    close = pd.DataFrame({"A": 50.0 + np.arange(len(idx), dtype=float)}, index=idx)
    min_history_days = 300

    elig_full = compute_eligibility_matrix(
        close=close,
        min_history_days=min_history_days,
        valid_lookback=252,
        min_valid_frac=0.98,
        min_price=1.0,
    )

    window_start = pd.Timestamp("2012-01-03")
    test_dt = pd.Timestamp("2012-07-02")
    close_hist = close.loc[window_start:]
    elig_wrong = compute_eligibility_matrix(
        close=close_hist,
        min_history_days=min_history_days,
        valid_lookback=252,
        min_valid_frac=0.98,
        min_price=1.0,
    )
    elig_correct = elig_full.loc[close_hist.index]

    assert bool(elig_correct.loc[test_dt, "A"])
    assert not bool(elig_wrong.loc[test_dt, "A"])


def test_momentum_requires_history_returns_nan_until_ready() -> None:
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    close = pd.DataFrame({"A": 100.0 + np.arange(len(idx), dtype=float)}, index=idx)
    out = momentum.compute(close=close, lookback_long=252, lookback_short=21)

    assert out.loc[idx[:252], "A"].isna().all()
    assert np.isfinite(float(out.loc[idx[-1], "A"]))
