from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.engine.runner import (
    _apply_universe_rebalance_skip,
    _prepare_close_panel,
    _rebalance_score_counts,
)
from quant_lab.strategies.topn import rebalance_mask
from quant_lab.strategies.topn import build_topn_weights


def test_rebalance_eligibility_counts_use_rebalance_dates_only() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    scores = pd.DataFrame(1.0, index=idx, columns=["A", "B", "C"])
    rb_mask = pd.Series(False, index=idx)
    rb_mask.iloc[::5] = True

    counts = _rebalance_score_counts(scores, rb_mask)
    assert len(counts) == int(rb_mask.sum())
    assert len(counts) != len(scores)


def test_weekly_rebalance_mask_is_not_daily() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    rb_mask = rebalance_mask(idx, "weekly")
    rb_count = int(rb_mask.sum())
    assert rb_count < len(idx)
    assert 15 <= rb_count <= 25


def test_prepare_close_panel_drops_all_nan_dates_when_none() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    close = pd.DataFrame(
        {
            "A": [10.0, 10.1, np.nan, 10.3, 10.4, 10.5],
            "B": [20.0, 20.1, np.nan, 20.3, 20.4, 20.5],
        },
        index=idx,
    )
    out = _prepare_close_panel(close, price_fill_mode="none")
    assert idx[2] not in out.index


def test_universe_skip_below_min_vs_cap_behavior() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    cols = [f"T{i}" for i in range(10)]
    close = pd.DataFrame({c: 100.0 + np.arange(len(idx), dtype=float) for c in cols}, index=idx)
    scores = pd.DataFrame(np.nan, index=idx, columns=cols)
    scores.loc[:, cols[:5]] = 1.0
    rb_mask = pd.Series(True, index=idx)

    scores_skip, _, _ = _apply_universe_rebalance_skip(
        scores=scores,
        rb_mask=rb_mask,
        universe_min_tickers=20,
        universe_skip_below_min_tickers=True,
    )
    w_skip = build_topn_weights(
        scores=scores_skip,
        close=close,
        top_n=20,
        rebalance="daily",
        weighting="equal",
        max_weight=1.0,
    )
    assert int((w_skip.abs() > 0).sum(axis=1).max()) == 0

    scores_cap, _, _ = _apply_universe_rebalance_skip(
        scores=scores,
        rb_mask=rb_mask,
        universe_min_tickers=20,
        universe_skip_below_min_tickers=False,
    )
    w_cap = build_topn_weights(
        scores=scores_cap,
        close=close,
        top_n=20,
        rebalance="daily",
        weighting="equal",
        max_weight=1.0,
    )
    holdings = (w_cap.abs() > 0).sum(axis=1)
    assert int(holdings.max()) <= 5
    assert int(holdings.max()) > 0
