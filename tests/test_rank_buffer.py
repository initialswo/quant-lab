from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.strategies.topn import build_topn_weights


def test_rank_buffer_zero_matches_existing_behavior() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    scores = pd.DataFrame(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 3.0, 2.0],
            [2.0, 1.0, 4.0, 3.0],
            [3.0, 2.0, 1.0, 4.0],
        ],
        index=idx,
        columns=cols,
    )
    w_ref = build_topn_weights(scores=scores, close=close, top_n=2, rebalance="daily", weighting="equal")
    w_buf0 = build_topn_weights(
        scores=scores,
        close=close,
        top_n=2,
        rank_buffer=0,
        rebalance="daily",
        weighting="equal",
    )
    assert np.allclose(w_ref.to_numpy(), w_buf0.to_numpy(), equal_nan=True)


def test_rank_buffer_holds_names_inside_band() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    scores = pd.DataFrame(
        [
            [10.0, 9.0, 8.0, 7.0],  # hold A,B
            [8.0, 9.0, 10.0, 7.0],  # A drops to rank 3 (inside buffer for top2+1)
            [7.0, 9.0, 10.0, 8.0],
        ],
        index=idx,
        columns=cols,
    )
    w = build_topn_weights(
        scores=scores,
        close=close,
        top_n=2,
        rank_buffer=1,
        rebalance="daily",
        weighting="equal",
    )
    # On 2nd rebalance date, A should still be held due to buffer.
    assert float(w.loc[idx[1], "A"]) > 0.0


def test_rank_buffer_sells_only_outside_band() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    scores = pd.DataFrame(
        [
            [10.0, 9.0, 8.0, 7.0],  # hold A,B
            [8.0, 9.0, 10.0, 7.0],  # A rank 3 -> keep with buffer=1
            [7.0, 9.0, 10.0, 8.0],  # A rank 4 -> sell (> top2+1)
        ],
        index=idx,
        columns=cols,
    )
    w = build_topn_weights(
        scores=scores,
        close=close,
        top_n=2,
        rank_buffer=1,
        rebalance="daily",
        weighting="equal",
    )
    assert float(w.loc[idx[1], "A"]) > 0.0
    assert np.isclose(float(w.loc[idx[2], "A"]), 0.0, atol=1e-12)

