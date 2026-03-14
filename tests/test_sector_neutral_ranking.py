from __future__ import annotations

import pandas as pd

from quant_lab.strategies.topn import build_topn_weights


def _toy_scores() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    cols = ["A1", "A2", "A3", "B1", "B2", "B3"]
    scores = pd.DataFrame(
        [
            [100.0, 90.0, 80.0, 10.0, 9.0, 8.0],
            [100.0, 90.0, 80.0, 10.0, 9.0, 8.0],
        ],
        index=idx,
        columns=cols,
    )
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    sector_map = {
        "A1": "Tech",
        "A2": "Tech",
        "A3": "Tech",
        "B1": "Health",
        "B2": "Health",
        "B3": "Health",
    }
    return scores, close, sector_map


def test_sector_neutral_changes_selection_vs_global() -> None:
    scores, close, sector_map = _toy_scores()

    w_global = build_topn_weights(
        scores=scores,
        close=close,
        top_n=2,
        rebalance="daily",
        weighting="equal",
        sector_by_ticker=sector_map,
        sector_neutral=False,
    )
    w_neutral = build_topn_weights(
        scores=scores,
        close=close,
        top_n=2,
        rebalance="daily",
        weighting="equal",
        sector_by_ticker=sector_map,
        sector_neutral=True,
    )

    # Global picks top two from Tech bucket (A1, A2).
    assert float(w_global.loc[scores.index[0], "A1"]) > 0.0
    assert float(w_global.loc[scores.index[0], "A2"]) > 0.0
    assert float(w_global.loc[scores.index[0], "B1"]) == 0.0

    # Sector-neutral rank normalization should admit top Health name (B1).
    assert float(w_neutral.loc[scores.index[0], "A1"]) > 0.0
    assert float(w_neutral.loc[scores.index[0], "B1"]) > 0.0
    assert float(w_neutral.loc[scores.index[0], "A2"]) == 0.0


def test_sector_neutral_deterministic() -> None:
    scores, close, sector_map = _toy_scores()
    w1 = build_topn_weights(
        scores=scores,
        close=close,
        top_n=3,
        rebalance="daily",
        weighting="equal",
        sector_by_ticker=sector_map,
        sector_neutral=True,
    )
    w2 = build_topn_weights(
        scores=scores,
        close=close,
        top_n=3,
        rebalance="daily",
        weighting="equal",
        sector_by_ticker=sector_map,
        sector_neutral=True,
    )
    pd.testing.assert_frame_equal(w1, w2)


def test_no_change_when_sector_neutral_false() -> None:
    scores, close, sector_map = _toy_scores()
    w_default = build_topn_weights(
        scores=scores,
        close=close,
        top_n=3,
        rebalance="daily",
        weighting="equal",
        sector_by_ticker=sector_map,
    )
    w_false = build_topn_weights(
        scores=scores,
        close=close,
        top_n=3,
        rebalance="daily",
        weighting="equal",
        sector_by_ticker=sector_map,
        sector_neutral=False,
    )
    pd.testing.assert_frame_equal(w_default, w_false)
