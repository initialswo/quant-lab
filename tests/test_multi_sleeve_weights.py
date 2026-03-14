from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.strategies.topn import build_multi_sleeve_weights, build_topn_weights


def test_single_sleeve_matches_topn_weights() -> None:
    idx = pd.date_range("2024-01-01", periods=12, freq="B")
    cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    score = pd.DataFrame(
        np.linspace(0.0, 1.0, len(idx) * len(cols)).reshape(len(idx), len(cols)),
        index=idx,
        columns=cols,
    )
    w_ref = build_topn_weights(scores=score, close=close, top_n=2, rebalance="daily")
    w_multi, sleeves = build_multi_sleeve_weights(
        sleeve_scores={"core": score},
        sleeve_allocations={"core": 1.0},
        sleeve_top_n={"core": 2},
        close=close,
        rebalance="daily",
    )
    assert np.allclose(w_ref.to_numpy(), w_multi.to_numpy(), equal_nan=True)
    assert "core" in sleeves


def test_multi_sleeve_overlap_aggregates_and_normalizes() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="B")
    cols = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    score_a = pd.DataFrame(
        [[5.0, 4.0, 3.0, 2.0, 1.0]] * len(idx),
        index=idx,
        columns=cols,
    )
    score_b = pd.DataFrame(
        [[5.0, 3.5, 4.5, 2.0, 1.0]] * len(idx),
        index=idx,
        columns=cols,
    )
    w, _ = build_multi_sleeve_weights(
        sleeve_scores={"s1": score_a, "s2": score_b},
        sleeve_allocations={"s1": 0.6, "s2": 0.4},
        sleeve_top_n={"s1": 2, "s2": 2},
        close=close,
        rebalance="daily",
    )
    dt = idx[-1]
    assert float(w.loc[dt].sum()) == pytest.approx(1.0, abs=1e-10)
    assert float(w.loc[dt, "A"]) > 0.0
    assert float(w.loc[dt, "B"]) > 0.0
    assert float(w.loc[dt, "C"]) > 0.0


def test_multi_sleeve_deterministic_and_no_future_dependency() -> None:
    idx = pd.date_range("2024-01-01", periods=15, freq="B")
    cols = ["A", "B", "C"]
    close = pd.DataFrame(
        {
            "A": np.linspace(100, 110, len(idx)),
            "B": np.linspace(100, 107, len(idx)),
            "C": np.linspace(100, 104, len(idx)),
        },
        index=idx,
    )
    base = pd.DataFrame(
        {
            "A": np.linspace(0.1, 0.9, len(idx)),
            "B": np.linspace(0.3, 0.8, len(idx)),
            "C": np.linspace(0.2, 0.7, len(idx)),
        },
        index=idx,
    )
    w1, _ = build_multi_sleeve_weights(
        sleeve_scores={"m": base, "q": base * 0.5},
        sleeve_allocations={"m": 0.5, "q": 0.5},
        sleeve_top_n={"m": 2, "q": 2},
        close=close,
        rebalance="daily",
    )
    w2, _ = build_multi_sleeve_weights(
        sleeve_scores={"m": base, "q": base * 0.5},
        sleeve_allocations={"m": 0.5, "q": 0.5},
        sleeve_top_n={"m": 2, "q": 2},
        close=close,
        rebalance="daily",
    )
    assert np.allclose(w1.to_numpy(), w2.to_numpy(), equal_nan=True)

    changed = base.copy()
    changed.loc[idx[-2:], "A"] = 999.0
    w_changed, _ = build_multi_sleeve_weights(
        sleeve_scores={"m": changed, "q": base * 0.5},
        sleeve_allocations={"m": 0.5, "q": 0.5},
        sleeve_top_n={"m": 2, "q": 2},
        close=close,
        rebalance="daily",
    )
    # Future score edits should not change earlier weights.
    assert np.allclose(
        w1.loc[idx[:-2]].to_numpy(),
        w_changed.loc[idx[:-2]].to_numpy(),
        equal_nan=True,
    )
