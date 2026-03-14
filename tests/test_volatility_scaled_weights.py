from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.strategies.topn import build_topn_weights


def test_volatility_scaled_disabled_matches_equal_weight() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    cols = ["A", "B", "C"]
    close = pd.DataFrame(
        {
            "A": np.linspace(100, 120, len(idx)),
            "B": np.linspace(100, 121, len(idx)),
            "C": np.linspace(100, 122, len(idx)),
        },
        index=idx,
    )
    scores = pd.DataFrame(1.0, index=idx, columns=cols)
    w0 = build_topn_weights(
        scores=scores,
        close=close,
        top_n=3,
        rebalance="daily",
        weighting="equal",
        volatility_scaled_weights=False,
    )
    w1 = build_topn_weights(
        scores=scores,
        close=close,
        top_n=3,
        rebalance="daily",
        weighting="equal",
        volatility_scaled_weights=False,
    )
    assert np.allclose(w0.to_numpy(), w1.to_numpy(), equal_nan=True)


def test_volatility_scaled_weights_favor_low_vol_names() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    close = pd.DataFrame(
        {
            "LOWV": np.linspace(100, 105, len(idx)),
            "HIGHV": 100 + np.sin(np.arange(len(idx))) * 8.0,
        },
        index=idx,
    )
    scores = pd.DataFrame({"LOWV": 1.0, "HIGHV": 1.0}, index=idx)
    w = build_topn_weights(
        scores=scores,
        close=close,
        top_n=2,
        rebalance="daily",
        weighting="equal",
        volatility_scaled_weights=True,
        vol_lookback=20,
    )
    dt = idx[-1]
    assert float(w.loc[dt, "LOWV"]) > float(w.loc[dt, "HIGHV"])


def test_volatility_scaled_weights_sum_to_one_when_invested() -> None:
    idx = pd.date_range("2024-01-01", periods=35, freq="B")
    close = pd.DataFrame(
        {
            "A": np.linspace(100, 110, len(idx)),
            "B": np.linspace(100, 108, len(idx)),
            "C": np.linspace(100, 106, len(idx)),
        },
        index=idx,
    )
    scores = pd.DataFrame(
        {
            "A": np.linspace(0.2, 0.8, len(idx)),
            "B": np.linspace(0.3, 0.7, len(idx)),
            "C": np.linspace(0.4, 0.6, len(idx)),
        },
        index=idx,
    )
    w = build_topn_weights(
        scores=scores,
        close=close,
        top_n=3,
        rebalance="daily",
        weighting="equal",
        volatility_scaled_weights=True,
        vol_lookback=20,
    )
    gross = w.sum(axis=1)
    invested = gross > 0.0
    assert np.allclose(gross.loc[invested].to_numpy(), np.ones(int(invested.sum())), atol=1e-10)

