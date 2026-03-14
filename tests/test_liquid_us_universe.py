from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.universe.liquid_us import build_liquid_us_universe


def test_liquid_us_universe_returns_boolean_matrix() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = pd.DataFrame(
        {
            "A": 20.0 + np.arange(len(idx), dtype=float),
            "B": 8.0 + np.arange(len(idx), dtype=float),
        },
        index=idx,
    )
    volumes = pd.DataFrame(1_000_000.0, index=idx, columns=prices.columns)

    elig = build_liquid_us_universe(
        prices=prices,
        volumes=volumes,
        min_price=10.0,
        min_avg_dollar_volume=10_000_000.0,
        adv_window=5,
        min_history=10,
    )
    assert elig.shape == prices.shape
    assert elig.index.equals(prices.index)
    assert elig.columns.tolist() == prices.columns.tolist()
    assert set(pd.unique(elig.to_numpy().ravel())) <= {True, False}


def test_liquid_us_universe_filters_applied() -> None:
    idx = pd.date_range("2024-01-01", periods=25, freq="B")
    prices = pd.DataFrame(
        {
            "PASS": 20.0 + np.arange(len(idx), dtype=float),
            "LOW_PRICE": 4.0 + np.arange(len(idx), dtype=float) * 0.01,
            "LOW_ADV": 30.0 + np.arange(len(idx), dtype=float),
            "SHORT_HIST": np.nan,
        },
        index=idx,
    )
    prices.loc[idx[10]:, "SHORT_HIST"] = 25.0 + np.arange(len(idx) - 10, dtype=float)
    volumes = pd.DataFrame(
        {
            "PASS": 1_000_000.0,
            "LOW_PRICE": 2_000_000.0,
            "LOW_ADV": 50_000.0,
            "SHORT_HIST": 1_000_000.0,
        },
        index=idx,
    )

    elig = build_liquid_us_universe(
        prices=prices,
        volumes=volumes,
        min_price=5.0,
        min_avg_dollar_volume=10_000_000.0,
        adv_window=5,
        min_history=10,
    )

    assert bool(elig["PASS"].iloc[-1])
    assert not bool(elig["LOW_PRICE"].iloc[-1])
    assert not bool(elig["LOW_ADV"].iloc[-1])
    assert not bool(elig["SHORT_HIST"].iloc[12])  # insufficient history near start
