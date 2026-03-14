from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.engine.metrics import compute_daily_mark_to_market


def _sample_close() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "B": [100.0, 99.5, 100.5, 101.0, 100.0, 100.5, 101.5, 102.0, 103.0, 104.0],
        },
        index=idx,
    )


def test_mark_to_market_nonzero_between_rebalances() -> None:
    close = _sample_close()
    idx = close.index
    weights_rebal = pd.DataFrame(
        {"A": [0.5, 0.5], "B": [0.5, 0.5]},
        index=pd.DatetimeIndex([idx[0], idx[5]]),
    )
    rebalance_dates = pd.DatetimeIndex([idx[0], idx[5]])

    _, net_ret, _ = compute_daily_mark_to_market(
        close=close,
        weights_rebal=weights_rebal,
        rebalance_dates=rebalance_dates,
        costs_bps=0.0,
    )

    between = net_ret.iloc[1:5]
    assert not np.allclose(between.to_numpy(), 0.0)


def test_costs_applied_only_on_rebalance_dates() -> None:
    close = _sample_close()
    idx = close.index
    weights_rebal = pd.DataFrame(
        {"A": [0.5, 0.8], "B": [0.5, 0.2]},
        index=pd.DatetimeIndex([idx[0], idx[5]]),
    )
    rebalance_dates = pd.DatetimeIndex([idx[0], idx[5]])

    _, net_ret, weights_daily = compute_daily_mark_to_market(
        close=close,
        weights_rebal=weights_rebal,
        rebalance_dates=rebalance_dates,
        costs_bps=25.0,
    )

    asset_ret = close.pct_change().fillna(0.0)
    gross_ret = (weights_daily.shift(1).fillna(0.0) * asset_ret).sum(axis=1)
    gross_ret.iloc[0] = 0.0
    cost_impact = gross_ret - net_ret
    nonzero_days = cost_impact[np.abs(cost_impact) > 1e-15].index

    assert set(nonzero_days).issubset(set(rebalance_dates))


def test_equity_consistency() -> None:
    close = _sample_close()
    idx = close.index
    weights_rebal = pd.DataFrame(
        {"A": [0.6, 0.4], "B": [0.4, 0.6]},
        index=pd.DatetimeIndex([idx[0], idx[5]]),
    )
    rebalance_dates = pd.DatetimeIndex([idx[0], idx[5]])

    equity, net_ret, _ = compute_daily_mark_to_market(
        close=close,
        weights_rebal=weights_rebal,
        rebalance_dates=rebalance_dates,
        costs_bps=10.0,
    )

    for i in range(1, len(equity)):
        lhs = float(equity.iloc[i])
        rhs = float(equity.iloc[i - 1] * (1.0 + net_ret.iloc[i]))
        assert np.isclose(lhs, rhs, atol=1e-12, rtol=1e-12)
