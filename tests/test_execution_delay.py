from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.strategies.topn import simulate_portfolio


def _toy_close() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    return pd.DataFrame(
        {
            "A": [100.0, 100.0, 100.0, 120.0, 120.0, 120.0],
            "B": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=idx,
    )


def _toy_weights(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    idx = close.index
    weights = pd.DataFrame(
        {"A": [0.0, 1.0], "B": [1.0, 0.0]},
        index=pd.DatetimeIndex([idx[0], idx[2]]),
    )
    rebalances = pd.DatetimeIndex([idx[0], idx[2]])
    return weights, rebalances


def test_execution_delay_zero_matches_default_behavior() -> None:
    close = _toy_close()
    weights, rebalances = _toy_weights(close)

    out_default = simulate_portfolio(
        close=close,
        weights=weights,
        costs_bps=0.0,
        rebalance_dates=rebalances,
    )
    out_zero = simulate_portfolio(
        close=close,
        weights=weights,
        costs_bps=0.0,
        rebalance_dates=rebalances,
        execution_delay_days=0,
    )
    assert np.allclose(
        out_default["DailyReturn"].to_numpy(),
        out_zero["DailyReturn"].to_numpy(),
        equal_nan=True,
    )


def test_execution_delay_shifts_implementation_and_cost_date() -> None:
    close = _toy_close()
    weights, rebalances = _toy_weights(close)

    out = simulate_portfolio(
        close=close,
        weights=weights,
        costs_bps=10.0,
        rebalance_dates=rebalances,
        execution_delay_days=1,
    )
    # With 1-day delay, implementation turnover (and costs) occur on day 1 and day 3.
    changed = out["Turnover"] > 1e-12
    changed_days = list(out.index[changed])
    assert changed_days == [close.index[1], close.index[3]]


def test_execution_delay_has_no_forward_leakage() -> None:
    close = _toy_close()
    weights, rebalances = _toy_weights(close)

    out0 = simulate_portfolio(
        close=close,
        weights=weights,
        costs_bps=0.0,
        rebalance_dates=rebalances,
        execution_delay_days=0,
    )
    out1 = simulate_portfolio(
        close=close,
        weights=weights,
        costs_bps=0.0,
        rebalance_dates=rebalances,
        execution_delay_days=1,
    )
    jump_day = close.index[3]
    # Asset A jumps +20% on jump_day. Delay=0 captures it; delay=1 does not.
    assert np.isclose(float(out0.loc[jump_day, "DailyReturn"]), 0.2, atol=1e-12)
    assert np.isclose(float(out1.loc[jump_day, "DailyReturn"]), 0.0, atol=1e-12)


def test_execution_delay_deterministic() -> None:
    close = _toy_close()
    weights, rebalances = _toy_weights(close)

    out1 = simulate_portfolio(
        close=close,
        weights=weights,
        costs_bps=5.0,
        rebalance_dates=rebalances,
        execution_delay_days=3,
    )
    out2 = simulate_portfolio(
        close=close,
        weights=weights,
        costs_bps=5.0,
        rebalance_dates=rebalances,
        execution_delay_days=3,
    )
    assert np.allclose(out1["DailyReturn"].to_numpy(), out2["DailyReturn"].to_numpy(), equal_nan=True)
    assert np.allclose(out1["Equity"].to_numpy(), out2["Equity"].to_numpy(), equal_nan=True)
