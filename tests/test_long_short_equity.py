from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.research.long_short_equity import (
    build_long_short_weights,
    run_long_short_backtest,
    simulate_long_short_portfolio,
)


def _sample_close(periods: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "A": np.linspace(100.0, 140.0, len(idx)),
            "B": np.linspace(100.0, 130.0, len(idx)),
            "C": np.linspace(100.0, 120.0, len(idx)),
            "D": np.linspace(100.0, 110.0, len(idx)),
            "E": np.linspace(100.0, 105.0, len(idx)),
            "F": np.linspace(100.0, 102.0, len(idx)),
        },
        index=idx,
    )


def _sample_scores(index: pd.DatetimeIndex) -> pd.DataFrame:
    cols = ["A", "B", "C", "D", "E", "F"]
    row = np.array([0.9, 0.7, 0.4, -0.1, -0.5, -0.9], dtype=float)
    vals = np.tile(row, (len(index), 1))
    return pd.DataFrame(vals, index=index, columns=cols)


def test_weights_sign_structure() -> None:
    close = _sample_close(periods=20)
    scores = _sample_scores(close.index)
    w = build_long_short_weights(
        scores=scores,
        close=close,
        long_n=2,
        short_n=2,
        rebalance="daily",
        weighting="equal",
    )
    for _, row in w.iterrows():
        pos = row[row > 0.0]
        neg = row[row < 0.0]
        assert len(pos) == 2
        assert len(neg) == 2


def test_side_sums_match_target_exposure() -> None:
    close = _sample_close(periods=20)
    scores = _sample_scores(close.index)
    w = build_long_short_weights(
        scores=scores,
        close=close,
        long_n=2,
        short_n=2,
        rebalance="daily",
        weighting="equal",
        gross_exposure=1.0,
        net_exposure=0.0,
    )
    long_sum = w.clip(lower=0.0).sum(axis=1)
    short_sum = w.clip(upper=0.0).sum(axis=1)
    assert np.allclose(long_sum.to_numpy(), 0.5, atol=1e-12)
    assert np.allclose(short_sum.to_numpy(), -0.5, atol=1e-12)


def test_equal_weighting_behavior() -> None:
    close = _sample_close(periods=10)
    scores = _sample_scores(close.index)
    w = build_long_short_weights(
        scores=scores,
        close=close,
        long_n=2,
        short_n=2,
        rebalance="daily",
        weighting="equal",
    )
    row = w.iloc[-1]
    assert row["A"] == pytest.approx(0.25, abs=1e-12)
    assert row["B"] == pytest.approx(0.25, abs=1e-12)
    assert row["E"] == pytest.approx(-0.25, abs=1e-12)
    assert row["F"] == pytest.approx(-0.25, abs=1e-12)


def test_inv_vol_weighting_runs_and_normalizes() -> None:
    close = _sample_close(periods=60)
    scores = _sample_scores(close.index)
    w = build_long_short_weights(
        scores=scores,
        close=close,
        long_n=3,
        short_n=2,
        rebalance="weekly",
        weighting="inv_vol",
    )
    long_sum = w.clip(lower=0.0).sum(axis=1)
    short_sum = w.clip(upper=0.0).sum(axis=1)
    invested = (w.abs().sum(axis=1) > 0.0)
    assert invested.any()
    assert np.allclose(long_sum.loc[invested].to_numpy(), 0.5, atol=1e-8)
    assert np.allclose(short_sum.loc[invested].to_numpy(), -0.5, atol=1e-8)


def test_simulation_outputs_required_columns() -> None:
    close = _sample_close(periods=40)
    scores = _sample_scores(close.index)
    w = build_long_short_weights(
        scores=scores,
        close=close,
        long_n=2,
        short_n=2,
        rebalance="weekly",
        weighting="equal",
    )
    sim = simulate_long_short_portfolio(
        close=close,
        weights=w,
        costs_bps=10.0,
        slippage_bps=0.0,
        execution_delay_days=0,
    )
    assert {"Equity", "DailyReturn", "Turnover", "EffectiveCostBps"}.issubset(set(sim.columns))


def test_end_to_end_runs() -> None:
    close = _sample_close(periods=90)
    scores = _sample_scores(close.index)
    sim, weights, summary = run_long_short_backtest(
        scores=scores,
        close=close,
        long_n=2,
        short_n=2,
        rebalance="weekly",
        weighting="inv_vol",
        costs_bps=10.0,
    )
    assert len(sim) == len(close)
    assert len(weights) == len(close)
    assert sim["Equity"].notna().all()
    assert float(sim["Equity"].iloc[0]) == pytest.approx(1.0, abs=1e-12)
    for k in ["CAGR", "Vol", "Sharpe", "MaxDD", "AnnualTurnover", "AvgGrossExposure", "AvgNetExposure"]:
        assert k in summary
