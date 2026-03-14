from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.research.sector_rotation import (
    build_sector_rotation_weights,
    compute_sector_momentum_signals,
    run_sector_rotation_backtest,
)


def _synthetic_close(periods: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "XLF": np.linspace(100.0, 135.0, len(idx)),
            "XLK": np.linspace(90.0, 160.0, len(idx)),
            "XLV": np.linspace(95.0, 120.0, len(idx)),
            "XLU": np.linspace(105.0, 110.0, len(idx)),
        },
        index=idx,
    )


def test_signal_generation_shape_and_lag_safety() -> None:
    close = _synthetic_close(periods=80)
    sig1 = compute_sector_momentum_signals(close=close, lookback=20, signal_type="relative")
    assert sig1.shape == close.shape
    assert sig1.index.equals(close.index)
    assert sig1.columns.tolist() == close.columns.tolist()
    assert sig1.iloc[:21].isna().all().all()

    close2 = close.copy()
    close2.iloc[-1, 0] = close2.iloc[-1, 0] * 10.0
    sig2 = compute_sector_momentum_signals(close=close2, lookback=20, signal_type="relative")
    assert np.allclose(sig1.to_numpy(), sig2.to_numpy(), equal_nan=True)


def test_top_n_selection_daily_equal_weights() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    scores = pd.DataFrame(
        {
            "A": [3.0, 3.0, 3.0, 3.0, 3.0],
            "B": [2.0, 2.0, 2.0, 2.0, 2.0],
            "C": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=idx,
    )
    close = pd.DataFrame(100.0, index=idx, columns=scores.columns)
    w = build_sector_rotation_weights(
        scores=scores,
        close=close,
        top_n=2,
        rebalance="daily",
        weighting="equal",
    )
    assert np.allclose(w["A"].to_numpy(), 0.5)
    assert np.allclose(w["B"].to_numpy(), 0.5)
    assert np.allclose(w["C"].to_numpy(), 0.0)


def test_weights_sum_to_one_when_invested() -> None:
    close = _synthetic_close(periods=100)
    scores = compute_sector_momentum_signals(close=close, lookback=20, signal_type="absolute")
    w = build_sector_rotation_weights(
        scores=scores,
        close=close,
        top_n=3,
        rebalance="monthly",
        weighting="inv_vol",
    )
    gross = w.sum(axis=1)
    invested = gross > 0.0
    assert invested.any()
    assert np.allclose(gross.loc[invested].to_numpy(), 1.0, atol=1e-10)
    assert (w >= -1e-12).all().all()


def test_backtest_has_required_output_columns() -> None:
    close = _synthetic_close(periods=110)
    sim, summary = run_sector_rotation_backtest(
        close=close,
        lookback=20,
        signal_type="relative",
        top_n=2,
        weighting="equal",
        rebalance="monthly",
        costs_bps=5.0,
    )
    assert {"Equity", "DailyReturn", "Turnover"}.issubset(set(sim.columns))
    for k in ["CAGR", "Vol", "Sharpe", "MaxDD", "AnnualTurnover"]:
        assert k in summary


def test_synthetic_end_to_end_runs() -> None:
    close = _synthetic_close(periods=130)
    sim, summary = run_sector_rotation_backtest(
        close=close,
        lookback=63,
        signal_type="absolute",
        top_n=3,
        weighting="inv_vol",
        rebalance="monthly",
        costs_bps=5.0,
    )
    assert len(sim) == len(close)
    assert sim["Equity"].notna().all()
    assert pytest.approx(float(sim["Equity"].iloc[0]), rel=0, abs=1e-12) == 1.0
    assert np.isfinite(float(summary["CAGR"]))
