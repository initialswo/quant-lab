from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.research.cross_asset_trend import (
    build_cross_asset_trend_weights,
    compute_trend_signal_12_1,
    resolve_available_assets,
)


def test_resolve_available_assets_prefers_exact_then_dotus() -> None:
    available = ["SPY.US", "GLD.US", "VNQ.US", "TLT"]
    used, mapping = resolve_available_assets(available_tickers=available, preferred_assets=["SPY", "TLT", "GLD"])
    assert used == ["SPY.US", "TLT", "GLD.US"]
    assert mapping == {"SPY": "SPY.US", "TLT": "TLT", "GLD": "GLD.US"}


def test_signal_formula_12_1() -> None:
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    close = pd.DataFrame({"A": np.linspace(100.0, 200.0, len(idx))}, index=idx)
    sig = compute_trend_signal_12_1(close=close, skip_days=21, lookback_days=252)
    dt = idx[-1]
    exp = float(close.loc[idx[-1 - 21], "A"] / close.loc[idx[-1 - 252], "A"] - 1.0)
    assert float(sig.loc[dt, "A"]) == pytest.approx(exp, rel=1e-12, abs=1e-12)


def test_monthly_positive_filter_and_zero_fallback() -> None:
    idx = pd.date_range("2024-01-01", periods=65, freq="B")
    sig = pd.DataFrame(
        {
            "A": np.linspace(0.2, 0.1, len(idx)),
            "B": np.linspace(-0.1, -0.2, len(idx)),
            "C": np.linspace(0.05, -0.05, len(idx)),
        },
        index=idx,
    )
    w = build_cross_asset_trend_weights(trend_signal=sig, rebalance="monthly", lag_days=1)
    # On first rebalance day lagged signal is NaN -> zero exposure fallback.
    assert float(w.iloc[0].sum()) == pytest.approx(0.0, abs=1e-12)
    # Later, only positive assets should be selected.
    nonzero_days = w.sum(axis=1) > 0
    if bool(nonzero_days.any()):
        chosen = (w.loc[nonzero_days] > 0).astype(int)
        assert (chosen["B"] == 0).all()


def test_deterministic_and_lag_safe_behavior() -> None:
    idx = pd.date_range("2024-01-01", periods=70, freq="B")
    sig = pd.DataFrame(
        {
            "A": np.linspace(0.01, 0.10, len(idx)),
            "B": np.linspace(0.02, 0.08, len(idx)),
            "C": np.linspace(-0.01, 0.03, len(idx)),
        },
        index=idx,
    )
    w1 = build_cross_asset_trend_weights(trend_signal=sig, rebalance="monthly", lag_days=1)
    w2 = build_cross_asset_trend_weights(trend_signal=sig, rebalance="monthly", lag_days=1)
    assert np.allclose(w1.to_numpy(), w2.to_numpy(), equal_nan=True)

    sig2 = sig.copy()
    sig2.iloc[-1, 0] = 999.0
    w3 = build_cross_asset_trend_weights(trend_signal=sig2, rebalance="monthly", lag_days=1)
    assert np.allclose(w1.iloc[:-1].to_numpy(), w3.iloc[:-1].to_numpy(), equal_nan=True)
