from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.research.sector_baskets import (
    build_monthly_topk_weights,
    build_sector_price_panel,
    build_sector_return_panel,
    compute_sector_momentum_12_1,
)


def test_sector_return_panel_equal_weight_and_min_constituents() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    close = pd.DataFrame(
        {
            "A": [100, 101, 102, 103, 104],
            "B": [100, 102, 104, 106, 108],
            "C": [100, 99, 98, 97, 96],
            "D": [100, 100, 100, 100, 100],
        },
        index=idx,
    )
    sector = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Health"}
    sec_ret, sec_cnt = build_sector_return_panel(close=close, sector_by_ticker=sector, min_constituents=2)
    # Tech day 2 return is avg(1%,2%) = 1.5%
    assert float(sec_ret.loc[idx[1], "Tech"]) == pytest.approx(0.015, abs=1e-12)
    assert int(sec_cnt.loc[idx[1], "Tech"]) == 2
    # With min_constituents=2 and first day NaN returns, first day must be NaN.
    assert np.isnan(float(sec_ret.loc[idx[0], "Tech"]))


def test_monthly_topk_selection_and_determinism() -> None:
    idx = pd.date_range("2024-01-01", periods=65, freq="B")
    sig = pd.DataFrame(
        {
            "Tech": np.linspace(0.10, 0.30, len(idx)),
            "Health": np.linspace(0.05, 0.10, len(idx)),
            "Fin": np.linspace(0.03, 0.08, len(idx)),
            "Util": np.linspace(0.01, 0.02, len(idx)),
        },
        index=idx,
    )
    w1 = build_monthly_topk_weights(signal=sig, top_k=2, rebalance="monthly", lag_days=1)
    w2 = build_monthly_topk_weights(signal=sig, top_k=2, rebalance="monthly", lag_days=1)
    assert np.allclose(w1.to_numpy(), w2.to_numpy(), equal_nan=True)
    gross = w1.sum(axis=1)
    invested = gross > 0.0
    assert np.allclose(gross.loc[invested].to_numpy(), np.ones(int(invested.sum())), atol=1e-12)


def test_no_lookahead_lag_behavior() -> None:
    idx = pd.date_range("2024-01-01", periods=70, freq="B")
    sig = pd.DataFrame(
        {
            "Tech": np.linspace(0.1, 0.2, len(idx)),
            "Health": np.linspace(0.08, 0.15, len(idx)),
            "Fin": np.linspace(0.05, 0.14, len(idx)),
        },
        index=idx,
    )
    w_base = build_monthly_topk_weights(signal=sig, top_k=1, rebalance="monthly", lag_days=1)
    sig_edit = sig.copy()
    sig_edit.iloc[-1, 0] = 999.0
    w_edit = build_monthly_topk_weights(signal=sig_edit, top_k=1, rebalance="monthly", lag_days=1)
    assert np.allclose(w_base.iloc[:-1].to_numpy(), w_edit.iloc[:-1].to_numpy(), equal_nan=True)


def test_momentum_formula_from_pseudo_prices() -> None:
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    rets = pd.DataFrame({"Tech": 0.001, "Health": 0.0005}, index=idx)
    p = build_sector_price_panel(rets, start_value=100.0)
    m = compute_sector_momentum_12_1(p, skip_days=21, lookback_days=252)
    dt = idx[-1]
    exp = float(p.loc[idx[-1 - 21], "Tech"] / p.loc[idx[-1 - 252], "Tech"] - 1.0)
    assert float(m.loc[dt, "Tech"]) == pytest.approx(exp, rel=1e-12, abs=1e-12)
