from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.factors.registry import compute_factor, list_factors


def _make_base_close(n: int = 260) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    a = 100.0 * np.power(1.0020, np.arange(n))  # stronger trend
    b = 100.0 * np.power(1.0010, np.arange(n))  # weaker trend
    c = 100.0 * np.power(0.9990, np.arange(n))  # down trend
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


def test_new_factor_names_discoverable() -> None:
    names = set(list_factors())
    assert "momentum_6_1" in names
    assert "momentum_3_1" in names
    assert "reversal_1m" in names
    assert "low_vol_60" in names


def test_factor_outputs_exist_on_toy_data() -> None:
    close = _make_base_close()
    for name in ["momentum_6_1", "momentum_3_1", "reversal_1m", "low_vol_60"]:
        out = compute_factor(name, close)
        assert isinstance(out, pd.DataFrame)
        assert out.shape == close.shape
        assert out.notna().any().any()


def test_momentum_6_1_and_3_1_rank_stronger_historical_return_higher() -> None:
    close = _make_base_close()
    m6 = compute_factor("momentum_6_1", close)
    m3 = compute_factor("momentum_3_1", close)
    dt6 = m6.dropna(how="all").index[-1]
    dt3 = m3.dropna(how="all").index[-1]
    assert m6.loc[dt6, "A"] > m6.loc[dt6, "B"] > m6.loc[dt6, "C"]
    assert m3.loc[dt3, "A"] > m3.loc[dt3, "B"] > m3.loc[dt3, "C"]


def test_reversal_1m_rewards_recent_losers() -> None:
    idx = pd.date_range("2021-01-01", periods=80, freq="B")
    w = np.concatenate([np.full(59, 100.0), np.linspace(100.0, 120.0, 21)])
    l = np.concatenate([np.full(59, 100.0), np.linspace(100.0, 80.0, 21)])
    close = pd.DataFrame({"WINNER": w, "LOSER": l}, index=idx)
    rev = compute_factor("reversal_1m", close)
    dt = rev.dropna(how="all").index[-1]
    assert rev.loc[dt, "LOSER"] > rev.loc[dt, "WINNER"]


def test_low_vol_60_rewards_smoother_series() -> None:
    idx = pd.date_range("2021-01-01", periods=220, freq="B")
    smooth = 100.0 * np.power(1.0008, np.arange(len(idx)))
    zig = smooth.copy()
    zig[1::2] = zig[1::2] * 1.02
    zig[::2] = zig[::2] * 0.98
    close = pd.DataFrame({"SMOOTH": smooth, "NOISY": zig}, index=idx)
    lv = compute_factor("low_vol_60", close)
    dt = lv.dropna(how="all").index[-1]
    assert lv.loc[dt, "SMOOTH"] > lv.loc[dt, "NOISY"]


def test_warmup_and_missing_values_handled_safely() -> None:
    close = _make_base_close()
    close.iloc[:10, 0] = np.nan
    close.iloc[50:70, 1] = np.nan

    m3 = compute_factor("momentum_3_1", close)
    lv60 = compute_factor("low_vol_60", close)

    assert m3.iloc[:63].isna().all().all()
    assert lv60.iloc[:60].isna().all().all()
    assert m3.notna().any().any()
    assert lv60.notna().any().any()

