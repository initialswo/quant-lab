from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.risk.regime import (
    build_regime_weight_series,
    compute_regime_label,
    parse_weight_map,
    should_apply_dynamic_factor_weights,
)


def test_parse_weight_map_basic_and_empty() -> None:
    assert parse_weight_map("") == {}
    assert parse_weight_map("   ") == {}
    out = parse_weight_map("momentum_12_1:0.7, low_vol_20:0.3")
    assert out == {"momentum_12_1": 0.7, "low_vol_20": 0.3}


def test_parse_weight_map_invalid_cases() -> None:
    with pytest.raises(ValueError):
        parse_weight_map("badtoken")
    with pytest.raises(ValueError):
        parse_weight_map("x:-0.1")
    with pytest.raises(ValueError):
        parse_weight_map("x:0,y:0")


def test_regime_label_is_shifted_by_one() -> None:
    idx = pd.date_range("2021-01-01", periods=8, freq="B")
    spy = pd.Series([100, 101, 102, 99, 100, 103, 104, 105], index=idx, dtype=float)

    label = compute_regime_label(
        spy_close=spy,
        score_index=idx,
        trend_sma=2,
        vol_lookback=2,
        vol_median_lookback=2,
    )

    sma = spy.rolling(2).mean()
    bull_trend = spy > sma
    rv = spy.pct_change().rolling(2).std(ddof=0) * np.sqrt(252.0)
    rv_med = rv.rolling(2).median()
    high_vol = rv > rv_med
    valid = sma.notna() & rv.notna() & rv_med.notna()
    pre = pd.Series(np.nan, index=idx, dtype=object)
    pre.loc[valid & bull_trend & (~high_vol)] = "bull"
    pre.loc[valid & ~(bull_trend & (~high_vol))] = "bear_or_volatile"

    # No-lookahead: shifted final label.
    assert pd.isna(label.iloc[0])
    for i in range(1, len(idx)):
        if pd.isna(label.iloc[i]):
            continue
        assert label.iloc[i] == pre.iloc[i - 1]


def test_regime_label_keeps_early_dates_nan_and_first_valid_after_shift() -> None:
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    spy = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx, dtype=float)
    trend_sma = 5
    vol_lookback = 3
    vol_median_lookback = 4

    label = compute_regime_label(
        spy_close=spy,
        score_index=idx,
        trend_sma=trend_sma,
        vol_lookback=vol_lookback,
        vol_median_lookback=vol_median_lookback,
    )

    sma = spy.rolling(trend_sma).mean()
    rv = spy.pct_change().rolling(vol_lookback).std(ddof=0) * np.sqrt(252.0)
    rv_med = rv.rolling(vol_median_lookback).median()
    valid = sma.notna() & rv.notna() & rv_med.notna()

    first_signal_dt = valid[valid].index[0]
    first_label_dt = idx[idx.get_loc(first_signal_dt) + 1]

    assert label.loc[idx[idx < first_label_dt]].isna().all()
    assert pd.notna(label.loc[first_label_dt])


def test_build_regime_weight_series_applies_bull_bear_and_normalizes() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    label = pd.Series(["bull", "bear_or_volatile", np.nan], index=idx, dtype=object)
    names = ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"]
    static = {
        "momentum_12_1": 0.43,
        "reversal_1m": 0.27,
        "low_vol_20": 0.15,
        "gross_profitability": 0.15,
    }
    bull = {
        "momentum_12_1": 0.48,
        "reversal_1m": 0.22,
        "low_vol_20": 0.10,
        "gross_profitability": 0.20,
    }
    bear = {
        "momentum_12_1": 0.28,
        "reversal_1m": 0.22,
        "low_vol_20": 0.30,
        "gross_profitability": 0.20,
    }
    ws = build_regime_weight_series(
        factor_names=names,
        static_weights=static,
        label=label,
        bull_weights=bull,
        bear_weights=bear,
    )
    wdf = pd.DataFrame({k: v for k, v in ws.items()}, index=idx)

    assert abs(float(wdf.loc[idx[0], "momentum_12_1"]) - 0.48) < 1e-12
    assert abs(float(wdf.loc[idx[1], "low_vol_20"]) - 0.30) < 1e-12
    assert abs(float(wdf.loc[idx[2], "reversal_1m"]) - 0.27) < 1e-12
    assert np.allclose(wdf.sum(axis=1).to_numpy(), np.ones(len(idx)))


def test_dynamic_weight_enablement_back_compat_and_deterministic() -> None:
    assert should_apply_dynamic_factor_weights(regime_filter=False, dynamic_factor_weights=False) is False
    assert should_apply_dynamic_factor_weights(regime_filter=True, dynamic_factor_weights=False) is True
    assert should_apply_dynamic_factor_weights(regime_filter=False, dynamic_factor_weights=True) is True
    assert should_apply_dynamic_factor_weights(regime_filter=True, dynamic_factor_weights=True) is True
