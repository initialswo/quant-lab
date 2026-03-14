"""Regime detection and dynamic factor-weight helpers."""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd


def parse_weight_map(raw: str) -> dict[str, float]:
    """Parse `factor:weight,...` into a validated nonnegative map."""
    if raw is None:
        return {}
    txt = raw.strip()
    if not txt:
        return {}

    out: dict[str, float] = {}
    for token in [x.strip() for x in txt.split(",") if x.strip()]:
        if ":" not in token:
            raise ValueError(f"Invalid regime weight token '{token}'. Expected factor:weight.")
        name, value = token.split(":", 1)
        factor = name.strip()
        if not factor:
            raise ValueError(f"Invalid regime weight token '{token}': empty factor name.")
        try:
            w = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid regime weight token '{token}': bad numeric weight.") from exc
        if w < 0:
            raise ValueError(f"Invalid regime weight token '{token}': weight must be >= 0.")
        out[factor] = w

    if out and float(sum(out.values())) <= 0.0:
        raise ValueError("Regime weight map cannot have all-zero weights.")
    return out


def weight_map_to_json(weight_map: dict[str, float]) -> str:
    """Stable JSON encoding for logging."""
    return json.dumps(weight_map, sort_keys=True, separators=(",", ":"))


def compute_regime_label(
    spy_close: pd.Series,
    score_index: pd.DatetimeIndex,
    trend_sma: int = 200,
    vol_lookback: int = 20,
    vol_median_lookback: int = 252,
) -> pd.Series:
    """
    Compute no-lookahead regime labels aligned to score index.

    Returned labels are shifted by 1 trading day before use.
    Values: 'bull', 'bear_or_volatile', or NaN.
    """
    if trend_sma <= 0 or vol_lookback <= 0 or vol_median_lookback <= 0:
        raise ValueError("Regime lookbacks must be > 0.")

    spy = pd.Series(spy_close).astype(float).sort_index()
    sma = spy.rolling(trend_sma).mean()
    bull_trend = spy > sma

    daily_ret = spy.pct_change()
    rv = daily_ret.rolling(vol_lookback).std(ddof=0) * np.sqrt(252.0)
    rv_med = rv.rolling(vol_median_lookback).median()
    high_vol = rv > rv_med

    valid = sma.notna() & rv.notna() & rv_med.notna()
    bull = bull_trend & (~high_vol)
    label = pd.Series(np.nan, index=spy.index, dtype=object)
    label.loc[valid & bull] = "bull"
    label.loc[valid & (~bull)] = "bear_or_volatile"
    # No-lookahead requirement: shift final label by one trading day.
    label = label.shift(1)
    return label.reindex(pd.DatetimeIndex(score_index))


def build_regime_weight_series(
    factor_names: Iterable[str],
    static_weights: dict[str, float],
    label: pd.Series,
    bull_weights: dict[str, float],
    bear_weights: dict[str, float],
) -> dict[str, pd.Series]:
    """
    Build per-date factor-weight series from regime labels.

    - NaN label -> static weights for that date.
    - Partial maps fall back factor-by-factor to static weights.
    - Per-date weights are normalized to sum to 1 (or static fallback if degenerate).
    """
    names = [str(x) for x in factor_names]
    idx = pd.DatetimeIndex(label.index)
    static = {n: float(static_weights.get(n, 0.0)) for n in names}
    static_sum = float(sum(static.values()))
    if static_sum <= 0.0:
        raise ValueError("Static factor weights must sum to > 0.")
    static = {k: v / static_sum for k, v in static.items()}

    out = {n: pd.Series(index=idx, dtype=float) for n in names}

    for dt in idx:
        lbl = label.loc[dt]
        if pd.isna(lbl):
            raw = dict(static)
        elif str(lbl) == "bull":
            raw = {n: float(bull_weights.get(n, static[n])) for n in names}
        else:
            raw = {n: float(bear_weights.get(n, static[n])) for n in names}

        total = float(sum(max(v, 0.0) for v in raw.values()))
        if total <= 0.0:
            raw = dict(static)
            total = float(sum(raw.values()))
        for n in names:
            out[n].loc[dt] = max(float(raw[n]), 0.0) / total

    return out


def should_apply_dynamic_factor_weights(
    regime_filter: bool,
    dynamic_factor_weights: bool,
) -> bool:
    """
    Resolve whether regime-aware factor weighting should be active.

    Backward-compatible behavior:
    - Existing regime_filter=True runs continue to use regime-aware weights.
    - dynamic_factor_weights=True enables regime-aware weights explicitly.
    """
    return bool(regime_filter or dynamic_factor_weights)
