"""Simple strategy allocator research functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_rows(weights: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(weights, dtype=float).copy()
    row_sum = w.sum(axis=1).replace(0.0, np.nan)
    w = w.div(row_sum, axis=0)
    return w


def static_weights(panel: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Return constant portfolio weights aligned to panel index."""
    cols = [str(c) for c in panel.columns]
    if set(weights.keys()) != set(cols):
        raise ValueError("weights keys must match panel columns exactly.")
    row = pd.Series({k: float(v) for k, v in weights.items()}, dtype=float).reindex(cols)
    if not np.isfinite(row.to_numpy(dtype=float)).all():
        raise ValueError("weights must be finite.")
    total = float(row.sum())
    if abs(total) <= 1e-12:
        raise ValueError("weights sum must be non-zero.")
    row = row / total
    out = pd.DataFrame([row.to_dict()] * len(panel.index), index=panel.index, columns=cols, dtype=float)
    out.index = pd.DatetimeIndex(out.index)
    out.index.name = "date"
    return out


def inverse_vol_allocator(panel: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """Compute lag-safe inverse-volatility strategy weights."""
    if int(lookback) <= 1:
        raise ValueError("lookback must be > 1.")
    p = pd.DataFrame(panel, dtype=float).copy()
    vol = p.rolling(int(lookback)).std(ddof=0)
    inv_vol = 1.0 / vol.replace(0.0, np.nan)
    w_target = _normalize_rows(inv_vol)
    equal_row = pd.Series(1.0 / float(len(p.columns)), index=p.columns, dtype=float)
    w_target = w_target.fillna(equal_row.to_dict())
    w = w_target.shift(1).fillna(equal_row.to_dict())
    w = _normalize_rows(w).fillna(equal_row.to_dict())
    return w.astype(float)


def smoothed_inverse_vol_allocator(
    panel: pd.DataFrame,
    lookback: int = 63,
    smoothing: float = 0.2,
) -> pd.DataFrame:
    """Compute exponentially smoothed inverse-volatility allocator weights."""
    alpha = float(smoothing)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("smoothing must be in (0, 1].")
    if int(lookback) <= 1:
        raise ValueError("lookback must be > 1.")

    p = pd.DataFrame(panel, dtype=float).copy()
    vol = p.rolling(int(lookback)).std(ddof=0)
    inv_vol = 1.0 / vol.replace(0.0, np.nan)
    target = _normalize_rows(inv_vol)
    equal_row = pd.Series(1.0 / float(len(p.columns)), index=p.columns, dtype=float)
    target = target.fillna(equal_row.to_dict())

    smoothed = pd.DataFrame(index=target.index, columns=target.columns, dtype=float)
    prev = target.iloc[0].astype(float).copy()
    smoothed.iloc[0] = prev
    for i in range(1, len(target.index)):
        curr = target.iloc[i].astype(float)
        prev = alpha * curr + (1.0 - alpha) * prev
        s = float(prev.sum())
        if not np.isfinite(s) or s == 0.0:
            prev = equal_row.copy()
        else:
            prev = prev / s
        smoothed.iloc[i] = prev

    w = smoothed.shift(1).fillna(equal_row.to_dict())
    w = _normalize_rows(w).fillna(equal_row.to_dict())
    return w.astype(float)


def simulate_allocator(panel_returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Simulate allocator returns/equity and include turnover series."""
    r = pd.DataFrame(panel_returns, dtype=float).copy()
    w = pd.DataFrame(weights, dtype=float).reindex(index=r.index, columns=r.columns)
    portfolio_return = (w.shift(1).fillna(0.0) * r).sum(axis=1).rename("returns")
    equity = (1.0 + portfolio_return.fillna(0.0)).cumprod().rename("equity")
    turnover = (0.5 * w.diff().abs().sum(axis=1)).fillna(0.0).rename("turnover")
    out = pd.concat([portfolio_return, equity, turnover], axis=1)
    out.index = pd.DatetimeIndex(out.index)
    out.index.name = "date"
    return out

