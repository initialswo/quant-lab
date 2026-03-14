"""Research-only cross-asset trend benchmark helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from quant_lab.strategies.topn import rebalance_mask


PREFERRED_ASSET_CANDIDATES: list[str] = [
    "SPY",
    "TLT",
    "GLD",
    "DBC",
    "VNQ",
    "SHY",
    "EFA",
    "EEM",
    "IEF",
    "LQD",
    "TIP",
    "UUP",
    "FXE",
]


def resolve_available_assets(
    available_tickers: Iterable[str],
    preferred_assets: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """
    Resolve available local symbols for preferred macro assets.

    Matching priority per preferred asset:
    1) exact symbol (e.g., SPY)
    2) `.US` suffix variant (e.g., SPY.US)
    """
    preferred = list(preferred_assets or PREFERRED_ASSET_CANDIDATES)
    avail_set = {str(x).strip().upper() for x in available_tickers}
    used: list[str] = []
    mapping: dict[str, str] = {}
    for root in preferred:
        r = str(root).strip().upper()
        if not r:
            continue
        for cand in [r, f"{r}.US"]:
            if cand in avail_set:
                used.append(cand)
                mapping[r] = cand
                break
    return used, mapping


def compute_trend_signal_12_1(
    close: pd.DataFrame,
    skip_days: int = 21,
    lookback_days: int = 252,
) -> pd.DataFrame:
    """Compute 12_1-style trend signal per asset: close[t-skip] / close[t-lookback] - 1."""
    if int(skip_days) <= 0 or int(lookback_days) <= int(skip_days):
        raise ValueError("require lookback_days > skip_days > 0")
    c = close.astype(float).sort_index()
    signal = c.shift(int(skip_days)) / c.shift(int(lookback_days)) - 1.0
    return signal.astype(float)


def build_cross_asset_trend_weights(
    trend_signal: pd.DataFrame,
    rebalance: str = "monthly",
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    Build equal-weight long-only weights on assets with positive lagged trend.

    Fallback when no positive-trend assets: zero exposure.
    """
    if int(lag_days) < 0:
        raise ValueError("lag_days must be >= 0")
    sig = trend_signal.astype(float).sort_index()
    sig_lag = sig.shift(int(lag_days))
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns, dtype=float)
    rb = rebalance_mask(pd.DatetimeIndex(sig.index), rebalance)
    current = pd.Series(0.0, index=sig.columns, dtype=float)
    for dt in sig.index:
        if bool(rb.loc[dt]):
            row = sig_lag.loc[dt]
            picked = row[(row > 0.0) & row.notna()].index.tolist()
            if not picked:
                current = pd.Series(0.0, index=sig.columns, dtype=float)
            else:
                nxt = pd.Series(0.0, index=sig.columns, dtype=float)
                nxt.loc[picked] = 1.0 / float(len(picked))
                current = nxt
        w.loc[dt] = current
    return w.astype(float)


def annual_turnover(weights: pd.DataFrame, rebalance: str = "monthly") -> float:
    """Approximate annual turnover from rebalance-day mean turnover."""
    w = weights.astype(float).sort_index()
    td = 0.5 * w.diff().abs().sum(axis=1).fillna(0.0)
    rb = rebalance_mask(pd.DatetimeIndex(w.index), rebalance)
    trb = td.loc[rb]
    if trb.empty:
        return float("nan")
    per_reb = float(trb.mean())
    reb_per_year = 12.0 if str(rebalance).lower() == "monthly" else 52.0
    if str(rebalance).lower() == "daily":
        reb_per_year = 252.0
    return float(per_reb * reb_per_year)
