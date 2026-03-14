"""Research helpers for synthetic sector baskets and sector-rotation signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_lab.data.fundamentals import normalize_ticker_symbol
from quant_lab.strategies.topn import rebalance_mask


def load_sector_mapping(
    tickers: list[str],
    sector_map_path: str = "projects/dashboard_legacy/data/sp500_tickers.csv",
) -> dict[str, str]:
    """Load ticker->sector mapping from a local CSV with Symbol/Ticker and Sector/GICS Sector columns."""
    p = Path(sector_map_path)
    if not p.exists():
        raise FileNotFoundError(f"sector map not found: {sector_map_path}")
    df = pd.read_csv(p)
    ticker_col = next((c for c in ["Ticker", "ticker", "Symbol", "symbol"] if c in df.columns), None)
    sector_col = next((c for c in ["Sector", "sector", "GICS Sector", "gics sector"] if c in df.columns), None)
    if ticker_col is None or sector_col is None:
        raise ValueError("sector map must include ticker/symbol and sector columns")

    mapping = {
        normalize_ticker_symbol(str(t)): str(s).strip() if str(s).strip() else "UNKNOWN"
        for t, s in zip(df[ticker_col], df[sector_col])
    }
    out = {}
    for t in tickers:
        nt = normalize_ticker_symbol(str(t))
        if nt in mapping:
            out[t] = mapping[nt]
    return out


def build_sector_return_panel(
    close: pd.DataFrame,
    sector_by_ticker: dict[str, str],
    min_constituents: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Date x Sector equal-weight daily return panel from stock close prices."""
    if int(min_constituents) <= 0:
        raise ValueError("min_constituents must be > 0")
    px = close.astype(float).sort_index()
    ret = px.pct_change()
    sectors = sorted({str(v) for v in sector_by_ticker.values() if str(v).strip()})
    sec_ret = pd.DataFrame(np.nan, index=px.index, columns=sectors, dtype=float)
    sec_cnt = pd.DataFrame(0, index=px.index, columns=sectors, dtype=int)

    for sec in sectors:
        members = [t for t, s in sector_by_ticker.items() if str(s) == sec and t in ret.columns]
        if not members:
            continue
        sub = ret[members]
        cnt = sub.notna().sum(axis=1).astype(int)
        mean_ret = sub.mean(axis=1, skipna=True)
        sec_cnt.loc[:, sec] = cnt
        sec_ret.loc[:, sec] = mean_ret.where(cnt >= int(min_constituents), np.nan)

    return sec_ret, sec_cnt


def build_sector_price_panel(
    sector_returns: pd.DataFrame,
    start_value: float = 100.0,
) -> pd.DataFrame:
    """Construct pseudo-price panel from sector returns."""
    if float(start_value) <= 0:
        raise ValueError("start_value must be > 0")
    r = sector_returns.astype(float).fillna(0.0)
    p = float(start_value) * (1.0 + r).cumprod()
    return p.astype(float)


def compute_sector_momentum_12_1(
    sector_price: pd.DataFrame,
    skip_days: int = 21,
    lookback_days: int = 252,
) -> pd.DataFrame:
    """Momentum signal: price[t-skip]/price[t-lookback] - 1."""
    if int(skip_days) <= 0 or int(lookback_days) <= int(skip_days):
        raise ValueError("require lookback_days > skip_days > 0")
    p = sector_price.astype(float)
    return p.shift(int(skip_days)) / p.shift(int(lookback_days)) - 1.0


def build_monthly_topk_weights(
    signal: pd.DataFrame,
    top_k: int = 3,
    rebalance: str = "monthly",
    lag_days: int = 1,
) -> pd.DataFrame:
    """Build deterministic monthly top-k equal weights using lagged signals."""
    if int(top_k) <= 0:
        raise ValueError("top_k must be > 0")
    if int(lag_days) < 0:
        raise ValueError("lag_days must be >= 0")

    sig = signal.astype(float).sort_index()
    sig_lag = sig.shift(int(lag_days))
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns, dtype=float)
    rb = rebalance_mask(pd.DatetimeIndex(sig.index), rebalance)
    current = pd.Series(0.0, index=sig.columns, dtype=float)
    for dt in sig.index:
        if bool(rb.loc[dt]):
            row = sig_lag.loc[dt].dropna()
            if row.empty:
                current = pd.Series(0.0, index=sig.columns, dtype=float)
            else:
                ordered = row.sort_values(ascending=False, kind="mergesort")
                chosen = ordered.head(int(top_k)).index.tolist()
                new_w = pd.Series(0.0, index=sig.columns, dtype=float)
                new_w.loc[chosen] = 1.0 / float(len(chosen))
                current = new_w
        w.loc[dt] = current
    return w.astype(float)


def annual_turnover(weights: pd.DataFrame, rebalance: str = "monthly") -> float:
    """Approximate annual turnover from rebalance-day average turnover."""
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
