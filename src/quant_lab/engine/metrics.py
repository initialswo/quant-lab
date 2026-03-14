"""Performance metrics."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def compute_daily_mark_to_market(
    close: pd.DataFrame,
    weights_rebal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    costs_bps: float,
    slippage_bps: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Compute daily mark-to-market portfolio equity from lagged holdings."""
    close = close.astype(float).sort_index()
    weights_daily = (
        weights_rebal.reindex(close.index)
        .ffill()
        .reindex(columns=close.columns, fill_value=0.0)
        .fillna(0.0)
        .astype(float)
    )

    asset_ret_raw = close.pct_change()
    w_prev = weights_daily.shift(1).fillna(0.0)
    missing_with_weight = asset_ret_raw.isna() & (w_prev.abs() > 0.0)
    missing_with_weight_count = int(missing_with_weight.to_numpy().sum())
    if missing_with_weight_count > 0:
        warnings.warn(
            "MTM: encountered NaN asset returns with non-zero prior weights; treating contributions as 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
    asset_ret = asset_ret_raw.fillna(0.0)
    gross_ret = (w_prev * asset_ret).sum(axis=1, min_count=1).fillna(0.0)
    if not gross_ret.empty:
        gross_ret.iloc[0] = 0.0

    turnover = 0.5 * weights_daily.diff().abs().sum(axis=1).fillna(0.0)
    rebalance_set = set(pd.DatetimeIndex(rebalance_dates))
    rebalance_mask = pd.Series(weights_daily.index.isin(rebalance_set), index=weights_daily.index)
    cost_ret = pd.Series(0.0, index=weights_daily.index, dtype=float)
    cost_per_turnover = float(costs_bps + slippage_bps) / 10000.0
    cost_ret.loc[rebalance_mask] = turnover.loc[rebalance_mask] * cost_per_turnover

    net_ret = gross_ret - cost_ret
    equity = (1.0 + net_ret).cumprod()
    return equity.rename("Equity"), net_ret.rename("DailyReturn"), weights_daily


def compute_metrics(daily_return_series: pd.Series) -> dict[str, float]:
    """Compute CAGR, annualized volatility, Sharpe, and max drawdown."""
    r = pd.Series(daily_return_series).dropna().astype(float)
    if r.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}

    n = len(r)
    growth = (1.0 + r).prod()
    cagr = growth ** (252.0 / n) - 1.0

    vol_raw = r.std(ddof=0)
    vol = vol_raw * np.sqrt(252.0)
    sharpe = (r.mean() / vol_raw) * np.sqrt(252.0) if vol_raw > 0 else np.nan

    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_dd = drawdown.min()

    return {"CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe), "MaxDD": float(max_dd)}
