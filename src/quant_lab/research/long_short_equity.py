"""Research helpers for a simple long/short equity sleeve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.strategies.topn import rebalance_mask, simulate_portfolio


def build_long_short_weights(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    long_n: int,
    short_n: int,
    rebalance: str,
    weighting: str = "equal",
    vol_lookback: int = 20,
    min_vol: float = 1e-6,
    gross_exposure: float = 1.0,
    net_exposure: float = 0.0,
) -> pd.DataFrame:
    """
    Build long/short basket weights on rebalance dates, then hold constant.

    Side targets:
    long_sum = (gross_exposure + net_exposure) / 2
    short_sum = -(gross_exposure - net_exposure) / 2
    """
    if int(long_n) <= 0 or int(short_n) <= 0:
        raise ValueError("long_n and short_n must both be > 0")
    if int(vol_lookback) <= 0:
        raise ValueError("vol_lookback must be > 0")
    if float(min_vol) <= 0:
        raise ValueError("min_vol must be > 0")
    if float(gross_exposure) <= 0:
        raise ValueError("gross_exposure must be > 0")
    if abs(float(net_exposure)) > float(gross_exposure):
        raise ValueError("abs(net_exposure) cannot exceed gross_exposure")

    method = str(weighting).lower().strip()
    if method not in {"equal", "inv_vol"}:
        raise ValueError("weighting must be 'equal' or 'inv_vol'")

    s = scores.astype(float).sort_index()
    px = close.astype(float).reindex(index=s.index, columns=s.columns)
    vol = px.pct_change().rolling(int(vol_lookback)).std(ddof=0).shift(1)

    long_target = (float(gross_exposure) + float(net_exposure)) / 2.0
    short_target_abs = (float(gross_exposure) - float(net_exposure)) / 2.0

    rb = rebalance_mask(pd.DatetimeIndex(s.index), rebalance)
    w = pd.DataFrame(0.0, index=s.index, columns=s.columns, dtype=float)
    current = pd.Series(0.0, index=s.columns, dtype=float)

    for dt in s.index:
        if bool(rb.loc[dt]):
            row = s.loc[dt].dropna().sort_values(ascending=False, kind="mergesort")
            if row.empty:
                current = pd.Series(0.0, index=s.columns, dtype=float)
                w.loc[dt] = current
                continue

            long_sel = row.head(int(long_n)).index.tolist()
            short_pool = row.drop(index=long_sel, errors="ignore")
            short_sel = short_pool.tail(int(short_n)).index.tolist()
            if not short_sel:
                short_sel = row.tail(int(short_n)).index.tolist()
                short_sel = [x for x in short_sel if x not in set(long_sel)]

            nxt = pd.Series(0.0, index=s.columns, dtype=float)
            if method == "equal":
                if long_sel:
                    nxt.loc[long_sel] = long_target / float(len(long_sel))
                if short_sel:
                    nxt.loc[short_sel] = -short_target_abs / float(len(short_sel))
            else:
                if long_sel:
                    long_vol = pd.to_numeric(vol.loc[dt, long_sel], errors="coerce")
                    long_valid = long_vol[(long_vol > 0.0) & long_vol.notna()]
                    if long_valid.empty:
                        nxt.loc[long_sel] = long_target / float(len(long_sel))
                    else:
                        inv = 1.0 / long_valid.clip(lower=float(min_vol))
                        inv = inv / float(inv.sum())
                        nxt.loc[inv.index] = inv.to_numpy(dtype=float) * float(long_target)
                if short_sel:
                    short_vol = pd.to_numeric(vol.loc[dt, short_sel], errors="coerce")
                    short_valid = short_vol[(short_vol > 0.0) & short_vol.notna()]
                    if short_valid.empty:
                        nxt.loc[short_sel] = -short_target_abs / float(len(short_sel))
                    else:
                        inv = 1.0 / short_valid.clip(lower=float(min_vol))
                        inv = inv / float(inv.sum())
                        nxt.loc[inv.index] = -inv.to_numpy(dtype=float) * float(short_target_abs)
            current = nxt
        w.loc[dt] = current
    return w.astype(float)


def simulate_long_short_portfolio(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    costs_bps: float,
    slippage_bps: float = 0.0,
    execution_delay_days: int = 0,
) -> pd.DataFrame:
    """
    Simulate causal long/short portfolio returns with turnover-based costs.

    portfolio_return[t] = sum_i(weights[t-1, i] * asset_return[t, i]) - cost[t]
    """
    w = weights.astype(float).sort_index()
    turnover = 0.5 * w.diff().abs().sum(axis=1).fillna(0.0)
    rb_dates = pd.DatetimeIndex(w.index[turnover > 0.0])
    sim = simulate_portfolio(
        close=close,
        weights=w,
        costs_bps=float(costs_bps),
        slippage_bps=float(slippage_bps),
        slippage_vol_mult=0.0,
        slippage_vol_lookback=20,
        rebalance_dates=rb_dates,
        execution_delay_days=int(execution_delay_days),
    )
    return sim


def _annual_turnover(weights: pd.DataFrame, rebalance: str) -> float:
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


def run_long_short_backtest(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    long_n: int,
    short_n: int,
    rebalance: str = "weekly",
    weighting: str = "equal",
    vol_lookback: int = 20,
    costs_bps: float = 10.0,
    slippage_bps: float = 0.0,
    execution_delay_days: int = 0,
    gross_exposure: float = 1.0,
    net_exposure: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Build long/short weights, simulate, and summarize performance."""
    weights = build_long_short_weights(
        scores=scores,
        close=close,
        long_n=int(long_n),
        short_n=int(short_n),
        rebalance=rebalance,
        weighting=weighting,
        vol_lookback=int(vol_lookback),
        min_vol=1e-6,
        gross_exposure=float(gross_exposure),
        net_exposure=float(net_exposure),
    )
    sim = simulate_long_short_portfolio(
        close=close.reindex(index=weights.index, columns=weights.columns),
        weights=weights,
        costs_bps=float(costs_bps),
        slippage_bps=float(slippage_bps),
        execution_delay_days=int(execution_delay_days),
    )
    m = compute_metrics(sim["DailyReturn"])
    summary = {
        "CAGR": float(m.get("CAGR", np.nan)),
        "Vol": float(m.get("Vol", np.nan)),
        "Sharpe": float(m.get("Sharpe", np.nan)),
        "MaxDD": float(m.get("MaxDD", np.nan)),
        "AnnualTurnover": float(_annual_turnover(weights, rebalance=rebalance)),
        "AvgGrossExposure": float(weights.abs().sum(axis=1).mean()),
        "AvgNetExposure": float(weights.sum(axis=1).mean()),
    }
    return sim.astype(float), weights.astype(float), summary
