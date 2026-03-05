import itertools

import numpy as np
import pandas as pd

from qre.metrics import max_drawdown, perf_stats
from qre.portfolio import apply_vol_target, sleeve_returns
from qre.signals import compute_momentum, compute_trend_strength, compute_vol_score, dual_momentum_alloc


def run_backtest(
    prices: pd.DataFrame,
    params: dict,
    return_series: bool = False,
    debug_selection: bool = False,
    log_holdings: bool = False,
) -> dict:
    returns = prices.pct_change()

    target_vol = params["target_vol"]
    roll_vol = params["roll_vol"]
    ma_trend = params["ma_trend"]
    lookback_mom = params["lookback_mom"]
    use_cov = params["use_cov"]
    leverage_hard_cap = params["leverage_hard_cap"]
    cash_vol_floor = params["cash_vol_floor"]
    bear_target_vol = float(params.get("bear_target_vol", 0.10))
    cost_bps = float(params.get("cost_bps", 0.0))
    cost_rate = cost_bps / 10000.0
    smoothing_alpha = float(params.get("smoothing_alpha", 1.0))
    weight_floor = float(params.get("weight_floor", 1e-4))
    top_k = int(params.get("top_k", 1))
    cash_ticker = str(params.get("cash_ticker", "SGOV"))
    abs_mom_threshold = float(params.get("abs_mom_threshold", 0.0))
    weight_method = str(params.get("weight_method", "inv_cov" if use_cov else "equal")).lower()

    cash_rets = pd.Series(0.0, index=returns.index)

    momentum = compute_momentum(prices, lookback=lookback_mom)
    allocation = dual_momentum_alloc(
        prices,
        momentum,
        rebalance="ME",
        smoothing_alpha=smoothing_alpha,
        weight_floor=weight_floor,
        top_k=top_k,
        use_cov=use_cov,
        roll_vol=roll_vol,
        weight_method=weight_method,
        cash_ticker=cash_ticker,
        abs_mom_threshold=abs_mom_threshold,
    )
    dm_returns = sleeve_returns(returns, allocation)

    # Monthly rebalance turnover from allocation change dates (trading-day aligned).
    turnover_raw = allocation.diff().abs().sum(axis=1)
    turnover_rebalance = turnover_raw[turnover_raw > 0.0]
    turnover = turnover_rebalance.iloc[1:] if len(turnover_rebalance) > 1 else turnover_rebalance.iloc[0:0]
    avg_turnover = float(turnover.mean()) if len(turnover) > 0 else 0.0
    ann_turnover = avg_turnover * 12.0
    total_turnover = float(turnover.sum()) if len(turnover) > 0 else 0.0
    rebalance_dates = turnover_rebalance.index

    trend_strength = compute_trend_strength(prices, ma_trend=ma_trend)
    effective_target_vol = pd.Series(float(target_vol), index=prices.index, dtype=float)
    effective_target_vol = effective_target_vol.where(trend_strength >= 0.0, float(bear_target_vol))
    trend_scale = 0.015
    trend_score = 1.0 / (1.0 + np.exp(-(trend_strength / trend_scale)))

    vol_score = compute_vol_score(returns, roll_vol=roll_vol)

    trend_score = trend_score.fillna(0.5)
    vol_score = vol_score.fillna(0.5)

    risk_budget = (1.0 - 0.10 * vol_score).clip(lower=0.60, upper=1.0)

    spy_w = (trend_score * risk_budget).clip(0.0, 1.0)
    dm_w = ((1.0 - trend_score) * risk_budget).clip(0.0, 1.0)
    cash_w = (1.0 - (spy_w + dm_w)).clip(0.0, 1.0)

    weights = pd.DataFrame({"SPY": spy_w, "DM": dm_w, "CASH": cash_w}, index=prices.index)

    gross_cap = 0.5 + 1.0 * trend_score
    gross_cap = gross_cap * (1.0 - 0.40 * vol_score)
    gross_cap = gross_cap.clip(lower=0.35, upper=leverage_hard_cap)

    sleeves = pd.DataFrame({
        "SPY": returns["SPY"],
        "DM": dm_returns,
        "CASH": cash_rets,
    })

    strat_gross = apply_vol_target(
        returns=sleeves,
        weights=weights,
        target_vol=effective_target_vol,
        roll_vol=roll_vol,
        gross_cap=gross_cap,
        use_cov=use_cov,
        leverage_hard_cap=leverage_hard_cap,
        cash_vol_floor=cash_vol_floor,
    )
    cost_series = turnover * cost_rate
    cost_series_daily = cost_series.reindex(strat_gross.index).fillna(0.0)
    strat = strat_gross - cost_series_daily

    if len(strat) < 1000:
        return {"ok": False}

    stats_net = perf_stats(strat)
    stats_gross = perf_stats(strat_gross)
    ann_cost_drag = float(cost_series.mean() * 12.0) if len(cost_series) > 0 else 0.0
    out = {
        "ok": True,
        **stats_net,
        "max_dd": max_drawdown(strat),
        "cost_bps": cost_bps,
        "ann_cost_drag": ann_cost_drag,
        "ann_ret_gross": float(stats_gross["ann_ret"]),
        "ann_ret_net": float(stats_net["ann_ret"]),
        "avg_turnover": avg_turnover,
        "ann_turnover": float(ann_turnover),
        "total_turnover": total_turnover,
        "target_vol": float(target_vol),
        "bear_target_vol": float(bear_target_vol),
        "roll_vol": int(roll_vol),
        "ma_trend": int(ma_trend),
        "lookback_mom": int(lookback_mom),
        "use_cov": bool(use_cov),
        "top_k": int(top_k),
        "weight_method": weight_method,
        "cash_ticker": cash_ticker,
        "abs_mom_threshold": abs_mom_threshold,
        "smoothing_alpha": smoothing_alpha,
        "weight_floor": weight_floor,
    }

    # Holdings summary is always included and remains compact.
    holdings_count = (
        (allocation.loc[rebalance_dates] >= weight_floor).sum(axis=1)
        if len(rebalance_dates) > 0
        else pd.Series(dtype=float)
    )
    if len(holdings_count) > 0:
        q = holdings_count.quantile([0.25, 0.5, 0.75])
        holdings_count_stats = {
            "min": int(holdings_count.min()),
            "p25": float(q.loc[0.25]),
            "median": float(q.loc[0.5]),
            "p75": float(q.loc[0.75]),
            "max": int(holdings_count.max()),
            "mean": float(holdings_count.mean()),
        }
        top_n_inferred = int(round(float(q.loc[0.5])))
    else:
        holdings_count_stats = {"min": 0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0, "mean": 0.0}
        top_n_inferred = 0

    out["holdings_count_series"] = [
        {"date": idx.strftime("%Y-%m-%d"), "count": int(val)}
        for idx, val in holdings_count.items()
    ]
    out["holdings_count_stats"] = holdings_count_stats
    out["top_n_inferred"] = top_n_inferred

    if return_series:
        out["returns"] = strat
        out["turnover_series"] = [
            {"date": idx.strftime("%Y-%m-%d"), "turnover": float(val)}
            for idx, val in turnover.items()
        ]
    if log_holdings or debug_selection:
        first_dates = list(rebalance_dates[:5])
        last_dates = list(rebalance_dates[-5:]) if len(rebalance_dates) > 0 else []
        top_turnover_dates = list(turnover.sort_values(ascending=False).head(5).index)

        selected_dates = sorted(set(first_dates + last_dates + top_turnover_dates))
        capped_snapshots = []
        for idx in selected_dates:
            row = allocation.loc[idx]
            holdings = [
                {"ticker": ticker, "w": float(w)}
                for ticker, w in row.items()
                if float(w) >= weight_floor
            ]
            capped_snapshots.append({"date": idx.strftime("%Y-%m-%d"), "holdings": holdings})

        if log_holdings:
            out["rebalance_holdings"] = capped_snapshots
        if debug_selection and not log_holdings:
            out["rebalance_holdings_preview"] = capped_snapshots

    return out


def sweep(prices: pd.DataFrame, grid: dict, dd_penalty_lambda: float) -> pd.DataFrame:
    results = []

    fields = ["target_vol", "ma_trend", "roll_vol", "lookback_mom", "use_cov", "top_k"]
    combos = itertools.product(
        grid["target_vols"],
        grid["ma_trends"],
        grid["roll_vols"],
        grid["lookback_moms"],
        grid["use_cov"],
        grid.get("top_ks", [1]),
    )

    for tv, ma, rv, mom, cov, top_k in combos:
        params = {
            "target_vol": tv,
            "roll_vol": rv,
            "ma_trend": ma,
            "lookback_mom": mom,
            "use_cov": cov,
            "top_k": top_k,
            "leverage_hard_cap": grid["leverage_hard_cap"],
            "cash_vol_floor": grid["cash_vol_floor"],
            "bear_target_vol": grid.get("bear_target_vol", 0.10),
            "cost_bps": grid.get("cost_bps", 0.0),
            "smoothing_alpha": grid.get("smoothing_alpha", 1.0),
            "weight_floor": grid.get("weight_floor", 1e-4),
            "cash_ticker": grid.get("cash_ticker", "SGOV"),
            "abs_mom_threshold": grid.get("abs_mom_threshold", 0.0),
            "weight_method": grid.get("weight_method", "inv_cov" if cov else "equal"),
        }
        out = run_backtest(prices, params=params, return_series=False)
        if out.get("ok"):
            results.append(out)

    if not results:
        return pd.DataFrame(columns=fields + ["ann_ret", "ann_vol", "sharpe", "max_dd", "score"])

    res = pd.DataFrame(results)
    res["score"] = res["sharpe"] - dd_penalty_lambda * res["max_dd"].abs()
    return res
