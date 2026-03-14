"""Research helpers for a growth-leader style long-only equity sleeve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.factors.combine import aggregate_factor_scores
from quant_lab.factors.registry import compute_factors
from quant_lab.factors.earnings_yield import compute as compute_earnings_yield
from quant_lab.factors.gross_profitability import compute as compute_gross_profitability
from quant_lab.strategies.topn import build_topn_weights, rebalance_mask, simulate_portfolio


def apply_growth_screen(
    fundamentals_aligned: dict[str, pd.DataFrame] | None,
    prices: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    min_price: float = 10.0,
    min_avg_dollar_volume: float = 5_000_000.0,
    adv_lookback: int = 20,
    require_positive_momentum: bool = False,
    momentum_lookback_long: int = 252,
    momentum_lookback_short: int = 21,
) -> pd.DataFrame:
    """
    Build a growth-leader style eligibility mask.

    Rules:
    - positive gross profitability
    - positive earnings yield if the required fundamentals are available
    - price > min_price
    - avg dollar volume >= min_avg_dollar_volume
    - optional positive momentum
    """
    close = prices.astype(float).sort_index()
    mask = pd.DataFrame(True, index=close.index, columns=close.columns, dtype=bool)

    gp = compute_gross_profitability(close=close, fundamentals_aligned=fundamentals_aligned)
    mask = mask & gp.gt(0.0).fillna(False)

    has_earnings_inputs = bool(
        fundamentals_aligned
        and "net_income" in fundamentals_aligned
        and "shares_outstanding" in fundamentals_aligned
    )
    if has_earnings_inputs:
        ey = compute_earnings_yield(close=close, fundamentals_aligned=fundamentals_aligned)
        mask = mask & ey.gt(0.0).fillna(False)

    mask = mask & close.gt(float(min_price)).fillna(False)

    if volume is not None:
        vol = volume.astype(float).reindex(index=close.index, columns=close.columns)
        adv = (close * vol).rolling(int(adv_lookback)).mean()
        mask = mask & adv.ge(float(min_avg_dollar_volume)).fillna(False)

    if bool(require_positive_momentum):
        mom = close.pct_change(int(momentum_lookback_long)) - close.pct_change(int(momentum_lookback_short))
        mask = mask & mom.gt(0.0).fillna(False)

    return mask.astype(bool)


def build_growth_scores(
    prices: pd.DataFrame,
    fundamentals_aligned: dict[str, pd.DataFrame] | None,
    screen_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build composite growth-leader scores using benchmark factor stack.

    Factors:
    - momentum_12_1
    - reversal_1m
    - low_vol_20
    - gross_profitability
    """
    close = prices.astype(float).sort_index()
    factor_names = ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"]
    factor_params = {
        "gross_profitability": {"fundamentals_aligned": fundamentals_aligned},
    }
    panels = compute_factors(factor_names=factor_names, close=close, factor_params=factor_params)
    if screen_mask is not None:
        elig = screen_mask.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
        panels = {k: v.where(elig) for k, v in panels.items()}
    weights = {name: 1.0 for name in factor_names}
    scores = aggregate_factor_scores(
        scores=panels,
        weights=weights,
        method="geometric_rank",
        require_all_factors=False,
    )
    return scores.astype(float)


def _annual_turnover(weights: pd.DataFrame, rebalance: str = "weekly") -> float:
    w = weights.astype(float).sort_index()
    td = 0.5 * w.diff().abs().sum(axis=1).fillna(0.0)
    rb = rebalance_mask(pd.DatetimeIndex(w.index), rebalance)
    trb = td.loc[rb]
    if trb.empty:
        return float("nan")
    per_reb = float(trb.mean())
    reb_per_year = 52.0 if str(rebalance).lower() == "weekly" else 12.0
    if str(rebalance).lower() == "daily":
        reb_per_year = 252.0
    return float(per_reb * reb_per_year)


def run_growth_leader_backtest(
    score_panel: pd.DataFrame,
    close_prices: pd.DataFrame,
    top_n: int,
    rebalance: str = "weekly",
    weighting: str = "equal",
    costs_bps: float = 10.0,
    vol_lookback: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run long-only top-N growth-leader backtest."""
    scores = score_panel.astype(float).sort_index()
    close = close_prices.astype(float).reindex(index=scores.index, columns=scores.columns)
    weights = build_topn_weights(
        scores=scores,
        close=close,
        top_n=int(top_n),
        rebalance=rebalance,
        weighting=weighting,
        vol_lookback=int(vol_lookback),
    )
    rb = rebalance_mask(pd.DatetimeIndex(weights.index), rebalance)
    rb_dates = pd.DatetimeIndex(weights.index[rb])
    sim = simulate_portfolio(
        close=close,
        weights=weights,
        costs_bps=float(costs_bps),
        slippage_bps=0.0,
        slippage_vol_mult=0.0,
        slippage_vol_lookback=20,
        rebalance_dates=rb_dates,
        execution_delay_days=0,
    )
    m = compute_metrics(sim["DailyReturn"])
    summary = {
        "CAGR": float(m.get("CAGR", np.nan)),
        "Vol": float(m.get("Vol", np.nan)),
        "Sharpe": float(m.get("Sharpe", np.nan)),
        "MaxDD": float(m.get("MaxDD", np.nan)),
        "AnnualTurnover": float(_annual_turnover(weights, rebalance=rebalance)),
    }
    return sim.astype(float), weights.astype(float), summary
