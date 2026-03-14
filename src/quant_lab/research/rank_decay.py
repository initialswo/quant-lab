"""Helpers for rank-decay / quantile portfolio experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.strategies.topn import rebalance_mask, simulate_portfolio


@dataclass(frozen=True)
class SleeveBacktestResult:
    """Container for one sleeve's simulation outputs and diagnostics."""

    sleeve: str
    weights: pd.DataFrame
    sim: pd.DataFrame
    summary: dict[str, float]
    names_by_rebalance: pd.Series
    rebalance_dates: pd.DatetimeIndex
    skipped_rebalance_dates: int


def _ordered_names(scores: pd.Series) -> list[str]:
    valid = pd.Series(scores).dropna().astype(float)
    if valid.empty:
        return []
    frame = valid.rename("score").reset_index()
    frame.columns = ["ticker", "score"]
    frame["ticker"] = frame["ticker"].astype(str)
    frame = frame.sort_values(["score", "ticker"], ascending=[True, True], kind="mergesort")
    return frame["ticker"].tolist()


def assign_quantile_buckets(
    scores: pd.Series,
    quantiles: int,
) -> dict[str, list[str]]:
    """
    Assign tickers into deterministic quantile buckets.

    Buckets are built from worst to best score, so Q1 is the lowest-score bucket
    and QN is the highest-score bucket. Ties are broken alphabetically by ticker.

    If fewer than `quantiles` names are available, the caller should skip the
    rebalance date rather than create empty sleeves.
    """
    if int(quantiles) < 2:
        raise ValueError("quantiles must be >= 2")

    ordered = _ordered_names(scores)
    if len(ordered) < int(quantiles):
        return {f"Q{i}": [] for i in range(1, int(quantiles) + 1)}

    splits = np.array_split(np.asarray(ordered, dtype=object), int(quantiles))
    return {
        f"Q{i + 1}": [str(x) for x in bucket.tolist()]
        for i, bucket in enumerate(splits)
    }


def build_rank_bucket_weights(
    scores: pd.DataFrame,
    quantiles: int,
    rebalance: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], dict[str, pd.DatetimeIndex], int]:
    """
    Build equal-weight long-only sleeves for each quantile bucket.

    On rebalance dates with fewer than `quantiles` valid names, the rebalance is
    skipped and prior holdings are carried forward. This avoids creating empty
    or ill-defined buckets in a small universe.
    """
    if int(quantiles) < 2:
        raise ValueError("quantiles must be >= 2")

    s = scores.astype(float).sort_index()
    rb = rebalance_mask(pd.DatetimeIndex(s.index), rebalance)
    sleeves = [f"Q{i}" for i in range(1, int(quantiles) + 1)]
    weights = {
        sleeve: pd.DataFrame(0.0, index=s.index, columns=s.columns, dtype=float)
        for sleeve in sleeves
    }
    names_by_date = {
        sleeve: pd.Series(np.nan, index=s.index, dtype=float)
        for sleeve in sleeves
    }
    executed_dates = {sleeve: [] for sleeve in sleeves}
    current = {
        sleeve: pd.Series(0.0, index=s.columns, dtype=float)
        for sleeve in sleeves
    }
    skipped = 0

    for dt in s.index:
        if bool(rb.loc[dt]):
            buckets = assign_quantile_buckets(s.loc[dt], quantiles=int(quantiles))
            if any(len(names) == 0 for names in buckets.values()):
                skipped += 1
            else:
                for sleeve in sleeves:
                    nxt = pd.Series(0.0, index=s.columns, dtype=float)
                    members = buckets[sleeve]
                    nxt.loc[members] = 1.0 / float(len(members))
                    current[sleeve] = nxt
                    names_by_date[sleeve].loc[dt] = float(len(members))
                    executed_dates[sleeve].append(pd.Timestamp(dt))
        for sleeve in sleeves:
            weights[sleeve].loc[dt] = current[sleeve]

    executed_idx = {
        sleeve: pd.DatetimeIndex(executed_dates[sleeve])
        for sleeve in sleeves
    }
    return weights, names_by_date, executed_idx, int(skipped)


def compute_turnover_stats(
    weights: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    rebalance: str,
) -> dict[str, float]:
    td = 0.5 * weights.astype(float).sort_index().diff().abs().sum(axis=1).fillna(0.0)
    rb_dates = pd.DatetimeIndex(rebalance_dates)
    td_reb = td.loc[td.index.isin(rb_dates)]
    if td_reb.empty:
        return {"avg_turnover": float("nan"), "annual_turnover": float("nan")}

    reb_per_year = 52.0
    freq = str(rebalance).lower().strip()
    if freq == "daily":
        reb_per_year = 252.0
    elif freq == "monthly":
        reb_per_year = 12.0
    elif freq == "biweekly":
        reb_per_year = 26.0
    return {
        "avg_turnover": float(td_reb.mean()),
        "annual_turnover": float(td_reb.mean() * reb_per_year),
    }


def run_rank_decay_backtest(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    quantiles: int = 5,
    rebalance: str = "weekly",
    costs_bps: float = 0.0,
    slippage_bps: float = 0.0,
    execution_delay_days: int = 0,
) -> dict[str, SleeveBacktestResult | pd.Series | int]:
    """Build quantile sleeves, simulate each sleeve, and derive the spread series."""
    s = scores.astype(float).sort_index()
    px = close.astype(float).reindex(index=s.index, columns=s.columns)
    sleeve_weights, names_by_date, executed_dates, skipped = build_rank_bucket_weights(
        scores=s,
        quantiles=int(quantiles),
        rebalance=rebalance,
    )

    sleeves = [f"Q{i}" for i in range(1, int(quantiles) + 1)]
    out: dict[str, SleeveBacktestResult | pd.Series | int] = {"skipped_rebalance_dates": int(skipped)}
    for sleeve in sleeves:
        sim = simulate_portfolio(
            close=px,
            weights=sleeve_weights[sleeve],
            costs_bps=float(costs_bps),
            slippage_bps=float(slippage_bps),
            slippage_vol_mult=0.0,
            slippage_vol_lookback=20,
            rebalance_dates=executed_dates[sleeve],
            execution_delay_days=int(execution_delay_days),
        )
        base = compute_metrics(sim["DailyReturn"])
        turnover = compute_turnover_stats(
            weights=sleeve_weights[sleeve],
            rebalance_dates=executed_dates[sleeve],
            rebalance=rebalance,
        )
        daily = pd.Series(sim["DailyReturn"], dtype=float)
        summary = {
            "CAGR": float(base.get("CAGR", np.nan)),
            "Vol": float(base.get("Vol", np.nan)),
            "Sharpe": float(base.get("Sharpe", np.nan)),
            "MaxDD": float(base.get("MaxDD", np.nan)),
            "HitRate": float((daily.dropna() > 0.0).mean()) if not daily.dropna().empty else float("nan"),
            "AvgTurnover": float(turnover["avg_turnover"]),
            "AnnualTurnover": float(turnover["annual_turnover"]),
            "NRebalanceDates": float(len(executed_dates[sleeve])),
            "MedianNames": float(names_by_date[sleeve].dropna().median())
            if not names_by_date[sleeve].dropna().empty
            else float("nan"),
        }
        out[sleeve] = SleeveBacktestResult(
            sleeve=sleeve,
            weights=sleeve_weights[sleeve],
            sim=sim.astype(float),
            summary=summary,
            names_by_rebalance=names_by_date[sleeve],
            rebalance_dates=executed_dates[sleeve],
            skipped_rebalance_dates=int(skipped),
        )

    top = out[f"Q{int(quantiles)}"]
    bottom = out["Q1"]
    assert isinstance(top, SleeveBacktestResult)
    assert isinstance(bottom, SleeveBacktestResult)
    spread = (
        pd.Series(top.sim["DailyReturn"], dtype=float) - pd.Series(bottom.sim["DailyReturn"], dtype=float)
    ).rename("Spread")
    out["Spread"] = spread.astype(float)
    return out


def monotonicity_score_from_cagr(values: list[float]) -> float:
    """
    Return a simple 0..1 monotonicity diagnostic from Q1..Qn CAGR values.

    The score is the Spearman rank correlation between bucket index and CAGR,
    transformed from [-1, 1] to [0, 1].
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2 or not np.isfinite(arr).all():
        return float("nan")
    idx = pd.Series(np.arange(1, arr.size + 1), dtype=float)
    cagr_rank = pd.Series(arr, dtype=float).rank(method="average")
    idx_rank = idx.rank(method="average")
    idx_std = float(idx_rank.std(ddof=0))
    cagr_std = float(cagr_rank.std(ddof=0))
    if idx_std <= 0.0 or cagr_std <= 0.0:
        return float("nan")
    score = float(((idx_rank - idx_rank.mean()) * (cagr_rank - cagr_rank.mean())).mean() / (idx_std * cagr_std))
    if not np.isfinite(score):
        return float("nan")
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
