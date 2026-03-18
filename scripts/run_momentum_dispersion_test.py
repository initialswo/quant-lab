#!/usr/bin/env python3
"""Run the post-fix momentum dispersion regime test against the canonical Quant Lab baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.engine.runner import (
    _collect_close_series,
    _collect_numeric_panel,
    _collect_price_series,
    _load_universe_seed_tickers,
    _prepare_close_panel,
    _price_panel_health_report,
    run_backtest,
)
from quant_lab.factors.registry import compute_factors
from quant_lab.strategies.topn import rebalance_mask
from quant_lab.universe.liquid_us import build_liquid_us_universe


RESULTS_ROOT = Path("results") / "momentum_dispersion_test"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_TOP_N = 75
DEFAULT_MAX_TICKERS = 2000
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
PRICE_FILL_MODE = "ffill"
MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 10_000_000.0
LIQUIDITY_LOOKBACK = 20
MIN_HISTORY_DAYS = 300
UNIVERSE_MIN_TICKERS = 20
MOMENTUM_FACTOR = "momentum_12_1"
BASELINE_FACTORS = ["gross_profitability", "reversal_1m"]
BASELINE_WEIGHTS = [0.7, 0.3]
BASELINE_AGGREGATION_METHOD = "linear"
DISPERSION_LABELS = ["low dispersion", "medium dispersion", "high dispersion"]
TIMESERIES_COLUMNS = [
    "date",
    "next_date",
    "n_eligible_stocks",
    "top_decile_mean_score",
    "bottom_decile_mean_score",
    "score_dispersion",
    "top_decile_forward_return",
    "bottom_decile_forward_return",
    "forward_return_spread",
    "market_forward_return",
    "dispersion_regime",
]
REGIME_SUMMARY_COLUMNS = [
    "dispersion_regime",
    "count_weeks",
    "mean_score_dispersion",
    "mean_forward_return_spread",
    "mean_market_return",
]
BASELINE_REGIME_COLUMNS = [
    "dispersion_regime",
    "count_weeks",
    "mean_weekly_return",
    "annualized_return",
    "annualized_volatility",
    "Sharpe",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--fundamentals-path", default="data/fundamentals/fundamentals_fmp.parquet")
    return parser.parse_args()



def _load_price_panels(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tickers = _load_universe_seed_tickers(
        universe=DEFAULT_UNIVERSE,
        max_tickers=int(DEFAULT_MAX_TICKERS),
        data_cache_dir=DATA_CACHE_DIR,
    )
    ohlcv_map, data_source_summary = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(start),
        end=str(end),
        cache_dir=DATA_CACHE_DIR,
        data_source=DATA_SOURCE,
        refresh=False,
        bulk_prepare=False,
    )
    close_cols, used_tickers, _, _, _ = _collect_close_series(ohlcv_map=ohlcv_map, requested_tickers=tickers)
    if len(used_tickers) < DEFAULT_TOP_N + 1:
        raise ValueError(
            f"Not enough tickers with data: got {len(used_tickers)}, need at least {DEFAULT_TOP_N + 1}."
        )

    close_raw = pd.concat(close_cols, axis=1, join="outer")
    adj_close_cols, _, _, _, _ = _collect_price_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Adj Close",
        fallback_field="Close",
    )
    adj_close_raw = pd.concat(adj_close_cols, axis=1, join="outer") if adj_close_cols else close_raw.copy()
    volume_raw = _collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Volume",
    )

    close = _prepare_close_panel(close_raw=close_raw, price_fill_mode=PRICE_FILL_MODE)
    adj_close = _prepare_close_panel(close_raw=adj_close_raw, price_fill_mode=PRICE_FILL_MODE)
    min_non_nan = int(0.8 * close.shape[1])
    close = close.dropna(thresh=min_non_nan)
    if close.empty:
        raise ValueError("No usable close-price history after alignment/fill.")

    _, broken_tickers, suspicious_tickers = _price_panel_health_report(
        close_raw=close_raw.reindex(index=close.index, columns=close.columns),
        close_filled=close,
    )
    if suspicious_tickers:
        print(
            "PRICE PANEL WARNING: long null runs detected "
            f"count={len(suspicious_tickers)} sample={suspicious_tickers[:10]}"
        )
    if broken_tickers:
        dropped = sorted(set(broken_tickers))
        print(
            "PRICE PANEL WARNING: dropping broken tickers "
            f"(close<=0 or max_abs_daily_return>200%) count={len(dropped)} sample={dropped[:10]}"
        )
        close = close.drop(columns=dropped, errors="ignore")
        adj_close = adj_close.drop(columns=dropped, errors="ignore")
        if close.shape[1] < DEFAULT_TOP_N + 1:
            raise ValueError(
                "Not enough tickers after dropping broken price series: "
                f"got {close.shape[1]}, need at least {DEFAULT_TOP_N + 1}."
            )

    volume = (
        volume_raw.reindex(index=close.index, columns=close.columns)
        if not volume_raw.empty
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    adj_close = adj_close.reindex(index=close.index, columns=close.columns).where(
        adj_close.reindex(index=close.index, columns=close.columns).notna(),
        close,
    )
    return close.astype(float), adj_close.astype(float), volume.astype(float), data_source_summary



def _compute_eligibility(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    eligibility = build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=MIN_PRICE,
        min_avg_dollar_volume=MIN_AVG_DOLLAR_VOLUME,
        adv_window=LIQUIDITY_LOOKBACK,
        min_history=MIN_HISTORY_DAYS,
    )
    return eligibility.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)



def _compute_momentum_panel(adj_close: pd.DataFrame) -> pd.DataFrame:
    raw_scores = compute_factors(
        factor_names=[MOMENTUM_FACTOR],
        close=adj_close,
        factor_params={},
    )
    return raw_scores[MOMENTUM_FACTOR].astype(float)



def _compute_weekly_forward_returns(adj_close: pd.DataFrame, rb_dates: pd.DatetimeIndex) -> pd.DataFrame:
    weekly_px = adj_close.reindex(rb_dates)
    return weekly_px.shift(-1).div(weekly_px).sub(1.0)



def _decile_mask(valid_scores: pd.Series) -> tuple[pd.Index, pd.Index]:
    if valid_scores.shape[0] < 10:
        return pd.Index([]), pd.Index([])
    ranks = valid_scores.rank(method="first")
    bins = pd.qcut(ranks, q=10, labels=False)
    bottom = bins.index[bins == 0]
    top = bins.index[bins == 9]
    return top, bottom



def _classify_dispersion_regimes(score_dispersion: pd.Series) -> pd.Series:
    valid = score_dispersion.dropna()
    if valid.empty:
        return pd.Series(index=score_dispersion.index, dtype="string")
    ranked = valid.rank(method="first")
    terciles = pd.qcut(ranked, q=3, labels=DISPERSION_LABELS)
    out = pd.Series(pd.NA, index=score_dispersion.index, dtype="string")
    out.loc[valid.index] = terciles.astype(str).values
    return out



def _build_dispersion_timeseries(
    momentum_scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    eligibility: pd.DataFrame,
    rb_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, dt in enumerate(rb_dates):
        next_dt = rb_dates[i + 1] if i + 1 < len(rb_dates) else pd.NaT
        scores = momentum_scores.loc[dt]
        elig = eligibility.loc[dt]
        valid_scores = scores.where(elig).dropna()
        n_eligible = int(valid_scores.shape[0])
        top_decile, bottom_decile = _decile_mask(valid_scores)

        row: dict[str, Any] = {
            "date": dt,
            "next_date": next_dt,
            "n_eligible_stocks": n_eligible,
            "top_decile_mean_score": float(valid_scores.loc[top_decile].mean()) if len(top_decile) else float("nan"),
            "bottom_decile_mean_score": float(valid_scores.loc[bottom_decile].mean()) if len(bottom_decile) else float("nan"),
            "score_dispersion": float("nan"),
            "top_decile_forward_return": float("nan"),
            "bottom_decile_forward_return": float("nan"),
            "forward_return_spread": float("nan"),
            "market_forward_return": float("nan"),
        }
        if len(top_decile) and len(bottom_decile):
            row["score_dispersion"] = float(row["top_decile_mean_score"] - row["bottom_decile_mean_score"])
        if pd.notna(next_dt) and len(top_decile) and len(bottom_decile):
            fwd = forward_returns.loc[dt]
            top_fwd = fwd.reindex(top_decile).dropna()
            bot_fwd = fwd.reindex(bottom_decile).dropna()
            market_fwd = fwd.reindex(valid_scores.index).dropna()
            row["top_decile_forward_return"] = float(top_fwd.mean()) if not top_fwd.empty else float("nan")
            row["bottom_decile_forward_return"] = float(bot_fwd.mean()) if not bot_fwd.empty else float("nan")
            if not top_fwd.empty and not bot_fwd.empty:
                row["forward_return_spread"] = float(row["top_decile_forward_return"] - row["bottom_decile_forward_return"])
            row["market_forward_return"] = float(market_fwd.mean()) if not market_fwd.empty else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows, columns=TIMESERIES_COLUMNS[:-1]).sort_values("date", kind="mergesort").reset_index(drop=True)
    out["dispersion_regime"] = _classify_dispersion_regimes(out["score_dispersion"])
    return out.loc[:, TIMESERIES_COLUMNS].copy()



def _summarize_regimes(timeseries: pd.DataFrame) -> pd.DataFrame:
    valid = timeseries.dropna(subset=["dispersion_regime", "forward_return_spread"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=REGIME_SUMMARY_COLUMNS)
    grouped = valid.groupby("dispersion_regime", observed=False)
    summary = grouped.agg(
        count_weeks=("date", "count"),
        mean_score_dispersion=("score_dispersion", "mean"),
        mean_forward_return_spread=("forward_return_spread", "mean"),
        mean_market_return=("market_forward_return", "mean"),
    ).reset_index()
    order = pd.Categorical(summary["dispersion_regime"], categories=DISPERSION_LABELS, ordered=True)
    return summary.assign(_order=order).sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)



def _run_baseline_backtest(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    cfg = {
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "universe_mode": "dynamic",
        "top_n": int(DEFAULT_TOP_N),
        "rebalance": DEFAULT_REBALANCE,
        "weighting": "equal",
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "max_tickers": DEFAULT_MAX_TICKERS,
        "data_source": DATA_SOURCE,
        "data_cache_dir": DATA_CACHE_DIR,
        "factor_name": list(BASELINE_FACTORS),
        "factor_names": list(BASELINE_FACTORS),
        "factor_weights": list(BASELINE_WEIGHTS),
        "portfolio_mode": "composite",
        "factor_aggregation_method": BASELINE_AGGREGATION_METHOD,
        "use_factor_normalization": True,
        "use_sector_neutralization": False,
        "use_size_neutralization": False,
        "orthogonalize_factors": False,
        "fundamentals_path": str(args.fundamentals_path),
        "save_artifacts": True,
    }
    return run_backtest(**cfg, run_cache={})



def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()



def _baseline_weekly_returns(backtest_outdir: str | Path, rb_dates: pd.DatetimeIndex) -> pd.DataFrame:
    equity = _read_csv(Path(backtest_outdir) / "equity.csv")
    if "Equity" not in equity.columns:
        raise ValueError(f"Missing Equity column in {Path(backtest_outdir) / 'equity.csv'}")
    eq = pd.to_numeric(equity["Equity"], errors="coerce").dropna().sort_index()
    sampled = eq.reindex(rb_dates).dropna()
    next_equity = sampled.shift(-1)
    weekly = next_equity.div(sampled).sub(1.0)
    out = pd.DataFrame({"date": sampled.index, "baseline_weekly_return": weekly.values})
    return out.iloc[:-1].reset_index(drop=True) if len(out) > 1 else out.iloc[0:0].copy()



def _summarize_baseline_by_regime(timeseries: pd.DataFrame, baseline_weekly: pd.DataFrame) -> pd.DataFrame:
    aligned = baseline_weekly.merge(
        timeseries.loc[:, ["date", "dispersion_regime"]],
        on="date",
        how="inner",
    ).dropna(subset=["dispersion_regime", "baseline_weekly_return"])
    if aligned.empty:
        return pd.DataFrame(columns=BASELINE_REGIME_COLUMNS)

    rows: list[dict[str, Any]] = []
    for label in DISPERSION_LABELS:
        sample = aligned.loc[aligned["dispersion_regime"] == label, "baseline_weekly_return"].astype(float)
        if sample.empty:
            rows.append(
                {
                    "dispersion_regime": label,
                    "count_weeks": 0,
                    "mean_weekly_return": float("nan"),
                    "annualized_return": float("nan"),
                    "annualized_volatility": float("nan"),
                    "Sharpe": float("nan"),
                }
            )
            continue
        mean_weekly = float(sample.mean())
        ann_return = float((1.0 + sample).prod() ** (52.0 / float(sample.shape[0])) - 1.0)
        ann_vol = float(sample.std(ddof=1) * np.sqrt(52.0)) if sample.shape[0] > 1 else float("nan")
        sharpe = float(sample.mean() / sample.std(ddof=1) * np.sqrt(52.0)) if sample.shape[0] > 1 and sample.std(ddof=1) > 0 else float("nan")
        rows.append(
            {
                "dispersion_regime": label,
                "count_weeks": int(sample.shape[0]),
                "mean_weekly_return": mean_weekly,
                "annualized_return": ann_return,
                "annualized_volatility": ann_vol,
                "Sharpe": sharpe,
            }
        )
    return pd.DataFrame(rows, columns=BASELINE_REGIME_COLUMNS)



def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return str(type(obj).__name__)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)



def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"



def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)



def _print_report(regime_summary: pd.DataFrame, baseline_by_regime: pd.DataFrame) -> None:
    print("")
    print("DISPERSION REGIME SUMMARY")
    print("-------------------------")
    print(f"{'Regime':18s} {'Weeks':>7s} {'ScoreDisp':>10s} {'FwdSpread':>10s} {'MktRet':>10s}")
    for _, row in regime_summary.iterrows():
        print(
            f"{str(row['dispersion_regime']):18s} "
            f"{int(row['count_weeks']):7d} "
            f"{_format_float(float(row['mean_score_dispersion'])):>10s} "
            f"{_format_float(float(row['mean_forward_return_spread'])):>10s} "
            f"{_format_float(float(row['mean_market_return'])):>10s}"
        )

    print("")
    print("BASELINE BY DISPERSION REGIME")
    print("-----------------------------")
    print(f"{'Regime':18s} {'Weeks':>7s} {'MeanWk':>10s} {'AnnRet':>10s} {'AnnVol':>10s} {'Sharpe':>8s}")
    for _, row in baseline_by_regime.iterrows():
        print(
            f"{str(row['dispersion_regime']):18s} "
            f"{int(row['count_weeks']):7d} "
            f"{_format_float(float(row['mean_weekly_return'])):>10s} "
            f"{_format_float(float(row['annualized_return'])):>10s} "
            f"{_format_float(float(row['annualized_volatility'])):>10s} "
            f"{_format_float(float(row['Sharpe'])):>8s}"
        )



def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("POST-FIX MOMENTUM DISPERSION TEST")
    print("---------------------------------")
    print(
        "Config: "
        f"universe={DEFAULT_UNIVERSE} rebalance={DEFAULT_REBALANCE} top_n={DEFAULT_TOP_N} costs_bps={DEFAULT_COSTS_BPS} "
        f"baseline_factors={','.join(BASELINE_FACTORS)} baseline_weights={BASELINE_WEIGHTS}"
    )
    print(
        "Note: score dispersion is top-decile minus bottom-decile mean raw momentum_12_1 on adjusted prices; "
        "forward spread is next-week adjusted return of top decile minus bottom decile."
    )

    t0 = time.perf_counter()
    close, adj_close, volume, data_source_summary = _load_price_panels(start=str(args.start), end=str(args.end))
    eligibility = _compute_eligibility(close=close, volume=volume)
    momentum_scores = _compute_momentum_panel(adj_close=adj_close)
    rb_mask = rebalance_mask(pd.DatetimeIndex(close.index), DEFAULT_REBALANCE).reindex(close.index).fillna(False)
    rb_dates = pd.DatetimeIndex(close.index[rb_mask])
    forward_returns = _compute_weekly_forward_returns(adj_close=adj_close, rb_dates=rb_dates)
    timeseries = _build_dispersion_timeseries(
        momentum_scores=momentum_scores.reindex(index=close.index, columns=close.columns),
        forward_returns=forward_returns,
        eligibility=eligibility,
        rb_dates=rb_dates,
    )
    regime_summary = _summarize_regimes(timeseries)

    baseline_summary, baseline_outdir = _run_baseline_backtest(args=args)
    baseline_weekly = _baseline_weekly_returns(backtest_outdir=baseline_outdir, rb_dates=rb_dates)
    baseline_by_regime = _summarize_baseline_by_regime(timeseries=timeseries, baseline_weekly=baseline_weekly)

    timeseries_path = run_dir / "weekly_dispersion_timeseries.csv"
    regime_summary_path = run_dir / "regime_summary.csv"
    baseline_by_regime_path = run_dir / "baseline_by_dispersion_regime.csv"
    manifest_path = run_dir / "manifest.json"

    timeseries.to_csv(timeseries_path, index=False, float_format="%.10g")
    regime_summary.to_csv(regime_summary_path, index=False, float_format="%.10g")
    baseline_by_regime.to_csv(baseline_by_regime_path, index=False, float_format="%.10g")

    high_regime = baseline_by_regime.loc[baseline_by_regime["dispersion_regime"] == "high dispersion"]
    low_regime = baseline_by_regime.loc[baseline_by_regime["dispersion_regime"] == "low dispersion"]
    high_sharpe = float(high_regime["Sharpe"].iloc[0]) if not high_regime.empty else float("nan")
    low_sharpe = float(low_regime["Sharpe"].iloc[0]) if not low_regime.empty else float("nan")

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_momentum_dispersion_test.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "rebalance": DEFAULT_REBALANCE,
        "eligibility_definition": {
            "universe": "liquid_us",
            "min_price": float(MIN_PRICE),
            "min_avg_dollar_volume": float(MIN_AVG_DOLLAR_VOLUME),
            "liquidity_lookback": int(LIQUIDITY_LOOKBACK),
            "min_history_days": int(MIN_HISTORY_DAYS),
        },
        "momentum_signal_definition": {
            "factor": MOMENTUM_FACTOR,
            "price_panel": "adj_close",
            "score_definition": "raw momentum_12_1 computed on adjusted close",
            "cross_section": "eligible liquid_us names on weekly rebalance dates",
            "bucket_method": "deciles via cross-sectional rank qcut",
            "primary_dispersion_metric": "top decile mean raw momentum score minus bottom decile mean raw momentum score",
            "forward_return_metric": "next weekly adjusted return of top decile minus bottom decile",
            "market_forward_return_metric": "equal-weight next weekly adjusted return across all eligible names",
            "regime_classification": "score_dispersion terciles ranked into low/medium/high dispersion",
        },
        "baseline_strategy": {
            "factors": list(BASELINE_FACTORS),
            "factor_weights": list(BASELINE_WEIGHTS),
            "factor_aggregation_method": BASELINE_AGGREGATION_METHOD,
            "top_n": int(DEFAULT_TOP_N),
            "rebalance": DEFAULT_REBALANCE,
            "costs_bps": float(DEFAULT_COSTS_BPS),
            "backtest_outdir": str(baseline_outdir),
            "backtest_summary": _to_serializable(baseline_summary),
        },
        "data_source_summary": _to_serializable(data_source_summary),
        "outputs": {
            "weekly_dispersion_timeseries": str(timeseries_path),
            "regime_summary": str(regime_summary_path),
            "baseline_by_dispersion_regime": str(baseline_by_regime_path),
            "manifest": str(manifest_path),
        },
        "high_vs_low_baseline_sharpe": {
            "high_dispersion_sharpe": high_sharpe,
            "low_dispersion_sharpe": low_sharpe,
        },
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    latest_dir = base_output_dir / "latest"
    _copy_latest(
        {
            timeseries_path.name: timeseries_path,
            regime_summary_path.name: regime_summary_path,
            baseline_by_regime_path.name: baseline_by_regime_path,
            manifest_path.name: manifest_path,
        },
        latest_dir,
    )

    _print_report(regime_summary=regime_summary, baseline_by_regime=baseline_by_regime)
    if pd.notna(high_sharpe) and pd.notna(low_sharpe):
        if high_sharpe > low_sharpe:
            interpretation = "High-dispersion environments look more favorable for trend / wave behavior in the baseline strategy."
        elif high_sharpe < low_sharpe:
            interpretation = "High-dispersion environments do not look more favorable; the baseline strategy is stronger in lower-dispersion weeks."
        else:
            interpretation = "High- and low-dispersion environments look similar for the baseline strategy."
    else:
        interpretation = "Insufficient regime data to compare high- and low-dispersion environments cleanly."
    print("")
    print(interpretation)
    print(f"Saved: {timeseries_path}")
    print(f"Saved: {regime_summary_path}")
    print(f"Saved: {baseline_by_regime_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_dir}")


if __name__ == "__main__":
    main()
