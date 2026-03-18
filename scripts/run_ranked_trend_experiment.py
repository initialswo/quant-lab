#!/usr/bin/env python3
"""Run a ranked 252-day trend sleeve using Top-N selection."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.engine import runner
from quant_lab.engine.metrics import compute_daily_mark_to_market, compute_metrics
from quant_lab.research.cross_asset_trend import annual_turnover
from quant_lab.strategies.topn import build_topn_weights, rebalance_mask
from quant_lab.universe.liquid_us import build_liquid_us_universe


RESULTS_ROOT = Path("results") / "ranked_trend_experiment"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_UNIVERSE_MODE = "dynamic"
DEFAULT_REBALANCE = "weekly"
DEFAULT_WEIGHTING = "inv_vol"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_TOP_N = 75
DEFAULT_MAX_TICKERS = 2000
DEFAULT_LOOKBACK = 252
DEFAULT_VOL_LOOKBACK = 20
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
STRATEGY_NAME = "trend_ranked_252_top75"
RESULT_COLUMNS = ["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--rebalance", choices=["daily", "weekly", "biweekly", "monthly"], default=DEFAULT_REBALANCE)
    parser.add_argument("--weighting", choices=["equal", "inv_vol"], default=DEFAULT_WEIGHTING)
    parser.add_argument("--costs_bps", type=float, default=DEFAULT_COSTS_BPS)
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--vol_lookback", type=int, default=DEFAULT_VOL_LOOKBACK)
    parser.add_argument("--max_tickers", type=int, default=DEFAULT_MAX_TICKERS)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    return parser.parse_args()


def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)


def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": [int(obj.shape[0]), int(obj.shape[1])],
            "columns_sample": [str(c) for c in list(obj.columns[:5])],
        }
    if isinstance(obj, pd.Series):
        return {"type": "Series", "length": int(obj.shape[0]), "name": str(obj.name)}
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def _load_price_panels(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[str], list[str]]:
    tickers = runner._load_universe_seed_tickers(
        universe=str(args.universe),
        max_tickers=int(args.max_tickers),
        data_cache_dir=DATA_CACHE_DIR,
    )
    ohlcv_map, data_summary = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(args.start),
        end=str(args.end),
        cache_dir=DATA_CACHE_DIR,
        data_source=DATA_SOURCE,
        refresh=False,
        bulk_prepare=False,
    )
    adj_cols, used_adj, missing_adj, rejected_adj, _ = runner._collect_price_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=tickers,
        field="Adj Close",
        fallback_field="Close",
    )
    close_cols, used_close, missing_close, rejected_close, _ = runner._collect_close_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=tickers,
    )
    if not adj_cols or not close_cols:
        raise ValueError(
            "No usable price history found for ranked trend experiment. "
            f"adj_missing={len(missing_adj)} adj_rejected={len(rejected_adj)} "
            f"close_missing={len(missing_close)} close_rejected={len(rejected_close)}"
        )

    adj_close = pd.concat(adj_cols, axis=1, join="outer")
    close = pd.concat(close_cols, axis=1, join="outer")
    adj_close = runner._prepare_close_panel(close_raw=adj_close, price_fill_mode="ffill").astype(float)
    close = runner._prepare_close_panel(close_raw=close, price_fill_mode="ffill").astype(float)

    common = [c for c in close.columns if c in set(adj_close.columns)]
    if not common:
        raise ValueError("No overlapping close and adjusted close columns were found.")
    close = close.loc[:, common]
    adj_close = adj_close.reindex(index=close.index, columns=common).astype(float)
    volume = runner._collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=common,
        field="Volume",
    ).reindex(index=close.index, columns=common).astype(float)
    used = sorted(set(used_adj).intersection(set(used_close)))
    missing = sorted(set(missing_adj).union(set(missing_close)))
    return close, adj_close, volume, data_summary, used, missing


def _build_ranked_weights(
    strength: pd.DataFrame,
    eligibility: pd.DataFrame,
    adj_close: pd.DataFrame,
    top_n: int,
    rebalance: str,
    weighting: str,
    vol_lookback: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    eligible_strength = strength.where(eligibility)
    lagged_scores = eligible_strength.shift(1)
    weights = build_topn_weights(
        scores=lagged_scores,
        close=adj_close,
        top_n=int(top_n),
        rebalance=str(rebalance),
        weighting=str(weighting),
        vol_lookback=int(vol_lookback),
        score_clip=5.0,
        score_floor=0.0,
        max_weight=1.0,
        sector_cap=0.0,
        sector_by_ticker=None,
        sector_neutral=False,
        rank_buffer=0,
        volatility_scaled_weights=False,
    ).astype(float)
    rb = rebalance_mask(pd.DatetimeIndex(weights.index), str(rebalance))
    rb_dates = pd.DatetimeIndex(weights.index[rb])
    selected_counts = (weights > 0.0).sum(axis=1).astype(int)
    return weights, selected_counts, rb_dates


def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)
    latest_dir = base_output_dir / "latest"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("RANKED TREND EXPERIMENT")
    print("-----------------------")
    print(
        "Config: "
        f"universe={args.universe} universe_mode={DEFAULT_UNIVERSE_MODE} rebalance={args.rebalance} "
        f"top_n={int(args.top_n)} weighting={args.weighting} costs_bps={float(args.costs_bps):.1f} "
        f"trend_strength={int(args.lookback)}-day return selection=top {int(args.top_n)} stocks by lagged cross-sectional return rank"
    )

    t0 = time.perf_counter()
    close, adj_close, volume, data_summary, used_tickers, missing_tickers = _load_price_panels(args=args)
    eligibility = build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=5.0,
        min_avg_dollar_volume=10_000_000.0,
        adv_window=20,
        min_history=252,
    )
    strength = adj_close.astype(float).pct_change(int(args.lookback)).astype(float)
    weights_rebal, selected_counts, rb_dates = _build_ranked_weights(
        strength=strength,
        eligibility=eligibility,
        adj_close=adj_close,
        top_n=int(args.top_n),
        rebalance=str(args.rebalance),
        weighting=str(args.weighting),
        vol_lookback=int(args.vol_lookback),
    )

    equity, daily_return, weights_daily = compute_daily_mark_to_market(
        close=adj_close,
        weights_rebal=weights_rebal,
        rebalance_dates=rb_dates,
        costs_bps=float(args.costs_bps),
        slippage_bps=0.0,
    )
    metrics = compute_metrics(daily_return)
    turnover = annual_turnover(weights_daily, rebalance=str(args.rebalance))

    results_df = pd.DataFrame(
        [
            {
                "Strategy": STRATEGY_NAME,
                "CAGR": float(metrics.get("CAGR", float("nan"))),
                "Vol": float(metrics.get("Vol", float("nan"))),
                "Sharpe": float(metrics.get("Sharpe", float("nan"))),
                "MaxDD": float(metrics.get("MaxDD", float("nan"))),
                "Turnover": float(turnover),
            }
        ],
        columns=RESULT_COLUMNS,
    )
    daily_returns_df = pd.DataFrame(
        {
            "Equity": equity.astype(float),
            "DailyReturn": daily_return.astype(float),
            "SelectedAssets": selected_counts.reindex(daily_return.index).fillna(0).astype(int),
        }
    )

    results_path = run_dir / "ranked_trend_results.csv"
    daily_returns_path = run_dir / "daily_returns.csv"
    rebalance_weights_path = run_dir / "rebalance_weights.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index_label="date", float_format="%.10g")
    weights_daily.loc[rb_dates].to_csv(rebalance_weights_path, index_label="date", float_format="%.10g")

    manifest = {
        "timestamp_utc": timestamp,
        "script_name": "scripts/run_ranked_trend_experiment.py",
        "results_dir": str(run_dir),
        "runtime_seconds": float(time.perf_counter() - t0),
        "date_range": {"start": str(args.start), "end": str(args.end)},
        "config": {
            "strategy_name": STRATEGY_NAME,
            "universe": str(args.universe),
            "universe_mode": DEFAULT_UNIVERSE_MODE,
            "rebalance": str(args.rebalance),
            "weighting": str(args.weighting),
            "costs_bps": float(args.costs_bps),
            "lookback": int(args.lookback),
            "vol_lookback": int(args.vol_lookback),
            "top_n": int(args.top_n),
            "selection_rule": "Rank stocks cross-sectionally by lagged 252-day return and hold the top_n names.",
        },
        "data_summary": _to_serializable(data_summary),
        "universe_summary": {
            "tickers_used": int(len(used_tickers)),
            "missing_tickers": int(len(missing_tickers)),
            "eligible_median": float(eligibility.sum(axis=1).median()) if not eligibility.empty else float("nan"),
            "selected_median": float(selected_counts.median()) if not selected_counts.empty else float("nan"),
        },
        "notes": [
            "Uses existing price data only; no new data source was added.",
            "The strategy reuses the existing Top-N framework with 252-day return as the ranking score.",
            "Engine logic is unchanged; this experiment is implemented entirely at the script layer.",
        ],
        "outputs": {
            "results": str(results_path),
            "daily_returns": str(daily_returns_path),
            "rebalance_weights": str(rebalance_weights_path),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _copy_latest(
        files={
            results_path.name: results_path,
            daily_returns_path.name: daily_returns_path,
            rebalance_weights_path.name: rebalance_weights_path,
            manifest_path.name: manifest_path,
        },
        latest_root=latest_dir,
    )

    print("")
    print(
        results_df.to_string(
            index=False,
            formatters={col: _format_float for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]},
        )
    )
    print("")
    print(f"Saved: {results_path}")
    print(f"Saved: {daily_returns_path}")
    print(f"Saved: {rebalance_weights_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_dir}")


if __name__ == "__main__":
    main()
