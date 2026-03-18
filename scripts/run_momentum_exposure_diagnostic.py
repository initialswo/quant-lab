#!/usr/bin/env python3
"""Measure the canonical baseline strategy's implicit momentum exposure."""

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
from quant_lab.engine.runner import run_backtest
from quant_lab.factors.momentum import compute as compute_momentum_12_1
from quant_lab.universe.liquid_us import build_liquid_us_universe


RESULTS_ROOT = Path("results") / "momentum_exposure_diagnostic"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_TOP_N = 75
DEFAULT_MAX_TICKERS = 2000
DEFAULT_DATA_SOURCE = "parquet"
DEFAULT_DATA_CACHE_DIR = "data/equities"
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
DEFAULT_PRICE_FILL_MODE = "ffill"
DEFAULT_UNIVERSE_MIN_HISTORY_DAYS = 300
DEFAULT_UNIVERSE_MIN_PRICE = 5.0
DEFAULT_MIN_AVG_DOLLAR_VOLUME = 10_000_000.0
DEFAULT_LIQUIDITY_LOOKBACK = 20
FACTOR_NAMES = ["gross_profitability", "reversal_1m"]
FACTOR_WEIGHTS = [0.7, 0.3]
FACTOR_AGGREGATION_METHOD = "linear"
SUMMARY_METRIC_ORDER = [
    "mean portfolio momentum",
    "mean universe momentum",
    "mean momentum exposure",
    "std exposure",
    "percent positive exposure",
]
PRINT_METRICS = [
    "mean portfolio momentum",
    "mean universe momentum",
    "mean momentum exposure",
    "percent positive exposure",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--data_source", default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--data_cache_dir", default=DEFAULT_DATA_CACHE_DIR)
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    parser.add_argument("--max_tickers", type=int, default=DEFAULT_MAX_TICKERS)
    return parser.parse_args()


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "universe_mode": "dynamic",
        "top_n": int(DEFAULT_TOP_N),
        "rebalance": DEFAULT_REBALANCE,
        "weighting": "equal",
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "max_tickers": int(args.max_tickers),
        "data_source": str(args.data_source),
        "data_cache_dir": str(args.data_cache_dir),
        "factor_name": list(FACTOR_NAMES),
        "factor_names": list(FACTOR_NAMES),
        "factor_weights": list(FACTOR_WEIGHTS),
        "portfolio_mode": "composite",
        "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
        "use_factor_normalization": True,
        "use_sector_neutralization": False,
        "use_size_neutralization": False,
        "orthogonalize_factors": False,
        "fundamentals_path": str(args.fundamentals_path),
        "save_artifacts": True,
    }


def _read_matrix_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def _load_runtime_price_panels(
    *,
    start: str,
    end: str,
    data_source: str,
    data_cache_dir: str,
    max_tickers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tickers = runner._load_universe_seed_tickers(
        universe=DEFAULT_UNIVERSE,
        max_tickers=int(max_tickers),
        data_cache_dir=str(data_cache_dir),
    )
    ohlcv_map, data_source_summary = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(start),
        end=str(end),
        cache_dir=str(data_cache_dir),
        data_source=str(data_source),
        refresh=False,
        bulk_prepare=False,
    )

    close_cols, used_tickers, missing_tickers, rejected_tickers, _ = runner._collect_close_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=tickers,
    )
    if not close_cols:
        raise ValueError(
            "No valid OHLCV data found for the diagnostic. "
            f"missing={len(missing_tickers)} rejected={len(rejected_tickers)}"
        )

    close_raw = pd.concat(close_cols, axis=1, join="outer")
    adj_close_cols, _, _, _, _ = runner._collect_price_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Adj Close",
        fallback_field="Close",
    )
    adj_close_raw = pd.concat(adj_close_cols, axis=1, join="outer") if adj_close_cols else close_raw.copy()
    volume_raw = runner._collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Volume",
    )

    close = runner._prepare_close_panel(close_raw=close_raw, price_fill_mode=DEFAULT_PRICE_FILL_MODE)
    adj_close = runner._prepare_close_panel(close_raw=adj_close_raw, price_fill_mode=DEFAULT_PRICE_FILL_MODE)

    min_non_nan = int(0.8 * close.shape[1])
    close = close.dropna(thresh=min_non_nan)
    if close.empty:
        raise ValueError("No usable close-price history remained after alignment and fill.")

    _, broken_tickers, _ = runner._price_panel_health_report(
        close_raw=close_raw.reindex(index=close.index, columns=close.columns),
        close_filled=close,
    )
    if broken_tickers:
        close = close.drop(columns=broken_tickers, errors="ignore")
        adj_close = adj_close.drop(columns=broken_tickers, errors="ignore")

    volume = (
        volume_raw.reindex(index=close.index, columns=close.columns)
        if not volume_raw.empty
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    adj_close = adj_close.reindex(index=close.index, columns=close.columns)
    adj_close = adj_close.where(adj_close.notna(), close)

    panel_summary = {
        "requested_tickers": int(len(tickers)),
        "loaded_tickers": int(len(used_tickers)),
        "missing_tickers": int(len(missing_tickers)),
        "rejected_tickers": int(len(rejected_tickers)),
        "dropped_broken_tickers": int(len(broken_tickers)),
        "final_tickers": int(close.shape[1]),
        "final_dates": int(close.shape[0]),
    }
    return close.astype(float), adj_close.astype(float), volume.astype(float), {
        "data_source_summary": data_source_summary,
        "panel_summary": panel_summary,
    }


def _build_liquid_us_eligibility(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    effective_min_history = max(
        int(DEFAULT_UNIVERSE_MIN_HISTORY_DAYS),
        int(runner._factor_required_history_days(factor_names=list(FACTOR_NAMES), factor_params_map={})),
    )
    return build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=float(DEFAULT_UNIVERSE_MIN_PRICE),
        min_avg_dollar_volume=float(DEFAULT_MIN_AVG_DOLLAR_VOLUME),
        adv_window=int(DEFAULT_LIQUIDITY_LOOKBACK),
        min_history=int(effective_min_history),
    )


def compute_momentum_exposure_timeseries(
    *,
    holdings: pd.DataFrame,
    eligibility: pd.DataFrame,
    momentum: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dt in pd.DatetimeIndex(rebalance_dates):
        if dt not in holdings.index or dt not in eligibility.index or dt not in momentum.index:
            continue

        weights = pd.to_numeric(holdings.loc[dt], errors="coerce").fillna(0.0)
        held_tickers = weights[weights.abs() > 0.0].index.tolist()
        eligible_row = eligibility.loc[dt].fillna(False).astype(bool)
        eligible_tickers = eligible_row[eligible_row].index.tolist()

        portfolio_momentum = pd.to_numeric(momentum.loc[dt, held_tickers], errors="coerce").dropna()
        universe_momentum = pd.to_numeric(momentum.loc[dt, eligible_tickers], errors="coerce").dropna()

        portfolio_mean = float(portfolio_momentum.mean()) if not portfolio_momentum.empty else float("nan")
        universe_mean = float(universe_momentum.mean()) if not universe_momentum.empty else float("nan")
        exposure = (
            float(portfolio_mean - universe_mean)
            if pd.notna(portfolio_mean) and pd.notna(universe_mean)
            else float("nan")
        )
        rows.append(
            {
                "date": pd.Timestamp(dt),
                "portfolio_momentum": portfolio_mean,
                "universe_momentum": universe_mean,
                "momentum_exposure": exposure,
                "portfolio_count": int(portfolio_momentum.shape[0]),
                "universe_count": int(universe_momentum.shape[0]),
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "date",
            "portfolio_momentum",
            "universe_momentum",
            "momentum_exposure",
            "portfolio_count",
            "universe_count",
        ],
    )
    if out.empty:
        return out
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def build_summary_frame(exposure_df: pd.DataFrame) -> pd.DataFrame:
    exposure = pd.to_numeric(exposure_df.get("momentum_exposure"), errors="coerce")
    portfolio = pd.to_numeric(exposure_df.get("portfolio_momentum"), errors="coerce")
    universe = pd.to_numeric(exposure_df.get("universe_momentum"), errors="coerce")

    metric_values = {
        "mean portfolio momentum": float(portfolio.mean()) if not portfolio.empty else float("nan"),
        "mean universe momentum": float(universe.mean()) if not universe.empty else float("nan"),
        "mean momentum exposure": float(exposure.mean()) if not exposure.empty else float("nan"),
        "std exposure": float(exposure.std()) if not exposure.empty else float("nan"),
        "percent positive exposure": (
            float((exposure > 0.0).mean() * 100.0) if exposure.notna().any() else float("nan")
        ),
    }
    return pd.DataFrame(
        [{"Metric": metric, "Value": metric_values[metric]} for metric in SUMMARY_METRIC_ORDER],
        columns=["Metric", "Value"],
    )


def _format_metric_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "-"
    if metric == "percent positive exposure":
        return f"{float(value):.2f}%"
    return f"{float(value):.6f}"


def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)


def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    run_config = build_run_config(args)

    summary, backtest_outdir = run_backtest(**run_config, run_cache=run_cache)
    backtest_dir = Path(backtest_outdir)
    holdings = _read_matrix_csv(backtest_dir / "holdings.csv").astype(float)
    composite_scores = _read_matrix_csv(backtest_dir / "composite_scores_snapshot.csv")
    rebalance_dates = pd.DatetimeIndex(composite_scores.index)

    close, adj_close, volume, load_diag = _load_runtime_price_panels(
        start=str(args.start),
        end=str(args.end),
        data_source=str(args.data_source),
        data_cache_dir=str(args.data_cache_dir),
        max_tickers=int(args.max_tickers),
    )
    eligibility = _build_liquid_us_eligibility(close=close, volume=volume)
    momentum = compute_momentum_12_1(close=adj_close)

    exposure_df = compute_momentum_exposure_timeseries(
        holdings=holdings,
        eligibility=eligibility,
        momentum=momentum,
        rebalance_dates=rebalance_dates,
    )
    summary_df = build_summary_frame(exposure_df)

    timeseries_path = run_dir / "momentum_exposure_timeseries.csv"
    summary_path = run_dir / "summary.csv"
    manifest_path = run_dir / "manifest.json"

    exposure_df.to_csv(timeseries_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")

    metric_map = {
        str(row["Metric"]): float(row["Value"])
        for row in summary_df.to_dict(orient="records")
    }
    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_momentum_exposure_diagnostic.py",
        "runtime_seconds": float(time.perf_counter() - t0),
        "start": str(args.start),
        "end": str(args.end),
        "strategy": {
            "factors": list(FACTOR_NAMES),
            "factor_weights": list(FACTOR_WEIGHTS),
            "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
            "top_n": int(DEFAULT_TOP_N),
            "rebalance": DEFAULT_REBALANCE,
            "universe": DEFAULT_UNIVERSE,
            "costs_bps": float(DEFAULT_COSTS_BPS),
        },
        "backtest": {
            "outdir": str(backtest_dir),
            "summary_path": str(backtest_dir / "summary.json"),
            "run_config": run_config,
            "summary": summary,
        },
        "diagnostic": {
            "momentum_signal": "momentum_12_1",
            "momentum_price_field": "adj_close",
            "eligibility_price_field": "close",
            "eligibility_volume_field": "volume",
            "eligibility_min_history_days": int(DEFAULT_UNIVERSE_MIN_HISTORY_DAYS),
            "eligibility_min_price": float(DEFAULT_UNIVERSE_MIN_PRICE),
            "eligibility_min_avg_dollar_volume": float(DEFAULT_MIN_AVG_DOLLAR_VOLUME),
            "eligibility_adv_window": int(DEFAULT_LIQUIDITY_LOOKBACK),
            "rebalance_dates_source": str(backtest_dir / "composite_scores_snapshot.csv"),
        },
        "metrics": metric_map,
        "load_diagnostics": load_diag,
        "outputs": {
            "timeseries": str(timeseries_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "notes": [
            "The canonical baseline is gross_profitability 0.70 plus reversal_1m 0.30 with linear aggregation.",
            "Portfolio momentum is the equal-weight mean raw momentum_12_1 across the names held on each rebalance date.",
            "Universe momentum is the equal-weight mean raw momentum_12_1 across runner-aligned liquid_us eligible stocks on each rebalance date.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    latest_root = base_output_dir / "latest"
    _copy_latest(
        files={
            "momentum_exposure_timeseries.csv": timeseries_path,
            "summary.csv": summary_path,
            "manifest.json": manifest_path,
        },
        latest_root=latest_root,
    )

    print("| Metric | Value |")
    print("|---|---:|")
    for metric in PRINT_METRICS:
        value = metric_map.get(metric, float("nan"))
        print(f"| {metric} | {_format_metric_value(metric, value)} |")

    print("")
    print(f"Saved: {timeseries_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_root}")


if __name__ == "__main__":
    main()
