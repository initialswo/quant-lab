#!/usr/bin/env python3
"""Compare the canonical quality sleeve against a multi-horizon pullback variant."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover
from quant_lab.strategies.topn import rebalance_mask


RESULTS_ROOT = Path("results") / "quality_multi_horizon_reversal_experiment"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_TOP_N = 75
DEFAULT_MAX_TICKERS = 2000
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
FACTOR_AGGREGATION_METHOD = "linear"
STRATEGIES = [
    {
        "strategy_name": "baseline",
        "factor_names": ["gross_profitability", "reversal_1m"],
        "factor_weights": [0.70, 0.30],
        "notes": "Canonical quality sleeve baseline.",
    },
    {
        "strategy_name": "multi_horizon_reversal_test",
        "factor_names": ["gross_profitability", "reversal_1m", "reversal_5d"],
        "factor_weights": [0.70, 0.20, 0.10],
        "notes": "Adds a shorter-horizon pullback layer on top of the 1-month reversal signal.",
    },
]
RESULT_COLUMNS = [
    "strategy_name",
    "factors",
    "factor_weights",
    "factor_aggregation_method",
    "start",
    "end",
    "universe",
    "rebalance",
    "top_n",
    "weighting",
    "costs_bps",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "Turnover",
    "hit_rate",
    "n_rebalance_dates",
    "median_selected_names",
]
SUMMARY_COLUMNS = ["strategy_name", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    parser.add_argument("--max_tickers", type=int, default=DEFAULT_MAX_TICKERS)
    return parser.parse_args()



def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()



def _artifact_stats(outdir: str | Path, rebalance: str, skipped_count: int) -> dict[str, float]:
    outdir_path = Path(outdir)
    equity = _read_csv(outdir_path / "equity.csv")
    holdings = _read_csv(outdir_path / "holdings.csv")
    daily_return = pd.to_numeric(equity.get("DailyReturn"), errors="coerce").dropna()
    rb_mask = rebalance_mask(pd.DatetimeIndex(holdings.index), str(rebalance)).reindex(holdings.index).fillna(False)
    selected_counts = (holdings.loc[rb_mask].abs() > 0.0).sum(axis=1).astype(float)
    scheduled_rebalances = int(rb_mask.sum())
    n_rebalance_dates = max(0, scheduled_rebalances - int(skipped_count))
    return {
        "hit_rate": float((daily_return > 0.0).mean()) if not daily_return.empty else float("nan"),
        "median_selected_names": float(selected_counts.median()) if not selected_counts.empty else float("nan"),
        "n_rebalance_dates": int(n_rebalance_dates),
    }



def _run_config(args: argparse.Namespace, strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "universe_mode": "dynamic",
        "top_n": DEFAULT_TOP_N,
        "rebalance": DEFAULT_REBALANCE,
        "weighting": "inv_vol",
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "max_tickers": int(args.max_tickers),
        "data_source": DATA_SOURCE,
        "data_cache_dir": DATA_CACHE_DIR,
        "factor_name": list(strategy["factor_names"]),
        "factor_names": list(strategy["factor_names"]),
        "factor_weights": list(strategy["factor_weights"]),
        "portfolio_mode": "composite",
        "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
        "use_factor_normalization": True,
        "use_sector_neutralization": False,
        "use_size_neutralization": False,
        "orthogonalize_factors": False,
        "fundamentals_path": str(args.fundamentals_path),
        "save_artifacts": True,
    }



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



def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"



def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)



def _print_summary(summary_df: pd.DataFrame) -> None:
    print("")
    print("QUALITY MULTI-HORIZON REVERSAL SUMMARY")
    print("--------------------------------------")
    print(
        f"{'Strategy':28s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Turnover':>10s}"
    )
    for _, row in summary_df.iterrows():
        print(
            f"{str(row['strategy_name']):28s} "
            f"{_format_float(float(row['CAGR'])):>8s} "
            f"{_format_float(float(row['Vol'])):>8s} "
            f"{_format_float(float(row['Sharpe'])):>8s} "
            f"{_format_float(float(row['MaxDD'])):>8s} "
            f"{_format_float(float(row['Turnover'])):>10s}"
        )



def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("QUALITY MULTI-HORIZON REVERSAL EXPERIMENT")
    print("-----------------------------------------")
    print(
        "Shared config: "
        f"universe={DEFAULT_UNIVERSE} rebalance={DEFAULT_REBALANCE} top_n={DEFAULT_TOP_N} "
        f"weighting=inv_vol costs_bps={DEFAULT_COSTS_BPS} factor_aggregation_method={FACTOR_AGGREGATION_METHOD}"
    )
    print("Comparison weights:")
    print("- baseline: gross_profitability=0.70, reversal_1m=0.30")
    print("- multi_horizon_reversal_test: gross_profitability=0.70, reversal_1m=0.20, reversal_5d=0.10")

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for strategy in STRATEGIES:
        cfg = _run_config(args=args, strategy=strategy)
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        stats = _artifact_stats(
            outdir=run_outdir,
            rebalance=DEFAULT_REBALANCE,
            skipped_count=int(summary.get("RebalanceSkippedCount", 0)),
        )
        rows.append(
            {
                "strategy_name": str(strategy["strategy_name"]),
                "factors": ",".join(strategy["factor_names"]),
                "factor_weights": ",".join(str(float(x)) for x in strategy["factor_weights"]),
                "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
                "start": str(args.start),
                "end": str(args.end),
                "universe": DEFAULT_UNIVERSE,
                "rebalance": DEFAULT_REBALANCE,
                "top_n": int(DEFAULT_TOP_N),
                "weighting": "inv_vol",
                "costs_bps": float(DEFAULT_COSTS_BPS),
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "Turnover": extract_annual_turnover(summary=summary, outdir=run_outdir),
                "hit_rate": float(stats["hit_rate"]),
                "n_rebalance_dates": int(stats["n_rebalance_dates"]),
                "median_selected_names": float(stats["median_selected_names"]),
            }
        )
        run_manifest.append(
            {
                "strategy_name": str(strategy["strategy_name"]),
                "notes": str(strategy["notes"]),
                "factor_names": list(strategy["factor_names"]),
                "factor_weights": list(strategy["factor_weights"]),
                "run_config": _to_serializable(cfg),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(
        by=["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort"
    ).reset_index(drop=True)
    summary_df = results_df.loc[:, SUMMARY_COLUMNS].copy()

    results_path = run_dir / "quality_multi_horizon_reversal_results.csv"
    summary_path = run_dir / "quality_multi_horizon_reversal_summary.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    baseline_row = results_df.loc[results_df["strategy_name"] == "baseline"].iloc[0]
    test_row = results_df.loc[results_df["strategy_name"] == "multi_horizon_reversal_test"].iloc[0]
    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_quality_multi_horizon_reversal_experiment.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "rebalance": DEFAULT_REBALANCE,
        "top_n": int(DEFAULT_TOP_N),
        "weighting": "inv_vol",
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
        "fundamentals_path": str(args.fundamentals_path),
        "strategies": [
            {
                "strategy_name": str(s["strategy_name"]),
                "factor_names": list(s["factor_names"]),
                "factor_weights": list(s["factor_weights"]),
                "notes": str(s["notes"]),
            }
            for s in STRATEGIES
        ],
        "comparison": {
            "baseline_sharpe": float(baseline_row["Sharpe"]),
            "test_sharpe": float(test_row["Sharpe"]),
            "sharpe_delta": float(test_row["Sharpe"] - baseline_row["Sharpe"]),
            "baseline_cagr": float(baseline_row["CAGR"]),
            "test_cagr": float(test_row["CAGR"]),
            "cagr_delta": float(test_row["CAGR"] - baseline_row["CAGR"]),
        },
        "output_dir": str(run_dir),
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "runtime_seconds": float(time.perf_counter() - t0),
        "notes": [
            "This experiment changes only the pullback factor set and factor weights inside the quality sleeve.",
            "It combines the existing 1-month and 5-day reversal factors using linear aggregation.",
            "All other portfolio and universe settings remain fixed versus the baseline.",
        ],
        "runs": _to_serializable(run_manifest),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    latest_root = base_output_dir / "latest"
    _copy_latest(
        files={
            results_path.name: results_path,
            summary_path.name: summary_path,
            manifest_path.name: manifest_path,
        },
        latest_root=latest_root,
    )

    _print_summary(summary_df)

    print("")
    print(
        "Delta vs baseline: "
        f"Sharpe={_format_float(float(test_row['Sharpe'] - baseline_row['Sharpe']))} "
        f"CAGR={_format_float(float(test_row['CAGR'] - baseline_row['CAGR']))}"
    )
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_root}")


if __name__ == "__main__":
    main()
