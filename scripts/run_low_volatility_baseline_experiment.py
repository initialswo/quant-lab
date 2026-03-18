#!/usr/bin/env python3
"""Run the baseline low-volatility sleeve experiment."""

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


RESULTS_ROOT = Path("results") / "low_volatility_baseline_experiment"
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
STRATEGY = {
    "strategy_name": "low_volatility_baseline",
    "factor_names": ["low_vol_20"],
    "factor_weights": [1.0],
    "notes": "Baseline low-volatility sleeve using the existing low_vol_20 factor.",
}
RESULT_COLUMNS = ["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    parser.add_argument("--max_tickers", type=int, default=DEFAULT_MAX_TICKERS)
    return parser.parse_args()



def _run_config(args: argparse.Namespace) -> dict[str, Any]:
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
        "factor_name": list(STRATEGY["factor_names"]),
        "factor_names": list(STRATEGY["factor_names"]),
        "factor_weights": list(STRATEGY["factor_weights"]),
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



def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("LOW VOLATILITY BASELINE EXPERIMENT")
    print("----------------------------------")
    print(
        "Config: "
        f"universe={DEFAULT_UNIVERSE} universe_mode=dynamic rebalance={DEFAULT_REBALANCE} "
        f"top_n={DEFAULT_TOP_N} weighting=inv_vol costs_bps={DEFAULT_COSTS_BPS} "
        f"factor_aggregation_method={FACTOR_AGGREGATION_METHOD} factor=low_vol_20"
    )

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    cfg = _run_config(args=args)
    summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)

    results_df = pd.DataFrame(
        [
            {
                "Strategy": str(STRATEGY["strategy_name"]),
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "Turnover": extract_annual_turnover(summary=summary, outdir=run_outdir),
            }
        ],
        columns=RESULT_COLUMNS,
    )

    results_path = run_dir / "low_volatility_baseline_results.csv"
    summary_path = run_dir / "low_volatility_baseline_summary.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(results_path, index=False)
    results_df.to_csv(summary_path, index=False)

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_low_volatility_baseline_experiment.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "universe_mode": "dynamic",
        "rebalance": DEFAULT_REBALANCE,
        "top_n": int(DEFAULT_TOP_N),
        "weighting": "inv_vol",
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "factor_names": list(STRATEGY["factor_names"]),
        "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
        "fundamentals_path": str(args.fundamentals_path),
        "output_dir": str(run_dir),
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "notes": [
            "This baseline low-volatility sleeve uses the existing low_vol_20 factor.",
            "The requested factor name low_volatility does not exist in the repo; this script reuses the existing low_vol_20 factor instead.",
        ],
        "runtime_seconds": float(time.perf_counter() - t0),
        "run": {
            "strategy_name": str(STRATEGY["strategy_name"]),
            "factor_weights": list(STRATEGY["factor_weights"]),
            "run_config": _to_serializable(cfg),
            "backtest_outdir": str(run_outdir),
            "summary_path": str(Path(run_outdir) / "summary.json"),
        },
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

    print("")
    print(
        results_df.to_string(
            index=False,
            formatters={col: _format_float for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]},
        )
    )
    print("")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_root}")


if __name__ == "__main__":
    main()
