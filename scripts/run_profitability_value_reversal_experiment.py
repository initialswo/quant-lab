#!/usr/bin/env python3
"""Run the post-fix profitability + value + reversal blend test against the canonical baseline."""

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


RESULTS_ROOT = Path("results") / "profitability_value_reversal_experiment"
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
STRATEGIES = [
    {
        "strategy_name": "baseline",
        "factor_names": ["gross_profitability", "reversal_1m"],
        "factor_weights": [0.7, 0.3],
    },
    {
        "strategy_name": "profitability_value_reversal",
        "factor_names": ["gross_profitability", "book_to_market", "reversal_1m"],
        "factor_weights": [0.5, 0.3, 0.2],
    },
]
RESULT_COLUMNS = [
    "strategy_name",
    "factor_names",
    "factor_weights",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "Turnover",
]
SUMMARY_COLUMNS = ["strategy_name", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    return parser.parse_args()



def _run_config(args: argparse.Namespace, strategy: dict[str, Any]) -> dict[str, Any]:
    return {
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
        "factor_name": list(strategy["factor_names"]),
        "factor_names": list(strategy["factor_names"]),
        "factor_weights": list(strategy["factor_weights"]),
        "portfolio_mode": "composite",
        "factor_aggregation_method": "linear",
        "use_factor_normalization": True,
        "use_sector_neutralization": False,
        "use_size_neutralization": False,
        "orthogonalize_factors": False,
        "fundamentals_path": str(args.fundamentals_path),
        "save_artifacts": True,
    }



def _result_row(strategy: dict[str, Any], summary: dict[str, Any], outdir: str) -> dict[str, Any]:
    return {
        "strategy_name": str(strategy["strategy_name"]),
        "factor_names": ",".join(strategy["factor_names"]),
        "factor_weights": ",".join(str(x) for x in strategy["factor_weights"]),
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "Vol": float(summary.get("Vol", float("nan"))),
        "Sharpe": float(summary.get("Sharpe", float("nan"))),
        "MaxDD": float(summary.get("MaxDD", float("nan"))),
        "Turnover": float(extract_annual_turnover(summary=summary, outdir=outdir)),
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
    print("PROFITABILITY + VALUE + REVERSAL SUMMARY")
    print("----------------------------------------")
    print(f"{'Strategy':30s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Turnover':>10s}")
    for _, row in summary_df.iterrows():
        print(
            f"{str(row['strategy_name']):30s} "
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

    print("POST-FIX PROFITABILITY + VALUE + REVERSAL EXPERIMENT")
    print("----------------------------------------------------")
    print(
        "Config: "
        f"top_n={DEFAULT_TOP_N} rebalance={DEFAULT_REBALANCE} universe={DEFAULT_UNIVERSE} costs_bps={DEFAULT_COSTS_BPS} "
        "factor_aggregation_method=linear"
    )
    print("Baseline: gross_profitability=0.70, reversal_1m=0.30")
    print("Blend: gross_profitability=0.50, book_to_market=0.30, reversal_1m=0.20")

    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []
    run_cache: dict[str, Any] = {}

    for strategy in STRATEGIES:
        cfg = _run_config(args=args, strategy=strategy)
        summary, outdir = run_backtest(**cfg, run_cache=run_cache)
        rows.append(_result_row(strategy=strategy, summary=summary, outdir=outdir))
        run_manifest.append(
            {
                "strategy_name": str(strategy["strategy_name"]),
                "run_config": _to_serializable(cfg),
                "backtest_outdir": str(outdir),
                "summary": _to_serializable(summary),
            }
        )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(
        by=["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort"
    ).reset_index(drop=True)
    summary_df = results_df.loc[:, SUMMARY_COLUMNS].copy()

    results_path = run_dir / "results.csv"
    summary_path = run_dir / "summary.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")

    baseline_sharpe = float(results_df.loc[results_df["strategy_name"] == "baseline", "Sharpe"].iloc[0])
    blend_sharpe = float(
        results_df.loc[results_df["strategy_name"] == "profitability_value_reversal", "Sharpe"].iloc[0]
    )

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_profitability_value_reversal_experiment.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "rebalance": DEFAULT_REBALANCE,
        "top_n": int(DEFAULT_TOP_N),
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "factor_aggregation_method": "linear",
        "fundamentals_path": str(args.fundamentals_path),
        "strategies": _to_serializable(STRATEGIES),
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "baseline_sharpe": baseline_sharpe,
        "blend_sharpe": blend_sharpe,
        "sharpe_difference_blend_minus_baseline": float(blend_sharpe - baseline_sharpe),
        "runtime_seconds": float(time.perf_counter() - t0),
        "notes": [
            "This is the post-fix profitability + value + reversal blend comparison against the canonical baseline.",
            "Signals and portfolio PnL run through the corrected run_backtest path using adjusted prices consistently.",
            "liquid_us eligibility remains identical to the active runtime.",
        ],
        "runs": _to_serializable(run_manifest),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

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
    print(f"Sharpe difference (blend - baseline): {_format_float(blend_sharpe - baseline_sharpe)}")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_root}")


if __name__ == "__main__":
    main()
