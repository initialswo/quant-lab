#!/usr/bin/env python3
"""Compare the baseline quality sleeve against a quality-plus-ROA variant."""

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


RESULTS_ROOT = Path("results") / "quality_roa_experiment"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_UNIVERSE_MODE = "dynamic"
DEFAULT_REBALANCE = "weekly"
DEFAULT_TOP_N = 75
DEFAULT_WEIGHTING = "inv_vol"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_MAX_TICKERS = 2000
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
DEFAULT_FUNDAMENTALS_FALLBACK_LAG_DAYS = 60
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
FACTOR_AGGREGATION_METHOD = "linear"
STRATEGIES: list[dict[str, Any]] = [
    {
        "strategy_name": "baseline_quality",
        "factor_names": ["gross_profitability", "reversal_1m", "reversal_5d"],
        "factor_weights": [0.70, 0.05, 0.25],
        "notes": "Baseline quality sleeve.",
    },
    {
        "strategy_name": "quality_with_roa",
        "factor_names": ["gross_profitability", "roa", "reversal_1m", "reversal_5d"],
        "factor_weights": [0.50, 0.20, 0.05, 0.25],
        "notes": "Adds ROA while preserving the existing reversal mix.",
    },
]
RESULT_COLUMNS = ["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    parser.add_argument("--max_tickers", type=int, default=DEFAULT_MAX_TICKERS)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    return parser.parse_args()


def _run_config(args: argparse.Namespace, strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "universe_mode": DEFAULT_UNIVERSE_MODE,
        "top_n": DEFAULT_TOP_N,
        "rebalance": DEFAULT_REBALANCE,
        "weighting": DEFAULT_WEIGHTING,
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
        "fundamentals_fallback_lag_days": DEFAULT_FUNDAMENTALS_FALLBACK_LAG_DAYS,
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


def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)


def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)
    latest_dir = base_output_dir / "latest"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("QUALITY ROA EXPERIMENT")
    print("----------------------")
    print(
        "Shared config: "
        f"universe={DEFAULT_UNIVERSE} universe_mode={DEFAULT_UNIVERSE_MODE} "
        f"rebalance={DEFAULT_REBALANCE} top_n={DEFAULT_TOP_N} weighting={DEFAULT_WEIGHTING} "
        f"costs_bps={DEFAULT_COSTS_BPS:.1f} factor_aggregation_method={FACTOR_AGGREGATION_METHOD}"
    )
    print("Comparison weights:")
    print("- baseline_quality: gross_profitability=0.70, reversal_1m=0.05, reversal_5d=0.25")
    print("- quality_with_roa: gross_profitability=0.50, roa=0.20, reversal_1m=0.05, reversal_5d=0.25")

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for strategy in STRATEGIES:
        strategy_name = str(strategy["strategy_name"])
        cfg = _run_config(args=args, strategy=strategy)
        print(f"Running {strategy_name}...")
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        rows.append(
            {
                "Strategy": strategy_name,
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "Turnover": float(extract_annual_turnover(summary=summary, outdir=run_outdir)),
            }
        )
        run_manifest.append(
            {
                "strategy_name": strategy_name,
                "notes": str(strategy["notes"]),
                "factor_names": list(strategy["factor_names"]),
                "factor_weights": list(strategy["factor_weights"]),
                "run_config": _to_serializable(cfg),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
                "engine_summary": _to_serializable(summary),
            }
        )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    order = {"baseline_quality": 0, "quality_with_roa": 1}
    results_df["_order"] = results_df["Strategy"].map(order).fillna(999)
    results_df = results_df.sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)

    delta_df = pd.DataFrame()
    baseline_row = results_df.loc[results_df["Strategy"].eq("baseline_quality")]
    roa_row = results_df.loc[results_df["Strategy"].eq("quality_with_roa")]
    if not baseline_row.empty and not roa_row.empty:
        base = baseline_row.iloc[0]
        comp = roa_row.iloc[0]
        delta_df = pd.DataFrame(
            [
                {
                    "Strategy": "quality_with_roa_vs_baseline",
                    "Delta_CAGR": float(comp["CAGR"] - base["CAGR"]),
                    "Delta_Vol": float(comp["Vol"] - base["Vol"]),
                    "Delta_Sharpe": float(comp["Sharpe"] - base["Sharpe"]),
                    "Delta_MaxDD": float(comp["MaxDD"] - base["MaxDD"]),
                    "Delta_Turnover": float(comp["Turnover"] - base["Turnover"]),
                }
            ]
        )

    summary_path = run_dir / "quality_roa_summary.csv"
    delta_path = run_dir / "quality_roa_delta_vs_baseline.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(summary_path, index=False, float_format="%.10g")
    if not delta_df.empty:
        delta_df.to_csv(delta_path, index=False, float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp,
                "results_dir": str(run_dir),
                "runtime_seconds": float(time.perf_counter() - t0),
                "date_range": {"start": str(args.start), "end": str(args.end)},
                "shared_config": {
                    "universe": DEFAULT_UNIVERSE,
                    "universe_mode": DEFAULT_UNIVERSE_MODE,
                    "rebalance": DEFAULT_REBALANCE,
                    "top_n": DEFAULT_TOP_N,
                    "weighting": DEFAULT_WEIGHTING,
                    "costs_bps": float(DEFAULT_COSTS_BPS),
                    "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
                },
                "strategies": [
                    {
                        "strategy_name": "baseline_quality",
                        "factor_names": ["gross_profitability", "reversal_1m", "reversal_5d"],
                        "factor_weights": [0.70, 0.05, 0.25],
                    },
                    {
                        "strategy_name": "quality_with_roa",
                        "factor_names": ["gross_profitability", "roa", "reversal_1m", "reversal_5d"],
                        "factor_weights": [0.50, 0.20, 0.05, 0.25],
                    },
                ],
                "notes": [
                    "Both strategies are run individually via run_backtest.",
                    "The existing ROA factor is used unchanged.",
                    "Engine logic is unchanged; this experiment is implemented entirely at the script layer.",
                ],
                "runs": run_manifest,
                "outputs": {
                    "summary": str(summary_path),
                    "delta_vs_baseline": str(delta_path) if not delta_df.empty else None,
                    "manifest": str(manifest_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    latest_files = {
        summary_path.name: summary_path,
        manifest_path.name: manifest_path,
    }
    if not delta_df.empty:
        latest_files[delta_path.name] = delta_path
    _copy_latest(files=latest_files, latest_root=latest_dir)

    print("")
    print(
        results_df.to_string(
            index=False,
            formatters={col: _format_float for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]},
        )
    )
    if not delta_df.empty:
        print("")
        delta_row = delta_df.iloc[0]
        print("Delta vs baseline:")
        print(
            f"CAGR={_format_float(float(delta_row['Delta_CAGR']))} "
            f"Vol={_format_float(float(delta_row['Delta_Vol']))} "
            f"Sharpe={_format_float(float(delta_row['Delta_Sharpe']))} "
            f"MaxDD={_format_float(float(delta_row['Delta_MaxDD']))} "
            f"Turnover={_format_float(float(delta_row['Delta_Turnover']))}"
        )
    print("")
    print(f"Saved: {summary_path}")
    if not delta_df.empty:
        print(f"Saved: {delta_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_dir}")


if __name__ == "__main__":
    main()
