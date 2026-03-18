#!/usr/bin/env python3
"""Run robustness checks for the leading multi-horizon quality strategy."""

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


RESULTS_ROOT = Path("results") / "quality_multi_horizon_robustness"
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
FACTOR_NAMES = ["gross_profitability", "reversal_1m", "reversal_5d"]
BASE_WEIGHTS = [0.70, 0.10, 0.20]
PERTURBED_WEIGHTS = [0.70, 0.05, 0.25]
FULL_PERIOD = (DEFAULT_START, DEFAULT_END)
SCENARIOS = [
    {
        "block": "Subperiod Tests",
        "strategy_name": "lead_strategy",
        "period_or_scenario": "2010-01-01→2016-12-31",
        "start": "2010-01-01",
        "end": "2016-12-31",
        "costs_bps": 10.0,
        "factor_weights": BASE_WEIGHTS,
    },
    {
        "block": "Subperiod Tests",
        "strategy_name": "lead_strategy",
        "period_or_scenario": "2016-01-01→2020-12-31",
        "start": "2016-01-01",
        "end": "2020-12-31",
        "costs_bps": 10.0,
        "factor_weights": BASE_WEIGHTS,
    },
    {
        "block": "Subperiod Tests",
        "strategy_name": "lead_strategy",
        "period_or_scenario": "2020-01-01→2024-12-31",
        "start": "2020-01-01",
        "end": "2024-12-31",
        "costs_bps": 10.0,
        "factor_weights": BASE_WEIGHTS,
    },
    {
        "block": "Cost Sensitivity",
        "strategy_name": "lead_strategy",
        "period_or_scenario": "Full period cost=10bps",
        "start": DEFAULT_START,
        "end": DEFAULT_END,
        "costs_bps": 10.0,
        "factor_weights": BASE_WEIGHTS,
    },
    {
        "block": "Cost Sensitivity",
        "strategy_name": "lead_strategy",
        "period_or_scenario": "Full period cost=20bps",
        "start": DEFAULT_START,
        "end": DEFAULT_END,
        "costs_bps": 20.0,
        "factor_weights": BASE_WEIGHTS,
    },
    {
        "block": "Slight Perturbation",
        "strategy_name": "perturbed_strategy",
        "period_or_scenario": "Full period perturbed weights",
        "start": DEFAULT_START,
        "end": DEFAULT_END,
        "costs_bps": 10.0,
        "factor_weights": PERTURBED_WEIGHTS,
    },
]
RESULT_COLUMNS = ["Strategy", "Period/Scenario", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    parser.add_argument("--max_tickers", type=int, default=DEFAULT_MAX_TICKERS)
    return parser.parse_args()



def _run_config(args: argparse.Namespace, scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": str(scenario["start"]),
        "end": str(scenario["end"]),
        "universe": DEFAULT_UNIVERSE,
        "universe_mode": "dynamic",
        "top_n": DEFAULT_TOP_N,
        "rebalance": DEFAULT_REBALANCE,
        "weighting": "inv_vol",
        "costs_bps": float(scenario["costs_bps"]),
        "max_tickers": int(args.max_tickers),
        "data_source": DATA_SOURCE,
        "data_cache_dir": DATA_CACHE_DIR,
        "factor_name": list(FACTOR_NAMES),
        "factor_names": list(FACTOR_NAMES),
        "factor_weights": list(scenario["factor_weights"]),
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



def _print_block(title: str, block_df: pd.DataFrame) -> None:
    print("")
    print(title.upper())
    print("-" * len(title))
    if block_df.empty:
        print("No results.")
        return
    print(block_df.to_string(index=False))



def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("QUALITY MULTI-HORIZON ROBUSTNESS")
    print("--------------------------------")
    print(
        "Base config: "
        f"universe={DEFAULT_UNIVERSE} universe_mode=dynamic rebalance={DEFAULT_REBALANCE} "
        f"top_n={DEFAULT_TOP_N} weighting=inv_vol factor_aggregation_method={FACTOR_AGGREGATION_METHOD} "
        f"factors={','.join(FACTOR_NAMES)}"
    )
    print(
        "Lead strategy weights: "
        f"gross_profitability={BASE_WEIGHTS[0]:.2f}, reversal_1m={BASE_WEIGHTS[1]:.2f}, reversal_5d={BASE_WEIGHTS[2]:.2f}"
    )

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        cfg = _run_config(args=args, scenario=scenario)
        print(
            f"Running [{scenario['block']}] {scenario['period_or_scenario']} "
            f"weights={scenario['factor_weights']} costs_bps={float(scenario['costs_bps']):.1f}"
        )
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        rows.append(
            {
                "Block": str(scenario["block"]),
                "Strategy": str(scenario["strategy_name"]),
                "Period/Scenario": str(scenario["period_or_scenario"]),
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "Turnover": extract_annual_turnover(summary=summary, outdir=run_outdir),
            }
        )
        run_manifest.append(
            {
                "block": str(scenario["block"]),
                "strategy_name": str(scenario["strategy_name"]),
                "period_or_scenario": str(scenario["period_or_scenario"]),
                "factor_weights": list(scenario["factor_weights"]),
                "costs_bps": float(scenario["costs_bps"]),
                "run_config": _to_serializable(cfg),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df[["Block", *RESULT_COLUMNS]].copy()
        results_df = results_df.sort_values(["Block", "Strategy", "Period/Scenario"], kind="mergesort").reset_index(drop=True)
    export_df = results_df.loc[:, RESULT_COLUMNS].copy() if not results_df.empty else pd.DataFrame(columns=RESULT_COLUMNS)

    results_path = run_dir / "quality_multi_horizon_robustness_results.csv"
    summary_path = run_dir / "quality_multi_horizon_robustness_summary.csv"
    manifest_path = run_dir / "manifest.json"

    export_df.to_csv(results_path, index=False)
    export_df.to_csv(summary_path, index=False)

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_quality_multi_horizon_robustness.py",
        "base_config": {
            "universe": DEFAULT_UNIVERSE,
            "universe_mode": "dynamic",
            "rebalance": DEFAULT_REBALANCE,
            "top_n": int(DEFAULT_TOP_N),
            "weighting": "inv_vol",
            "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
            "factor_names": list(FACTOR_NAMES),
            "baseline_costs_bps": float(DEFAULT_COSTS_BPS),
            "fundamentals_path": str(args.fundamentals_path),
        },
        "lead_strategy_weights": list(BASE_WEIGHTS),
        "perturbed_weights": list(PERTURBED_WEIGHTS),
        "scenarios": [
            {
                "block": str(s["block"]),
                "strategy_name": str(s["strategy_name"]),
                "period_or_scenario": str(s["period_or_scenario"]),
                "start": str(s["start"]),
                "end": str(s["end"]),
                "costs_bps": float(s["costs_bps"]),
                "factor_weights": list(s["factor_weights"]),
            }
            for s in SCENARIOS
        ],
        "output_dir": str(run_dir),
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "runtime_seconds": float(time.perf_counter() - t0),
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

    for block_name in ["Subperiod Tests", "Cost Sensitivity", "Slight Perturbation"]:
        block_df = export_df.loc[results_df["Block"] == block_name] if not results_df.empty else pd.DataFrame(columns=RESULT_COLUMNS)
        _print_block(block_name, block_df)

    print("")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_root}")


if __name__ == "__main__":
    main()
