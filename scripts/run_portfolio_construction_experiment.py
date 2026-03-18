#!/usr/bin/env python3
"""Compare portfolio construction variants for the canonical baseline alpha engine."""

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


RESULTS_ROOT = Path("results") / "portfolio_construction_experiment"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_TOP_N = 75
DEFAULT_MAX_TICKERS = 2000
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
FACTOR_NAMES = ["gross_profitability", "reversal_1m"]
FACTOR_WEIGHTS = [0.7, 0.3]
FACTOR_AGGREGATION_METHOD = "linear"
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
VARIANT_SPECS = [
    {"strategy": "equal_weight", "weighting": "equal", "max_weight": 0.15},
    {"strategy": "inv_vol_weight", "weighting": "inv_vol", "max_weight": 0.15},
    {"strategy": "capped_inv_vol_weight", "weighting": "inv_vol", "max_weight": 0.05},
]
RESULT_COLUMNS = ["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    return parser.parse_args()


def build_run_config(args: argparse.Namespace, strategy: str) -> dict[str, Any]:
    spec = next((row for row in VARIANT_SPECS if str(row["strategy"]) == str(strategy)), None)
    if spec is None:
        raise ValueError(f"Unknown strategy variant: {strategy}")
    return {
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "universe_mode": "dynamic",
        "top_n": int(DEFAULT_TOP_N),
        "rebalance": DEFAULT_REBALANCE,
        "weighting": str(spec["weighting"]),
        "max_weight": float(spec["max_weight"]),
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "max_tickers": DEFAULT_MAX_TICKERS,
        "data_source": DATA_SOURCE,
        "data_cache_dir": DATA_CACHE_DIR,
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


def _read_holdings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing holdings artifact: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index().astype(float)


def annual_turnover_from_holdings(path: str | Path) -> float:
    holdings = _read_holdings(Path(path) / "holdings.csv")
    turnover = 0.5 * holdings.diff().abs().sum(axis=1).fillna(0.0)
    return float(turnover.mean() * 252.0) if not turnover.empty else float("nan")


def build_results_row(strategy: str, summary: dict[str, Any], outdir: str | Path) -> dict[str, Any]:
    return {
        "Strategy": str(strategy),
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "Vol": float(summary.get("Vol", float("nan"))),
        "Sharpe": float(summary.get("Sharpe", float("nan"))),
        "MaxDD": float(summary.get("MaxDD", float("nan"))),
        "Turnover": annual_turnover_from_holdings(outdir),
    }


def build_summary_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    ordered = results_df.sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    best = ordered.iloc[0]
    equal_row = results_df.loc[results_df["Strategy"] == "equal_weight"].iloc[0]
    improved = str(best["Strategy"]) != "equal_weight" and float(best["Sharpe"]) > float(equal_row["Sharpe"])
    return pd.DataFrame(
        [
            {
                "best_strategy": str(best["Strategy"]),
                "best_sharpe": float(best["Sharpe"]),
                "best_cagr": float(best["CAGR"]),
                "improves_over_equal_weight": bool(improved),
                "sharpe_delta_vs_equal_weight": float(best["Sharpe"] - equal_row["Sharpe"]),
                "cagr_delta_vs_equal_weight": float(best["CAGR"] - equal_row["CAGR"]),
            }
        ]
    )


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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    print("PORTFOLIO CONSTRUCTION EXPERIMENT")
    print("---------------------------------")
    print(
        "Config: "
        f"universe={DEFAULT_UNIVERSE} rebalance={DEFAULT_REBALANCE} costs_bps={DEFAULT_COSTS_BPS} "
        f"top_n={DEFAULT_TOP_N} factors={','.join(FACTOR_NAMES)} factor_weights={FACTOR_WEIGHTS} "
        f"factor_aggregation_method={FACTOR_AGGREGATION_METHOD}"
    )

    for spec in VARIANT_SPECS:
        strategy = str(spec["strategy"])
        cfg = build_run_config(args=args, strategy=strategy)
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        rows.append(build_results_row(strategy=strategy, summary=summary, outdir=run_outdir))
        run_manifest.append(
            {
                "strategy": strategy,
                "run_config": cfg,
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_df = results_df.sort_values(["Strategy"], kind="mergesort").reset_index(drop=True)
    summary_df = build_summary_frame(results_df)

    results_path = run_dir / "results.csv"
    summary_path = run_dir / "summary.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")

    best_strategy = str(summary_df.loc[0, "best_strategy"])
    improved = bool(summary_df.loc[0, "improves_over_equal_weight"])
    if improved:
        improvement_text = "A volatility-aware weighting method improves the canonical baseline on Sharpe."
    else:
        improvement_text = "No tested weighting method improves the canonical baseline on Sharpe."
    best_text = f"Best construction by Sharpe: {best_strategy}."

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_portfolio_construction_experiment.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "rebalance": DEFAULT_REBALANCE,
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "top_n": int(DEFAULT_TOP_N),
        "fundamentals_path": str(args.fundamentals_path),
        "factors": list(FACTOR_NAMES),
        "factor_weights": list(FACTOR_WEIGHTS),
        "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
        "variants": list(VARIANT_SPECS),
        "output_dir": str(run_dir),
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "assessment": {
            "improvement_text": improvement_text,
            "best_text": best_text,
        },
        "runtime_seconds": float(time.perf_counter() - t0),
        "runs": run_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    latest_root = base_output_dir / "latest"
    _copy_latest(
        files={
            "results.csv": results_path,
            "summary.csv": summary_path,
            "manifest.json": manifest_path,
        },
        latest_root=latest_root,
    )

    print("")
    print("| Strategy | CAGR | Vol | Sharpe | MaxDD | Turnover |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in results_df.to_dict(orient="records"):
        print(
            f"| {row['Strategy']} | {_format_float(float(row['CAGR']))} | {_format_float(float(row['Vol']))} | "
            f"{_format_float(float(row['Sharpe']))} | {_format_float(float(row['MaxDD']))} | "
            f"{_format_float(float(row['Turnover']))} |"
        )

    print("")
    print(improvement_text)
    print(best_text)
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_root}")


if __name__ == "__main__":
    main()
