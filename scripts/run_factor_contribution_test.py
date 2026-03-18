#!/usr/bin/env python3
"""Run a post-fix factor contribution test for the canonical baseline strategy."""

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


RESULTS_ROOT = Path("results") / "factor_contribution_test"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_TOP_N = 75
DEFAULT_MAX_TICKERS = 2000
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
FACTOR_NAMES = ["gross_profitability", "reversal_1m"]
FACTOR_AGGREGATION_METHOD = "linear"
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
STRATEGIES = [
    ("baseline", [0.70, 0.30]),
    ("profitability_only", [1.00, 0.00]),
    ("reversal_only", [0.00, 1.00]),
    ("equal_weight", [0.50, 0.50]),
]
RESULT_COLUMNS = [
    "strategy_name",
    "profitability_weight",
    "reversal_weight",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "Turnover",
    "hit_rate",
    "n_rebalance_dates",
    "median_selected_names",
]
SUMMARY_COLUMNS = [
    "strategy_name",
    "profitability_weight",
    "reversal_weight",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "Turnover",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
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



def _run_config(args: argparse.Namespace, weights: list[float]) -> dict[str, Any]:
    return {
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "universe_mode": "dynamic",
        "top_n": DEFAULT_TOP_N,
        "rebalance": DEFAULT_REBALANCE,
        "weighting": "equal",
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "max_tickers": DEFAULT_MAX_TICKERS,
        "data_source": DATA_SOURCE,
        "data_cache_dir": DATA_CACHE_DIR,
        "factor_name": list(FACTOR_NAMES),
        "factor_names": list(FACTOR_NAMES),
        "factor_weights": list(weights),
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
    print("FACTOR CONTRIBUTION SUMMARY")
    print("---------------------------")
    print(
        f"{'Strategy':18s} {'GP':>6s} {'Rev':>6s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Turnover':>10s}"
    )
    for _, row in summary_df.iterrows():
        print(
            f"{str(row['strategy_name']):18s} "
            f"{_format_float(float(row['profitability_weight'])):>6s} "
            f"{_format_float(float(row['reversal_weight'])):>6s} "
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

    print("POST-FIX FACTOR CONTRIBUTION TEST")
    print("---------------------------------")
    print(
        "Config: "
        f"universe={DEFAULT_UNIVERSE} rebalance={DEFAULT_REBALANCE} costs_bps={DEFAULT_COSTS_BPS} top_n={DEFAULT_TOP_N} "
        f"factors={','.join(FACTOR_NAMES)} factor_aggregation_method={FACTOR_AGGREGATION_METHOD}"
    )

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for strategy_name, weights in STRATEGIES:
        cfg = _run_config(args=args, weights=weights)
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        stats = _artifact_stats(
            outdir=run_outdir,
            rebalance=DEFAULT_REBALANCE,
            skipped_count=int(summary.get("RebalanceSkippedCount", 0)),
        )
        rows.append(
            {
                "strategy_name": str(strategy_name),
                "profitability_weight": float(weights[0]),
                "reversal_weight": float(weights[1]),
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
                "strategy_name": str(strategy_name),
                "weights": list(weights),
                "run_config": _to_serializable(cfg),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_df = results_df.sort_values(["Sharpe", "strategy_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    summary_df = results_df[SUMMARY_COLUMNS].copy()
    best_row = summary_df.iloc[0].to_dict()

    results_path = run_dir / "factor_contribution_results.csv"
    summary_path = run_dir / "factor_contribution_summary.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")

    profit_only = summary_df.loc[summary_df["strategy_name"].eq("profitability_only")]
    reversal_only = summary_df.loc[summary_df["strategy_name"].eq("reversal_only")]
    baseline = summary_df.loc[summary_df["strategy_name"].eq("baseline")]
    equal_weight = summary_df.loc[summary_df["strategy_name"].eq("equal_weight")]

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_factor_contribution_test.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "rebalance": DEFAULT_REBALANCE,
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "top_n": int(DEFAULT_TOP_N),
        "fundamentals_path": str(args.fundamentals_path),
        "factors": list(FACTOR_NAMES),
        "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
        "strategies": [{"strategy_name": name, "weights": weights} for name, weights in STRATEGIES],
        "output_dir": str(run_dir),
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "best_configuration": _to_serializable(best_row),
        "runtime_seconds": float(time.perf_counter() - t0),
        "notes": [
            "This is the post-fix factor contribution test for the canonical baseline factors.",
            "It uses the corrected runner-level adjusted-price PnL path and warehouse-backed legacy research store.",
            "All runs use linear aggregation so supplied factor weights are honored.",
        ],
        "runs": run_manifest,
    }
    manifest_path.write_text(json.dumps(_to_serializable(manifest), indent=2), encoding="utf-8")

    latest_root = base_output_dir / "latest"
    _copy_latest(
        files={
            "factor_contribution_results.csv": results_path,
            "factor_contribution_summary.csv": summary_path,
            "manifest.json": manifest_path,
        },
        latest_root=latest_root,
    )

    _print_summary(summary_df=summary_df)
    print("")
    print(
        "Best configuration: "
        f"{best_row['strategy_name']} Sharpe={_format_float(float(best_row['Sharpe']))} "
        f"CAGR={_format_float(float(best_row['CAGR']))} Vol={_format_float(float(best_row['Vol']))} "
        f"MaxDD={_format_float(float(best_row['MaxDD']))} Turnover={_format_float(float(best_row['Turnover']))}"
    )
    if not profit_only.empty and not reversal_only.empty:
        profit_sharpe = float(profit_only.iloc[0]['Sharpe'])
        reversal_sharpe = float(reversal_only.iloc[0]['Sharpe'])
        dominant = 'gross_profitability' if profit_sharpe >= reversal_sharpe else 'reversal_1m'
        print(f"Factor contribution: {dominant} contributes more on a standalone basis.")
    if not baseline.empty and not profit_only.empty:
        baseline_sharpe = float(baseline.iloc[0]['Sharpe'])
        profit_sharpe = float(profit_only.iloc[0]['Sharpe'])
        diff = baseline_sharpe - profit_sharpe
        if diff > 0.02:
            print(f"Reversal meaningfully improves profitability: baseline Sharpe exceeds profitability_only by {_format_float(diff)}.")
        elif diff > 0.0:
            print(f"Reversal improves profitability modestly: baseline Sharpe exceeds profitability_only by {_format_float(diff)}.")
        else:
            print(f"Reversal does not improve profitability in this test: baseline Sharpe trails profitability_only by {_format_float(abs(diff))}.")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_root}")


if __name__ == "__main__":
    main()
