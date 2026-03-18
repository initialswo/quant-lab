#!/usr/bin/env python3
"""Run the post-fix three-factor blend test against the canonical corrected baseline."""

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


RESULTS_ROOT = Path("results") / "three_factor_blend_experiment"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_MAX_TICKERS = 2000
DEFAULT_TOP_N = 75
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
FACTOR_NAMES = ["gross_profitability", "reversal_1m", "momentum_12_1"]
FACTOR_AGGREGATION_METHOD = "linear"
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
CANONICAL_BASELINE_SHARPE = 0.8023
WEIGHT_SPECS = [
    ("baseline", [0.70, 0.30, 0.00]),
    ("blend_1", [0.60, 0.25, 0.15]),
    ("blend_2", [0.50, 0.30, 0.20]),
    ("blend_3", [0.50, 0.40, 0.10]),
    ("blend_4", [0.40, 0.40, 0.20]),
]
RESULT_COLUMNS = [
    "strategy_name",
    "profitability_weight",
    "reversal_weight",
    "momentum_weight",
    "factors",
    "factor_aggregation_method",
    "start",
    "end",
    "universe",
    "rebalance",
    "top_n",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "Turnover",
    "hit_rate",
    "n_rebalance_dates",
    "median_selected_names",
    "beats_corrected_baseline",
]
SUMMARY_COLUMNS = [
    "strategy_name",
    "profitability_weight",
    "reversal_weight",
    "momentum_weight",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "Turnover",
    "beats_corrected_baseline",
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



def _result_row(
    strategy_name: str,
    weights: list[float],
    summary: dict[str, Any],
    outdir: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    stats = _artifact_stats(
        outdir=outdir,
        rebalance=DEFAULT_REBALANCE,
        skipped_count=int(summary.get("RebalanceSkippedCount", 0)),
    )
    sharpe = float(summary.get("Sharpe", float("nan")))
    return {
        "strategy_name": str(strategy_name),
        "profitability_weight": float(weights[0]),
        "reversal_weight": float(weights[1]),
        "momentum_weight": float(weights[2]),
        "factors": ",".join(FACTOR_NAMES),
        "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
        "start": str(start),
        "end": str(end),
        "universe": DEFAULT_UNIVERSE,
        "rebalance": DEFAULT_REBALANCE,
        "top_n": int(DEFAULT_TOP_N),
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "Vol": float(summary.get("Vol", float("nan"))),
        "Sharpe": sharpe,
        "MaxDD": float(summary.get("MaxDD", float("nan"))),
        "Turnover": extract_annual_turnover(summary=summary, outdir=outdir),
        "hit_rate": float(stats["hit_rate"]),
        "n_rebalance_dates": int(stats["n_rebalance_dates"]),
        "median_selected_names": float(stats["median_selected_names"]),
        "beats_corrected_baseline": bool(pd.notna(sharpe) and sharpe > CANONICAL_BASELINE_SHARPE),
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
    print("THREE-FACTOR BLEND SUMMARY")
    print("--------------------------")
    print(
        f"{'Strategy':12s} {'GP':>6s} {'Rev':>6s} {'Mom':>6s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Turnover':>10s}  Beat?"
    )
    for _, row in summary_df.iterrows():
        beat = "YES" if bool(row["beats_corrected_baseline"]) else "NO"
        print(
            f"{str(row['strategy_name']):12s} "
            f"{_format_float(float(row['profitability_weight'])):>6s} "
            f"{_format_float(float(row['reversal_weight'])):>6s} "
            f"{_format_float(float(row['momentum_weight'])):>6s} "
            f"{_format_float(float(row['CAGR'])):>8s} "
            f"{_format_float(float(row['Vol'])):>8s} "
            f"{_format_float(float(row['Sharpe'])):>8s} "
            f"{_format_float(float(row['MaxDD'])):>8s} "
            f"{_format_float(float(row['Turnover'])):>10s}  "
            f"{beat}"
        )



def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("POST-FIX THREE-FACTOR BLEND TEST")
    print("--------------------------------")
    print(
        "Config: "
        f"universe={DEFAULT_UNIVERSE} rebalance={DEFAULT_REBALANCE} costs_bps={DEFAULT_COSTS_BPS} top_n={DEFAULT_TOP_N} "
        f"factors={','.join(FACTOR_NAMES)} factor_aggregation_method={FACTOR_AGGREGATION_METHOD} baseline_sharpe={CANONICAL_BASELINE_SHARPE:.4f}"
    )

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for strategy_name, weights in WEIGHT_SPECS:
        cfg = _run_config(args=args, weights=weights)
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        rows.append(
            _result_row(
                strategy_name=strategy_name,
                weights=weights,
                summary=summary,
                outdir=run_outdir,
                start=str(args.start),
                end=str(args.end),
            )
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

    results_path = run_dir / "three_factor_blend_results.csv"
    summary_path = run_dir / "three_factor_blend_summary.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")

    best_row = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    beats_baseline = bool(summary_df["beats_corrected_baseline"].any()) if not summary_df.empty else False
    beaters = summary_df.loc[summary_df["beats_corrected_baseline"], "strategy_name"].astype(str).tolist() if not summary_df.empty else []

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_three_factor_blend_experiment.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "rebalance": DEFAULT_REBALANCE,
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "top_n": int(DEFAULT_TOP_N),
        "fundamentals_path": str(args.fundamentals_path),
        "factors": list(FACTOR_NAMES),
        "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
        "canonical_baseline_sharpe": float(CANONICAL_BASELINE_SHARPE),
        "output_dir": str(run_dir),
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "best_configuration": _to_serializable(best_row),
        "beats_corrected_baseline": bool(beats_baseline),
        "beating_strategies": beaters,
        "runtime_seconds": float(time.perf_counter() - t0),
        "notes": [
            "This is the post-fix three-factor blend test against the canonical corrected baseline.",
            "It reuses the corrected runner-level adjusted-price PnL path and warehouse-backed legacy research store.",
            "All strategies run with linear factor aggregation so supplied weights are honored.",
        ],
        "runs": run_manifest,
    }
    manifest_path.write_text(json.dumps(_to_serializable(manifest), indent=2), encoding="utf-8")

    latest_root = base_output_dir / "latest"
    _copy_latest(
        files={
            "three_factor_blend_results.csv": results_path,
            "three_factor_blend_summary.csv": summary_path,
            "manifest.json": manifest_path,
        },
        latest_root=latest_root,
    )

    _print_summary(summary_df=summary_df)
    print("")
    if best_row:
        print(
            "Best configuration: "
            f"{best_row['strategy_name']} "
            f"Sharpe={_format_float(float(best_row['Sharpe']))} "
            f"CAGR={_format_float(float(best_row['CAGR']))} "
            f"Vol={_format_float(float(best_row['Vol']))} "
            f"MaxDD={_format_float(float(best_row['MaxDD']))} "
            f"Turnover={_format_float(float(best_row['Turnover']))}"
        )
    if beats_baseline:
        print(
            "Momentum improved the corrected baseline: "
            + ", ".join(beaters)
            + f" beat baseline Sharpe {CANONICAL_BASELINE_SHARPE:.4f}."
        )
    else:
        print(f"Momentum did not improve the corrected baseline Sharpe of {CANONICAL_BASELINE_SHARPE:.4f}.")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_root}")


if __name__ == "__main__":
    main()
