#!/usr/bin/env python3
"""Run post-fix stability comparisons for the canonical baseline and blend_1."""

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


RESULTS_ROOT = Path("results") / "baseline_vs_blend1_stability"
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_TOP_N = 75
DEFAULT_MAX_TICKERS = 2000
FACTOR_NAMES = ["gross_profitability", "reversal_1m", "momentum_12_1"]
FACTOR_AGGREGATION_METHOD = "linear"
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
STRATEGIES = [
    ("baseline", [0.70, 0.30, 0.00]),
    ("blend_1", [0.60, 0.25, 0.15]),
]
PERIODS = [
    ("2005-2009", "2005-01-01", "2009-12-31"),
    ("2010-2016", "2010-01-01", "2016-12-31"),
    ("2017-2024", "2017-01-01", "2024-12-31"),
    ("full_sample", "2010-01-01", "2024-12-31"),
]
RESULT_COLUMNS = [
    "period",
    "period_start",
    "period_end",
    "strategy_name",
    "profitability_weight",
    "reversal_weight",
    "momentum_weight",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "Turnover",
    "hit_rate",
    "n_rebalance_dates",
    "median_selected_names",
    "outperformed_baseline_in_period",
]
SUMMARY_COLUMNS = [
    "period",
    "strategy_name",
    "profitability_weight",
    "reversal_weight",
    "momentum_weight",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "Turnover",
    "outperformed_baseline_in_period",
]
PERIOD_ORDER = {name: idx for idx, (name, _, _) in enumerate(PERIODS)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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



def _run_config(start: str, end: str, fundamentals_path: str, weights: list[float]) -> dict[str, Any]:
    return {
        "start": str(start),
        "end": str(end),
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
        "fundamentals_path": str(fundamentals_path),
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
    print("STABILITY SUMMARY")
    print("-----------------")
    print(
        f"{'Period':11s} {'Strategy':9s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Turnover':>10s}"
    )
    for _, row in summary_df.iterrows():
        print(
            f"{str(row['period']):11s} "
            f"{str(row['strategy_name']):9s} "
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

    print("POST-FIX BASELINE VS BLEND_1 STABILITY TEST")
    print("-------------------------------------------")
    print(
        "Config: "
        f"universe={DEFAULT_UNIVERSE} rebalance={DEFAULT_REBALANCE} costs_bps={DEFAULT_COSTS_BPS} top_n={DEFAULT_TOP_N} "
        f"factors={','.join(FACTOR_NAMES)} factor_aggregation_method={FACTOR_AGGREGATION_METHOD}"
    )

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for period_name, start, end in PERIODS:
        period_rows: list[dict[str, Any]] = []
        for strategy_name, weights in STRATEGIES:
            cfg = _run_config(start=start, end=end, fundamentals_path=str(args.fundamentals_path), weights=weights)
            summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
            stats = _artifact_stats(
                outdir=run_outdir,
                rebalance=DEFAULT_REBALANCE,
                skipped_count=int(summary.get("RebalanceSkippedCount", 0)),
            )
            row = {
                "period": str(period_name),
                "period_start": str(start),
                "period_end": str(end),
                "strategy_name": str(strategy_name),
                "profitability_weight": float(weights[0]),
                "reversal_weight": float(weights[1]),
                "momentum_weight": float(weights[2]),
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "Turnover": extract_annual_turnover(summary=summary, outdir=run_outdir),
                "hit_rate": float(stats["hit_rate"]),
                "n_rebalance_dates": int(stats["n_rebalance_dates"]),
                "median_selected_names": float(stats["median_selected_names"]),
                "outperformed_baseline_in_period": False,
            }
            period_rows.append(row)
            run_manifest.append(
                {
                    "period": str(period_name),
                    "strategy_name": str(strategy_name),
                    "weights": list(weights),
                    "run_config": _to_serializable(cfg),
                    "backtest_outdir": str(run_outdir),
                    "summary_path": str(Path(run_outdir) / "summary.json"),
                }
            )
        baseline_row = next((r for r in period_rows if r["strategy_name"] == "baseline"), None)
        if baseline_row is not None:
            baseline_sharpe = float(baseline_row["Sharpe"])
            for row in period_rows:
                row["outperformed_baseline_in_period"] = bool(
                    row["strategy_name"] != "baseline" and pd.notna(row["Sharpe"]) and row["Sharpe"] > baseline_sharpe
                )
        rows.extend(period_rows)

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_df["_period_order"] = results_df["period"].map(PERIOD_ORDER)
    results_df["_strategy_order"] = results_df["strategy_name"].map({"baseline": 0, "blend_1": 1})
    results_df = results_df.sort_values(["_period_order", "_strategy_order"], kind="mergesort").drop(columns=["_period_order", "_strategy_order"]).reset_index(drop=True)
    summary_df = results_df[SUMMARY_COLUMNS].copy()

    results_path = run_dir / "baseline_vs_blend1_stability_results.csv"
    summary_path = run_dir / "baseline_vs_blend1_stability_summary.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")

    period_outcomes = []
    for period_name, _, _ in PERIODS:
        part = results_df.loc[results_df["period"].eq(period_name)].copy()
        base = part.loc[part["strategy_name"].eq("baseline")]
        blend = part.loc[part["strategy_name"].eq("blend_1")]
        if base.empty or blend.empty:
            continue
        base_sharpe = float(base.iloc[0]["Sharpe"])
        blend_sharpe = float(blend.iloc[0]["Sharpe"])
        period_outcomes.append({
            "period": period_name,
            "baseline_sharpe": base_sharpe,
            "blend_1_sharpe": blend_sharpe,
            "blend_1_beats_baseline": bool(pd.notna(blend_sharpe) and blend_sharpe > base_sharpe),
        })

    blend_wins = [x for x in period_outcomes if x["blend_1_beats_baseline"]]
    consistent = len(blend_wins) == len(period_outcomes) and len(period_outcomes) > 0

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_baseline_vs_blend1_stability.py",
        "universe": DEFAULT_UNIVERSE,
        "rebalance": DEFAULT_REBALANCE,
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "top_n": int(DEFAULT_TOP_N),
        "fundamentals_path": str(args.fundamentals_path),
        "factors": list(FACTOR_NAMES),
        "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
        "periods": [{"period": p, "start": s, "end": e} for p, s, e in PERIODS],
        "strategies": [{"strategy_name": name, "weights": w} for name, w in STRATEGIES],
        "output_dir": str(run_dir),
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "period_outcomes": period_outcomes,
        "blend_1_consistently_beats_baseline": bool(consistent),
        "runtime_seconds": float(time.perf_counter() - t0),
        "notes": [
            "This compares the canonical corrected baseline and blend_1 across multiple historical windows.",
            "It uses the corrected runner-level adjusted-price PnL path and warehouse-backed legacy research store.",
            "All runs use linear aggregation so supplied factor weights are honored.",
        ],
        "runs": run_manifest,
    }
    manifest_path.write_text(json.dumps(_to_serializable(manifest), indent=2), encoding="utf-8")

    latest_root = base_output_dir / "latest"
    _copy_latest(
        files={
            "baseline_vs_blend1_stability_results.csv": results_path,
            "baseline_vs_blend1_stability_summary.csv": summary_path,
            "manifest.json": manifest_path,
        },
        latest_root=latest_root,
    )

    _print_summary(summary_df=summary_df)
    print("")
    if consistent:
        print("Conclusion: blend_1 beats the baseline consistently across all tested periods.")
        print("Momentum improvement appears stable across the tested regimes.")
    else:
        winning_periods = [x['period'] for x in blend_wins]
        if winning_periods:
            print("Conclusion: blend_1 does not beat the baseline consistently across all periods.")
            print("Momentum improvement appears regime-dependent; blend_1 wins in: " + ", ".join(winning_periods))
        else:
            print("Conclusion: blend_1 does not beat the baseline in any tested period.")
            print("Momentum improvement does not appear stable in this comparison.")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_root}")


if __name__ == "__main__":
    main()
