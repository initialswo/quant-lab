"""Run a controlled baseline comparison of core factor composites."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover
from quant_lab.strategies.topn import rebalance_mask


RESULTS_ROOT = Path("results") / "core_factor_composites"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_TOP_N = 10
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_MAX_TICKERS = 2000
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
RESULT_COLUMNS = [
    "strategy_name",
    "factors",
    "start",
    "end",
    "universe",
    "top_n",
    "rebalance",
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
STRATEGY_SPECS: list[dict[str, Any]] = [
    {"strategy_name": "momentum_only", "factors": ["momentum_12_1"]},
    {"strategy_name": "reversal_only", "factors": ["reversal_1m"]},
    {"strategy_name": "profitability_only", "factors": ["gross_profitability"]},
    {
        "strategy_name": "momentum_profitability",
        "factors": ["momentum_12_1", "gross_profitability"],
    },
    {
        "strategy_name": "reversal_profitability",
        "factors": ["reversal_1m", "gross_profitability"],
    },
    {
        "strategy_name": "momentum_reversal_profitability",
        "factors": ["momentum_12_1", "reversal_1m", "gross_profitability"],
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--rebalance", choices=["daily", "weekly", "biweekly", "monthly"], default=DEFAULT_REBALANCE)
    parser.add_argument("--costs_bps", type=float, default=DEFAULT_COSTS_BPS)
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()



def _equal_weights(factors: list[str]) -> list[float]:
    if not factors:
        raise ValueError("factor set must be non-empty")
    return [1.0 / float(len(factors))] * len(factors)



def _run_config(args: argparse.Namespace, factors: list[str]) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "start": str(args.start),
        "end": str(args.end),
        "universe": str(args.universe),
        "universe_mode": "dynamic",
        "top_n": int(args.top_n),
        "rebalance": str(args.rebalance),
        "weighting": "equal",
        "costs_bps": float(args.costs_bps),
        "max_tickers": DEFAULT_MAX_TICKERS,
        "data_source": DATA_SOURCE,
        "data_cache_dir": DATA_CACHE_DIR,
        "factor_name": list(factors),
        "factor_names": list(factors),
        "factor_weights": _equal_weights(factors),
        "portfolio_mode": "composite",
        "factor_aggregation_method": "mean_rank",
        "use_factor_normalization": True,
        "use_sector_neutralization": False,
        "use_size_neutralization": False,
        "orthogonalize_factors": False,
        "fundamentals_path": str(args.fundamentals_path),
        "save_artifacts": True,
    }
    return cfg



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



def _result_row(
    summary: dict[str, Any],
    outdir: str,
    strategy_name: str,
    factors: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stats = _artifact_stats(
        outdir=outdir,
        rebalance=str(args.rebalance),
        skipped_count=int(summary.get("RebalanceSkippedCount", 0)),
    )
    return {
        "strategy_name": str(strategy_name),
        "factors": ",".join(factors),
        "start": str(args.start),
        "end": str(args.end),
        "universe": str(args.universe),
        "top_n": int(args.top_n),
        "rebalance": str(args.rebalance),
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "Vol": float(summary.get("Vol", float("nan"))),
        "Sharpe": float(summary.get("Sharpe", float("nan"))),
        "MaxDD": float(summary.get("MaxDD", float("nan"))),
        "Turnover": extract_annual_turnover(summary=summary, outdir=outdir),
        "hit_rate": float(stats["hit_rate"]),
        "n_rebalance_dates": int(stats["n_rebalance_dates"]),
        "median_selected_names": float(stats["median_selected_names"]),
    }



def _summary_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = results_df[["strategy_name", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]].copy()
    return summary.reindex(columns=SUMMARY_COLUMNS)



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



def main() -> None:
    args = _parse_args()
    if str(args.universe).strip().lower() != "liquid_us":
        raise ValueError("This baseline script currently supports only --universe liquid_us.")
    if int(args.top_n) <= 0:
        raise ValueError("--top_n must be > 0")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(str(args.output_dir)) if str(args.output_dir).strip() else RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    results_rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    print("")
    print("CORE FACTOR COMPOSITES")
    print("----------------------")
    for spec in STRATEGY_SPECS:
        strategy_name = str(spec["strategy_name"])
        factors = list(spec["factors"])
        cfg = _run_config(args=args, factors=factors)
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        row = _result_row(
            summary=summary,
            outdir=run_outdir,
            strategy_name=strategy_name,
            factors=factors,
            args=args,
        )
        results_rows.append(row)
        run_manifest.append(
            {
                "strategy_name": strategy_name,
                "factors": factors,
                "run_config": _to_serializable(cfg),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )
        print(
            f"{strategy_name:32s} "
            f"Sharpe={_format_float(float(row['Sharpe']))} "
            f"CAGR={_format_float(float(row['CAGR']))} "
            f"MaxDD={_format_float(float(row['MaxDD']))} "
            f"Turnover={_format_float(float(row['Turnover']))}"
        )

    results_df = pd.DataFrame(results_rows, columns=RESULT_COLUMNS)
    results_df = results_df.sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    summary_df = _summary_frame(results_df).sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(drop=True)

    results_path = output_dir / "core_factor_results.csv"
    summary_path = output_dir / "core_factor_summary.csv"
    manifest_path = output_dir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_core_factor_composites.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": str(args.universe),
        "top_n": int(args.top_n),
        "rebalance": str(args.rebalance),
        "costs_bps": float(args.costs_bps),
        "fundamentals_path": str(args.fundamentals_path),
        "strategies": STRATEGY_SPECS,
        "composite_method": "mean_rank",
        "weighting": "equal",
        "output_dir": str(output_dir),
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "runtime_seconds": float(time.perf_counter() - t0),
        "notes": [
            "All strategies run through run_backtest with the liquid_us dynamic-universe path.",
            "Composite aggregation reuses the runner's factor_aggregation_method='mean_rank'.",
            "No custom preprocessing, neutralization, or weighting is added in this script.",
        ],
        "runs": run_manifest,
    }
    manifest_path.write_text(json.dumps(_to_serializable(manifest), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
