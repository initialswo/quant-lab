"""Run weighted gross_profitability/reversal_1m blend experiments."""

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


RESULTS_ROOT = Path("results") / "weighted_factor_blends"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_TOP_N = 50
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_MAX_TICKERS = 2000
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
FACTOR_SET = ["gross_profitability", "reversal_1m"]
WEIGHT_SPECS: list[dict[str, Any]] = [
    {"strategy_name": "profitability_100", "profitability_weight": 1.0, "reversal_weight": 0.0},
    {"strategy_name": "profitability_90_reversal_10", "profitability_weight": 0.9, "reversal_weight": 0.1},
    {"strategy_name": "profitability_80_reversal_20", "profitability_weight": 0.8, "reversal_weight": 0.2},
    {"strategy_name": "profitability_70_reversal_30", "profitability_weight": 0.7, "reversal_weight": 0.3},
    {"strategy_name": "profitability_60_reversal_40", "profitability_weight": 0.6, "reversal_weight": 0.4},
    {"strategy_name": "profitability_50_reversal_50", "profitability_weight": 0.5, "reversal_weight": 0.5},
    {"strategy_name": "reversal_100", "profitability_weight": 0.0, "reversal_weight": 1.0},
]
RESULT_COLUMNS = [
    "strategy_name",
    "profitability_weight",
    "reversal_weight",
    "CAGR",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "turnover",
    "hit_rate",
    "n_rebalance_dates",
    "median_selected_names",
]
SUMMARY_COLUMNS = [
    "strategy_name",
    "profitability_weight",
    "reversal_weight",
    "sharpe",
    "CAGR",
    "max_drawdown",
    "turnover",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--rebalance", choices=["daily", "weekly", "biweekly", "monthly"], default=DEFAULT_REBALANCE)
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--costs_bps", type=float, default=DEFAULT_COSTS_BPS)
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def _run_config(args: argparse.Namespace, profitability_weight: float, reversal_weight: float) -> dict[str, Any]:
    return {
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
        "factor_name": list(FACTOR_SET),
        "factor_names": list(FACTOR_SET),
        "factor_weights": [float(profitability_weight), float(reversal_weight)],
        "portfolio_mode": "composite",
        "factor_aggregation_method": "linear",
        "use_factor_normalization": True,
        "use_sector_neutralization": False,
        "use_size_neutralization": False,
        "orthogonalize_factors": False,
        "fundamentals_path": str(args.fundamentals_path),
        "save_artifacts": True,
    }


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
    outdir: str | Path,
    strategy_name: str,
    profitability_weight: float,
    reversal_weight: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stats = _artifact_stats(
        outdir=outdir,
        rebalance=str(args.rebalance),
        skipped_count=int(summary.get("RebalanceSkippedCount", 0)),
    )
    return {
        "strategy_name": str(strategy_name),
        "profitability_weight": float(profitability_weight),
        "reversal_weight": float(reversal_weight),
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "ann_vol": float(summary.get("Vol", float("nan"))),
        "sharpe": float(summary.get("Sharpe", float("nan"))),
        "max_drawdown": float(summary.get("MaxDD", float("nan"))),
        "turnover": extract_annual_turnover(summary=summary, outdir=outdir),
        "hit_rate": float(stats["hit_rate"]),
        "n_rebalance_dates": int(stats["n_rebalance_dates"]),
        "median_selected_names": float(stats["median_selected_names"]),
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


def main() -> None:
    args = _parse_args()
    if int(args.top_n) <= 0:
        raise ValueError("--top_n must be > 0")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(str(args.output_dir)) if str(args.output_dir).strip() else RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []
    daily_returns_map: dict[str, str] = {}
    holdings_map: dict[str, str] = {}

    print("")
    print("WEIGHTED FACTOR BLENDS")
    print("----------------------")
    for spec in WEIGHT_SPECS:
        strategy_name = str(spec["strategy_name"])
        profitability_weight = float(spec["profitability_weight"])
        reversal_weight = float(spec["reversal_weight"])
        cfg = _run_config(
            args=args,
            profitability_weight=profitability_weight,
            reversal_weight=reversal_weight,
        )
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        row = _result_row(
            summary=summary,
            outdir=run_outdir,
            strategy_name=strategy_name,
            profitability_weight=profitability_weight,
            reversal_weight=reversal_weight,
            args=args,
        )
        rows.append(row)
        run_manifest.append(
            {
                "strategy_name": strategy_name,
                "factor_set": list(FACTOR_SET),
                "weights": {
                    "gross_profitability": profitability_weight,
                    "reversal_1m": reversal_weight,
                },
                "run_config": _to_serializable(cfg),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )

        equity_src = Path(run_outdir) / "equity.csv"
        holdings_src = Path(run_outdir) / "holdings.csv"
        equity_dst = output_dir / f"{strategy_name}_daily_returns.csv"
        holdings_dst = output_dir / f"{strategy_name}_holdings.csv"
        if equity_src.exists():
            shutil.copy2(equity_src, equity_dst)
            daily_returns_map[strategy_name] = str(equity_dst)
        if holdings_src.exists():
            shutil.copy2(holdings_src, holdings_dst)
            holdings_map[strategy_name] = str(holdings_dst)

        print(
            f"{strategy_name:32s} "
            f"Sharpe={_format_float(float(row['sharpe']))} "
            f"CAGR={_format_float(float(row['CAGR']))} "
            f"MaxDD={_format_float(float(row['max_drawdown']))} "
            f"Turnover={_format_float(float(row['turnover']))}"
        )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_df = results_df.sort_values(["sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    summary_df = results_df.loc[:, SUMMARY_COLUMNS].copy()
    summary_df = summary_df.sort_values(["sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(drop=True)

    runtime_seconds = time.perf_counter() - t0
    results_path = output_dir / "weighted_factor_blends_results.csv"
    summary_path = output_dir / "weighted_factor_blends_summary.csv"
    manifest_path = output_dir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_weighted_factor_blends.py",
        "factor_set": list(FACTOR_SET),
        "weights_tested": [
            {
                "strategy_name": str(spec["strategy_name"]),
                "gross_profitability": float(spec["profitability_weight"]),
                "reversal_1m": float(spec["reversal_weight"]),
            }
            for spec in WEIGHT_SPECS
        ],
        "backtest_settings": {
            "start": str(args.start),
            "end": str(args.end),
            "universe": str(args.universe),
            "universe_mode": "dynamic",
            "rebalance": str(args.rebalance),
            "top_n": int(args.top_n),
            "weighting": "equal",
            "costs_bps": float(args.costs_bps),
            "data_source": DATA_SOURCE,
            "data_cache_dir": DATA_CACHE_DIR,
            "fundamentals_path": str(args.fundamentals_path),
            "factor_aggregation_method": "linear",
            "use_factor_normalization": True,
        },
        "output_dir": str(output_dir),
        "output_paths": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "daily_returns": daily_returns_map,
            "holdings": holdings_map,
        },
        "runtime_seconds": float(runtime_seconds),
        "runs": run_manifest,
    }
    manifest_path.write_text(json.dumps(_to_serializable(manifest), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
