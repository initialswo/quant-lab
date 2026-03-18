#!/usr/bin/env python3
"""Sweep portfolio weights between quality and momentum sleeves."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest


RESULTS_ROOT = Path("results") / "quality_momentum_weight_sweep"
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
SLEEVE_SPECS: list[dict[str, Any]] = [
    {
        "strategy_name": "quality_only",
        "factor_names": ["gross_profitability", "reversal_1m", "reversal_5d"],
        "factor_weights": [0.70, 0.05, 0.25],
    },
    {
        "strategy_name": "momentum_only",
        "factor_names": ["momentum_12_1"],
        "factor_weights": [1.0],
    },
]
PORTFOLIO_SPECS: list[dict[str, Any]] = [
    {
        "strategy_name": "quality_50_momentum_50",
        "quality_weight": 0.50,
        "momentum_weight": 0.50,
    },
    {
        "strategy_name": "quality_70_momentum_30",
        "quality_weight": 0.70,
        "momentum_weight": 0.30,
    },
    {
        "strategy_name": "quality_80_momentum_20",
        "quality_weight": 0.80,
        "momentum_weight": 0.20,
    },
    {
        "strategy_name": "quality_90_momentum_10",
        "quality_weight": 0.90,
        "momentum_weight": 0.10,
    },
]
RESULT_COLUMNS = ["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    parser.add_argument("--max_tickers", type=int, default=DEFAULT_MAX_TICKERS)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    return parser.parse_args()



def _run_config(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
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
        "factor_name": list(spec["factor_names"]),
        "factor_names": list(spec["factor_names"]),
        "factor_weights": list(spec["factor_weights"]),
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



def _read_equity_frame(outdir: Path) -> pd.DataFrame:
    path = outdir / "equity.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing equity artifact: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()



def _load_daily_return(outdir: Path, name: str) -> pd.Series:
    equity = _read_equity_frame(outdir)
    if "DailyReturn" in equity.columns:
        values = pd.to_numeric(equity["DailyReturn"], errors="coerce").fillna(0.0)
        return values.rename(name)
    if "Equity" not in equity.columns:
        raise ValueError(f"equity.csv at {outdir} must include DailyReturn or Equity")
    daily_return = pd.to_numeric(equity["Equity"], errors="coerce").pct_change().fillna(0.0)
    return daily_return.rename(name)



def _metrics_row(strategy_name: str, daily_return: pd.Series) -> dict[str, Any]:
    metrics = compute_metrics(pd.to_numeric(daily_return, errors="coerce").fillna(0.0).astype(float))
    return {
        "Strategy": str(strategy_name),
        "CAGR": float(metrics.get("CAGR", float("nan"))),
        "Vol": float(metrics.get("Vol", float("nan"))),
        "Sharpe": float(metrics.get("Sharpe", float("nan"))),
        "MaxDD": float(metrics.get("MaxDD", float("nan"))),
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("QUALITY MOMENTUM WEIGHT SWEEP")
    print("-----------------------------")
    print(
        "Base config: "
        f"universe={DEFAULT_UNIVERSE} universe_mode={DEFAULT_UNIVERSE_MODE} "
        f"rebalance={DEFAULT_REBALANCE} top_n={DEFAULT_TOP_N} weighting={DEFAULT_WEIGHTING} "
        f"costs_bps={DEFAULT_COSTS_BPS:.1f} factor_aggregation_method={FACTOR_AGGREGATION_METHOD}"
    )
    print(
        "Quality sleeve: "
        "gross_profitability=0.70, reversal_1m=0.05, reversal_5d=0.25"
    )
    print("Momentum sleeve: momentum_12_1=1.00")

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    sleeve_returns: list[pd.Series] = []
    sleeve_manifest: list[dict[str, Any]] = []

    for spec in SLEEVE_SPECS:
        strategy_name = str(spec["strategy_name"])
        cfg = _run_config(args=args, spec=spec)
        print(f"Running {strategy_name}...")
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        daily_return = _load_daily_return(Path(run_outdir), strategy_name)
        sleeve_returns.append(daily_return)
        sleeve_manifest.append(
            {
                "strategy_name": strategy_name,
                "factor_names": list(spec["factor_names"]),
                "factor_weights": list(spec["factor_weights"]),
                "run_config": _to_serializable(cfg),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
                "engine_summary": _to_serializable(summary),
            }
        )

    sleeve_returns_df = pd.concat(sleeve_returns, axis=1).sort_index().fillna(0.0)
    portfolio_returns: list[pd.Series] = []
    metrics_rows: list[dict[str, Any]] = []
    portfolio_manifest: list[dict[str, Any]] = []

    for spec in PORTFOLIO_SPECS:
        strategy_name = str(spec["strategy_name"])
        portfolio_return = (
            sleeve_returns_df["quality_only"] * float(spec["quality_weight"])
            + sleeve_returns_df["momentum_only"] * float(spec["momentum_weight"])
        ).rename(strategy_name)
        portfolio_returns.append(portfolio_return)
        metrics_rows.append(_metrics_row(strategy_name=strategy_name, daily_return=portfolio_return))
        portfolio_manifest.append(
            {
                "strategy_name": strategy_name,
                "quality_weight": float(spec["quality_weight"]),
                "momentum_weight": float(spec["momentum_weight"]),
                "combination_method": "Combine sleeve daily returns after individual backtests.",
            }
        )

    portfolio_returns_df = pd.concat(portfolio_returns, axis=1).sort_index().fillna(0.0)
    results_df = pd.DataFrame(metrics_rows, columns=RESULT_COLUMNS)

    metrics_path = run_dir / "metrics.csv"
    sleeve_returns_path = run_dir / "sleeve_daily_returns.csv"
    portfolio_returns_path = run_dir / "portfolio_daily_returns.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(metrics_path, index=False, float_format="%.10g")
    sleeve_returns_df.to_csv(sleeve_returns_path, index_label="date", float_format="%.10g")
    portfolio_returns_df.to_csv(portfolio_returns_path, index_label="date", float_format="%.10g")
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
                "sleeves": [
                    {
                        "strategy_name": "quality_only",
                        "factor_names": ["gross_profitability", "reversal_1m", "reversal_5d"],
                        "factor_weights": [0.70, 0.05, 0.25],
                    },
                    {
                        "strategy_name": "momentum_only",
                        "factor_names": ["momentum_12_1"],
                        "factor_weights": [1.0],
                    },
                ],
                "portfolio_weight_sweep": portfolio_manifest,
                "notes": [
                    "Both sleeves are run once and reused across all tested portfolio blends.",
                    "Portfolio returns are combined post-backtest from sleeve daily return series.",
                    "Engine logic is unchanged; this experiment is implemented entirely at the script layer.",
                ],
                "runs": sleeve_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _copy_latest(
        files={
            "metrics.csv": metrics_path,
            "sleeve_daily_returns.csv": sleeve_returns_path,
            "portfolio_daily_returns.csv": portfolio_returns_path,
            "manifest.json": manifest_path,
        },
        latest_root=base_output_dir / "latest",
    )

    table = results_df.copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD"]:
        table[col] = table[col].map(_format_float)
    print("")
    print(table.to_string(index=False))
    print("")
    print(f"Saved results: {run_dir}")


if __name__ == "__main__":
    main()
