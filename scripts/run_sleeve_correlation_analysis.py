#!/usr/bin/env python3
"""Analyze diversification relationships between quality, momentum, and low-vol sleeves."""

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


RESULTS_ROOT = Path("results") / "sleeve_correlation_analysis"
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
    {
        "strategy_name": "low_vol_only",
        "factor_names": ["low_vol_20"],
        "factor_weights": [1.0],
    },
]


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



def _format_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.map(_format_float))



def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("SLEEVE CORRELATION ANALYSIS")
    print("--------------------------")
    print(
        "Config: "
        f"universe={DEFAULT_UNIVERSE} universe_mode={DEFAULT_UNIVERSE_MODE} "
        f"rebalance={DEFAULT_REBALANCE} top_n={DEFAULT_TOP_N} weighting={DEFAULT_WEIGHTING} "
        f"costs_bps={DEFAULT_COSTS_BPS:.1f} factor_aggregation_method={FACTOR_AGGREGATION_METHOD}"
    )

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    sleeve_returns: list[pd.Series] = []
    run_manifest: list[dict[str, Any]] = []

    for spec in SLEEVE_SPECS:
        strategy_name = str(spec["strategy_name"])
        cfg = _run_config(args=args, spec=spec)
        print(f"Running {strategy_name}...")
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        daily_return = _load_daily_return(Path(run_outdir), strategy_name)
        sleeve_returns.append(daily_return)
        run_manifest.append(
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

    daily_returns_df = pd.concat(sleeve_returns, axis=1).sort_index().fillna(0.0)
    daily_returns_df.index.name = "date"
    correlation_df = daily_returns_df.corr()
    covariance_df = daily_returns_df.cov()

    daily_returns_path = run_dir / "daily_returns.csv"
    correlation_path = run_dir / "correlation_matrix.csv"
    covariance_path = run_dir / "covariance_matrix.csv"
    manifest_path = run_dir / "manifest.json"

    daily_returns_df.to_csv(daily_returns_path, float_format="%.10g")
    correlation_df.to_csv(correlation_path, float_format="%.10g")
    covariance_df.to_csv(covariance_path, float_format="%.10g")
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
                    {
                        "strategy_name": "low_vol_only",
                        "factor_names": ["low_vol_20"],
                        "factor_weights": [1.0],
                    },
                ],
                "outputs": {
                    "daily_returns": str(daily_returns_path),
                    "correlation_matrix": str(correlation_path),
                    "covariance_matrix": str(covariance_path),
                    "manifest": str(manifest_path),
                },
                "notes": [
                    "Goal: understand diversification relationships between sleeves.",
                    "Matrices are computed from aligned daily sleeve return series after standalone backtests.",
                    "Engine logic is unchanged; this experiment is implemented entirely at the script layer.",
                ],
                "runs": run_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _copy_latest(
        files={
            "daily_returns.csv": daily_returns_path,
            "correlation_matrix.csv": correlation_path,
            "covariance_matrix.csv": covariance_path,
            "manifest.json": manifest_path,
        },
        latest_root=base_output_dir / "latest",
    )

    print("")
    print("CORRELATION MATRIX")
    print("------------------")
    print(_format_df(correlation_df).to_string())
    print("")
    print(f"Saved: {correlation_path}")
    print(f"Saved: {covariance_path}")
    print(f"Saved: {daily_returns_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {base_output_dir / 'latest'}")


if __name__ == "__main__":
    main()
