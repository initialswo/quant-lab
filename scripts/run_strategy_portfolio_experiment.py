"""Run a simple multi-strategy sleeve portfolio experiment."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest


RESULTS_ROOT = Path("results") / "strategy_portfolio_experiment"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_TOP_N = 50
DEFAULT_COSTS_BPS = 10.0
DEFAULT_MAX_TICKERS = 2000
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
STRATEGY_SPECS: list[dict[str, Any]] = [
    {
        "strategy_name": "quality",
        "return_column": "quality_return",
        "factor_names": ["gross_profitability"],
        "factor_weights": [1.0],
        "allocation": 0.50,
    },
    {
        "strategy_name": "momentum",
        "return_column": "momentum_return",
        "factor_names": ["momentum_12_1"],
        "factor_weights": [1.0],
        "allocation": 0.30,
    },
    {
        "strategy_name": "reversal",
        "return_column": "reversal_return",
        "factor_names": ["reversal_1m"],
        "factor_weights": [1.0],
        "allocation": 0.20,
    },
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


def _run_config(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
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
        "factor_name": list(spec["factor_names"]),
        "factor_names": list(spec["factor_names"]),
        "factor_weights": list(spec["factor_weights"]),
        "portfolio_mode": "composite",
        "factor_aggregation_method": "linear",
        "use_factor_normalization": True,
        "use_sector_neutralization": False,
        "use_size_neutralization": False,
        "orthogonalize_factors": False,
        "fundamentals_path": str(args.fundamentals_path),
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
        return pd.to_numeric(equity["DailyReturn"], errors="coerce").fillna(0.0).rename(name)
    if "Equity" not in equity.columns:
        raise ValueError(f"equity.csv at {outdir} must include DailyReturn or Equity")
    daily_return = pd.to_numeric(equity["Equity"], errors="coerce").pct_change().fillna(0.0)
    return daily_return.rename(name)



def _strategy_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "annual_vol": float(summary.get("Vol", float("nan"))),
        "sharpe": float(summary.get("Sharpe", float("nan"))),
        "max_drawdown": float(summary.get("MaxDD", float("nan"))),
    }



def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"



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



def main() -> None:
    args = _parse_args()
    if int(args.top_n) <= 0:
        raise ValueError("--top_n must be > 0")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(str(args.output_dir)) if str(args.output_dir).strip() else RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    strategy_returns: list[pd.Series] = []
    strategy_manifest: list[dict[str, Any]] = []
    strategy_metric_rows: list[dict[str, Any]] = []

    print("")
    print("STRATEGY PORTFOLIO EXPERIMENT")
    print("-----------------------------")
    print("")
    print("Strategy Performance")
    for spec in STRATEGY_SPECS:
        cfg = _run_config(args=args, spec=spec)
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        daily_return = _load_daily_return(Path(run_outdir), str(spec["return_column"]))
        strategy_returns.append(daily_return)
        metrics = _strategy_metrics(summary)
        strategy_metric_rows.append(
            {
                "strategy_name": str(spec["strategy_name"]),
                "allocation": float(spec["allocation"]),
                **metrics,
            }
        )
        strategy_manifest.append(
            {
                "strategy_name": str(spec["strategy_name"]),
                "factor_names": list(spec["factor_names"]),
                "factor_weights": list(spec["factor_weights"]),
                "allocation": float(spec["allocation"]),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
                "run_config": _to_serializable(cfg),
            }
        )
        print(f"{str(spec['strategy_name']).title():12s} Sharpe={_format_float(metrics['sharpe'])}")

    returns_df = pd.concat(strategy_returns, axis=1).sort_index().fillna(0.0)
    returns_df.index.name = "date"
    corr_matrix = returns_df.corr()
    weights = pd.Series(
        {str(spec["return_column"]): float(spec["allocation"]) for spec in STRATEGY_SPECS},
        dtype=float,
    )
    portfolio_return = returns_df.mul(weights, axis=1).sum(axis=1).rename("portfolio_return")
    portfolio_equity = (1.0 + portfolio_return).cumprod().rename("portfolio_equity")
    portfolio_metrics = compute_metrics(portfolio_return)

    print("")
    print("Correlation Matrix")
    print(
        f"Quality vs Momentum  {_format_float(float(corr_matrix.loc['quality_return', 'momentum_return']))}"
    )
    print(
        f"Quality vs Reversal  {_format_float(float(corr_matrix.loc['quality_return', 'reversal_return']))}"
    )
    print(
        f"Momentum vs Reversal {_format_float(float(corr_matrix.loc['momentum_return', 'reversal_return']))}"
    )

    print("")
    print("Portfolio Allocation")
    print("Quality   50%")
    print("Momentum  30%")
    print("Reversal  20%")

    print("")
    print("Portfolio Performance")
    print(f"Sharpe={_format_float(float(portfolio_metrics.get('Sharpe', float('nan'))))}")
    print(f"CAGR={_format_float(float(portfolio_metrics.get('CAGR', float('nan'))))}")
    print(f"MaxDD={_format_float(float(portfolio_metrics.get('MaxDD', float('nan'))))}")

    strategy_returns_path = output_dir / "strategy_returns.csv"
    correlation_path = output_dir / "correlation_matrix.csv"
    portfolio_returns_path = output_dir / "portfolio_returns.csv"
    portfolio_equity_path = output_dir / "portfolio_equity.csv"
    portfolio_summary_path = output_dir / "portfolio_summary.csv"
    manifest_path = output_dir / "manifest.json"

    returns_df.reset_index().to_csv(strategy_returns_path, index=False, float_format="%.10g")
    corr_matrix.to_csv(correlation_path, float_format="%.10g")
    portfolio_return.to_frame().reset_index().to_csv(portfolio_returns_path, index=False, float_format="%.10g")
    portfolio_equity.to_frame().reset_index().to_csv(portfolio_equity_path, index=False, float_format="%.10g")
    pd.DataFrame(
        [
            {
                "CAGR": float(portfolio_metrics.get("CAGR", float("nan"))),
                "annual_vol": float(portfolio_metrics.get("Vol", float("nan"))),
                "sharpe": float(portfolio_metrics.get("Sharpe", float("nan"))),
                "max_drawdown": float(portfolio_metrics.get("MaxDD", float("nan"))),
            }
        ]
    ).to_csv(portfolio_summary_path, index=False, float_format="%.10g")

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_strategy_portfolio_experiment.py",
        "strategies_used": [
            {
                "strategy_name": str(spec["strategy_name"]),
                "factor_names": list(spec["factor_names"]),
                "allocation": float(spec["allocation"]),
            }
            for spec in STRATEGY_SPECS
        ],
        "weights": {
            "quality": 0.50,
            "momentum": 0.30,
            "reversal": 0.20,
        },
        "backtest_configuration": {
            "start": str(args.start),
            "end": str(args.end),
            "universe": str(args.universe),
            "universe_mode": "dynamic",
            "rebalance": str(args.rebalance),
            "top_n": int(args.top_n),
            "costs_bps": float(args.costs_bps),
            "fundamentals_path": str(args.fundamentals_path),
            "weighting": "equal",
        },
        "output_paths": {
            "strategy_returns": str(strategy_returns_path),
            "correlation_matrix": str(correlation_path),
            "portfolio_returns": str(portfolio_returns_path),
            "portfolio_equity": str(portfolio_equity_path),
            "portfolio_summary": str(portfolio_summary_path),
            "manifest": str(manifest_path),
        },
        "runtime_seconds": float(time.perf_counter() - t0),
        "strategy_runs": strategy_manifest,
        "strategy_performance": strategy_metric_rows,
        "portfolio_performance": {
            "CAGR": float(portfolio_metrics.get("CAGR", float("nan"))),
            "annual_vol": float(portfolio_metrics.get("Vol", float("nan"))),
            "sharpe": float(portfolio_metrics.get("Sharpe", float("nan"))),
            "max_drawdown": float(portfolio_metrics.get("MaxDD", float("nan"))),
        },
    }
    manifest_path.write_text(json.dumps(_to_serializable(manifest), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
