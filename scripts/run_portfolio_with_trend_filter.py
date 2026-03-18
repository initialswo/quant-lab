#!/usr/bin/env python3
"""Run a 90/10 quality-momentum portfolio with a post-backtest SPY trend filter overlay."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest


RESULTS_ROOT = Path("results") / "portfolio_trend_filter"
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
SMA_WINDOW = 200
PORTFOLIO_WEIGHTS = {"quality_only": 0.90, "momentum_only": 0.10}
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


def _load_spy_close(start: str, end: str) -> tuple[pd.Series, dict[str, Any]]:
    ohlcv_map, data_summary = fetch_ohlcv_with_summary(
        tickers=["SPY"],
        start=str(start),
        end=str(end),
        cache_dir=DATA_CACHE_DIR,
        data_source=DATA_SOURCE,
        refresh=False,
        bulk_prepare=False,
    )
    spy = ohlcv_map.get("SPY")
    if spy is None and ohlcv_map:
        spy = next(iter(ohlcv_map.values()))
    if spy is None or spy.empty:
        raise ValueError("Unable to load SPY price history for the trend filter overlay.")
    field = "Adj Close" if "Adj Close" in spy.columns else "Close"
    close = pd.to_numeric(spy[field], errors="coerce").dropna().astype(float)
    if close.empty:
        raise ValueError("SPY price history is empty after coercion.")
    return close.rename("SPY_Close"), data_summary


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
    latest_dir = base_output_dir / "latest"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("PORTFOLIO TREND FILTER")
    print("----------------------")
    print(
        "Base portfolio: quality 0.90 / momentum 0.10 with "
        "quality sleeve = gross_profitability 0.70, reversal_1m 0.05, reversal_5d 0.25 and momentum sleeve = momentum_12_1 1.00"
    )
    print(
        "Shared config: "
        f"universe={DEFAULT_UNIVERSE} universe_mode={DEFAULT_UNIVERSE_MODE} "
        f"rebalance={DEFAULT_REBALANCE} top_n={DEFAULT_TOP_N} weighting={DEFAULT_WEIGHTING} "
        f"costs_bps={DEFAULT_COSTS_BPS:.1f} factor_aggregation_method={FACTOR_AGGREGATION_METHOD}"
    )
    print(
        "Overlay: compute SPY 200-day moving average and apply a causal gate to daily portfolio returns; "
        "when SPY close is not above its 200-day SMA, portfolio return is set to 0.0."
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

    sleeve_returns_df = pd.concat(sleeve_returns, axis=1).sort_index().fillna(0.0)
    original_return = (
        sleeve_returns_df["quality_only"] * float(PORTFOLIO_WEIGHTS["quality_only"])
        + sleeve_returns_df["momentum_only"] * float(PORTFOLIO_WEIGHTS["momentum_only"])
    ).rename("original_portfolio")

    spy_close_raw, spy_data_summary = _load_spy_close(start=str(args.start), end=str(args.end))
    spy_close = spy_close_raw.reindex(original_return.index).ffill().rename("SPY_Close")
    spy_sma_200 = spy_close.rolling(SMA_WINDOW).mean().rename("SPY_SMA_200")
    risk_on = (spy_close > spy_sma_200).shift(1).reindex(original_return.index).fillna(False).astype(bool).rename("RiskOn")
    filtered_return = original_return.where(risk_on, 0.0).rename("filtered_portfolio")

    results_df = pd.DataFrame(
        [
            _metrics_row(strategy_name="original_portfolio", daily_return=original_return),
            _metrics_row(strategy_name="filtered_portfolio", daily_return=filtered_return),
        ],
        columns=RESULT_COLUMNS,
    )
    daily_returns_df = pd.concat(
        [sleeve_returns_df, original_return, filtered_return, spy_close, spy_sma_200, risk_on.astype(int)],
        axis=1,
    ).sort_index()

    summary_path = run_dir / "portfolio_trend_filter_summary.csv"
    daily_returns_path = run_dir / "portfolio_trend_filter_daily_returns.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(summary_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index_label="date", float_format="%.10g")
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
                "base_portfolio": {
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
                    "weights": dict(PORTFOLIO_WEIGHTS),
                    "combination_method": "Combine sleeve daily returns after individual backtests; no shared capital.",
                },
                "trend_filter": {
                    "benchmark": "SPY",
                    "price_field": str(spy_close_raw.name),
                    "sma_window": int(SMA_WINDOW),
                    "rule": "risk_on[t] = SPY_close[t] > SMA200[t]; apply risk_on.shift(1) to portfolio daily returns",
                    "active_days": int(risk_on.sum()),
                    "inactive_days": int((~risk_on).sum()),
                },
                "spy_data_summary": _to_serializable(spy_data_summary),
                "notes": [
                    "The base portfolio is unchanged from the requested 90/10 quality-momentum blend.",
                    "The trend overlay is applied post-backtest on the combined portfolio daily return series.",
                    "The SPY filter uses a lagged daily signal to remain causal and avoid lookahead bias.",
                    "Engine logic is unchanged; this experiment is implemented entirely at the script layer.",
                ],
                "runs": run_manifest,
                "outputs": {
                    "summary": str(summary_path),
                    "daily_returns": str(daily_returns_path),
                    "manifest": str(manifest_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _copy_latest(
        files={
            summary_path.name: summary_path,
            daily_returns_path.name: daily_returns_path,
            manifest_path.name: manifest_path,
        },
        latest_root=latest_dir,
    )

    print("")
    print(
        results_df.to_string(
            index=False,
            formatters={col: _format_float for col in ["CAGR", "Vol", "Sharpe", "MaxDD"]},
        )
    )
    print("")
    print(f"Saved: {summary_path}")
    print(f"Saved: {daily_returns_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_dir}")


if __name__ == "__main__":
    main()
