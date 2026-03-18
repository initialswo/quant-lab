#!/usr/bin/env python3
"""Run the post-fix dispersion-based exposure scaling test on the canonical baseline strategy."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.engine.runner import run_backtest


RESULTS_ROOT = Path("results") / "dispersion_exposure_scaling_test"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_TOP_N = 75
DEFAULT_MAX_TICKERS = 2000
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
DISPERSION_PATH = Path("results") / "momentum_dispersion_test" / "latest" / "weekly_dispersion_timeseries.csv"
FACTOR_NAMES = ["gross_profitability", "reversal_1m"]
FACTOR_WEIGHTS = [0.7, 0.3]
FACTOR_AGGREGATION_METHOD = "linear"
REGIME_ORDER = ["low", "medium", "high"]
EXPOSURE_BY_REGIME = {"low": 1.0, "medium": 1.0, "high": 0.5}
RETURN_COLUMNS = ["date", "next_date", "dispersion_regime", "exposure", "weekly_return"]
SUMMARY_COLUMNS = ["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "weeks"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--dispersion_path", default=str(DISPERSION_PATH))
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    return parser.parse_args()



def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return pd.read_csv(path)



def _read_indexed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()



def _normalize_regime(label: str | float | None) -> str | None:
    if label is None or pd.isna(label):
        return None
    text = str(label).strip().lower()
    if text.startswith("low"):
        return "low"
    if text.startswith("medium"):
        return "medium"
    if text.startswith("high"):
        return "high"
    return text or None



def _load_dispersion(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    expected = {"date", "next_date", "dispersion_regime"}
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(f"Dispersion file missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], utc=False)
    df["next_date"] = pd.to_datetime(df["next_date"], utc=False)
    df["dispersion_regime"] = df["dispersion_regime"].map(_normalize_regime)
    df = df.dropna(subset=["date", "next_date", "dispersion_regime"]).copy()
    df = df.loc[df["dispersion_regime"].isin(REGIME_ORDER)].copy()
    df = df.sort_values(["date"], kind="mergesort").reset_index(drop=True)
    return df.loc[:, ["date", "next_date", "dispersion_regime"]].copy()



def _run_baseline_backtest(start: str, end: str, fundamentals_path: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    cfg = {
        "start": str(start),
        "end": str(end),
        "universe": DEFAULT_UNIVERSE,
        "universe_mode": "dynamic",
        "top_n": int(DEFAULT_TOP_N),
        "rebalance": DEFAULT_REBALANCE,
        "weighting": "equal",
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
        "fundamentals_path": str(fundamentals_path),
        "save_artifacts": True,
    }
    summary, outdir = run_backtest(**cfg, run_cache={})
    return summary, outdir, cfg



def _build_aligned_weekly_returns(backtest_outdir: str | Path, dispersion: pd.DataFrame) -> pd.DataFrame:
    equity = _read_indexed_csv(Path(backtest_outdir) / "equity.csv")
    if "Equity" not in equity.columns:
        raise ValueError(f"Missing Equity column in {Path(backtest_outdir) / 'equity.csv'}")
    eq = pd.to_numeric(equity["Equity"], errors="coerce").dropna().sort_index()

    rows: list[dict[str, Any]] = []
    for _, row in dispersion.iterrows():
        dt = pd.Timestamp(row["date"])
        next_dt = pd.Timestamp(row["next_date"])
        if dt not in eq.index or next_dt not in eq.index:
            continue
        start_eq = float(eq.loc[dt])
        end_eq = float(eq.loc[next_dt])
        if not np.isfinite(start_eq) or not np.isfinite(end_eq) or start_eq <= 0.0:
            continue
        regime = str(row["dispersion_regime"])
        rows.append(
            {
                "date": dt,
                "next_date": next_dt,
                "dispersion_regime": regime,
                "exposure": 1.0,
                "weekly_return": float(end_eq / start_eq - 1.0),
            }
        )
    return pd.DataFrame(rows, columns=RETURN_COLUMNS)



def _apply_exposure_scaling(baseline_returns: pd.DataFrame) -> pd.DataFrame:
    scaled = baseline_returns.copy()
    scaled["exposure"] = scaled["dispersion_regime"].map(EXPOSURE_BY_REGIME).astype(float)
    scaled["weekly_return"] = scaled["exposure"].astype(float) * scaled["weekly_return"].astype(float)
    return scaled



def _compute_max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns.astype(float)).cumprod()
    peak = equity.cummax()
    drawdown = equity.div(peak).sub(1.0)
    return float(drawdown.min()) if not drawdown.empty else float("nan")



def _performance_row(strategy: str, returns_df: pd.DataFrame) -> dict[str, Any]:
    sample = returns_df["weekly_return"].astype(float).dropna()
    weeks = int(sample.shape[0])
    if weeks == 0:
        return {"Strategy": strategy, "CAGR": float("nan"), "Vol": float("nan"), "Sharpe": float("nan"), "MaxDD": float("nan"), "weeks": 0}
    cagr = float((1.0 + sample).prod() ** (52.0 / float(weeks)) - 1.0)
    vol = float(sample.std(ddof=1) * np.sqrt(52.0)) if weeks > 1 else float("nan")
    sharpe = float(sample.mean() / sample.std(ddof=1) * np.sqrt(52.0)) if weeks > 1 and sample.std(ddof=1) > 0 else float("nan")
    maxdd = _compute_max_drawdown(sample)
    return {"Strategy": strategy, "CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd, "weeks": weeks}



def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
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
    print("DISPERSION EXPOSURE SCALING SUMMARY")
    print("-----------------------------------")
    print(f"{'Strategy':22s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s}")
    for _, row in summary_df.iterrows():
        print(
            f"{str(row['Strategy']):22s} "
            f"{_format_float(float(row['CAGR'])):>8s} "
            f"{_format_float(float(row['Vol'])):>8s} "
            f"{_format_float(float(row['Sharpe'])):>8s} "
            f"{_format_float(float(row['MaxDD'])):>8s}"
        )



def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("POST-FIX DISPERSION EXPOSURE SCALING TEST")
    print("-----------------------------------------")
    print(
        "Config: "
        f"factors={','.join(FACTOR_NAMES)} factor_weights={FACTOR_WEIGHTS} top_n={DEFAULT_TOP_N} "
        f"rebalance={DEFAULT_REBALANCE} weighting=equal costs_bps={DEFAULT_COSTS_BPS} universe={DEFAULT_UNIVERSE}"
    )
    print(f"Using dispersion labels from: {args.dispersion_path}")
    print("Exposure rule: low=1.0 medium=1.0 high=0.5")

    t0 = time.perf_counter()
    dispersion = _load_dispersion(Path(str(args.dispersion_path)))
    backtest_summary, backtest_outdir, run_cfg = _run_baseline_backtest(
        start=str(args.start),
        end=str(args.end),
        fundamentals_path=str(args.fundamentals_path),
    )
    baseline_returns = _build_aligned_weekly_returns(backtest_outdir=backtest_outdir, dispersion=dispersion)
    scaled_returns = _apply_exposure_scaling(baseline_returns=baseline_returns)

    summary_df = pd.DataFrame(
        [
            _performance_row("baseline", baseline_returns),
            _performance_row("dispersion_scaled", scaled_returns),
        ],
        columns=SUMMARY_COLUMNS,
    )

    baseline_path = run_dir / "baseline_returns.csv"
    scaled_path = run_dir / "scaled_returns.csv"
    summary_path = run_dir / "performance_summary.csv"
    manifest_path = run_dir / "manifest.json"

    baseline_returns.to_csv(baseline_path, index=False, float_format="%.10g")
    scaled_returns.to_csv(scaled_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")

    baseline_sharpe = float(summary_df.loc[summary_df["Strategy"] == "baseline", "Sharpe"].iloc[0])
    scaled_sharpe = float(summary_df.loc[summary_df["Strategy"] == "dispersion_scaled", "Sharpe"].iloc[0])

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_dispersion_exposure_scaling_test.py",
        "dispersion_source": str(args.dispersion_path),
        "dispersion_label_mapping": {
            "low dispersion": "low",
            "medium dispersion": "medium",
            "high dispersion": "high",
        },
        "exposure_rule": dict(EXPOSURE_BY_REGIME),
        "start": str(args.start),
        "end": str(args.end),
        "baseline_strategy": {
            "factors": list(FACTOR_NAMES),
            "factor_weights": list(FACTOR_WEIGHTS),
            "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
            "top_n": int(DEFAULT_TOP_N),
            "rebalance": DEFAULT_REBALANCE,
            "weighting": "equal",
            "costs_bps": float(DEFAULT_COSTS_BPS),
            "universe": DEFAULT_UNIVERSE,
            "price_panel_for_signals": "adj_close via corrected run_backtest",
            "price_panel_for_pnl": "adj_close via corrected run_backtest",
            "eligibility": "liquid_us dynamic universe via active runtime",
            "fundamentals_path": str(args.fundamentals_path),
            "run_config": _to_serializable(run_cfg),
            "backtest_summary": _to_serializable(backtest_summary),
            "backtest_outdir": str(backtest_outdir),
        },
        "outputs": {
            "baseline_returns": str(baseline_path),
            "scaled_returns": str(scaled_path),
            "performance_summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "sharpe_difference_scaled_minus_baseline": float(scaled_sharpe - baseline_sharpe),
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    latest_dir = base_output_dir / "latest"
    _copy_latest(
        {
            baseline_path.name: baseline_path,
            scaled_path.name: scaled_path,
            summary_path.name: summary_path,
            manifest_path.name: manifest_path,
        },
        latest_dir,
    )

    _print_summary(summary_df)
    print("")
    print(f"Sharpe difference (scaled - baseline): {_format_float(scaled_sharpe - baseline_sharpe)}")
    print(f"Saved: {baseline_path}")
    print(f"Saved: {scaled_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_dir}")


if __name__ == "__main__":
    main()
