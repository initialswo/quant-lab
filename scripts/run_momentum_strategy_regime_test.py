#!/usr/bin/env python3
"""Run the post-fix pure momentum strategy across saved momentum-dispersion regimes."""

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


RESULTS_ROOT = Path("results") / "momentum_strategy_regime_test"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_TOP_N = 75
DEFAULT_MAX_TICKERS = 2000
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
DISPERSION_PATH = Path("results") / "momentum_dispersion_test" / "latest" / "weekly_dispersion_timeseries.csv"
FACTOR_NAME = "momentum_12_1"
REGIME_ORDER = ["low", "medium", "high"]
DETAIL_COLUMNS = ["date", "next_date", "dispersion_regime", "momentum_weekly_return"]
SUMMARY_COLUMNS = ["Regime", "Weeks", "Annualized Return", "Vol", "Sharpe", "mean_weekly_return"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--dispersion_path", default=str(DISPERSION_PATH))
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



def _run_momentum_backtest(start: str, end: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
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
        "factor_name": FACTOR_NAME,
        "factor_names": [FACTOR_NAME],
        "factor_weights": [1.0],
        "portfolio_mode": "composite",
        "factor_aggregation_method": "linear",
        "use_factor_normalization": True,
        "use_sector_neutralization": False,
        "use_size_neutralization": False,
        "orthogonalize_factors": False,
        "save_artifacts": True,
    }
    summary, outdir = run_backtest(**cfg, run_cache={})
    return summary, outdir, cfg



def _build_weekly_returns(backtest_outdir: str | Path, dispersion: pd.DataFrame) -> pd.DataFrame:
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
        rows.append(
            {
                "date": dt,
                "next_date": next_dt,
                "dispersion_regime": str(row["dispersion_regime"]),
                "momentum_weekly_return": float(end_eq / start_eq - 1.0),
            }
        )
    return pd.DataFrame(rows, columns=DETAIL_COLUMNS)



def _summarize_by_regime(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        sample = detail.loc[detail["dispersion_regime"] == regime, "momentum_weekly_return"].astype(float)
        if sample.empty:
            rows.append(
                {
                    "Regime": regime,
                    "Weeks": 0,
                    "Annualized Return": float("nan"),
                    "Vol": float("nan"),
                    "Sharpe": float("nan"),
                    "mean_weekly_return": float("nan"),
                }
            )
            continue
        ann_return = float((1.0 + sample).prod() ** (52.0 / float(sample.shape[0])) - 1.0)
        ann_vol = float(sample.std(ddof=1) * np.sqrt(52.0)) if sample.shape[0] > 1 else float("nan")
        sharpe = float(sample.mean() / sample.std(ddof=1) * np.sqrt(52.0)) if sample.shape[0] > 1 and sample.std(ddof=1) > 0 else float("nan")
        rows.append(
            {
                "Regime": regime,
                "Weeks": int(sample.shape[0]),
                "Annualized Return": ann_return,
                "Vol": ann_vol,
                "Sharpe": sharpe,
                "mean_weekly_return": float(sample.mean()),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)



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
    print("PURE MOMENTUM BY DISPERSION REGIME")
    print("----------------------------------")
    print(f"{'Regime':10s} {'Weeks':>7s} {'AnnRet':>10s} {'Vol':>10s} {'Sharpe':>8s}")
    for _, row in summary_df.iterrows():
        print(
            f"{str(row['Regime']):10s} "
            f"{int(row['Weeks']):7d} "
            f"{_format_float(float(row['Annualized Return'])):>10s} "
            f"{_format_float(float(row['Vol'])):>10s} "
            f"{_format_float(float(row['Sharpe'])):>8s}"
        )



def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("POST-FIX MOMENTUM STRATEGY REGIME TEST")
    print("--------------------------------------")
    print(
        "Config: "
        f"factor={FACTOR_NAME} top_n={DEFAULT_TOP_N} rebalance={DEFAULT_REBALANCE} "
        f"weighting=equal costs_bps={DEFAULT_COSTS_BPS} universe={DEFAULT_UNIVERSE}"
    )
    print(f"Using dispersion labels from: {args.dispersion_path}")

    t0 = time.perf_counter()
    dispersion = _load_dispersion(Path(str(args.dispersion_path)))
    backtest_summary, backtest_outdir, run_cfg = _run_momentum_backtest(start=str(args.start), end=str(args.end))
    detail_df = _build_weekly_returns(backtest_outdir=backtest_outdir, dispersion=dispersion)
    summary_df = _summarize_by_regime(detail_df)

    detail_path = run_dir / "momentum_regime_performance.csv"
    summary_path = run_dir / "regime_summary.csv"
    manifest_path = run_dir / "manifest.json"

    detail_df.to_csv(detail_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")

    best_row = summary_df.sort_values(["Sharpe", "Annualized Return"], ascending=[False, False], kind="mergesort").iloc[0]
    high_row = summary_df.loc[summary_df["Regime"] == "high"]
    high_sharpe = float(high_row["Sharpe"].iloc[0]) if not high_row.empty else float("nan")
    best_regime = str(best_row["Regime"])

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_momentum_strategy_regime_test.py",
        "dispersion_source": str(args.dispersion_path),
        "dispersion_label_mapping": {
            "low dispersion": "low",
            "medium dispersion": "medium",
            "high dispersion": "high",
        },
        "start": str(args.start),
        "end": str(args.end),
        "strategy": {
            "factor": FACTOR_NAME,
            "top_n": int(DEFAULT_TOP_N),
            "rebalance": DEFAULT_REBALANCE,
            "weighting": "equal",
            "costs_bps": float(DEFAULT_COSTS_BPS),
            "universe": DEFAULT_UNIVERSE,
            "price_panel_for_signals": "adj_close via corrected run_backtest",
            "price_panel_for_pnl": "adj_close via corrected run_backtest",
            "eligibility": "liquid_us dynamic universe via active runtime",
            "run_config": _to_serializable(run_cfg),
            "backtest_summary": _to_serializable(backtest_summary),
            "backtest_outdir": str(backtest_outdir),
        },
        "outputs": {
            "momentum_regime_performance": str(detail_path),
            "regime_summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "best_regime_by_sharpe": best_regime,
        "high_dispersion_sharpe": high_sharpe,
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    latest_dir = base_output_dir / "latest"
    _copy_latest(
        {
            detail_path.name: detail_path,
            summary_path.name: summary_path,
            manifest_path.name: manifest_path,
        },
        latest_dir,
    )

    _print_summary(summary_df)
    if best_regime == "high":
        note = "Pure momentum performs best in high-dispersion environments in this run."
    else:
        note = f"Pure momentum does not perform best in high-dispersion environments here; the best regime is {best_regime}."
    print("")
    print(note)
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_dir}")


if __name__ == "__main__":
    main()
