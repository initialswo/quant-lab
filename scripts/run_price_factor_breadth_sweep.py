"""Portfolio breadth sweep for standalone price-factor sleeves at small Top-N values."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover
from quant_lab.research.windows import get_price_window, resolve_window


RESULTS_ROOT = Path("results") / "price_factor_breadth_sweep"
DEFAULT_WINDOW = get_price_window()
START = str(DEFAULT_WINDOW["start"])
END = str(DEFAULT_WINDOW["end"])
UNIVERSE = "liquid_us"
UNIVERSE_MODE = "dynamic"
REBALANCE = "weekly"
WEIGHTING = "equal"
COSTS_BPS = 10.0
MAX_TICKERS = 300
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
FUNDAMENTALS_FALLBACK_LAG_DAYS = 60
FACTORS: list[str] = [
    "momentum_12_1",
    "reversal_1m",
    "low_vol_20",
]
TOP_N_VALUES: list[int] = [5, 10, 15, 20, 30]
RESULT_COLUMNS: list[str] = [
    "Strategy",
    "top_n",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "Turnover",
]


def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)


def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def _base_config() -> dict[str, Any]:
    return {
        "start": START,
        "end": END,
        "universe": UNIVERSE,
        "universe_mode": UNIVERSE_MODE,
        "rebalance": REBALANCE,
        "weighting": WEIGHTING,
        "costs_bps": COSTS_BPS,
        "max_tickers": MAX_TICKERS,
        "data_source": DATA_SOURCE,
        "data_cache_dir": DATA_CACHE_DIR,
        "fundamentals_path": FUNDAMENTALS_PATH,
        "fundamentals_fallback_lag_days": FUNDAMENTALS_FALLBACK_LAG_DAYS,
        "save_artifacts": True,
    }


def _run_config(factor_name: str, top_n: int) -> dict[str, Any]:
    cfg = dict(_base_config())
    cfg["factor_name"] = [str(factor_name)]
    cfg["factor_names"] = [str(factor_name)]
    cfg["factor_weights"] = [1.0]
    cfg["top_n"] = int(top_n)
    return cfg


def _result_row(summary: dict[str, Any], outdir: str, factor_name: str, top_n: int) -> dict[str, Any]:
    return {
        "Strategy": str(factor_name),
        "top_n": int(top_n),
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "Vol": float(summary.get("Vol", float("nan"))),
        "Sharpe": float(summary.get("Sharpe", float("nan"))),
        "MaxDD": float(summary.get("MaxDD", float("nan"))),
        "Turnover": extract_annual_turnover(summary=summary, outdir=outdir),
    }


def _build_manifest(outdir: Path, runs: list[dict[str, Any]], max_tickers: int) -> dict[str, Any]:
    return {
        "timestamp_utc": outdir.name,
        "results_dir": str(outdir),
        "date_range": {"start": START, "end": END},
        "universe": UNIVERSE,
        "universe_mode": UNIVERSE_MODE,
        "rebalance": REBALANCE,
        "weighting": WEIGHTING,
        "costs_bps": COSTS_BPS,
        "max_tickers": max_tickers,
        "factors": FACTORS,
        "top_n_values": TOP_N_VALUES,
        "notes": [
            "Breadth sweep focuses on standalone price-factor sleeves.",
            "Each run uses the standard liquid_us dynamic universe and weekly equal-weight Top-N construction.",
        ],
        "runs": runs,
    }


def run_sweep() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for factor_name in FACTORS:
        for top_n in TOP_N_VALUES:
            cfg = _run_config(factor_name=factor_name, top_n=top_n)
            print(f"Running factor={factor_name} top_n={top_n}")
            summary, outdir = run_backtest(**cfg, run_cache=run_cache)
            rows.append(_result_row(summary=summary, outdir=outdir, factor_name=factor_name, top_n=top_n))
            run_manifest.append(
                {
                    "factor_name": str(factor_name),
                    "top_n": int(top_n),
                    "backtest_outdir": str(outdir),
                    "summary_path": str(Path(outdir) / "summary.json"),
                    "run_config": cfg,
                }
            )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_df = results_df.sort_values(["Strategy", "top_n"], kind="mergesort").reset_index(drop=True)
    return results_df, run_manifest


def write_outputs(
    results_df: pd.DataFrame,
    run_manifest: list[dict[str, Any]],
    results_root: Path,
) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / ts
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df = results_df.sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(
        drop=True
    )

    results_path = outdir / "price_factor_breadth_results.csv"
    summary_path = outdir / "price_factor_breadth_summary.csv"
    manifest_path = outdir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")
    manifest_path.write_text(
        json.dumps(_build_manifest(outdir=outdir, runs=run_manifest, max_tickers=MAX_TICKERS), indent=2),
        encoding="utf-8",
    )
    _copy_latest(
        files={
            "price_factor_breadth_results.csv": results_path,
            "price_factor_breadth_summary.csv": summary_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    display = summary_df.copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]:
        display[col] = display[col].map(_format_float)
    print("")
    print("PRICE FACTOR BREADTH SWEEP")
    print("--------------------------")
    print(display.to_string(index=False))
    print("")
    print(f"Saved results: {outdir}")
    return outdir


def main() -> None:
    global START, END, MAX_TICKERS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--max_tickers", type=int, default=MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    START, END = resolve_window(
        default_window=DEFAULT_WINDOW,
        start=args.start,
        end=args.end,
    )
    MAX_TICKERS = int(args.max_tickers)

    results_df, run_manifest = run_sweep()
    write_outputs(
        results_df=results_df,
        run_manifest=run_manifest,
        results_root=Path(str(args.results_root)),
    )


if __name__ == "__main__":
    main()
