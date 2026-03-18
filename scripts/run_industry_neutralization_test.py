#!/usr/bin/env python3
"""Run the post-fix industry-neutralization test against the canonical baseline strategy."""

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


RESULTS_ROOT = Path("results") / "industry_neutralization_test"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_TOP_N = 75
DEFAULT_MAX_TICKERS = 2000
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
SECURITY_MASTER_PATH = Path("data/warehouse/security_master.parquet")
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
FACTOR_NAMES = ["gross_profitability", "reversal_1m"]
FACTOR_WEIGHTS = [0.7, 0.3]
FACTOR_AGGREGATION_METHOD = "linear"
STRATEGIES = [
    {
        "strategy_name": "baseline",
        "factor_neutralization": None,
        "notes": "Canonical corrected baseline.",
    },
    {
        "strategy_name": "industry_neutral",
        "factor_neutralization": "sector",
        "notes": "Composite-score neutralization using industry_ff48 buckets from warehouse security_master.",
    },
]
RESULT_COLUMNS = [
    "strategy_name",
    "factor_neutralization",
    "factors",
    "factor_weights",
    "factor_aggregation_method",
    "start",
    "end",
    "universe",
    "rebalance",
    "top_n",
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    parser.add_argument("--security-master-path", default=str(SECURITY_MASTER_PATH))
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



def _build_industry_map(security_master_path: Path, output_path: Path) -> dict[str, int]:
    sm = pd.read_parquet(security_master_path)
    required = {"canonical_symbol", "industry_ff48"}
    missing = required.difference(sm.columns)
    if missing:
        raise ValueError(f"security_master missing required columns: {sorted(missing)}")
    out = sm[["canonical_symbol", "industry_ff48"]].copy()
    out["Ticker"] = out["canonical_symbol"].astype(str).str.strip().str.upper()
    out["Sector"] = out["industry_ff48"].astype("string").fillna("UNKNOWN")
    out = out[["Ticker", "Sector"]].drop_duplicates(subset=["Ticker"], keep="last").sort_values("Ticker").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    value_counts = out["Sector"].value_counts(dropna=False)
    return {
        "rows": int(len(out)),
        "unknown_count": int(value_counts.get("UNKNOWN", 0)),
        "mapped_count": int(len(out) - int(value_counts.get("UNKNOWN", 0))),
    }



def _run_config(args: argparse.Namespace, strategy: dict[str, Any], sector_map_path: Path) -> dict[str, Any]:
    cfg = {
        "start": str(args.start),
        "end": str(args.end),
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
        "factor_weights": list(FACTOR_WEIGHTS),
        "portfolio_mode": "composite",
        "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
        "use_factor_normalization": True,
        "use_sector_neutralization": False,
        "use_size_neutralization": False,
        "orthogonalize_factors": False,
        "fundamentals_path": str(args.fundamentals_path),
        "save_artifacts": True,
    }
    if strategy.get("factor_neutralization") is not None:
        cfg["factor_neutralization"] = str(strategy["factor_neutralization"])
        cfg["sector_map"] = str(sector_map_path)
    return cfg



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
    print("INDUSTRY NEUTRALIZATION SUMMARY")
    print("-------------------------------")
    print(f"{'Strategy':18s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Turnover':>10s}")
    for _, row in summary_df.iterrows():
        print(
            f"{str(row['strategy_name']):18s} "
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

    industry_map_path = run_dir / "industry_ff48_sector_map.csv"
    industry_map_stats = _build_industry_map(Path(args.security_master_path), industry_map_path)

    print("POST-FIX INDUSTRY NEUTRALIZATION TEST")
    print("------------------------------------")
    print(
        "Config: "
        f"universe={DEFAULT_UNIVERSE} rebalance={DEFAULT_REBALANCE} costs_bps={DEFAULT_COSTS_BPS} top_n={DEFAULT_TOP_N} "
        f"factors={','.join(FACTOR_NAMES)} factor_weights={FACTOR_WEIGHTS} factor_aggregation_method={FACTOR_AGGREGATION_METHOD}"
    )
    print(
        "Note: industry_neutral uses the runner composite-score neutralization path with industry_ff48 buckets "
        f"from {args.security_master_path}."
    )

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for strategy in STRATEGIES:
        cfg = _run_config(args=args, strategy=strategy, sector_map_path=industry_map_path)
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        stats = _artifact_stats(
            outdir=run_outdir,
            rebalance=DEFAULT_REBALANCE,
            skipped_count=int(summary.get("RebalanceSkippedCount", 0)),
        )
        rows.append(
            {
                "strategy_name": str(strategy["strategy_name"]),
                "factor_neutralization": "none" if strategy.get("factor_neutralization") is None else str(strategy["factor_neutralization"]),
                "factors": ",".join(FACTOR_NAMES),
                "factor_weights": ",".join(str(x) for x in FACTOR_WEIGHTS),
                "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
                "start": str(args.start),
                "end": str(args.end),
                "universe": DEFAULT_UNIVERSE,
                "rebalance": DEFAULT_REBALANCE,
                "top_n": int(DEFAULT_TOP_N),
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "Turnover": extract_annual_turnover(summary=summary, outdir=run_outdir),
                "hit_rate": float(stats["hit_rate"]),
                "n_rebalance_dates": int(stats["n_rebalance_dates"]),
                "median_selected_names": float(stats["median_selected_names"]),
            }
        )
        run_manifest.append(
            {
                "strategy_name": str(strategy["strategy_name"]),
                "notes": str(strategy["notes"]),
                "run_config": _to_serializable(cfg),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(
        by=["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort"
    ).reset_index(drop=True)
    summary_df = results_df.loc[:, SUMMARY_COLUMNS].copy()

    results_path = run_dir / "industry_neutralization_results.csv"
    summary_path = run_dir / "industry_neutralization_summary.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    baseline_sharpe = float(results_df.loc[results_df["strategy_name"] == "baseline", "Sharpe"].iloc[0])
    neutral_sharpe = float(results_df.loc[results_df["strategy_name"] == "industry_neutral", "Sharpe"].iloc[0])
    sharpe_pct_change = float(((neutral_sharpe / baseline_sharpe) - 1.0) * 100.0) if baseline_sharpe != 0 else float("nan")

    best_row = summary_df.iloc[0]
    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_industry_neutralization_test.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "rebalance": DEFAULT_REBALANCE,
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "top_n": int(DEFAULT_TOP_N),
        "factors": list(FACTOR_NAMES),
        "factor_weights": list(FACTOR_WEIGHTS),
        "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
        "fundamentals_path": str(args.fundamentals_path),
        "security_master_path": str(args.security_master_path),
        "industry_map_path": str(industry_map_path),
        "industry_map_stats": industry_map_stats,
        "output_dir": str(run_dir),
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "industry_map": str(industry_map_path),
        },
        "baseline_sharpe": baseline_sharpe,
        "industry_neutral_sharpe": neutral_sharpe,
        "sharpe_pct_change": sharpe_pct_change,
        "best_configuration": {
            "strategy_name": str(best_row["strategy_name"]),
            "Sharpe": float(best_row["Sharpe"]),
            "CAGR": float(best_row["CAGR"]),
            "Vol": float(best_row["Vol"]),
            "MaxDD": float(best_row["MaxDD"]),
            "Turnover": float(best_row["Turnover"]),
        },
        "runtime_seconds": float(time.perf_counter() - t0),
        "notes": [
            "This is the post-fix industry-neutralization test against the canonical corrected baseline.",
            "The industry-neutral leg uses the existing runner composite-score neutralization path with industry_ff48 treated as the sector bucket.",
            "Unmapped industry_ff48 names are assigned to UNKNOWN in the temporary map so they remain tradable.",
        ],
        "runs": _to_serializable(run_manifest),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    latest_root = base_output_dir / "latest"
    _copy_latest(
        files={
            results_path.name: results_path,
            summary_path.name: summary_path,
            manifest_path.name: manifest_path,
            industry_map_path.name: industry_map_path,
        },
        latest_root=latest_root,
    )

    _print_summary(summary_df)
    print("")
    print(f"Baseline Sharpe: {_format_float(baseline_sharpe)}")
    print(f"Industry-neutral Sharpe: {_format_float(neutral_sharpe)}")
    print(f"Sharpe change: {_format_float(sharpe_pct_change)}%")
    if pd.notna(sharpe_pct_change) and sharpe_pct_change > 0.0:
        print("Interpretation: neutralization improves performance, which suggests some baseline alpha may come from industry tilts rather than only stock selection.")
    elif pd.notna(sharpe_pct_change) and sharpe_pct_change < 0.0:
        print("Interpretation: neutralization hurts performance, which suggests part of the baseline alpha is tied to industry exposure rather than being purely stock-specific.")
    else:
        print("Interpretation: neutralization leaves Sharpe unchanged, which suggests the baseline alpha is largely stock-specific rather than industry-driven.")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_root}")


if __name__ == "__main__":
    main()
