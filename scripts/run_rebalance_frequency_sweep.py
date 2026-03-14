"""Rebalance-frequency sweep for the benchmark factor strategy."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover


FACTORS: list[str] = [
    "momentum_12_1",
    "reversal_1m",
    "low_vol_20",
    "gross_profitability",
]
FACTOR_WEIGHTS: list[float] = [0.25, 0.25, 0.25, 0.25]
REBALANCE_FREQUENCIES: list[str] = ["weekly", "biweekly", "monthly"]
UNIVERSE_CONFIGS: dict[str, dict[str, Any]] = {
    "sp500": {},
    "liquid_us": {"universe_mode": "dynamic"},
}
RESULT_COLUMNS: list[str] = [
    "universe",
    "rebalance_frequency",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "AnnualTurnover",
    "EligMedian",
    "TradableMedian",
    "RebalanceSkipped",
]

BASE_CONFIG: dict[str, Any] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
    "top_n": 50,
    "costs_bps": 10.0,
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "factor_name": FACTORS,
    "factor_names": FACTORS,
    "factor_weights": FACTOR_WEIGHTS,
    "save_artifacts": True,
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
}


def build_run_config(universe: str, rebalance_frequency: str) -> dict[str, Any]:
    if universe not in UNIVERSE_CONFIGS:
        raise ValueError(f"Unsupported universe: {universe}")
    cfg = dict(BASE_CONFIG)
    cfg.update(UNIVERSE_CONFIGS[universe])
    cfg["universe"] = str(universe)
    cfg["rebalance"] = str(rebalance_frequency)
    return cfg


def summary_to_result_row(
    summary: dict[str, Any],
    outdir: str,
    universe: str,
    rebalance_frequency: str,
) -> dict[str, Any]:
    return {
        "universe": str(universe),
        "rebalance_frequency": str(rebalance_frequency),
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "Vol": float(summary.get("Vol", float("nan"))),
        "Sharpe": float(summary.get("Sharpe", float("nan"))),
        "MaxDD": float(summary.get("MaxDD", float("nan"))),
        "AnnualTurnover": extract_annual_turnover(summary=summary, outdir=outdir),
        "EligMedian": float(summary.get("EligibleOnRebalanceMedian", float("nan"))),
        "TradableMedian": float(summary.get("FinalTradableOnRebalanceMedian", float("nan"))),
        "RebalanceSkipped": int(summary.get("RebalanceSkippedCount", 0)),
    }


def build_sharpe_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(index=pd.Index([], name="rebalance_frequency"))
    pivot = results_df.pivot(index="rebalance_frequency", columns="universe", values="Sharpe")
    pivot = pivot.reindex(REBALANCE_FREQUENCIES)
    pivot.columns.name = None
    return pivot.reset_index()


def run_sweep() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for universe in UNIVERSE_CONFIGS:
        for rebalance_frequency in REBALANCE_FREQUENCIES:
            cfg = build_run_config(universe=universe, rebalance_frequency=rebalance_frequency)
            print(f"Running universe={universe} rebalance_frequency={rebalance_frequency}")
            summary, outdir = run_backtest(**cfg, run_cache=run_cache)
            rows.append(
                summary_to_result_row(
                    summary=summary,
                    outdir=outdir,
                    universe=universe,
                    rebalance_frequency=rebalance_frequency,
                )
            )
            run_manifest.append(
                {
                    "universe": str(universe),
                    "rebalance_frequency": str(rebalance_frequency),
                    "backtest_outdir": str(outdir),
                    "summary_path": str(Path(outdir) / "summary.json"),
                    "run_config": cfg,
                }
            )

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    order = {name: i for i, name in enumerate(REBALANCE_FREQUENCIES)}
    df["_order"] = df["rebalance_frequency"].map(order)
    df = df.sort_values(["_order", "universe"], kind="mergesort").drop(columns="_order").reset_index(drop=True)
    return df, run_manifest


def write_outputs(
    results_df: pd.DataFrame,
    run_manifest: list[dict[str, Any]],
    results_root: Path = Path("results") / "rebalance_frequency_sweep",
) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df = build_sharpe_summary(results_df)

    results_path = outdir / "rebalance_frequency_results.csv"
    summary_path = outdir / "rebalance_frequency_summary.csv"
    manifest_path = outdir / "rebalance_frequency_manifest.json"
    latest_results_path = results_root / "rebalance_frequency_results_latest.csv"
    latest_summary_path = results_root / "rebalance_frequency_summary_latest.csv"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")
    results_df.to_csv(latest_results_path, index=False, float_format="%.10g")
    summary_df.to_csv(latest_summary_path, index=False, float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp,
                "results_dir": str(outdir),
                "base_config": BASE_CONFIG,
                "universes": list(UNIVERSE_CONFIGS.keys()),
                "rebalance_frequencies": REBALANCE_FREQUENCIES,
                "runs": run_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("REBALANCE FREQUENCY SWEEP")
    print("-------------------------")
    print(results_df.to_string(index=False))
    print("")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {latest_results_path}")
    print(f"Saved: {latest_summary_path}")
    print(f"Saved: {manifest_path}")
    return outdir


def main() -> None:
    results_df, run_manifest = run_sweep()
    write_outputs(results_df=results_df, run_manifest=run_manifest)


if __name__ == "__main__":
    main()
