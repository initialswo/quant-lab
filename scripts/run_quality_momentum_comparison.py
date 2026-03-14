"""Quality-momentum comparison study versus the benchmark 4-factor stack."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover


STRATEGIES: list[dict[str, Any]] = [
    {
        "strategy_name": "quality_momentum",
        "factors": ["gross_profitability", "momentum_12_1"],
    },
    {
        "strategy_name": "quality_momentum_pullback",
        "factors": ["gross_profitability", "momentum_12_1", "reversal_1m"],
    },
    {
        "strategy_name": "benchmark_4factor",
        "factors": ["gross_profitability", "momentum_12_1", "reversal_1m", "low_vol_20"],
    },
]
UNIVERSE_CONFIGS: dict[str, dict[str, Any]] = {
    "sp500": {},
    "liquid_us": {"universe_mode": "dynamic"},
}
RESULT_COLUMNS: list[str] = [
    "universe",
    "strategy_name",
    "factor_set",
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
    "rebalance": "weekly",
    "costs_bps": 10.0,
    "max_tickers": 2000,
    "weighting": "equal",
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "save_artifacts": True,
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
}


def format_factor_set(factors: list[str]) -> str:
    return ",".join(str(x) for x in factors)


def equal_weights(factors: list[str]) -> list[float]:
    if not factors:
        raise ValueError("factor set must be non-empty")
    return [1.0 / float(len(factors))] * len(factors)


def build_run_config(universe: str, factors: list[str]) -> dict[str, Any]:
    if universe not in UNIVERSE_CONFIGS:
        raise ValueError(f"Unsupported universe: {universe}")
    cfg = dict(BASE_CONFIG)
    cfg.update(UNIVERSE_CONFIGS[universe])
    cfg["universe"] = str(universe)
    cfg["factor_name"] = list(factors)
    cfg["factor_names"] = list(factors)
    cfg["factor_weights"] = equal_weights(factors)
    return cfg


def summary_to_result_row(
    summary: dict[str, Any],
    outdir: str,
    universe: str,
    strategy_name: str,
    factors: list[str],
) -> dict[str, Any]:
    return {
        "universe": str(universe),
        "strategy_name": str(strategy_name),
        "factor_set": format_factor_set(factors),
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
        return pd.DataFrame(index=pd.Index([], name="strategy_name"))
    pivot = results_df.pivot(index="strategy_name", columns="universe", values="Sharpe")
    pivot = pivot.reindex([str(s["strategy_name"]) for s in STRATEGIES])
    pivot.columns.name = None
    return pivot.reset_index()


def run_sweep() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for universe in UNIVERSE_CONFIGS:
        for strategy in STRATEGIES:
            factors = list(strategy["factors"])
            strategy_name = str(strategy["strategy_name"])
            cfg = build_run_config(universe=universe, factors=factors)
            print(f"Running universe={universe} strategy={strategy_name}")
            summary, outdir = run_backtest(**cfg, run_cache=run_cache)
            rows.append(
                summary_to_result_row(
                    summary=summary,
                    outdir=outdir,
                    universe=universe,
                    strategy_name=strategy_name,
                    factors=factors,
                )
            )
            run_manifest.append(
                {
                    "universe": str(universe),
                    "strategy_name": strategy_name,
                    "factor_set": format_factor_set(factors),
                    "backtest_outdir": str(outdir),
                    "summary_path": str(Path(outdir) / "summary.json"),
                    "run_config": cfg,
                }
            )

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    order = {str(s["strategy_name"]): i for i, s in enumerate(STRATEGIES)}
    df["_order"] = df["strategy_name"].map(order)
    df = df.sort_values(["_order", "universe"], kind="mergesort").drop(columns="_order").reset_index(drop=True)
    return df, run_manifest


def write_outputs(
    results_df: pd.DataFrame,
    run_manifest: list[dict[str, Any]],
    runtime_seconds: float,
    results_root: Path = Path("results") / "quality_momentum_comparison",
) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df = build_sharpe_summary(results_df)

    results_path = outdir / "quality_momentum_results.csv"
    summary_path = outdir / "quality_momentum_summary.csv"
    manifest_path = outdir / "quality_momentum_manifest.json"
    latest_results_path = results_root / "quality_momentum_results_latest.csv"
    latest_summary_path = results_root / "quality_momentum_summary_latest.csv"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")
    results_df.to_csv(latest_results_path, index=False, float_format="%.10g")
    summary_df.to_csv(latest_summary_path, index=False, float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp,
                "results_dir": str(outdir),
                "runtime_seconds": float(runtime_seconds),
                "base_config": BASE_CONFIG,
                "universes": list(UNIVERSE_CONFIGS.keys()),
                "strategies": STRATEGIES,
                "runs": run_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("QUALITY MOMENTUM COMPARISON")
    print("---------------------------")
    print(results_df.to_string(index=False))
    print("")
    print("SHARPE PIVOT")
    print("------------")
    print(summary_df.to_string(index=False))
    print("")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {latest_results_path}")
    print(f"Saved: {latest_summary_path}")
    print(f"Saved: {manifest_path}")
    return outdir


def main() -> None:
    t0 = time.perf_counter()
    results_df, run_manifest = run_sweep()
    runtime_seconds = time.perf_counter() - t0
    write_outputs(results_df=results_df, run_manifest=run_manifest, runtime_seconds=runtime_seconds)


if __name__ == "__main__":
    main()
