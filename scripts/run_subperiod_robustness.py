"""Subperiod robustness sweep for the benchmark factor strategy."""

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
WEIGHT_SETS: list[list[float]] = [
    [0.25, 0.25, 0.25, 0.25],
    [0.30, 0.20, 0.10, 0.40],
    [0.20, 0.30, 0.10, 0.40],
]
SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2010-12-31"),
    ("2010-01-01", "2015-12-31"),
    ("2015-01-01", "2020-12-31"),
    ("2020-01-01", "2024-12-31"),
]
UNIVERSE_CONFIGS: dict[str, dict[str, Any]] = {
    "sp500": {},
    "liquid_us": {"universe_mode": "dynamic"},
}
RESULT_COLUMNS: list[str] = [
    "universe",
    "period",
    "weight_set",
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
    "top_n": 50,
    "rebalance": "weekly",
    "costs_bps": 10.0,
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "factor_name": FACTORS,
    "factor_names": FACTORS,
    "save_artifacts": True,
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
}


def format_weight_set(weights: list[float]) -> str:
    return ",".join(f"{float(x):.2f}" for x in weights)


def format_period(start: str, end: str) -> str:
    return f"{start}→{end}"


def build_run_config(universe: str, factor_weights: list[float], start: str, end: str) -> dict[str, Any]:
    if universe not in UNIVERSE_CONFIGS:
        raise ValueError(f"Unsupported universe: {universe}")
    cfg = dict(BASE_CONFIG)
    cfg.update(UNIVERSE_CONFIGS[universe])
    cfg["universe"] = str(universe)
    cfg["factor_weights"] = list(float(x) for x in factor_weights)
    cfg["start"] = str(start)
    cfg["end"] = str(end)
    return cfg


def summary_to_result_row(
    summary: dict[str, Any],
    outdir: str,
    universe: str,
    start: str,
    end: str,
    factor_weights: list[float],
) -> dict[str, Any]:
    return {
        "universe": str(universe),
        "period": format_period(start, end),
        "weight_set": format_weight_set(factor_weights),
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
        return pd.DataFrame(index=pd.Index([], name="period"))
    pivot = results_df.pivot_table(index=["universe", "period"], columns="weight_set", values="Sharpe")
    pivot = pivot.sort_index()
    pivot.columns.name = None
    return pivot.reset_index()


def run_sweep() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for universe in UNIVERSE_CONFIGS:
        for start, end in SUBPERIODS:
            for factor_weights in WEIGHT_SETS:
                cfg = build_run_config(
                    universe=universe,
                    factor_weights=factor_weights,
                    start=start,
                    end=end,
                )
                weight_label = format_weight_set(factor_weights)
                period_label = format_period(start, end)
                print(f"Running universe={universe} period={period_label} weight_set={weight_label}")
                summary, outdir = run_backtest(**cfg, run_cache=run_cache)
                rows.append(
                    summary_to_result_row(
                        summary=summary,
                        outdir=outdir,
                        universe=universe,
                        start=start,
                        end=end,
                        factor_weights=factor_weights,
                    )
                )
                run_manifest.append(
                    {
                        "universe": str(universe),
                        "period": period_label,
                        "weight_set": weight_label,
                        "backtest_outdir": str(outdir),
                        "summary_path": str(Path(outdir) / "summary.json"),
                        "run_config": cfg,
                    }
                )

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df = df.sort_values(["universe", "period", "weight_set"], kind="mergesort").reset_index(drop=True)
    return df, run_manifest


def write_outputs(
    results_df: pd.DataFrame,
    run_manifest: list[dict[str, Any]],
    results_root: Path = Path("results") / "subperiod_robustness",
) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df = build_sharpe_summary(results_df)

    results_path = outdir / "subperiod_results.csv"
    summary_path = outdir / "subperiod_summary.csv"
    manifest_path = outdir / "subperiod_manifest.json"
    latest_results_path = results_root / "subperiod_results_latest.csv"
    latest_summary_path = results_root / "subperiod_summary_latest.csv"

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
                "weight_sets": [format_weight_set(w) for w in WEIGHT_SETS],
                "subperiods": [{"start": s, "end": e, "label": format_period(s, e)} for s, e in SUBPERIODS],
                "runs": run_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("SUBPERIOD ROBUSTNESS")
    print("--------------------")
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
