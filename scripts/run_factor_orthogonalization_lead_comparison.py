"""Focused comparison: lead 4-factor candidate with/without factor orthogonalization."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.runner import run_backtest


SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]

SECTOR_MAP_SOURCE = "projects/dashboard_legacy/data/sp500_tickers.csv"

BASE_CONFIG: dict[str, object] = {
    "top_n": 50,
    "rebalance": "monthly",
    "start": "2005-01-01",
    "end": "2024-12-31",
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "factor_name": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
    "factor_names": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
    "factor_weights": [0.43, 0.27, 0.15, 0.15],
    "target_vol": 0.14,
    "port_vol_lookback": 20,
    "max_leverage": 1.5,
    "regime_filter": True,
    "regime_benchmark": "SPY",
    "bear_exposure_scale": 1.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
    "costs_bps": 10.0,
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
    "sector_map": SECTOR_MAP_SOURCE,
}


def _run_one(cfg: dict[str, object], run_cache: dict[str, Any]) -> dict[str, Any]:
    summary, _ = run_backtest(**cfg, run_cache=run_cache)
    return summary


def _fmt(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def main() -> None:
    run_cache: dict[str, Any] = {}
    variants = [
        {"Variant": "Lead4F_Baseline", "orthogonalize_factors": False},
        {"Variant": "Lead4F_Orthogonalized", "orthogonalize_factors": True},
    ]

    rows: list[dict[str, Any]] = []
    sub_rows: list[dict[str, Any]] = []

    for variant in variants:
        cfg = dict(BASE_CONFIG)
        cfg["orthogonalize_factors"] = bool(variant["orthogonalize_factors"])
        full = _run_one(cfg, run_cache=run_cache)
        vname = str(variant["Variant"])
        rows.append(
            {
                "Variant": vname,
                "OrthogonalizeFactors": bool(variant["orthogonalize_factors"]),
                "CAGR": float(full.get("CAGR", float("nan"))),
                "Vol": float(full.get("Vol", float("nan"))),
                "Sharpe": float(full.get("Sharpe", float("nan"))),
                "MaxDD": float(full.get("MaxDD", float("nan"))),
                "Outdir": str(full.get("Outdir", "")),
            }
        )

        for s, e in SUBPERIODS:
            cfg_sub = dict(cfg)
            cfg_sub["start"] = s
            cfg_sub["end"] = e
            sub = _run_one(cfg_sub, run_cache=run_cache)
            sub_rows.append(
                {
                    "Variant": vname,
                    "OrthogonalizeFactors": bool(variant["orthogonalize_factors"]),
                    "Start": s,
                    "End": e,
                    "Sharpe": float(sub.get("Sharpe", float("nan"))),
                }
            )

    df = pd.DataFrame(rows)
    sub_df = pd.DataFrame(sub_rows)
    stats = (
        sub_df.groupby("Variant", dropna=False)["Sharpe"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "MeanSubSharpe",
                "std": "StdSubSharpe",
                "min": "MinSubSharpe",
                "max": "MaxSubSharpe",
            }
        )
    )
    df = df.merge(stats, on="Variant", how="left")
    df["StabilityScore"] = df["MeanSubSharpe"] - 0.5 * df["StdSubSharpe"]

    disp_cols = [
        "Variant",
        "OrthogonalizeFactors",
        "CAGR",
        "Vol",
        "Sharpe",
        "MaxDD",
        "MeanSubSharpe",
        "StdSubSharpe",
        "MinSubSharpe",
        "MaxSubSharpe",
        "StabilityScore",
    ]
    disp = df[disp_cols].copy()
    for c in [
        "CAGR",
        "Vol",
        "Sharpe",
        "MaxDD",
        "MeanSubSharpe",
        "StdSubSharpe",
        "MinSubSharpe",
        "MaxSubSharpe",
        "StabilityScore",
    ]:
        disp[c] = disp[c].map(_fmt)

    print("LEAD 4FACTOR: BASELINE VS ORTHOGONALIZED")
    print("-----------------------------------------")
    print(disp.to_string(index=False))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results") / f"factor_orthogonalization_lead_comparison_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    full_csv = outdir / "lead_factor_orthogonalization_comparison.csv"
    sub_csv = outdir / "lead_factor_orthogonalization_subperiods.csv"
    summary_json = outdir / "lead_factor_orthogonalization_summary.json"

    df.to_csv(full_csv, index=False, float_format="%.10g")
    sub_df.to_csv(sub_csv, index=False, float_format="%.10g")
    summary_json.write_text(
        json.dumps(
            {
                "base_config": BASE_CONFIG,
                "variants": variants,
                "subperiods": [{"start": s, "end": e} for s, e in SUBPERIODS],
                "comparison": df.to_dict(orient="records"),
                "subperiod_results": sub_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print(f"Saved comparison: {full_csv}")
    print(f"Saved subperiods: {sub_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()

