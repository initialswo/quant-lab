"""Compare equal-weight vs volatility-scaled weighting for the lead strategy."""

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

BASE_CONFIG: dict[str, object] = {
    "top_n": 50,
    "rank_buffer": 20,
    "rebalance": "monthly",
    "start": "2005-01-01",
    "end": "2024-12-31",
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "factor_name": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
    "factor_names": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
    "factor_weights": [0.43, 0.27, 0.15, 0.15],
    "dynamic_factor_weights": True,
    "regime_filter": False,
    "regime_benchmark": "SPY",
    "regime_bull_weights": "momentum_12_1:0.48,reversal_1m:0.22,low_vol_20:0.10,gross_profitability:0.20",
    "regime_bear_weights": "momentum_12_1:0.28,reversal_1m:0.22,low_vol_20:0.30,gross_profitability:0.20",
    "target_vol": 0.14,
    "port_vol_lookback": 20,
    "max_leverage": 1.5,
    "bear_exposure_scale": 1.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
    "costs_bps": 10.0,
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
}


def _run_one(cfg: dict[str, object], run_cache: dict[str, Any]) -> dict[str, Any]:
    summary, _ = run_backtest(**cfg, run_cache=run_cache)
    return summary


def main() -> None:
    run_cache: dict[str, Any] = {}
    variants = [
        {"Variant": "Lead_EqualWeight", "volatility_scaled_weights": False},
        {"Variant": "Lead_VolScaled", "volatility_scaled_weights": True},
    ]

    rows: list[dict[str, Any]] = []
    sub_rows: list[dict[str, Any]] = []

    for variant in variants:
        cfg = dict(BASE_CONFIG)
        cfg["volatility_scaled_weights"] = bool(variant["volatility_scaled_weights"])
        full = _run_one(cfg, run_cache=run_cache)
        vname = str(variant["Variant"])
        rows.append(
            {
                "Variant": vname,
                "VolatilityScaledWeights": bool(variant["volatility_scaled_weights"]),
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
                    "Start": s,
                    "End": e,
                    "Sharpe": float(sub.get("Sharpe", float("nan"))),
                }
            )

    df = pd.DataFrame(rows)
    sub_df = pd.DataFrame(sub_rows)
    stats = (
        sub_df.groupby("Variant", dropna=False)["Sharpe"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "MeanSubSharpe", "std": "StdSubSharpe"})
    )
    df = df.merge(stats, on="Variant", how="left")
    df["StabilityScore"] = df["MeanSubSharpe"] - 0.5 * df["StdSubSharpe"]

    outdir = Path("results") / "volatility_weighting_test"
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"comparison_{ts}.csv"
    json_path = outdir / f"comparison_{ts}.json"
    latest_csv = outdir / "comparison_latest.csv"
    latest_json = outdir / "comparison_latest.json"
    df.to_csv(csv_path, index=False, float_format="%.10g")
    df.to_csv(latest_csv, index=False, float_format="%.10g")
    payload = {
        "base_config": BASE_CONFIG,
        "variants": variants,
        "comparison": df.to_dict(orient="records"),
        "subperiod_results": sub_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("VOLATILITY WEIGHTING COMPARISON")
    print("-------------------------------")
    print(
        df[
            [
                "Variant",
                "CAGR",
                "Vol",
                "Sharpe",
                "MaxDD",
                "StabilityScore",
            ]
        ].to_string(index=False)
    )
    print("")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {latest_csv}")
    print(f"Saved: {latest_json}")


if __name__ == "__main__":
    main()

