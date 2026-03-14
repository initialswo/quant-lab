"""Sweep bear-regime exposure scale on fixed core stack + liquidity architecture."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_lab.engine.runner import run_backtest


FULL_START = "2005-01-01"
FULL_END = "2024-12-31"
SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]

BASE_CONFIG: dict[str, object] = {
    "start": FULL_START,
    "end": FULL_END,
    "factor_name": ["momentum_12_1", "reversal_1m", "low_vol_20"],
    "factor_names": ["momentum_12_1", "reversal_1m", "low_vol_20"],
    "factor_weights": [0.5, 0.3, 0.2],
    "top_n": 50,
    "rebalance": "monthly",
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "costs_bps": 10.0,
    "target_vol": 0.15,
    "port_vol_lookback": 20,
    "max_leverage": 1.5,
    "regime_filter": True,
    "regime_benchmark": "SPY",
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
}

VARIANTS: list[dict[str, object]] = [
    {"Variant": "Bear_1.00", "bear_exposure_scale": 1.00},
    {"Variant": "Bear_0.75", "bear_exposure_scale": 0.75},
    {"Variant": "Bear_0.50", "bear_exposure_scale": 0.50},
    {"Variant": "Bear_0.25", "bear_exposure_scale": 0.25},
]


def _fmt_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def _period_label(start: str, end: str) -> str:
    return f"{start[:4]}-{end[:4]}"


def _run_once(start: str, end: str, variant_cfg: dict[str, object]) -> dict:
    cfg = dict(BASE_CONFIG)
    cfg.update(variant_cfg)
    cfg["start"] = start
    cfg["end"] = end
    cfg.pop("Variant", None)
    summary, _ = run_backtest(**cfg)
    if str(summary.get("DataSource", "")).lower() != "parquet":
        raise ValueError(f"Unexpected DataSource={summary.get('DataSource')}; expected parquet.")
    if int(summary.get("DataFetchedOrRefreshed", 0)) != 0:
        raise ValueError(
            "Backtest unexpectedly refreshed/fetched external data: "
            f"DataFetchedOrRefreshed={summary.get('DataFetchedOrRefreshed')}"
        )
    return summary


def _full_row(variant: str, summary: dict) -> dict:
    return {
        "Variant": variant,
        "BearExposureScale": float(summary.get("BearExposureScale", float("nan"))),
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "Vol": float(summary.get("Vol", float("nan"))),
        "Sharpe": float(summary.get("Sharpe", float("nan"))),
        "MaxDD": float(summary.get("MaxDD", float("nan"))),
        "LeverageAvg": float(summary.get("LeverageAvg", float("nan"))),
        "LeverageMax": float(summary.get("LeverageMax", float("nan"))),
        "regime_pct_bull": float(summary.get("regime_pct_bull", float("nan"))),
        "regime_pct_bear_or_volatile": float(
            summary.get("regime_pct_bear_or_volatile", float("nan"))
        ),
        "LiquidityEligibleMedian": float(summary.get("LiquidityEligibleMedian", float("nan"))),
        "SanityWarningsCount": int(summary.get("SanityWarningsCount", 0)),
        "Outdir": str(summary.get("Outdir", "")),
    }


def _subperiod_row(variant: str, start: str, end: str, summary: dict) -> dict:
    return {
        "Variant": variant,
        "Period": _period_label(start, end),
        "Start": start,
        "End": end,
        "Sharpe": float(summary.get("Sharpe", float("nan"))),
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "Vol": float(summary.get("Vol", float("nan"))),
        "MaxDD": float(summary.get("MaxDD", float("nan"))),
        "BearExposureScale": float(summary.get("BearExposureScale", float("nan"))),
        "Outdir": str(summary.get("Outdir", "")),
    }


def _stability_table(subperiod_df: pd.DataFrame) -> pd.DataFrame:
    grp = subperiod_df.groupby("Variant", dropna=False)["Sharpe"]
    stats = grp.agg(["mean", "std", "min", "max"]).reset_index()
    return stats.rename(
        columns={
            "mean": "MeanSubSharpe",
            "std": "StdSubSharpe",
            "min": "MinSubSharpe",
            "max": "MaxSubSharpe",
        }
    )


def main() -> None:
    full_rows: list[dict] = []
    subperiod_rows: list[dict] = []

    for variant_cfg in VARIANTS:
        variant = str(variant_cfg["Variant"])
        full = _run_once(start=FULL_START, end=FULL_END, variant_cfg=variant_cfg)
        full_rows.append(_full_row(variant, full))
        for start, end in SUBPERIODS:
            part = _run_once(start=start, end=end, variant_cfg=variant_cfg)
            subperiod_rows.append(_subperiod_row(variant, start, end, part))

    full_df = pd.DataFrame(full_rows)
    subperiod_df = pd.DataFrame(subperiod_rows)
    stability_df = _stability_table(subperiod_df)

    disp_full = full_df[
        ["Variant", "BearExposureScale", "CAGR", "Vol", "Sharpe", "MaxDD"]
    ].copy()
    disp_full.columns = ["Variant", "BearScale", "CAGR", "Vol", "Sharpe", "MaxDD"]
    for c in ["BearScale", "CAGR", "Vol", "Sharpe", "MaxDD"]:
        disp_full[c] = disp_full[c].map(_fmt_float)

    disp_sub = stability_df[
        ["Variant", "MeanSubSharpe", "StdSubSharpe", "MinSubSharpe", "MaxSubSharpe"]
    ].copy()
    for c in ["MeanSubSharpe", "StdSubSharpe", "MinSubSharpe", "MaxSubSharpe"]:
        disp_sub[c] = disp_sub[c].map(_fmt_float)

    print("BEAR EXPOSURE SWEEP — FULL PERIOD")
    print("---------------------------------")
    print(disp_full.to_string(index=False))
    print("")
    print("BEAR EXPOSURE SWEEP — SUBPERIOD STABILITY")
    print("-----------------------------------------")
    print(disp_sub.to_string(index=False))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"bear_exposure_sweep_{ts}.csv"
    out_json = outdir / f"bear_exposure_sweep_{ts}.json"
    out_sub_csv = outdir / f"bear_exposure_subperiods_{ts}.csv"

    full_df.to_csv(out_csv, index=False, float_format="%.10g")
    subperiod_df.to_csv(out_sub_csv, index=False, float_format="%.10g")
    out_json.write_text(
        json.dumps(
            {
                "config": {
                    "full_period": {"start": FULL_START, "end": FULL_END},
                    "subperiods": [{"start": s, "end": e} for s, e in SUBPERIODS],
                    "base": BASE_CONFIG,
                    "variants": VARIANTS,
                },
                "full_period_results": full_rows,
                "subperiod_results": subperiod_rows,
                "subperiod_stability": stability_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved Subperiod CSV: {out_sub_csv}")


if __name__ == "__main__":
    main()
