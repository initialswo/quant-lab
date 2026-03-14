"""Compare best architecture performance before vs after liquidity filtering."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_lab.engine.runner import run_backtest


SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]


def _fmt(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def _period_label(start: str, end: str) -> str:
    return f"{start[:4]}-{end[:4]}"


def _base_cfg() -> dict[str, object]:
    return {
        "start": "2005-01-01",
        "end": "2024-12-31",
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
    }


def _variants() -> list[dict[str, object]]:
    return [
        {
            "Variant": "NoLiquidityFilter",
            "min_price": 0.0,
            "min_avg_dollar_volume": 0.0,
            "liquidity_lookback": 20,
        },
        {
            "Variant": "LiquidityFilter",
            "min_price": 5.0,
            "min_avg_dollar_volume": 5_000_000.0,
            "liquidity_lookback": 20,
        },
    ]


def _run_one(start: str, end: str, variant: dict[str, object]) -> dict:
    cfg = _base_cfg()
    cfg.update(variant)
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


def main() -> None:
    rows: list[dict] = []
    sub_rows: list[dict] = []

    for variant in _variants():
        name = str(variant["Variant"])
        full = _run_one(start="2005-01-01", end="2024-12-31", variant=variant)
        rows.append(
            {
                "Variant": name,
                "Tickers": int(full.get("TickersUsed", 0)),
                "LiqEligibleMed": float(full.get("LiquidityEligibleMedian", float("nan"))),
                "CAGR": float(full.get("CAGR", float("nan"))),
                "Vol": float(full.get("Vol", float("nan"))),
                "Sharpe": float(full.get("Sharpe", float("nan"))),
                "MaxDD": float(full.get("MaxDD", float("nan"))),
                "MinPrice": float(full.get("MinPrice", float(variant.get("min_price", 0.0)))),
                "MinAvgDollarVolume": float(
                    full.get("MinAvgDollarVolume", float(variant.get("min_avg_dollar_volume", 0.0)))
                ),
                "LiquidityLookback": int(
                    full.get("LiquidityLookback", int(variant.get("liquidity_lookback", 20)))
                ),
                "Outdir": str(full.get("Outdir", "")),
            }
        )
        for start, end in SUBPERIODS:
            part = _run_one(start=start, end=end, variant=variant)
            sub_rows.append(
                {
                    "Variant": name,
                    "Period": _period_label(start, end),
                    "Start": start,
                    "End": end,
                    "Sharpe": float(part.get("Sharpe", float("nan"))),
                }
            )

    df = pd.DataFrame(rows)
    sub_df = pd.DataFrame(sub_rows)
    sub_stats = (
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

    print("LIQUIDITY FILTER COMPARISON")
    print("---------------------------")
    disp = df[["Variant", "Tickers", "LiqEligibleMed", "CAGR", "Vol", "Sharpe", "MaxDD"]].copy()
    for c in ["LiqEligibleMed", "CAGR", "Vol", "Sharpe", "MaxDD"]:
        disp[c] = disp[c].map(_fmt)
    print(disp.to_string(index=False))

    print("")
    print("SUBPERIOD SHARPE STABILITY")
    print("--------------------------")
    disp_sub = sub_stats.copy()
    for c in ["MeanSubSharpe", "StdSubSharpe", "MinSubSharpe", "MaxSubSharpe"]:
        disp_sub[c] = disp_sub[c].map(_fmt)
    print(disp_sub.to_string(index=False))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / f"liquidity_filter_comparison_{ts}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.10g")
    print("")
    print(f"Saved CSV: {out}")


if __name__ == "__main__":
    main()
