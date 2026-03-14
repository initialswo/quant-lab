"""Run fixed subperiod backtests and print/save a compact summary table."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_lab.engine.runner import run_backtest


DEFAULT_PERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]


def _parse_csv_list(raw: str) -> list[str]:
    out = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError("Expected a non-empty comma-separated list.")
    return out


def _parse_csv_floats(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return vals


def _fmt_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def _period_label(start: str, end: str) -> str:
    return f"{start[:4]}-{end[:4]}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run fixed subperiod backtests for model validation.")
    p.add_argument("--factor", default="momentum_12_1,reversal_1m,low_vol_20")
    p.add_argument("--factor_weights", default="0.5,0.3,0.2")
    p.add_argument("--top_n", type=int, default=50)
    p.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default="monthly")
    p.add_argument("--max_tickers", type=int, default=2000)
    p.add_argument("--data_source", default="parquet", choices=["parquet"])
    p.add_argument("--data_root", default="data/equities")
    p.add_argument("--costs_bps", type=float, default=10.0)
    p.add_argument("--target_vol", type=float, default=0.0)
    p.add_argument("--port_vol_lookback", type=int, default=20)
    p.add_argument("--max_leverage", type=float, default=1.0)
    p.add_argument("--min_price", type=float, default=0.0)
    p.add_argument("--min_avg_dollar_volume", type=float, default=0.0)
    p.add_argument("--liquidity_lookback", type=int, default=20)
    p.add_argument("--regime_filter", type=int, choices=[0, 1], default=0)
    p.add_argument("--regime_benchmark", default="SPY")
    p.add_argument("--bear_exposure_scale", type=float, default=1.0)
    p.add_argument("--save_json", type=int, choices=[0, 1], default=1)
    return p


def main() -> None:
    args = build_parser().parse_args()

    factor_names = _parse_csv_list(args.factor)
    factor_weights = _parse_csv_floats(args.factor_weights)
    if len(factor_names) != len(factor_weights):
        raise ValueError("factor and factor_weights lengths must match.")

    rows: list[dict] = []
    for start, end in DEFAULT_PERIODS:
        summary, _ = run_backtest(
            start=start,
            end=end,
            max_tickers=int(args.max_tickers),
            top_n=int(args.top_n),
            rebalance=str(args.rebalance),
            costs_bps=float(args.costs_bps),
            data_source=str(args.data_source),
            data_cache_dir=str(args.data_root),
            factor_name=factor_names,
            factor_names=factor_names,
            factor_weights=factor_weights,
            target_vol=float(args.target_vol),
            port_vol_lookback=int(args.port_vol_lookback),
            max_leverage=float(args.max_leverage),
            min_price=float(args.min_price),
            min_avg_dollar_volume=float(args.min_avg_dollar_volume),
            liquidity_lookback=int(args.liquidity_lookback),
            regime_filter=bool(args.regime_filter),
            regime_benchmark=str(args.regime_benchmark),
            bear_exposure_scale=float(args.bear_exposure_scale),
        )
        if str(summary.get("DataSource", "")).lower() != "parquet":
            raise ValueError(f"Unexpected DataSource={summary.get('DataSource')}; expected parquet.")
        if int(summary.get("DataFetchedOrRefreshed", 0)) != 0:
            raise ValueError(
                "Backtest unexpectedly refreshed/fetched external data: "
                f"DataFetchedOrRefreshed={summary.get('DataFetchedOrRefreshed')}"
            )
        row = {
            "Period": _period_label(start, end),
            "Start": start,
            "End": end,
            "TickersUsed": int(summary.get("TickersUsed", 0)),
            "MissingTickersCount": int(summary.get("MissingTickersCount", 0)),
            "CAGR": float(summary.get("CAGR", float("nan"))),
            "Vol": float(summary.get("Vol", float("nan"))),
            "Sharpe": float(summary.get("Sharpe", float("nan"))),
            "MaxDD": float(summary.get("MaxDD", float("nan"))),
            "SanityWarningsCount": int(summary.get("SanityWarningsCount", 0)),
            "TargetVol": float(summary.get("TargetVol", float(args.target_vol))),
            "PortVolLookback": int(summary.get("PortVolLookback", int(args.port_vol_lookback))),
            "MaxLeverage": float(summary.get("MaxLeverage", float(args.max_leverage))),
            "MinPrice": float(summary.get("MinPrice", float(args.min_price))),
            "MinAvgDollarVolume": float(
                summary.get("MinAvgDollarVolume", float(args.min_avg_dollar_volume))
            ),
            "LiquidityLookback": int(summary.get("LiquidityLookback", int(args.liquidity_lookback))),
            "LiquidityEligibleMedian": float(summary.get("LiquidityEligibleMedian", float("nan"))),
            "LiquidityFilteredOutMedian": float(
                summary.get("LiquidityFilteredOutMedian", float("nan"))
            ),
            "LeverageAvg": float(summary.get("LeverageAvg", float("nan"))),
            "LeverageMax": float(summary.get("LeverageMax", float("nan"))),
            "RawVol": float(summary.get("RawVol", float("nan"))),
            "FinalRealizedVol": float(summary.get("FinalRealizedVol", float("nan"))),
            "VolTargetingEnabled": bool(summary.get("VolTargetingEnabled", False)),
            "RegimeFilter": bool(summary.get("RegimeFilter", bool(args.regime_filter))),
            "RegimeBenchmark": str(summary.get("RegimeBenchmark", str(args.regime_benchmark))),
            "BearExposureScale": float(summary.get("BearExposureScale", float(args.bear_exposure_scale))),
            "regime_pct_bull": float(summary.get("regime_pct_bull", float("nan"))),
            "regime_pct_bear_or_volatile": float(
                summary.get("regime_pct_bear_or_volatile", float("nan"))
            ),
            "Outdir": str(summary.get("Outdir", "")),
            "DataSource": str(summary.get("DataSource", "")),
            "DataFetchedOrRefreshed": int(summary.get("DataFetchedOrRefreshed", 0)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("SUBPERIOD BACKTEST SUMMARY")
    print("--------------------------")
    disp = df[
        [
            "Period",
            "TickersUsed",
            "MissingTickersCount",
            "CAGR",
            "Vol",
            "Sharpe",
            "MaxDD",
            "SanityWarningsCount",
            "TargetVol",
            "PortVolLookback",
            "MaxLeverage",
            "MinPrice",
            "MinAvgDollarVolume",
            "LiquidityLookback",
            "LiquidityEligibleMedian",
            "LiquidityFilteredOutMedian",
            "LeverageAvg",
            "LeverageMax",
            "RawVol",
            "FinalRealizedVol",
            "VolTargetingEnabled",
            "RegimeFilter",
            "RegimeBenchmark",
            "BearExposureScale",
            "regime_pct_bull",
            "regime_pct_bear_or_volatile",
        ]
    ].copy()
    disp.columns = [
        "Period",
        "Tickers",
        "Missing",
        "CAGR",
        "Vol",
        "Sharpe",
        "MaxDD",
        "Warnings",
        "TgtVol",
        "VolLb",
        "MaxLev",
        "MinPx",
        "MinADV",
        "LiqLb",
        "LiqEligMed",
        "LiqOutMed",
        "LevAvg",
        "LevMax",
        "RawVol",
        "FinalVol",
        "VT",
        "Regime",
        "Benchmark",
        "BearScale",
        "BullPct",
        "BearPct",
    ]
    for c in [
        "CAGR",
        "Vol",
        "Sharpe",
        "MaxDD",
        "TgtVol",
        "MaxLev",
        "MinPx",
        "MinADV",
        "LiqLb",
        "LiqEligMed",
        "LiqOutMed",
        "LevAvg",
        "LevMax",
        "RawVol",
        "FinalVol",
        "BearScale",
        "BullPct",
        "BearPct",
    ]:
        disp[c] = disp[c].map(_fmt_float)
    print(disp.to_string(index=False))

    warn_rows = df[
        (df["SanityWarningsCount"] > 0) | (df["CAGR"] > 1.0) | (df["Vol"] > 1.0)
    ].copy()
    if not warn_rows.empty:
        print("\nWARNING BANNER")
        for _, r in warn_rows.iterrows():
            print(
                f"[{r['Period']}] warnings={int(r['SanityWarningsCount'])} "
                f"CAGR={_fmt_float(float(r['CAGR']))} Vol={_fmt_float(float(r['Vol']))}"
            )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_csv = Path("results") / f"subperiod_summary_{ts}.csv"
    out_json = Path("results") / f"subperiod_summary_{ts}.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.10g")
    print(f"\nSaved CSV: {out_csv}")
    if bool(args.save_json):
        out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
