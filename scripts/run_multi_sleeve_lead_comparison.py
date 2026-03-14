"""Research comparison: lead single-composite vs multi-sleeve portfolio architecture."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest
from quant_lab.strategies.topn import rebalance_mask


SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]

MULTI_SLEEVE_CONFIG = {
    "sleeves": [
        {
            "name": "momentum",
            "factors": ["momentum_12_1", "reversal_1m"],
            "factor_weights": [0.65, 0.35],
            "allocation": 0.50,
            "top_n": 25,
        },
        {
            "name": "defensive",
            "factors": ["low_vol_20"],
            "factor_weights": [1.0],
            "allocation": 0.25,
            "top_n": 15,
        },
        {
            "name": "quality",
            "factors": ["gross_profitability"],
            "factor_weights": [1.0],
            "allocation": 0.25,
            "top_n": 15,
        },
    ]
}

BASE_CONFIG: dict[str, Any] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
    "max_tickers": 2000,
    "top_n": 50,
    "rank_buffer": 20,
    "rebalance": "monthly",
    "costs_bps": 10.0,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "factor_name": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
    "factor_names": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
    "factor_weights": [0.43, 0.27, 0.15, 0.15],
    "target_vol": 0.14,
    "port_vol_lookback": 20,
    "max_leverage": 1.5,
    "regime_filter": False,
    "regime_benchmark": "SPY",
    "regime_bull_weights": "momentum_12_1:0.48,reversal_1m:0.22,low_vol_20:0.10,gross_profitability:0.20",
    "regime_bear_weights": "momentum_12_1:0.28,reversal_1m:0.22,low_vol_20:0.30,gross_profitability:0.20",
    "bear_exposure_scale": 1.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
    "save_artifacts": True,
}


def _subperiod_stats(outdir: str) -> tuple[float, float, float, float]:
    eq_path = Path(outdir) / "equity_curve.csv"
    eq = pd.read_csv(eq_path, parse_dates=["date"]).set_index("date").sort_index()
    r = pd.to_numeric(eq["returns"], errors="coerce").fillna(0.0)
    vals: list[float] = []
    for s, e in SUBPERIODS:
        rr = r.loc[(r.index >= pd.Timestamp(s)) & (r.index <= pd.Timestamp(e))]
        m = compute_metrics(rr)
        vals.append(float(m.get("Sharpe", float("nan"))))
    ser = pd.Series(vals, dtype=float)
    return float(ser.mean()), float(ser.std(ddof=0)), float(ser.min()), float(ser.max())


def _holding_and_turnover(outdir: str, rebalance: str) -> tuple[float, float]:
    hp = Path(outdir) / "holdings.csv"
    if not hp.exists():
        return float("nan"), float("nan")
    h = pd.read_csv(hp, parse_dates=["date"]).set_index("date").sort_index().astype(float)
    avg_holdings = float((h > 0.0).sum(axis=1).mean())
    td = 0.5 * h.diff().abs().sum(axis=1).fillna(0.0)
    rb = rebalance_mask(pd.DatetimeIndex(h.index), rebalance)
    trb = td.loc[rb]
    turnover = float(trb.mean()) if not trb.empty else float("nan")
    return avg_holdings, turnover


def main() -> None:
    run_cache: dict[str, Any] = {}
    variants = [
        {
            "Variant": "Lead_Composite",
            "portfolio_mode": "composite",
            "dynamic_factor_weights": True,
            "multi_sleeve_config": None,
        },
        {
            "Variant": "Lead_MultiSleeve",
            "portfolio_mode": "multi_sleeve",
            "dynamic_factor_weights": False,
            "multi_sleeve_config": MULTI_SLEEVE_CONFIG,
        },
    ]

    rows: list[dict[str, Any]] = []
    subperiod_rows: list[dict[str, Any]] = []

    for i, variant in enumerate(variants, start=1):
        cfg = dict(BASE_CONFIG)
        cfg["portfolio_mode"] = str(variant["portfolio_mode"])
        cfg["dynamic_factor_weights"] = bool(variant["dynamic_factor_weights"])
        if variant["multi_sleeve_config"] is not None:
            cfg["multi_sleeve_config"] = dict(variant["multi_sleeve_config"])

        print(
            f"[{i}/{len(variants)}] {variant['Variant']} "
            f"mode={cfg['portfolio_mode']} dynamic_factor_weights={cfg['dynamic_factor_weights']}"
        )
        summary, outdir = run_backtest(**cfg, run_cache=run_cache)
        mean_sub, std_sub, min_sub, max_sub = _subperiod_stats(outdir=outdir)
        avg_holdings, turnover = _holding_and_turnover(outdir=outdir, rebalance=str(cfg["rebalance"]))
        row = {
            "Variant": str(variant["Variant"]),
            "PortfolioMode": str(cfg["portfolio_mode"]),
            "DynamicFactorWeights": bool(cfg["dynamic_factor_weights"]),
            "CAGR": float(summary.get("CAGR", float("nan"))),
            "Vol": float(summary.get("Vol", float("nan"))),
            "Sharpe": float(summary.get("Sharpe", float("nan"))),
            "MaxDD": float(summary.get("MaxDD", float("nan"))),
            "MeanSubSharpe": mean_sub,
            "StdSubSharpe": std_sub,
            "MinSubSharpe": min_sub,
            "MaxSubSharpe": max_sub,
            "StabilityScore": float(mean_sub - 0.5 * std_sub),
            "AvgHoldings": avg_holdings,
            "Turnover": turnover,
            "Outdir": str(outdir),
        }
        rows.append(row)
        for s, e in SUBPERIODS:
            eq = pd.read_csv(Path(outdir) / "equity_curve.csv", parse_dates=["date"]).set_index("date").sort_index()
            rr = pd.to_numeric(eq["returns"], errors="coerce").fillna(0.0)
            rr = rr.loc[(rr.index >= pd.Timestamp(s)) & (rr.index <= pd.Timestamp(e))]
            m = compute_metrics(rr)
            subperiod_rows.append(
                {
                    "Variant": str(variant["Variant"]),
                    "PortfolioMode": str(cfg["portfolio_mode"]),
                    "Start": s,
                    "End": e,
                    "Sharpe": float(m.get("Sharpe", float("nan"))),
                }
            )
        print(
            f"    Sharpe={row['Sharpe']:.4f} CAGR={row['CAGR']:.4f} "
            f"MaxDD={row['MaxDD']:.4f} Stability={row['StabilityScore']:.4f}"
        )

    df = pd.DataFrame(rows)
    sub_df = pd.DataFrame(subperiod_rows)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results") / f"multi_sleeve_lead_comparison_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    comp_csv = outdir / "multi_sleeve_comparison.csv"
    sub_csv = outdir / "multi_sleeve_subperiods.csv"
    summary_json = outdir / "multi_sleeve_summary.json"

    df.to_csv(comp_csv, index=False, float_format="%.10g")
    sub_df.to_csv(sub_csv, index=False, float_format="%.10g")
    summary_json.write_text(
        json.dumps(
            {
                "base_config": BASE_CONFIG,
                "multi_sleeve_config": MULTI_SLEEVE_CONFIG,
                "subperiods": [{"start": s, "end": e} for s, e in SUBPERIODS],
                "results": rows,
                "subperiod_results": subperiod_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nMULTI-SLEEVE VS COMPOSITE")
    print("-------------------------")
    print(
        df[
            [
                "Variant",
                "PortfolioMode",
                "CAGR",
                "Vol",
                "Sharpe",
                "MaxDD",
                "MeanSubSharpe",
                "StdSubSharpe",
                "MinSubSharpe",
                "MaxSubSharpe",
                "StabilityScore",
                "AvgHoldings",
                "Turnover",
            ]
        ].to_string(index=False)
    )
    print("")
    print(f"Saved: {comp_csv}")
    print(f"Saved: {sub_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
