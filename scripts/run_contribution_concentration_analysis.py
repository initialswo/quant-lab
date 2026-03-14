"""Contribution concentration analysis for the lead strategy."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.data.loaders import load_ohlcv_for_research
from quant_lab.engine.runner import run_backtest
from quant_lab.research.contribution import (
    compute_daily_ticker_contributions,
    summarize_contribution_concentration,
)


BASE_CONFIG: dict[str, Any] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
    "max_tickers": 2000,
    "top_n": 50,
    "rebalance": "monthly",
    "costs_bps": 10.0,
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
    "bear_exposure_scale": 1.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
    "save_artifacts": True,
}


def _load_holdings(holdings_path: Path) -> pd.DataFrame:
    h = pd.read_csv(holdings_path, index_col="date", parse_dates=["date"])
    h.index = pd.DatetimeIndex(h.index)
    return h.astype(float).sort_index()


def _load_close_panel(
    start: str,
    end: str,
    tickers: list[str],
) -> pd.DataFrame:
    data = load_ohlcv_for_research(
        start=start,
        end=end,
        universe="sp500",
        max_tickers=2000,
        store_root="data/equities",
    )
    close = data.panels.get("close", pd.DataFrame()).astype(float)
    close = close.reindex(columns=tickers)
    return close


def main() -> None:
    summary, run_outdir = run_backtest(**BASE_CONFIG)
    run_out = Path(str(run_outdir))
    holdings_path = run_out / "holdings.csv"
    if not holdings_path.exists():
        raise FileNotFoundError(f"holdings artifact not found: {holdings_path}")

    holdings = _load_holdings(holdings_path)
    close = _load_close_panel(
        start=str(BASE_CONFIG["start"]),
        end=str(BASE_CONFIG["end"]),
        tickers=list(holdings.columns),
    )
    close = close.reindex(index=holdings.index, columns=holdings.columns)
    contrib_daily = compute_daily_ticker_contributions(close=close, realized_weights=holdings)
    contrib_by_ticker = contrib_daily.sum(axis=0).sort_values(ascending=False).astype(float)

    total_signed = float(contrib_by_ticker.sum())
    total_abs = float(contrib_by_ticker.abs().sum())
    abs_share = (contrib_by_ticker.abs() / total_abs) if total_abs > 0.0 else contrib_by_ticker * np.nan
    signed_share = (
        (contrib_by_ticker / total_signed)
        if (np.isfinite(total_signed) and abs(total_signed) > 1e-12)
        else contrib_by_ticker * np.nan
    )
    top = pd.DataFrame(
        {
            "Ticker": contrib_by_ticker.index,
            "CumulativeContribution": contrib_by_ticker.to_numpy(dtype=float),
            "AbsContribution": contrib_by_ticker.abs().to_numpy(dtype=float),
            "AbsShare": abs_share.reindex(contrib_by_ticker.index).to_numpy(dtype=float),
            "SignedShareOfTotal": signed_share.reindex(contrib_by_ticker.index).to_numpy(dtype=float),
        }
    )
    top20 = top.head(20).copy()

    conc = summarize_contribution_concentration(contrib_by_ticker)
    summary_payload: dict[str, Any] = {
        "run_outdir": str(run_out),
        "config": BASE_CONFIG,
        "metrics": {
            "CAGR": float(summary.get("CAGR", float("nan"))),
            "Vol": float(summary.get("Vol", float("nan"))),
            "Sharpe": float(summary.get("Sharpe", float("nan"))),
            "MaxDD": float(summary.get("MaxDD", float("nan"))),
        },
        "contribution": {
            "ticker_count": int(len(contrib_by_ticker)),
            "total_cumulative_contribution_signed": total_signed,
            "total_cumulative_contribution_abs": total_abs,
            **conc,
            "q05_signed": float(contrib_by_ticker.quantile(0.05)),
            "q25_signed": float(contrib_by_ticker.quantile(0.25)),
            "q50_signed": float(contrib_by_ticker.quantile(0.50)),
            "q75_signed": float(contrib_by_ticker.quantile(0.75)),
            "q95_signed": float(contrib_by_ticker.quantile(0.95)),
        },
    }

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results") / f"contribution_concentration_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    contrib_path = outdir / "contribution_by_ticker.csv"
    top20_path = outdir / "top_contributors.csv"
    summary_path = outdir / "contribution_summary.json"

    top.to_csv(contrib_path, index=False, float_format="%.10g")
    top20.to_csv(top20_path, index=False, float_format="%.10g")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("CONTRIBUTION CONCENTRATION SUMMARY")
    print("----------------------------------")
    print(f"Top1 abs share:  {summary_payload['contribution']['top1_share_abs']:.4f}")
    print(f"Top5 abs share:  {summary_payload['contribution']['top5_share_abs']:.4f}")
    print(f"Top10 abs share: {summary_payload['contribution']['top10_share_abs']:.4f}")
    print(f"Top20 abs share: {summary_payload['contribution']['top20_share_abs']:.4f}")
    print(f"Herfindahl(abs): {summary_payload['contribution']['herfindahl_abs']:.6f}")
    print(f"Effective N(abs): {summary_payload['contribution']['effective_n_abs']:.2f}")
    print("")
    print("Top 20 contributors:")
    print(top20.to_string(index=False))
    print("")
    print(f"Saved: {contrib_path}")
    print(f"Saved: {top20_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

