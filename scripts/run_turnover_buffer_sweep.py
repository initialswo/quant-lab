"""Research sweep for Top-N rank buffer turnover control."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.runner import run_backtest
from quant_lab.strategies.topn import rebalance_mask


BUFFER_VALUES = [0, 10, 20, 30]

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


def _compute_turnover_from_holdings(outdir: str, rebalance: str) -> float:
    h_path = Path(outdir) / "holdings.csv"
    if not h_path.exists():
        return float("nan")
    h = pd.read_csv(h_path, parse_dates=["date"]).set_index("date").sort_index().astype(float)
    td = 0.5 * h.diff().abs().sum(axis=1).fillna(0.0)
    rb = rebalance_mask(pd.DatetimeIndex(h.index), rebalance)
    trb = td.loc[rb]
    return float(trb.mean()) if not trb.empty else float("nan")


def main() -> None:
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for b in BUFFER_VALUES:
        cfg = dict(BASE_CONFIG)
        cfg["rank_buffer"] = int(b)
        summary, outdir = run_backtest(**cfg, run_cache=run_cache)
        turnover = _compute_turnover_from_holdings(outdir=outdir, rebalance=str(BASE_CONFIG["rebalance"]))
        rows.append(
            {
                "RankBuffer": int(b),
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "Turnover": turnover,
                "Outdir": str(summary.get("Outdir", outdir)),
            }
        )

    df = pd.DataFrame(rows).sort_values("RankBuffer").reset_index(drop=True)
    print("TURNOVER BUFFER SWEEP")
    print("---------------------")
    print(df.to_string(index=False))

    base_out = Path("results") / "turnover_buffer_sweep"
    base_out.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = base_out / f"turnover_buffer_sweep_{ts}.csv"
    summary_path = base_out / f"turnover_buffer_sweep_{ts}.json"
    latest_csv = base_out / "turnover_buffer_sweep_latest.csv"
    latest_json = base_out / "turnover_buffer_sweep_latest.json"

    df.to_csv(csv_path, index=False, float_format="%.10g")
    df.to_csv(latest_csv, index=False, float_format="%.10g")
    summary_payload = {
        "config": BASE_CONFIG,
        "buffers": BUFFER_VALUES,
        "results": rows,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("")
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {latest_csv}")
    print(f"Saved: {latest_json}")


if __name__ == "__main__":
    main()
