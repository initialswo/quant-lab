"""Research sweep: composite-score neutralization modes for the lead strategy."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest


MODES: list[str | None] = [None, "beta", "sector", "beta_sector", "beta_sector_size"]
SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]

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
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
    "save_artifacts": True,
}


def _subperiod_stats(outdir: str) -> tuple[float, float, float, float]:
    eq = pd.read_csv(Path(outdir) / "equity_curve.csv", parse_dates=["date"]).set_index("date").sort_index()
    r = pd.to_numeric(eq["returns"], errors="coerce").fillna(0.0)
    vals: list[float] = []
    for s, e in SUBPERIODS:
        rr = r.loc[(r.index >= pd.Timestamp(s)) & (r.index <= pd.Timestamp(e))]
        m = compute_metrics(rr)
        vals.append(float(m.get("Sharpe", float("nan"))))
    s = pd.Series(vals, dtype=float)
    return float(s.mean()), float(s.std(ddof=0)), float(s.min()), float(s.max())


def main() -> None:
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for i, mode in enumerate(MODES, start=1):
        cfg = dict(BASE_CONFIG)
        cfg["factor_neutralization"] = mode
        label = "none" if mode is None else str(mode)
        print(f"[{i}/{len(MODES)}] neutralization={label}")
        summary, outdir = run_backtest(**cfg, run_cache=run_cache)
        mean_sub, std_sub, min_sub, max_sub = _subperiod_stats(outdir=outdir)
        rows.append(
            {
                "FactorNeutralization": label,
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "MeanSubSharpe": mean_sub,
                "StdSubSharpe": std_sub,
                "MinSubSharpe": min_sub,
                "MaxSubSharpe": max_sub,
                "StabilityScore": float(mean_sub - 0.5 * std_sub),
                "Outdir": str(outdir),
            }
        )
        print(
            f"    Sharpe={rows[-1]['Sharpe']:.4f} CAGR={rows[-1]['CAGR']:.4f} "
            f"MaxDD={rows[-1]['MaxDD']:.4f}"
        )

    df = pd.DataFrame(rows)
    base = Path("results") / "factor_neutralization_test"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = base / f"factor_neutralization_sweep_{ts}.csv"
    json_path = base / f"factor_neutralization_sweep_{ts}.json"
    latest_csv = base / "factor_neutralization_sweep_latest.csv"
    latest_json = base / "factor_neutralization_sweep_latest.json"
    df.to_csv(csv_path, index=False, float_format="%.10g")
    df.to_csv(latest_csv, index=False, float_format="%.10g")
    payload = {
        "base_config": BASE_CONFIG,
        "modes": ["none", "beta", "sector", "beta_sector", "beta_sector_size"],
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nFACTOR NEUTRALIZATION SWEEP")
    print("---------------------------")
    print(
        df[
            [
                "FactorNeutralization",
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
