"""Compare factor aggregation methods for the lead configuration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest


METHODS = ["linear", "mean_rank", "geometric_rank"]
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
    "universe": "sp500",
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


def _stability(outdir: str) -> tuple[float, float, float, float, float]:
    eq = pd.read_csv(Path(outdir) / "equity_curve.csv", parse_dates=["date"]).set_index("date").sort_index()
    r = pd.to_numeric(eq["returns"], errors="coerce").fillna(0.0)
    vals: list[float] = []
    for s, e in SUBPERIODS:
        rr = r.loc[(r.index >= pd.Timestamp(s)) & (r.index <= pd.Timestamp(e))]
        m = compute_metrics(rr)
        vals.append(float(m.get("Sharpe", float("nan"))))
    s = pd.Series(vals, dtype=float)
    mean_sub = float(s.mean())
    std_sub = float(s.std(ddof=0))
    min_sub = float(s.min())
    max_sub = float(s.max())
    stability = float(mean_sub - 0.5 * std_sub)
    return mean_sub, std_sub, min_sub, max_sub, stability


def main() -> None:
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for i, method in enumerate(METHODS, start=1):
        cfg = dict(BASE_CONFIG)
        cfg["factor_aggregation_method"] = method
        print(f"[{i}/{len(METHODS)}] method={method}")
        summary, outdir = run_backtest(**cfg, run_cache=run_cache)
        mean_sub, std_sub, min_sub, max_sub, stability = _stability(outdir)
        rows.append(
            {
                "FactorAggregationMethod": method,
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "MeanSubSharpe": mean_sub,
                "StdSubSharpe": std_sub,
                "MinSubSharpe": min_sub,
                "MaxSubSharpe": max_sub,
                "StabilityScore": stability,
                "Outdir": str(outdir),
            }
        )
        print(
            f"    Sharpe={rows[-1]['Sharpe']:.4f} CAGR={rows[-1]['CAGR']:.4f} "
            f"MaxDD={rows[-1]['MaxDD']:.4f} Stability={rows[-1]['StabilityScore']:.4f}"
        )

    df = pd.DataFrame(rows)
    base = Path("results") / "rank_aggregation_test"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = base / "rank_aggregation_comparison.csv"
    summary_path = base / "rank_aggregation_summary.json"
    hist_csv = base / f"rank_aggregation_comparison_{ts}.csv"
    hist_json = base / f"rank_aggregation_summary_{ts}.json"

    df.to_csv(csv_path, index=False, float_format="%.10g")
    df.to_csv(hist_csv, index=False, float_format="%.10g")
    payload = {
        "base_config": BASE_CONFIG,
        "methods": METHODS,
        "subperiods": [{"start": s, "end": e} for s, e in SUBPERIODS],
        "results": rows,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    hist_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nRANK AGGREGATION COMPARISON")
    print("---------------------------")
    print(
        df[
            [
                "FactorAggregationMethod",
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
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
