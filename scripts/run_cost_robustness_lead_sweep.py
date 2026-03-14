"""Cost/slippage robustness sweep for the current lead strategy."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest


COSTS_BPS = [5, 10, 20, 35, 50]
SLIPPAGE_BPS = [0, 5, 10, 20]
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


def _subperiod_sharpes(outdir: str) -> dict[str, float]:
    eq_path = Path(outdir) / "equity_curve.csv"
    eq = pd.read_csv(eq_path, parse_dates=["date"]).set_index("date").sort_index()
    r = pd.to_numeric(eq["returns"], errors="coerce").fillna(0.0)
    sharpe_map: dict[str, float] = {}
    for s, e in SUBPERIODS:
        rr = r.loc[(r.index >= pd.Timestamp(s)) & (r.index <= pd.Timestamp(e))]
        m = compute_metrics(rr)
        sharpe_map[f"{s}_{e}"] = float(m.get("Sharpe", float("nan")))
    return sharpe_map


def main() -> None:
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    combos = [(c, s) for c in COSTS_BPS for s in SLIPPAGE_BPS]
    total = len(combos)

    for i, (costs_bps, slippage_bps) in enumerate(combos, start=1):
        cfg = dict(BASE_CONFIG)
        cfg["costs_bps"] = float(costs_bps)
        cfg["slippage_bps"] = float(slippage_bps)
        cfg["slippage_vol_mult"] = 0.0
        summary, outdir = run_backtest(**cfg, run_cache=run_cache)
        sharpes = list(_subperiod_sharpes(outdir).values())
        s = pd.Series(sharpes, dtype=float)
        row = {
            "CostsBps": int(costs_bps),
            "SlippageBps": int(slippage_bps),
            "BaseReference": bool(int(costs_bps) == 10 and int(slippage_bps) == 0),
            "CAGR": float(summary.get("CAGR", float("nan"))),
            "Vol": float(summary.get("Vol", float("nan"))),
            "Sharpe": float(summary.get("Sharpe", float("nan"))),
            "MaxDD": float(summary.get("MaxDD", float("nan"))),
            "MeanSubSharpe": float(s.mean()),
            "StdSubSharpe": float(s.std(ddof=0)),
            "MinSubSharpe": float(s.min()),
            "MaxSubSharpe": float(s.max()),
            "StabilityScore": float(s.mean() - 0.5 * s.std(ddof=0)),
            "Outdir": str(outdir),
        }
        rows.append(row)
        print(
            f"[{i}/{total}] costs_bps={costs_bps} slippage_bps={slippage_bps} "
            f"Sharpe={row['Sharpe']:.4f} CAGR={row['CAGR']:.4f} MaxDD={row['MaxDD']:.4f}"
        )

    df = pd.DataFrame(rows).sort_values(["CostsBps", "SlippageBps"]).reset_index(drop=True)
    outdir = Path("results") / f"cost_robustness_lead_sweep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    full_csv = outdir / "cost_robustness_full.csv"
    summary_json = outdir / "cost_robustness_summary.json"
    df.to_csv(full_csv, index=False, float_format="%.10g")

    baseline = df.loc[(df["CostsBps"] == 10) & (df["SlippageBps"] == 0)].iloc[0].to_dict()
    payload = {
        "config": BASE_CONFIG,
        "costs_bps_grid": COSTS_BPS,
        "slippage_bps_grid": SLIPPAGE_BPS,
        "slippage_method": (
            "Total rebalance-day cost per unit turnover = "
            "(costs_bps + slippage_bps + slippage_vol_mult*weighted_asset_vol_bps)/10000. "
            "This sweep sets slippage_vol_mult=0 and varies slippage_bps only."
        ),
        "baseline_reference": baseline,
        "results": df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    disp_cols = [
        "CostsBps",
        "SlippageBps",
        "BaseReference",
        "CAGR",
        "Vol",
        "Sharpe",
        "MaxDD",
        "StabilityScore",
    ]
    print("\nCOST ROBUSTNESS TABLE")
    print("---------------------")
    print(df[disp_cols].to_string(index=False))
    print("")
    print(f"Saved: {full_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
