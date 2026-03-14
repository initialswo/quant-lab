"""Strict out-of-sample walk-forward validation for the lead strategy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest


TRAIN_START = "2005-01-01"
WINDOWS: list[tuple[str, str, str, str]] = [
    ("2005-01-01", "2012-12-31", "2013-01-01", "2014-12-31"),
    ("2005-01-01", "2014-12-31", "2015-01-01", "2016-12-31"),
    ("2005-01-01", "2016-12-31", "2017-01-01", "2018-12-31"),
    ("2005-01-01", "2018-12-31", "2019-01-01", "2020-12-31"),
    ("2005-01-01", "2020-12-31", "2021-01-01", "2022-12-31"),
    ("2005-01-01", "2022-12-31", "2023-01-01", "2024-12-31"),
]

BASE_CONFIG: dict[str, Any] = {
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


def main() -> None:
    outdir = Path("results") / "walkforward_lead_strategy"
    outdir.mkdir(parents=True, exist_ok=True)
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for train_start, train_end, test_start, test_end in WINDOWS:
        cfg = dict(BASE_CONFIG)
        cfg["start"] = TRAIN_START
        cfg["end"] = test_end
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)

        eq = pd.read_csv(Path(run_outdir) / "equity_curve.csv", parse_dates=["date"])
        eq = eq.set_index("date").sort_index()
        test_ret = eq.loc[(eq.index >= pd.Timestamp(test_start)) & (eq.index <= pd.Timestamp(test_end)), "returns"]
        m = compute_metrics(test_ret)

        rows.append(
            {
                "TrainStart": train_start,
                "TrainEnd": train_end,
                "TestStart": test_start,
                "TestEnd": test_end,
                "Obs": int(test_ret.dropna().shape[0]),
                "CAGR": float(m.get("CAGR", float("nan"))),
                "Vol": float(m.get("Vol", float("nan"))),
                "Sharpe": float(m.get("Sharpe", float("nan"))),
                "MaxDD": float(m.get("MaxDD", float("nan"))),
                "RunOutdir": str(run_outdir),
                "FullRunSharpe": float(summary.get("Sharpe", float("nan"))),
            }
        )

    results = pd.DataFrame(rows)
    results_path = outdir / "walkforward_results.csv"
    results.to_csv(results_path, index=False, float_format="%.10g")

    sharpe = pd.to_numeric(results["Sharpe"], errors="coerce")
    summary_payload = {
        "config": BASE_CONFIG,
        "train_start_anchor": TRAIN_START,
        "windows": rows,
        "oos_summary": {
            "avg_oos_sharpe": float(sharpe.mean()),
            "median_oos_sharpe": float(sharpe.median()),
            "worst_window_sharpe": float(sharpe.min()),
            "best_window_sharpe": float(sharpe.max()),
            "window_count": int(sharpe.notna().sum()),
        },
    }
    summary_path = outdir / "walkforward_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("WALKFORWARD OOS SUMMARY")
    print("-----------------------")
    print(results[["TrainEnd", "TestStart", "TestEnd", "CAGR", "Vol", "Sharpe", "MaxDD"]].to_string(index=False))
    print("")
    print(f"Average OOS Sharpe: {summary_payload['oos_summary']['avg_oos_sharpe']:.4f}")
    print(f"Median OOS Sharpe:  {summary_payload['oos_summary']['median_oos_sharpe']:.4f}")
    print(f"Worst OOS Sharpe:   {summary_payload['oos_summary']['worst_window_sharpe']:.4f}")
    print(f"Best OOS Sharpe:    {summary_payload['oos_summary']['best_window_sharpe']:.4f}")
    print("")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

