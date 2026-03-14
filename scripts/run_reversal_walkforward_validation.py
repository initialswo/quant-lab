"""Strict expanding-window OOS validation for reversal Strategy 3 candidate."""

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

# Candidate from reversal experiment grid:
# factor=reversal_1m, top_n=50, rank_buffer=20, rebalance=weekly, execution_delay_days=0
BASE_CONFIG: dict[str, Any] = {
    "max_tickers": 2000,
    "top_n": 50,
    "rank_buffer": 20,
    "rebalance": "weekly",
    "execution_delay_days": 0,
    "costs_bps": 10.0,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "factor_name": ["reversal_1m"],
    "factor_names": ["reversal_1m"],
    "factor_weights": [1.0],
    "factor_aggregation_method": "geometric_rank",
    "dynamic_factor_weights": True,
    "target_vol": 0.14,
    "port_vol_lookback": 20,
    "max_leverage": 1.5,
    "regime_filter": False,
    "regime_benchmark": "SPY",
    "bear_exposure_scale": 1.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
    "save_artifacts": True,
}


def main() -> None:
    outdir = Path("results") / "reversal_walkforward_validation"
    outdir.mkdir(parents=True, exist_ok=True)

    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(WINDOWS, start=1):
        cfg = dict(BASE_CONFIG)
        cfg["start"] = TRAIN_START
        cfg["end"] = test_end

        print(f"[{i}/{len(WINDOWS)}] Train {train_start}..{train_end} -> Test {test_start}..{test_end}")
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)

        eq = pd.read_csv(Path(run_outdir) / "equity_curve.csv", parse_dates=["date"]).set_index("date").sort_index()
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
    oos_summary = {
        "avg_oos_sharpe": float(sharpe.mean()),
        "median_oos_sharpe": float(sharpe.median()),
        "worst_window_sharpe": float(sharpe.min()),
        "best_window_sharpe": float(sharpe.max()),
        "window_count": int(sharpe.notna().sum()),
    }

    summary_payload = {
        "config": BASE_CONFIG,
        "train_start_anchor": TRAIN_START,
        "windows": rows,
        "oos_summary": oos_summary,
    }
    summary_path = outdir / "walkforward_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    cols = ["TrainEnd", "TestStart", "TestEnd", "CAGR", "Vol", "Sharpe", "MaxDD"]
    display_df = results[cols].copy()
    print("\nREVERSAL WALKFORWARD OOS RESULTS")
    print("--------------------------------")
    print(display_df.to_string(index=False))
    print("\nOOS SHARPE SUMMARY")
    print("------------------")
    print(f"Average: {oos_summary['avg_oos_sharpe']:.4f}")
    print(f"Median:  {oos_summary['median_oos_sharpe']:.4f}")
    print(f"Worst:   {oos_summary['worst_window_sharpe']:.4f}")
    print(f"Best:    {oos_summary['best_window_sharpe']:.4f}")
    print("\nSaved artifacts:")
    print(f"- {results_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
