"""Test sleeve allocation stability across fixed subperiods."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import run_composite_vs_sleeves as composite
from quant_lab.engine.runner import run_backtest


RESULTS_ROOT = Path("results") / "sleeve_subperiod_test"
SUBPERIODS: list[tuple[str, str]] = [
    ("2000-01-01", "2009-12-31"),
    ("2010-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]
ALLOCATIONS: dict[str, list[float]] = {
    "equal": [0.25, 0.25, 0.25, 0.25],
    "rev_tilt_2": [0.15, 0.40, 0.10, 0.35],
}
SLEEVE_ORDER: list[str] = [
    "sleeve_momentum",
    "sleeve_reversal",
    "sleeve_low_vol",
    "sleeve_gross_profitability",
]


def _composite_run_config(start: str, end: str) -> dict[str, Any]:
    spec = next(
        item for item in composite.STRATEGY_SPECS if str(item["strategy"]) == "composite_benchmark"
    )
    cfg = composite._run_config(spec)
    cfg["start"] = str(start)
    cfg["end"] = str(end)
    return cfg


def _run_period(
    start: str,
    end: str,
    max_tickers: int,
    run_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    original_start = composite.START
    original_end = composite.END
    original_max_tickers = composite.MAX_TICKERS
    try:
        composite.START = str(start)
        composite.END = str(end)
        composite.MAX_TICKERS = int(max_tickers)

        summary, outdir = run_backtest(**_composite_run_config(start=start, end=end), run_cache=run_cache)
        composite_payload = composite._strategy_payload(
            name="composite_benchmark",
            summary=summary,
            outdir=outdir,
        )

        sleeve_payloads, _ = composite.run_sleeve_backtests(run_cache=run_cache)
        sleeve_payloads = sorted(sleeve_payloads, key=lambda p: SLEEVE_ORDER.index(str(p["name"])))

        rows: list[dict[str, Any]] = []
        rows.append(
            composite._portfolio_metrics(
                name="composite_benchmark",
                daily_return=composite_payload["daily_return"],
                turnover=composite_payload["daily_turnover"],
            )
        )
        for name, weights in ALLOCATIONS.items():
            payload = composite.weighted_sleeve_portfolio(
                sleeve_payloads=sleeve_payloads,
                weights=weights,
                name=name,
            )
            row = composite._portfolio_metrics(
                name=name,
                daily_return=payload["daily_return"],
                turnover=payload["daily_turnover"],
            )
            rows.append(row)
        return rows
    finally:
        composite.START = original_start
        composite.END = original_end
        composite.MAX_TICKERS = original_max_tickers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_tickers", type=int, default=composite.MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    results_root = Path(str(args.results_root))
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / ts
    outdir.mkdir(parents=True, exist_ok=True)

    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for start, end in SUBPERIODS:
        period_label = f"{start} -> {end}"
        print(f"Running {period_label}...")
        for row in _run_period(
            start=start,
            end=end,
            max_tickers=int(args.max_tickers),
            run_cache=run_cache,
        ):
            rows.append(
                {
                    "Period": period_label,
                    "Start": start,
                    "End": end,
                    "Strategy": str(row["Strategy"]),
                    "CAGR": float(row["CAGR"]),
                    "Vol": float(row["Vol"]),
                    "Sharpe": float(row["Sharpe"]),
                    "MaxDD": float(row["MaxDD"]),
                    "TotalReturn": float(row["TotalReturn"]),
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df.sort_values(
        ["Start", "Strategy"],
        kind="mergesort",
    ).reset_index(drop=True)

    metrics_path = outdir / "metrics_by_period.csv"
    manifest_path = outdir / "manifest.json"
    metrics_df.to_csv(metrics_path, index=False, float_format="%.10g")
    manifest = {
        "run_timestamp_utc": ts,
        "results_dir": str(outdir),
        "universe": composite.UNIVERSE,
        "universe_mode": composite.UNIVERSE_MODE,
        "top_n": composite.TOP_N,
        "rebalance": composite.REBALANCE,
        "weighting": composite.WEIGHTING,
        "costs_bps": composite.COSTS_BPS,
        "max_tickers": int(args.max_tickers),
        "subperiods": [{"start": s, "end": e} for s, e in SUBPERIODS],
        "allocations": ALLOCATIONS,
        "notes": [
            "equal and rev_tilt_2 are return-level sleeve combinations.",
            "composite_benchmark uses the existing equal-weight four-factor composite strategy.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    composite._copy_latest(
        files={
            "metrics_by_period.csv": metrics_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    table = metrics_df[["Period", "Strategy", "CAGR", "Vol", "Sharpe", "MaxDD"]].copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD"]:
        table[col] = table[col].map(composite._format_float)
    print("SLEEVE SUBPERIOD TEST")
    print("---------------------")
    print(table.to_string(index=False))
    print("")
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
