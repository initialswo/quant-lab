"""Sweep capital allocations across the four benchmark factor sleeves."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import run_composite_vs_sleeves as composite


RESULTS_ROOT = Path("results") / "sleeve_allocation_sweep"
DOC_LOG_PATH = Path("docs/experiment_log.md")
ALLOCATIONS: dict[str, list[float]] = {
    "equal": [0.25, 0.25, 0.25, 0.25],
    "rev_tilt_1": [0.20, 0.35, 0.15, 0.30],
    "rev_tilt_2": [0.15, 0.40, 0.10, 0.35],
    "defensive": [0.20, 0.25, 0.35, 0.20],
    "profit_tilt": [0.20, 0.25, 0.15, 0.40],
    "momentum_tilt": [0.35, 0.25, 0.15, 0.25],
}
SLEEVE_ORDER: list[str] = [
    "sleeve_momentum",
    "sleeve_reversal",
    "sleeve_low_vol",
    "sleeve_gross_profitability",
]


def _append_experiment_log(
    outdir: Path,
    metrics_df: pd.DataFrame,
    start: str,
    end: str,
    max_tickers: int,
    command: str,
    smoke_mode: bool,
) -> None:
    best = metrics_df.iloc[0]
    equal_row = metrics_df.set_index("Allocation").loc["equal"]
    prefix = "Sleeve Allocation Sweep" if not smoke_mode else "Sleeve Allocation Sweep (Smoke Validation)"
    entry = f"""

## {datetime.now(timezone.utc).strftime("%Y-%m-%d")} — {prefix}

### Objective
Test whether simple capital-allocation tilts across the four benchmark factor sleeves improve the equal-weight sleeve portfolio.

### Command
```bash
{command}
```

### Setup
- Universe: `{composite.UNIVERSE}`
- Universe mode: `{composite.UNIVERSE_MODE}`
- Dates: `{start}` to `{end}`
- Rebalance: `{composite.REBALANCE}`
- Top N: `{composite.TOP_N}`
- Weighting: `{composite.WEIGHTING}`
- Costs: `{composite.COSTS_BPS:.0f} bps`
- Max tickers: `{max_tickers}`
- Sleeves: `momentum_12_1`, `reversal_1m`, `low_vol_20`, `gross_profitability`

### Key Results
- Best allocation by Sharpe: `{best['Allocation']}` (`Sharpe={best['Sharpe']:.4f}`, `CAGR={best['CAGR']:.4f}`, `MaxDD={best['MaxDD']:.4f}`)
- Equal allocation baseline: `Sharpe={equal_row['Sharpe']:.4f}`, `CAGR={equal_row['CAGR']:.4f}`, `MaxDD={equal_row['MaxDD']:.4f}`
- Best minus equal delta:
  - CAGR: `{best['CAGR'] - equal_row['CAGR']:.4f}`
  - Sharpe: `{best['Sharpe'] - equal_row['Sharpe']:.4f}`
  - MaxDD: `{best['MaxDD'] - equal_row['MaxDD']:.4f}`
  - Turnover: `{best['Turnover'] - equal_row['Turnover']:.4f}`

### Initial Interpretation
- Allocation tilts {'improved' if best['Sharpe'] > equal_row['Sharpe'] else 'did not improve'} Sharpe versus equal sleeves in this run.
- Best result came from `{best['Allocation']}`.
- Results bundle: `{outdir}`
"""
    with DOC_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry.rstrip() + "\n")


def _build_manifest(
    outdir: Path,
    sleeve_manifest: list[dict[str, Any]],
    start: str,
    end: str,
    max_tickers: int,
) -> dict[str, Any]:
    return {
        "run_timestamp_utc": outdir.name,
        "results_dir": str(outdir),
        "date_range": {"start": start, "end": end},
        "universe": composite.UNIVERSE,
        "universe_mode": composite.UNIVERSE_MODE,
        "top_n": composite.TOP_N,
        "rebalance": composite.REBALANCE,
        "weighting": composite.WEIGHTING,
        "costs_bps": composite.COSTS_BPS,
        "max_tickers": max_tickers,
        "sleeve_order": SLEEVE_ORDER,
        "allocations": ALLOCATIONS,
        "notes": [
            "Portfolio returns are weighted sums of sleeve daily return series.",
            "Portfolio turnover is the matching weighted sum of sleeve daily turnover series.",
        ],
        "sleeve_runs": sleeve_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=composite.START)
    parser.add_argument("--end", default=composite.END)
    parser.add_argument("--max_tickers", type=int, default=composite.MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    parser.add_argument("--skip_log", action="store_true")
    args = parser.parse_args()
    command = "PYTHONPATH=src python scripts/run_sleeve_allocation_sweep.py"
    if str(args.start) != composite.START:
        command += f" --start {args.start}"
    if str(args.end) != composite.END:
        command += f" --end {args.end}"
    if int(args.max_tickers) != composite.MAX_TICKERS:
        command += f" --max_tickers {int(args.max_tickers)}"
    if str(args.results_root) != str(RESULTS_ROOT):
        command += f" --results_root {args.results_root}"
    if bool(args.skip_log):
        command += " --skip_log"

    composite.START = str(args.start)
    composite.END = str(args.end)
    composite.MAX_TICKERS = int(args.max_tickers)

    results_root = Path(str(args.results_root))
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / ts
    outdir.mkdir(parents=True, exist_ok=True)

    run_cache: dict[str, Any] = {}
    sleeve_payloads, sleeve_manifest = composite.run_sleeve_backtests(run_cache=run_cache)
    sleeve_payloads = sorted(sleeve_payloads, key=lambda p: SLEEVE_ORDER.index(str(p["name"])))

    portfolio_payloads: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for allocation_name, weights in ALLOCATIONS.items():
        payload = composite.weighted_sleeve_portfolio(
            sleeve_payloads=sleeve_payloads,
            weights=weights,
            name=allocation_name,
        )
        portfolio_payloads.append(payload)
        metrics_rows.append(
            composite._portfolio_metrics(
                name=allocation_name,
                daily_return=payload["daily_return"],
                turnover=payload["daily_turnover"],
            )
        )

    metrics_df = pd.DataFrame(metrics_rows).rename(columns={"Strategy": "Allocation"})
    metrics_df = metrics_df.sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(
        drop=True
    )
    daily_returns_df = (
        pd.concat([p["daily_return"] for p in portfolio_payloads], axis=1).sort_index().fillna(0.0)
    )

    metrics_path = outdir / "metrics.csv"
    daily_returns_path = outdir / "daily_returns.csv"
    manifest_path = outdir / "manifest.json"

    metrics_df.to_csv(metrics_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index_label="date", float_format="%.10g")
    manifest = _build_manifest(
        outdir=outdir,
        sleeve_manifest=sleeve_manifest,
        start=str(args.start),
        end=str(args.end),
        max_tickers=int(args.max_tickers),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    composite._copy_latest(
        files={
            "metrics.csv": metrics_path,
            "daily_returns.csv": daily_returns_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    display = metrics_df.copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover", "TotalReturn"]:
        display[col] = display[col].map(composite._format_float)
    print("SLEEVE ALLOCATION SWEEP")
    print("-----------------------")
    print(display[["Allocation", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover", "TotalReturn"]].to_string(index=False))
    print("")
    print(f"Saved results: {outdir}")

    if not bool(args.skip_log):
        _append_experiment_log(
            outdir=outdir,
            metrics_df=metrics_df,
            start=str(args.start),
            end=str(args.end),
            max_tickers=int(args.max_tickers),
            command=command,
            smoke_mode=bool(args.skip_log),
        )


if __name__ == "__main__":
    main()
