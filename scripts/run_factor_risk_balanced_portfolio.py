"""Run a risk-balanced version of the six-factor benchmark portfolio."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_composite_vs_sleeves as composite
import run_factor_benchmark_portfolio as benchmark


RESULTS_ROOT = Path("results") / "factor_risk_balanced"


def _risk_balanced_payload(sleeve_payloads: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.Series]:
    returns_df = pd.concat([p["daily_return"] for p in sleeve_payloads], axis=1).sort_index().fillna(0.0)
    vol = returns_df.std(ddof=0).replace(0.0, np.nan)
    inv_vol = 1.0 / vol
    weights = (inv_vol / float(inv_vol.sum())).astype(float)

    payload = composite.weighted_sleeve_portfolio(
        sleeve_payloads=sleeve_payloads,
        weights=weights.reindex(returns_df.columns).to_numpy(dtype=float),
        name="factor_benchmark_risk_balanced",
    )
    return payload, weights


def _build_manifest(
    outdir: Path,
    sleeve_manifest: list[dict[str, Any]],
    risk_balanced_weights: pd.Series,
    max_tickers: int,
) -> dict[str, Any]:
    equal_weight = float(1.0 / len(benchmark.FACTOR_NAMES))
    return {
        "timestamp_utc": outdir.name,
        "results_dir": str(outdir),
        "date_range": {"start": benchmark.START, "end": benchmark.END},
        "universe": "liquid_us",
        "universe_mode": "dynamic",
        "rebalance": "weekly",
        "top_n": 50,
        "weighting": "equal",
        "costs_bps": 10.0,
        "max_tickers": max_tickers,
        "factors": benchmark.FACTOR_NAMES,
        "portfolio_definitions": {
            "factor_benchmark_equal_weight": {
                "weights": {factor: equal_weight for factor in benchmark.FACTOR_NAMES},
                "construction": "return-level equal-weight sleeve combination",
            },
            "factor_benchmark_risk_balanced": {
                "weights": {
                    str(name).replace("sleeve_", ""): float(weight) for name, weight in risk_balanced_weights.items()
                },
                "construction": "return-level inverse full-sample sleeve volatility weights",
            },
        },
        "notes": [
            "Sleeve definitions match run_factor_benchmark_portfolio.py.",
            "Risk-balanced weights are computed as inverse full-sample daily sleeve volatility and then normalized.",
            "Weights are static after estimation; no allocator turnover or dynamic rebalancing is applied.",
        ],
        "sleeve_runs": sleeve_manifest,
    }


def _print_interpretation(metrics_df: pd.DataFrame) -> None:
    by_name = metrics_df.set_index("Strategy")
    equal_weight = by_name.loc["factor_benchmark_equal_weight"]
    risk_bal = by_name.loc["factor_benchmark_risk_balanced"]
    sharpe_note = (
        f"Risk balancing improved Sharpe ({risk_bal['Sharpe']:.4f} vs {equal_weight['Sharpe']:.4f})."
        if float(risk_bal["Sharpe"]) > float(equal_weight["Sharpe"])
        else f"Risk balancing did not improve Sharpe ({risk_bal['Sharpe']:.4f} vs {equal_weight['Sharpe']:.4f})."
    )
    dd_note = (
        f"Drawdown improved ({risk_bal['MaxDD']:.4f} vs {equal_weight['MaxDD']:.4f})."
        if float(risk_bal["MaxDD"]) > float(equal_weight["MaxDD"])
        else f"Drawdown did not improve ({risk_bal['MaxDD']:.4f} vs {equal_weight['MaxDD']:.4f})."
    )
    print("INTERPRETATION")
    print("--------------")
    print(sharpe_note)
    print(dd_note)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=benchmark.START)
    parser.add_argument("--end", default=benchmark.END)
    parser.add_argument("--max_tickers", type=int, default=benchmark.MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    benchmark.START = str(args.start)
    benchmark.END = str(args.end)
    benchmark.MAX_TICKERS = int(args.max_tickers)

    results_root = Path(str(args.results_root))
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / ts
    outdir.mkdir(parents=True, exist_ok=True)

    special_factor_params = benchmark._build_special_factor_params(
        start=benchmark.START,
        end=benchmark.END,
        max_tickers=benchmark.MAX_TICKERS,
    )
    sleeve_payloads, sleeve_manifest = benchmark._run_sleeves(special_factor_params=special_factor_params)

    equal_payload = benchmark._combined_payload(sleeve_payloads=sleeve_payloads)
    risk_payload, risk_weights = _risk_balanced_payload(sleeve_payloads=sleeve_payloads)
    payloads = [equal_payload, risk_payload]

    metrics_df = pd.DataFrame(
        [
            composite._portfolio_metrics(
                name=payload["name"],
                daily_return=payload["daily_return"],
                turnover=payload["daily_turnover"],
            )
            for payload in payloads
        ]
    ).set_index("Strategy").loc[
        ["factor_benchmark_equal_weight", "factor_benchmark_risk_balanced"]
    ].reset_index()

    daily_returns_df = pd.concat([p["daily_return"] for p in payloads], axis=1).sort_index().fillna(0.0)

    results_path = outdir / "risk_balanced_results.csv"
    daily_returns_path = outdir / "daily_returns.csv"
    manifest_path = outdir / "manifest.json"

    metrics_df.to_csv(results_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index_label="date", float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                outdir=outdir,
                sleeve_manifest=sleeve_manifest,
                risk_balanced_weights=risk_weights,
                max_tickers=benchmark.MAX_TICKERS,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    composite._copy_latest(
        files={
            "risk_balanced_results.csv": results_path,
            "daily_returns.csv": daily_returns_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    table = metrics_df[["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]].copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]:
        table[col] = table[col].map(composite._format_float)
    print("FACTOR RISK-BALANCED TEST")
    print("-------------------------")
    print("")
    print(table.to_string(index=False))
    print("")
    _print_interpretation(metrics_df)
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
