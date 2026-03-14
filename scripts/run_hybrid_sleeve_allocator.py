"""Compare fixed, inverse-vol, and hybrid sleeve allocators."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_composite_vs_sleeves as composite
import run_volatility_balanced_sleeves as vol_bal


RESULTS_ROOT = Path("results") / "hybrid_sleeve_allocator"
VOL_WINDOW = 63
STRATEGY_ORDER: list[str] = [
    "rev_tilt_2",
    "volatility_balanced_sleeves",
    "hybrid_25",
    "hybrid_50",
    "hybrid_75",
]
HYBRID_DEFS: dict[str, float] = {
    "hybrid_25": 0.25,
    "hybrid_50": 0.50,
    "hybrid_75": 0.75,
}
FACTOR_LABELS: dict[str, str] = {
    "sleeve_momentum": "momentum_12_1",
    "sleeve_reversal": "reversal_1m",
    "sleeve_gross_profitability": "gross_profitability",
    "sleeve_low_vol": "low_vol_20",
}


def _fixed_weight_frame(index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    fixed = pd.Series(vol_bal.REV_TILT_2_WEIGHTS, index=vol_bal.SLEEVE_ORDER, dtype=float)
    return pd.DataFrame(
        np.tile(fixed.reindex(columns).to_numpy(dtype=float), (len(index), 1)),
        index=index,
        columns=columns,
        dtype=float,
    )


def _allocator_payload(
    name: str,
    sleeve_returns: pd.DataFrame,
    sleeve_turnover: pd.DataFrame,
    weights_daily: pd.DataFrame,
) -> dict[str, Any]:
    row_sums = weights_daily.sum(axis=1)
    if not np.allclose(row_sums.to_numpy(dtype=float), 1.0, atol=1e-8):
        raise ValueError(f"{name}: sleeve weights must sum to 1 each day")

    weights_prev = weights_daily.shift(1).fillna(0.0)
    gross_return = (weights_prev * sleeve_returns).sum(axis=1).rename(name)
    internal_turnover = (weights_prev * sleeve_turnover).sum(axis=1).rename(f"{name}_internal_turnover")
    allocator_turnover = (0.5 * weights_daily.diff().abs().sum(axis=1)).fillna(0.0).rename(
        f"{name}_allocator_turnover"
    )
    allocator_cost = allocator_turnover * (composite.COSTS_BPS / 10000.0)
    net_return = (gross_return - allocator_cost).rename(name)
    total_turnover = (internal_turnover + allocator_turnover).rename(name)
    equity = (1.0 + net_return).cumprod().rename("Equity")
    return {
        "name": name,
        "summary": {},
        "outdir": None,
        "equity": pd.DataFrame({"Equity": equity, "DailyReturn": net_return, "Turnover": total_turnover}),
        "holdings": None,
        "daily_return": net_return,
        "daily_turnover": total_turnover,
        "weights_daily": weights_daily,
    }


def _hybrid_weights(
    fixed_weights: pd.DataFrame,
    inv_vol_weights: pd.DataFrame,
    dynamic_share: float,
) -> pd.DataFrame:
    hybrid = ((1.0 - dynamic_share) * fixed_weights) + (dynamic_share * inv_vol_weights)
    return hybrid.div(hybrid.sum(axis=1), axis=0).astype(float)


def _build_manifest(
    outdir: Path,
    sleeve_manifest: list[dict[str, Any]],
    start: str,
    end: str,
    max_tickers: int,
    vol_window: int,
) -> dict[str, Any]:
    return {
        "run_timestamp_utc": outdir.name,
        "results_dir": str(outdir),
        "date_range": {"start": start, "end": end},
        "universe": composite.UNIVERSE,
        "universe_mode": composite.UNIVERSE_MODE,
        "rebalance": composite.REBALANCE,
        "top_n": composite.TOP_N,
        "weighting": composite.WEIGHTING,
        "costs_bps": composite.COSTS_BPS,
        "max_tickers": max_tickers,
        "vol_window_days": vol_window,
        "fixed_sleeve_weights": {
            "momentum_12_1": 0.15,
            "reversal_1m": 0.40,
            "low_vol_20": 0.10,
            "gross_profitability": 0.35,
        },
        "strategy_definitions": {
            "rev_tilt_2": {"fixed_share": 1.0, "dynamic_share": 0.0},
            "volatility_balanced_sleeves": {"fixed_share": 0.0, "dynamic_share": 1.0},
            "hybrid_25": {"fixed_share": 0.75, "dynamic_share": 0.25},
            "hybrid_50": {"fixed_share": 0.50, "dynamic_share": 0.50},
            "hybrid_75": {"fixed_share": 0.25, "dynamic_share": 0.75},
        },
        "notes": [
            "Sleeve returns are reused from the existing single-factor sleeve backtests.",
            "Inverse-vol weights use a 63-day rolling sleeve return volatility shifted by one trading day.",
            "Allocator turnover is charged at 10 bps; underlying sleeve returns already include stock-level trading costs.",
        ],
        "sleeve_runs": sleeve_manifest,
    }


def _print_interpretation(metrics_df: pd.DataFrame) -> None:
    by_name = metrics_df.set_index("Strategy")
    baseline = by_name.loc["rev_tilt_2"]
    best = metrics_df.iloc[0]
    hybrids = by_name.loc[list(HYBRID_DEFS.keys())]
    hybrid_best_sharpe = float(hybrids["Sharpe"].max())
    hybrid_best_drawdown = float(hybrids["MaxDD"].max())
    sharpe_note = (
        f"A hybrid beat rev_tilt_2 on Sharpe ({hybrid_best_sharpe:.4f} vs {baseline['Sharpe']:.4f})."
        if hybrid_best_sharpe > float(baseline["Sharpe"])
        else f"No hybrid beat rev_tilt_2 on Sharpe ({baseline['Sharpe']:.4f} remains best among fixed/hybrid candidates)."
    )
    dd_note = (
        f"Hybrids improved drawdown versus rev_tilt_2 (best hybrid MaxDD {hybrid_best_drawdown:.4f} vs {baseline['MaxDD']:.4f})."
        if hybrid_best_drawdown > float(baseline["MaxDD"])
        else f"Hybrids did not materially improve drawdown versus rev_tilt_2 ({baseline['MaxDD']:.4f})."
    )
    pure = by_name.loc["volatility_balanced_sleeves"]
    defensive_note = (
        f"Pure inverse-vol remained more defensive, with lower Vol and lower Sharpe than rev_tilt_2 ({pure['Vol']:.4f}/{pure['Sharpe']:.4f} vs {baseline['Vol']:.4f}/{baseline['Sharpe']:.4f})."
        if float(pure["Sharpe"]) < float(baseline["Sharpe"]) and float(pure["CAGR"]) < float(baseline["CAGR"])
        else "Pure inverse-vol did not look clearly too defensive in this run."
    )
    print("INTERPRETATION")
    print("--------------")
    print(sharpe_note)
    print(dd_note)
    print(defensive_note)
    print(f"Best Sharpe overall: {best['Strategy']} ({best['Sharpe']:.4f}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=composite.START)
    parser.add_argument("--end", default=composite.END)
    parser.add_argument("--max_tickers", type=int, default=composite.MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    parser.add_argument("--vol_window", type=int, default=VOL_WINDOW)
    args = parser.parse_args()

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
    sleeve_payloads = sorted(sleeve_payloads, key=lambda p: vol_bal.SLEEVE_ORDER.index(str(p["name"])))
    sleeve_returns = pd.concat([p["daily_return"] for p in sleeve_payloads], axis=1).sort_index().fillna(0.0)
    sleeve_turnover = pd.concat([p["daily_turnover"] for p in sleeve_payloads], axis=1).sort_index().fillna(0.0)

    fixed_weights = _fixed_weight_frame(index=sleeve_returns.index, columns=sleeve_returns.columns)
    inv_vol_weights = vol_bal._inverse_vol_weights(sleeve_returns=sleeve_returns, window=int(args.vol_window))

    payloads: list[dict[str, Any]] = [
        composite.weighted_sleeve_portfolio(
            sleeve_payloads=sleeve_payloads,
            weights=vol_bal.REV_TILT_2_WEIGHTS,
            name="rev_tilt_2",
        ),
        _allocator_payload(
            name="volatility_balanced_sleeves",
            sleeve_returns=sleeve_returns,
            sleeve_turnover=sleeve_turnover,
            weights_daily=inv_vol_weights,
        ),
    ]
    payloads.extend(
        _allocator_payload(
            name=name,
            sleeve_returns=sleeve_returns,
            sleeve_turnover=sleeve_turnover,
            weights_daily=_hybrid_weights(
                fixed_weights=fixed_weights,
                inv_vol_weights=inv_vol_weights,
                dynamic_share=dynamic_share,
            ),
        )
        for name, dynamic_share in HYBRID_DEFS.items()
    )

    metrics_df = pd.DataFrame(
        [
            composite._portfolio_metrics(
                name=payload["name"],
                daily_return=payload["daily_return"],
                turnover=payload["daily_turnover"],
            )
            for payload in payloads
        ]
    ).sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(drop=True)

    daily_returns_df = pd.concat([p["daily_return"] for p in payloads], axis=1).sort_index().fillna(0.0)
    weights_export = pd.concat(
        [
            fixed_weights.rename(columns=FACTOR_LABELS).add_prefix("rev_tilt_2__"),
            *[
                payload["weights_daily"].rename(columns=FACTOR_LABELS).add_prefix(f"{payload['name']}__")
                for payload in payloads
                if "weights_daily" in payload
            ],
        ],
        axis=1,
    ).sort_index()

    metrics_path = outdir / "hybrid_allocator_results.csv"
    daily_returns_path = outdir / "daily_returns.csv"
    sleeve_weights_path = outdir / "sleeve_weights.csv"
    manifest_path = outdir / "manifest.json"

    metrics_df.to_csv(metrics_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index_label="date", float_format="%.10g")
    weights_export.to_csv(sleeve_weights_path, index_label="date", float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                outdir=outdir,
                sleeve_manifest=sleeve_manifest,
                start=str(args.start),
                end=str(args.end),
                max_tickers=int(args.max_tickers),
                vol_window=int(args.vol_window),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    composite._copy_latest(
        files={
            "hybrid_allocator_results.csv": metrics_path,
            "daily_returns.csv": daily_returns_path,
            "sleeve_weights.csv": sleeve_weights_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    display = metrics_df.copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover", "TotalReturn"]:
        display[col] = display[col].map(composite._format_float)
    print("HYBRID SLEEVE ALLOCATOR TEST")
    print("----------------------------")
    print(display[["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover", "TotalReturn"]].to_string(index=False))
    print("")
    _print_interpretation(metrics_df)
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
