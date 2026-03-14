"""Compare a fixed sleeve blend against a volatility-balanced sleeve allocator."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_composite_vs_sleeves as composite


RESULTS_ROOT = Path("results") / "volatility_balanced_sleeves"
VOL_WINDOW = 63
SLEEVE_ORDER: list[str] = [
    "sleeve_momentum",
    "sleeve_reversal",
    "sleeve_gross_profitability",
    "sleeve_low_vol",
]
REV_TILT_2_WEIGHTS = [0.15, 0.40, 0.35, 0.10]


def _rebalance_mask(index: pd.Index) -> pd.Series:
    rb = composite.rebalance_mask(pd.DatetimeIndex(index), composite.REBALANCE)
    return pd.Series(rb, index=index, dtype=bool)


def _equal_weight_row(columns: pd.Index) -> pd.Series:
    return pd.Series(1.0 / len(columns), index=columns, dtype=float)


def _inverse_vol_weights(
    sleeve_returns: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    vol = sleeve_returns.rolling(window=window, min_periods=window).std(ddof=0).shift(1)
    rb_mask = _rebalance_mask(sleeve_returns.index)
    target = pd.DataFrame(np.nan, index=sleeve_returns.index, columns=sleeve_returns.columns, dtype=float)
    inv_vol = 1.0 / vol.replace(0.0, np.nan)
    target.loc[rb_mask] = inv_vol.loc[rb_mask]

    eq = _equal_weight_row(target.columns)
    row_sums = target.sum(axis=1, min_count=1)
    valid = row_sums > 0.0
    target = target.div(row_sums.where(valid), axis=0)
    target.loc[rb_mask & ~valid] = eq.to_numpy(dtype=float)

    first_idx = target.index[0]
    if pd.isna(target.loc[first_idx]).all():
        target.loc[first_idx] = eq.to_numpy(dtype=float)

    return target.ffill().fillna(eq.to_dict()).astype(float)


def _vol_balanced_payload(
    sleeve_payloads: list[dict[str, Any]],
    window: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ordered = sorted(sleeve_payloads, key=lambda p: SLEEVE_ORDER.index(str(p["name"])))
    returns_df = pd.concat([p["daily_return"] for p in ordered], axis=1).sort_index().fillna(0.0)
    turnover_df = pd.concat([p["daily_turnover"] for p in ordered], axis=1).sort_index().fillna(0.0)
    weights_daily = _inverse_vol_weights(returns_df, window=window)

    weights_prev = weights_daily.shift(1).fillna(0.0)
    combined_return = (weights_prev * returns_df).sum(axis=1).rename("volatility_balanced_sleeves")
    internal_turnover = (weights_prev * turnover_df).sum(axis=1).rename("internal_turnover")
    allocator_turnover = (0.5 * weights_daily.diff().abs().sum(axis=1)).fillna(0.0).rename("allocator_turnover")
    allocator_cost = allocator_turnover * (composite.COSTS_BPS / 10000.0)
    net_return = (combined_return - allocator_cost).rename("volatility_balanced_sleeves")
    total_turnover = (internal_turnover + allocator_turnover).rename("volatility_balanced_sleeves")
    equity = (1.0 + net_return).cumprod().rename("Equity")

    payload = {
        "name": "volatility_balanced_sleeves",
        "summary": {},
        "outdir": None,
        "equity": pd.DataFrame(
            {
                "Equity": equity,
                "DailyReturn": net_return,
                "Turnover": total_turnover,
            }
        ),
        "holdings": None,
        "daily_return": net_return,
        "daily_turnover": total_turnover,
    }
    weights_out = weights_daily.rename(
        columns={
            "sleeve_momentum": "momentum_12_1",
            "sleeve_reversal": "reversal_1m",
            "sleeve_gross_profitability": "gross_profitability",
            "sleeve_low_vol": "low_vol_20",
        }
    )
    return payload, weights_out


def _manifest(
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
        "top_n": composite.TOP_N,
        "rebalance": composite.REBALANCE,
        "weighting": composite.WEIGHTING,
        "costs_bps": composite.COSTS_BPS,
        "max_tickers": max_tickers,
        "sleeves": [
            "momentum_12_1",
            "reversal_1m",
            "gross_profitability",
            "low_vol_20",
        ],
        "rev_tilt_2_weights": {
            "momentum_12_1": REV_TILT_2_WEIGHTS[0],
            "reversal_1m": REV_TILT_2_WEIGHTS[1],
            "gross_profitability": REV_TILT_2_WEIGHTS[2],
            "low_vol_20": REV_TILT_2_WEIGHTS[3],
        },
        "volatility_balanced": {
            "vol_window_days": vol_window,
            "weight_formula": "inverse trailing sleeve volatility",
            "weight_timing": "rolling std shifted by one trading day and rebalanced weekly",
            "allocator_cost_bps": composite.COSTS_BPS,
        },
        "notes": [
            "Sleeve returns come from the existing single-factor sleeve backtests.",
            "rev_tilt_2 is a fixed return-level blend of sleeve daily returns.",
            "volatility_balanced_sleeves uses dynamic inverse-vol sleeve weights and applies allocator turnover costs at 10 bps.",
            "Underlying sleeve returns already include their own stock-level transaction costs.",
        ],
        "sleeve_runs": sleeve_manifest,
    }


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
    sleeve_payloads = sorted(sleeve_payloads, key=lambda p: SLEEVE_ORDER.index(str(p["name"])))

    fixed_payload = composite.weighted_sleeve_portfolio(
        sleeve_payloads=sleeve_payloads,
        weights=REV_TILT_2_WEIGHTS,
        name="rev_tilt_2",
    )
    vol_payload, sleeve_weights = _vol_balanced_payload(sleeve_payloads=sleeve_payloads, window=int(args.vol_window))

    payloads = [fixed_payload, vol_payload]
    metrics_rows = [
        composite._portfolio_metrics(
            name=payload["name"],
            daily_return=payload["daily_return"],
            turnover=payload["daily_turnover"],
        )
        for payload in payloads
    ]
    metrics_df = pd.DataFrame(metrics_rows).set_index("Strategy").loc[
        ["rev_tilt_2", "volatility_balanced_sleeves"]
    ].reset_index()
    daily_returns_df = pd.concat([p["daily_return"] for p in payloads], axis=1).sort_index().fillna(0.0)

    metrics_path = outdir / "vol_balanced_results.csv"
    manifest_path = outdir / "manifest.json"
    daily_returns_path = outdir / "daily_returns.csv"
    sleeve_weights_path = outdir / "sleeve_weights.csv"

    metrics_df.to_csv(metrics_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index_label="date", float_format="%.10g")
    sleeve_weights.to_csv(sleeve_weights_path, index_label="date", float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            _manifest(
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
            "vol_balanced_results.csv": metrics_path,
            "manifest.json": manifest_path,
            "daily_returns.csv": daily_returns_path,
            "sleeve_weights.csv": sleeve_weights_path,
        },
        latest_root=results_root / "latest",
    )

    display = metrics_df.copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover", "TotalReturn"]:
        display[col] = display[col].map(composite._format_float)
    print("VOLATILITY BALANCED SLEEVE TEST")
    print("--------------------------------")
    print(display[["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover", "TotalReturn"]].to_string(index=False))
    print("")
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
