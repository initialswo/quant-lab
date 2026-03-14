"""Compare the static rev_tilt_2 sleeve mix against a dynamic sleeve allocator."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_composite_vs_sleeves as composite
from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover


RESULTS_ROOT = Path("results") / "rev_tilt_2_dynamic_allocator"
START = "2000-01-01"
END = "2024-12-31"
MAX_TICKERS = 300
SHARPE_WINDOW = 63
MIN_WEIGHT = 0.10
MAX_WEIGHT = 0.40

SLEEVE_SPECS: list[dict[str, Any]] = [
    {
        "strategy_name": "sleeve_reversal",
        "factor_name": "reversal_1m",
        "factor_params": {},
        "notes": "Standalone reversal sleeve.",
    },
    {
        "strategy_name": "sleeve_gross_profitability",
        "factor_name": "gross_profitability",
        "factor_params": {},
        "notes": "Standalone profitability sleeve.",
    },
    {
        "strategy_name": "sleeve_momentum",
        "factor_name": "momentum_12_1",
        "factor_params": {},
        "notes": "Standalone momentum sleeve.",
    },
    {
        "strategy_name": "sleeve_low_vol",
        "factor_name": "low_vol_20",
        "factor_params": {},
        "notes": "Requested volatility_20 ascending sleeve implemented with existing low_vol_20 factor.",
    },
]

ORIGINAL_WEIGHTS: dict[str, float] = {
    "sleeve_reversal": 0.40,
    "sleeve_gross_profitability": 0.30,
    "sleeve_momentum": 0.20,
    "sleeve_low_vol": 0.10,
}

SLEEVE_LABELS: dict[str, str] = {
    "sleeve_reversal": "reversal_1m",
    "sleeve_gross_profitability": "gross_profitability",
    "sleeve_momentum": "momentum_12_1",
    "sleeve_low_vol": "volatility_20",
}


def _base_config() -> dict[str, Any]:
    return {
        "start": START,
        "end": END,
        "universe": composite.UNIVERSE,
        "universe_mode": composite.UNIVERSE_MODE,
        "top_n": 50,
        "rebalance": "weekly",
        "weighting": composite.WEIGHTING,
        "costs_bps": composite.COSTS_BPS,
        "max_tickers": MAX_TICKERS,
        "data_source": composite.DATA_SOURCE,
        "data_cache_dir": composite.DATA_CACHE_DIR,
        "fundamentals_path": composite.FUNDAMENTALS_PATH,
        "fundamentals_fallback_lag_days": composite.FUNDAMENTALS_FALLBACK_LAG_DAYS,
        "save_artifacts": True,
    }


def _run_config(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(_base_config())
    factor_name = str(spec["factor_name"])
    cfg["factor_name"] = [factor_name]
    cfg["factor_names"] = [factor_name]
    cfg["factor_weights"] = [1.0]
    if spec["factor_params"]:
        cfg["factor_params"] = {factor_name: dict(spec["factor_params"])}
    return cfg


def _run_sleeves() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_cache: dict[str, Any] = {}
    payloads: list[dict[str, Any]] = []
    manifest_runs: list[dict[str, Any]] = []

    for spec in SLEEVE_SPECS:
        strategy_name = str(spec["strategy_name"])
        print(f"Running {strategy_name}...")
        summary, run_outdir = run_backtest(**_run_config(spec), run_cache=run_cache)
        payload = composite._strategy_payload(name=strategy_name, summary=summary, outdir=run_outdir)
        payloads.append(payload)
        manifest_runs.append(
            {
                "strategy_name": strategy_name,
                "factor_name": str(spec["factor_name"]),
                "notes": str(spec["notes"]),
                "annual_turnover_from_summary": extract_annual_turnover(summary=summary, outdir=run_outdir),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )
    return payloads, manifest_runs


def _softmax(scores: pd.Series) -> pd.Series:
    values = pd.to_numeric(scores, errors="coerce").astype(float)
    centered = values - float(values.max())
    exp_vals = np.exp(centered)
    weights = exp_vals / float(exp_vals.sum())
    return pd.Series(weights, index=values.index, dtype=float)


def _bounded_rescale(
    raw_weights: pd.Series,
    min_weight: float,
    max_weight: float,
) -> pd.Series:
    weights = pd.to_numeric(raw_weights, errors="coerce").astype(float).copy()
    n = len(weights)
    if n == 0:
        raise ValueError("raw_weights must be non-empty")
    if n * min_weight > 1.0 + 1e-12:
        raise ValueError("min_weight is infeasible for the number of sleeves")
    if n * max_weight < 1.0 - 1e-12:
        raise ValueError("max_weight is infeasible for the number of sleeves")

    weights = weights / float(weights.sum())
    free = pd.Series(True, index=weights.index, dtype=bool)
    bounded = pd.Series(np.nan, index=weights.index, dtype=float)
    remaining = 1.0

    while bool(free.any()):
        free_weights = weights.loc[free]
        scaled = free_weights / float(free_weights.sum()) * remaining
        low = scaled < (min_weight - 1e-12)
        high = scaled > (max_weight + 1e-12)
        if not bool(low.any() or high.any()):
            bounded.loc[free] = scaled
            break
        if bool(low.any()):
            idx = low.index[low]
            bounded.loc[idx] = min_weight
            free.loc[idx] = False
            remaining -= float(min_weight * len(idx))
        if bool(high.any()):
            idx = high.index[high]
            bounded.loc[idx] = max_weight
            free.loc[idx] = False
            remaining -= float(max_weight * len(idx))
        if remaining < -1e-12:
            raise ValueError("weight bounds left negative remaining mass")

    bounded = bounded.fillna(0.0)
    bounded = bounded / float(bounded.sum())
    return bounded.astype(float)


def _dynamic_weights(
    sleeve_returns: pd.DataFrame,
    fallback_weights: pd.Series,
    window: int,
    min_weight: float,
    max_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rolling_mean = sleeve_returns.rolling(window=window, min_periods=window).mean()
    rolling_std = sleeve_returns.rolling(window=window, min_periods=window).std(ddof=0)
    trailing_sharpe = (rolling_mean / rolling_std.replace(0.0, np.nan)) * np.sqrt(252.0)
    scores = trailing_sharpe.shift(1)

    rb_mask = pd.Series(
        composite.rebalance_mask(pd.DatetimeIndex(sleeve_returns.index), "weekly"),
        index=sleeve_returns.index,
        dtype=bool,
    )
    target = pd.DataFrame(np.nan, index=sleeve_returns.index, columns=sleeve_returns.columns, dtype=float)

    for dt in sleeve_returns.index[rb_mask]:
        row = pd.to_numeric(scores.loc[dt], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if row.notna().sum() == len(row):
            raw = _softmax(row)
            target.loc[dt] = _bounded_rescale(raw, min_weight=min_weight, max_weight=max_weight).to_numpy(dtype=float)
        else:
            target.loc[dt] = fallback_weights.reindex(target.columns).to_numpy(dtype=float)

    first_idx = target.index[0]
    if pd.isna(target.loc[first_idx]).all():
        target.loc[first_idx] = fallback_weights.reindex(target.columns).to_numpy(dtype=float)

    weights_daily = target.ffill().fillna(fallback_weights.to_dict()).astype(float)
    return weights_daily, scores


def _dynamic_allocator_payload(
    sleeve_returns: pd.DataFrame,
    sleeve_turnover: pd.DataFrame,
    weights_daily: pd.DataFrame,
) -> dict[str, Any]:
    row_sums = weights_daily.sum(axis=1)
    if not np.allclose(row_sums.to_numpy(dtype=float), 1.0, atol=1e-8):
        raise ValueError("Dynamic sleeve weights must sum to 1")

    weights_prev = weights_daily.shift(1).fillna(0.0)
    gross_return = (weights_prev * sleeve_returns).sum(axis=1).rename("rev_tilt_2_dynamic")
    internal_turnover = (weights_prev * sleeve_turnover).sum(axis=1).rename("internal_turnover")
    allocator_turnover = (0.5 * weights_daily.diff().abs().sum(axis=1)).fillna(0.0).rename("allocator_turnover")
    allocator_cost = allocator_turnover * (composite.COSTS_BPS / 10000.0)
    net_return = (gross_return - allocator_cost).rename("rev_tilt_2_dynamic")
    total_turnover = (internal_turnover + allocator_turnover).rename("rev_tilt_2_dynamic")
    equity = (1.0 + net_return).cumprod().rename("Equity")
    return {
        "name": "rev_tilt_2_dynamic",
        "summary": {},
        "outdir": None,
        "equity": pd.DataFrame({"Equity": equity, "DailyReturn": net_return, "Turnover": total_turnover}),
        "holdings": None,
        "daily_return": net_return,
        "daily_turnover": total_turnover,
        "weights_daily": weights_daily,
        "allocator_turnover": allocator_turnover,
    }


def _build_manifest(
    outdir: Path,
    sleeve_manifest: list[dict[str, Any]],
    start: str,
    end: str,
    max_tickers: int,
) -> dict[str, Any]:
    return {
        "timestamp_utc": outdir.name,
        "results_dir": str(outdir),
        "date_range": {"start": start, "end": end},
        "universe": composite.UNIVERSE,
        "universe_mode": composite.UNIVERSE_MODE,
        "top_n": 50,
        "rebalance": "weekly",
        "weighting": composite.WEIGHTING,
        "costs_bps": composite.COSTS_BPS,
        "max_tickers": max_tickers,
        "strategy_definitions": {
            "rev_tilt_2_original": {
                SLEEVE_LABELS[name]: weight for name, weight in ORIGINAL_WEIGHTS.items()
            },
            "rev_tilt_2_dynamic": {
                "score": "trailing 63-day sleeve Sharpe from daily returns",
                "transformation": "softmax followed by bounded proportional rescaling",
                "min_weight": MIN_WEIGHT,
                "max_weight": MAX_WEIGHT,
                "rebalance_frequency": "weekly",
            },
        },
        "notes": [
            "Portfolios are return-level combinations of sleeve backtests produced by run_backtest.",
            "Requested volatility_20 ascending sleeve is implemented with existing low_vol_20 factor.",
            "Dynamic weights use trailing 63-day sleeve Sharpe shifted by one trading day.",
            "Allocator turnover is charged at the same 10 bps cost assumption as the sleeve backtests.",
            "When 63-day Sharpe history is unavailable, dynamic weights fall back to the original static rev_tilt_2 mix.",
        ],
        "sleeve_runs": sleeve_manifest,
    }


def _print_interpretation(metrics_df: pd.DataFrame) -> None:
    by_name = metrics_df.set_index("Strategy")
    original = by_name.loc["rev_tilt_2_original"]
    dynamic = by_name.loc["rev_tilt_2_dynamic"]
    sharpe_improved = float(dynamic["Sharpe"]) > float(original["Sharpe"])
    dd_improved = float(dynamic["MaxDD"]) > float(original["MaxDD"])
    turnover_increased = float(dynamic["Turnover"]) > float(original["Turnover"])

    print("INTERPRETATION")
    print("--------------")
    print(
        f"- Did Sharpe improve: {'yes' if sharpe_improved else 'no'} "
        f"({dynamic['Sharpe']:.4f} vs {original['Sharpe']:.4f})."
    )
    print(
        f"- Did drawdown improve: {'yes' if dd_improved else 'no'} "
        f"({dynamic['MaxDD']:.4f} vs {original['MaxDD']:.4f})."
    )
    print(
        f"- Did turnover increase: {'yes' if turnover_increased else 'no'} "
        f"({dynamic['Turnover']:.4f} vs {original['Turnover']:.4f})."
    )


def main() -> None:
    global START, END, MAX_TICKERS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--max_tickers", type=int, default=MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    parser.add_argument("--sharpe_window", type=int, default=SHARPE_WINDOW)
    args = parser.parse_args()

    START = str(args.start)
    END = str(args.end)
    MAX_TICKERS = int(args.max_tickers)

    results_root = Path(str(args.results_root))
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / ts
    outdir.mkdir(parents=True, exist_ok=True)

    sleeve_payloads, sleeve_manifest = _run_sleeves()
    sleeve_order = list(ORIGINAL_WEIGHTS.keys())
    sleeve_payloads = sorted(sleeve_payloads, key=lambda p: sleeve_order.index(str(p["name"])))

    sleeve_returns = pd.concat([p["daily_return"] for p in sleeve_payloads], axis=1).sort_index().fillna(0.0)
    sleeve_turnover = pd.concat([p["daily_turnover"] for p in sleeve_payloads], axis=1).sort_index().fillna(0.0)
    fallback_weights = pd.Series(ORIGINAL_WEIGHTS, dtype=float).reindex(sleeve_returns.columns)

    dynamic_weights, sleeve_scores = _dynamic_weights(
        sleeve_returns=sleeve_returns,
        fallback_weights=fallback_weights,
        window=int(args.sharpe_window),
        min_weight=MIN_WEIGHT,
        max_weight=MAX_WEIGHT,
    )

    original_payload = composite.weighted_sleeve_portfolio(
        sleeve_payloads=[next(p for p in sleeve_payloads if p["name"] == name) for name in sleeve_order],
        weights=[ORIGINAL_WEIGHTS[name] for name in sleeve_order],
        name="rev_tilt_2_original",
    )
    dynamic_payload = _dynamic_allocator_payload(
        sleeve_returns=sleeve_returns,
        sleeve_turnover=sleeve_turnover,
        weights_daily=dynamic_weights,
    )
    portfolio_payloads = [original_payload, dynamic_payload]

    metrics_df = pd.DataFrame(
        [
            composite._portfolio_metrics(
                name=payload["name"],
                daily_return=payload["daily_return"],
                turnover=payload["daily_turnover"],
            )
            for payload in portfolio_payloads
        ]
    )
    metrics_df = (
        metrics_df.set_index("Strategy")
        .loc[["rev_tilt_2_original", "rev_tilt_2_dynamic"]]
        .reset_index()
    )

    daily_returns_df = pd.concat([p["daily_return"] for p in portfolio_payloads], axis=1).sort_index().fillna(0.0)

    original_weight_frame = pd.DataFrame(
        np.tile(fallback_weights.to_numpy(dtype=float), (len(dynamic_weights.index), 1)),
        index=dynamic_weights.index,
        columns=dynamic_weights.columns,
        dtype=float,
    )
    sleeve_weights_df = pd.concat(
        [
            original_weight_frame.rename(columns=SLEEVE_LABELS).add_prefix("rev_tilt_2_original__"),
            dynamic_weights.rename(columns=SLEEVE_LABELS).add_prefix("rev_tilt_2_dynamic__"),
            sleeve_scores.rename(columns=SLEEVE_LABELS).add_prefix("score__"),
        ],
        axis=1,
    ).sort_index()

    results_path = outdir / "rev_tilt_2_dynamic_results.csv"
    daily_returns_path = outdir / "daily_returns.csv"
    sleeve_weights_path = outdir / "sleeve_weights.csv"
    manifest_path = outdir / "manifest.json"

    metrics_df.to_csv(results_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index_label="date", float_format="%.10g")
    sleeve_weights_df.to_csv(sleeve_weights_path, index_label="date", float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                outdir=outdir,
                sleeve_manifest=sleeve_manifest,
                start=START,
                end=END,
                max_tickers=MAX_TICKERS,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    table = metrics_df[["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]].copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]:
        table[col] = table[col].map(composite._format_float)

    print("REV_TILT_2 DYNAMIC ALLOCATOR TEST")
    print("---------------------------------")
    print("")
    print(table.to_string(index=False))
    print("")
    _print_interpretation(metrics_df)
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
