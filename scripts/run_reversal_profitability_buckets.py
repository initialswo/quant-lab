"""Measure reversal performance within profitability quintiles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import run_composite_vs_sleeves as composite
from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.engine import runner
from quant_lab.engine.runner import run_backtest
from quant_lab.factors.registry import compute_factors
from quant_lab.research.sweep_metrics import extract_annual_turnover
from quant_lab.strategies.topn import rebalance_mask
from quant_lab.universe.liquid_us import build_liquid_us_universe


RESULTS_ROOT = Path("results") / "reversal_profitability_buckets"
START = "2000-01-01"
END = "2024-12-31"
MAX_TICKERS = 2000
QUANTILES = 5
STRATEGIES: list[str] = [f"reversal_profit_q{i}" for i in range(1, QUANTILES + 1)]


def _base_config() -> dict[str, Any]:
    return {
        "start": START,
        "end": END,
        "universe": "liquid_us",
        "universe_mode": "dynamic",
        "top_n": composite.TOP_N,
        "rebalance": composite.REBALANCE,
        "weighting": composite.WEIGHTING,
        "costs_bps": composite.COSTS_BPS,
        "max_tickers": MAX_TICKERS,
        "data_source": composite.DATA_SOURCE,
        "data_cache_dir": composite.DATA_CACHE_DIR,
        "fundamentals_path": composite.FUNDAMENTALS_PATH,
        "fundamentals_fallback_lag_days": composite.FUNDAMENTALS_FALLBACK_LAG_DAYS,
        "save_artifacts": True,
        "factor_name": ["reversal_1m"],
        "factor_names": ["reversal_1m"],
        "factor_weights": [1.0],
        "universe_min_tickers": 1,
        "universe_skip_below_min_tickers": False,
    }


def _load_price_volume_panels(start: str, end: str, max_tickers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = runner._load_universe_seed_tickers(
        universe="liquid_us",
        max_tickers=int(max_tickers),
        data_cache_dir=composite.DATA_CACHE_DIR,
    )
    ohlcv_map, _ = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(start),
        end=str(end),
        cache_dir=composite.DATA_CACHE_DIR,
        data_source=composite.DATA_SOURCE,
        refresh=False,
        bulk_prepare=False,
    )
    close_cols, used_tickers, missing_tickers, rejected_tickers, _ = runner._collect_close_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=tickers,
    )
    if not close_cols:
        raise ValueError(
            "No valid OHLCV data found. "
            f"missing={len(missing_tickers)} rejected={len(rejected_tickers)}"
        )
    close = pd.concat(close_cols, axis=1, join="outer")
    close = runner._prepare_close_panel(close_raw=close, price_fill_mode="ffill").astype(float)
    volume = runner._collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Volume",
    ).reindex(index=close.index, columns=close.columns).astype(float)
    return close, volume


def _build_bucket_memberships(
    start: str,
    end: str,
    max_tickers: int,
    outdir: Path,
) -> dict[str, Path]:
    close, volume = _load_price_volume_panels(start=start, end=end, max_tickers=max_tickers)
    factor_params = runner._augment_factor_params_with_fundamentals(
        factor_names=["gross_profitability"],
        factor_params_map={},
        close=close,
        fundamentals_path=composite.FUNDAMENTALS_PATH,
        fundamentals_fallback_lag_days=composite.FUNDAMENTALS_FALLBACK_LAG_DAYS,
    )
    gp = compute_factors(
        factor_names=["gross_profitability"],
        close=close,
        factor_params=factor_params,
    )["gross_profitability"].astype(float)
    eligibility = build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=1.0,
        min_avg_dollar_volume=0.0,
        adv_window=20,
        min_history=300,
    )
    rb = rebalance_mask(pd.DatetimeIndex(close.index), composite.REBALANCE)
    rb_dates = pd.DatetimeIndex(close.index[rb])
    labels = [f"Q{i}" for i in range(1, QUANTILES + 1)]
    buckets = {
        label: pd.DataFrame(False, index=rb_dates, columns=close.columns, dtype=bool) for label in labels
    }

    for dt in rb_dates:
        gp_row = gp.loc[dt]
        elig_row = eligibility.loc[dt].fillna(False).astype(bool)
        valid = gp_row[elig_row & gp_row.notna()].astype(float)
        if int(valid.shape[0]) < QUANTILES:
            continue
        try:
            q = pd.qcut(valid.rank(method="first"), q=QUANTILES, labels=labels)
        except ValueError:
            continue
        for label in labels:
            picked = q.index[q.astype(str) == label]
            buckets[label].loc[dt, picked] = True

    out: dict[str, Path] = {}
    for label in labels:
        path = outdir / f"profitability_{label.lower()}_membership.csv"
        buckets[label].astype(int).to_csv(path, index_label="date")
        out[label] = path
    return out


def _result_row(summary: dict[str, Any], outdir: str, strategy_name: str) -> dict[str, Any]:
    payload = composite._strategy_payload(name=strategy_name, summary=summary, outdir=outdir)
    annual_turnover = extract_annual_turnover(summary=summary, outdir=outdir)
    row = composite._portfolio_metrics(
        name=strategy_name,
        daily_return=payload["daily_return"],
        turnover=payload["daily_turnover"],
        annual_turnover_override=annual_turnover,
    )
    return {
        "Strategy": str(strategy_name),
        "CAGR": float(row["CAGR"]),
        "Vol": float(row["Vol"]),
        "Sharpe": float(row["Sharpe"]),
        "MaxDD": float(row["MaxDD"]),
        "Turnover": float(row["Turnover"]),
        "HitRate": float(row["HitRate"]),
        "TotalReturn": float(row["TotalReturn"]),
    }


def main() -> None:
    global START, END, MAX_TICKERS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--max_tickers", type=int, default=MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    START = str(args.start)
    END = str(args.end)
    MAX_TICKERS = int(args.max_tickers)

    results_root = Path(str(args.results_root))
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / ts
    outdir.mkdir(parents=True, exist_ok=True)

    bucket_paths = _build_bucket_memberships(
        start=START,
        end=END,
        max_tickers=MAX_TICKERS,
        outdir=outdir,
    )

    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    manifest_runs: list[dict[str, Any]] = []
    for i, strategy_name in enumerate(STRATEGIES, start=1):
        label = f"Q{i}"
        print(f"Running {strategy_name}...")
        cfg = dict(_base_config())
        cfg["historical_membership_path"] = str(bucket_paths[label])
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        rows.append(_result_row(summary=summary, outdir=run_outdir, strategy_name=strategy_name))
        manifest_runs.append(
            {
                "strategy_name": strategy_name,
                "profitability_bucket": label,
                "membership_path": str(bucket_paths[label]),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(
        ["Sharpe", "CAGR"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    results_path = outdir / "reversal_profitability_results.csv"
    manifest_path = outdir / "manifest.json"
    results_df.to_csv(results_path, index=False, float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            {
                "timestamp_utc": ts,
                "results_dir": str(outdir),
                "date_range": {"start": START, "end": END},
                "universe": "liquid_us",
                "universe_mode": "dynamic",
                "top_n": composite.TOP_N,
                "rebalance": composite.REBALANCE,
                "weighting": composite.WEIGHTING,
                "costs_bps": composite.COSTS_BPS,
                "max_tickers": MAX_TICKERS,
                "quantiles": QUANTILES,
                "runs": manifest_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    composite._copy_latest(
        files={
            "reversal_profitability_results.csv": results_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    table = results_df[["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD"]].copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD"]:
        table[col] = table[col].map(composite._format_float)
    print("")
    print("REVERSAL BY PROFITABILITY BUCKET")
    print("--------------------------------")
    print("")
    print(table.to_string(index=False))
    print("")
    print(f"Saved: {results_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
