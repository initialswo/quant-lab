"""Run a long-only asset growth anomaly sleeve."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import run_composite_vs_sleeves as composite
from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel, load_fundamentals_file
from quant_lab.engine import runner
from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover


RESULTS_ROOT = Path("results") / "asset_growth_sleeve"
START = "2000-01-01"
END = "2024-12-31"
MAX_TICKERS = 300

STRATEGIES: list[dict[str, Any]] = [
    {
        "strategy_name": "asset_growth_sleeve",
        "factor_name": "asset_growth",
        "factor_params": {},
        "notes": "Low asset growth preferred via negative PIT asset growth score.",
    }
]


def _base_config() -> dict[str, Any]:
    return {
        "start": START,
        "end": END,
        "universe": composite.UNIVERSE,
        "universe_mode": composite.UNIVERSE_MODE,
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
        "use_factor_normalization": False,
    }


def _run_config(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(_base_config())
    cfg["factor_name"] = [str(spec["factor_name"])]
    cfg["factor_names"] = [str(spec["factor_name"])]
    cfg["factor_weights"] = [1.0]
    if spec["factor_params"]:
        cfg["factor_params"] = {str(spec["factor_name"]): dict(spec["factor_params"])}
    return cfg


def _load_close_panel(start: str, end: str, max_tickers: int) -> pd.DataFrame:
    tickers = runner._load_universe_seed_tickers(
        universe=composite.UNIVERSE,
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
    close_cols, _, missing_tickers, rejected_tickers, _ = runner._collect_close_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=tickers,
    )
    if not close_cols:
        raise ValueError(
            "No valid OHLCV data found. "
            f"missing={len(missing_tickers)} rejected={len(rejected_tickers)}"
        )
    close = pd.concat(close_cols, axis=1, join="outer")
    return runner._prepare_close_panel(close_raw=close, price_fill_mode="ffill").astype(float)


def _build_asset_growth_params(start: str, end: str, max_tickers: int) -> dict[str, Any]:
    close = _load_close_panel(start=start, end=end, max_tickers=max_tickers)
    fundamentals = load_fundamentals_file(
        path=composite.FUNDAMENTALS_PATH,
        fallback_lag_days=int(composite.FUNDAMENTALS_FALLBACK_LAG_DAYS),
    )
    aligned = align_fundamentals_to_daily_panel(
        fundamentals=fundamentals,
        price_index=pd.DatetimeIndex(close.index),
        price_columns=close.columns,
    )
    return {"fundamentals_aligned": aligned, "lag_days": 252}


def _result_row(summary: dict[str, Any], outdir: str, strategy_name: str) -> tuple[dict[str, Any], pd.Series]:
    payload = composite._strategy_payload(name=strategy_name, summary=summary, outdir=outdir)
    annual_turnover = extract_annual_turnover(summary=summary, outdir=outdir)
    row = composite._portfolio_metrics(
        name=strategy_name,
        daily_return=payload["daily_return"],
        turnover=payload["daily_turnover"],
        annual_turnover_override=annual_turnover,
    )
    return (
        {
            "Strategy": str(strategy_name),
            "CAGR": float(row["CAGR"]),
            "Vol": float(row["Vol"]),
            "Sharpe": float(row["Sharpe"]),
            "MaxDD": float(row["MaxDD"]),
            "Turnover": float(row["Turnover"]),
            "TotalReturn": float(row["TotalReturn"]),
            "HitRate": float(row["HitRate"]),
        },
        payload["daily_return"],
    )


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

    asset_growth_params = _build_asset_growth_params(start=START, end=END, max_tickers=MAX_TICKERS)
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    daily_returns: list[pd.Series] = []
    manifest_runs: list[dict[str, Any]] = []

    for spec in STRATEGIES:
        strategy_name = str(spec["strategy_name"])
        print(f"Running {strategy_name}...")
        spec_run = dict(spec)
        spec_run["factor_params"] = asset_growth_params
        cfg = _run_config(spec_run)
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        row, daily_return = _result_row(summary=summary, outdir=run_outdir, strategy_name=strategy_name)
        rows.append(row)
        daily_returns.append(daily_return)
        manifest_runs.append(
            {
                "strategy_name": strategy_name,
                "factor_name": str(spec_run["factor_name"]),
                "notes": str(spec_run["notes"]),
                "use_factor_normalization": False,
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort")
    results_df = results_df.reset_index(drop=True)
    daily_returns_df = pd.concat(daily_returns, axis=1).sort_index().fillna(0.0)

    results_path = outdir / "asset_growth_results.csv"
    daily_returns_path = outdir / "daily_returns.csv"
    manifest_path = outdir / "manifest.json"
    results_df.to_csv(results_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index_label="date", float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            {
                "timestamp_utc": ts,
                "results_dir": str(outdir),
                "date_range": {"start": START, "end": END},
                "universe": composite.UNIVERSE,
                "universe_mode": composite.UNIVERSE_MODE,
                "top_n": composite.TOP_N,
                "rebalance": composite.REBALANCE,
                "weighting": composite.WEIGHTING,
                "costs_bps": composite.COSTS_BPS,
                "max_tickers": MAX_TICKERS,
                "factor_definition": "score = -((total_assets_t / total_assets_t_252) - 1)",
                "notes": [
                    "run_backtest is used with default causal execution behavior.",
                    "asset_growth uses PIT-aligned total_assets and requires both current and 252-day lag observations.",
                    "use_factor_normalization is disabled so the factor score ordering remains the raw negative asset growth signal.",
                ],
                "runs": manifest_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    composite._copy_latest(
        files={
            "asset_growth_results.csv": results_path,
            "daily_returns.csv": daily_returns_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    table = results_df[["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]].copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]:
        table[col] = table[col].map(composite._format_float)
    print("ASSET GROWTH SLEEVE TEST")
    print("------------------------")
    print("")
    print(table.to_string(index=False))
    print("")
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
