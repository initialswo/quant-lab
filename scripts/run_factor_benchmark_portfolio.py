"""Run an equal-weight factor benchmark portfolio built from standalone sleeves."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_composite_vs_sleeves as composite
from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel, load_fundamentals_file
from quant_lab.engine import runner
from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover


RESULTS_ROOT = Path("results") / "factor_benchmark"
START = "2000-01-01"
END = "2024-12-31"
MAX_TICKERS = 300
FACTOR_NAMES: list[str] = [
    "momentum_12_1",
    "reversal_1m",
    "low_vol_20",
    "gross_profitability",
    "book_to_market",
    "asset_growth",
]

SLEEVE_SPECS: list[dict[str, Any]] = [
    {"strategy_name": "sleeve_momentum", "factor_name": "momentum_12_1", "notes": "Standalone momentum sleeve."},
    {"strategy_name": "sleeve_reversal", "factor_name": "reversal_1m", "notes": "Standalone reversal sleeve."},
    {"strategy_name": "sleeve_low_vol", "factor_name": "low_vol_20", "notes": "Standalone low-vol sleeve."},
    {
        "strategy_name": "sleeve_gross_profitability",
        "factor_name": "gross_profitability",
        "notes": "Standalone profitability sleeve.",
    },
    {
        "strategy_name": "sleeve_book_to_market",
        "factor_name": "book_to_market",
        "notes": "Classic value sleeve using PIT shareholders_equity / market_cap.",
    },
    {
        "strategy_name": "sleeve_asset_growth",
        "factor_name": "asset_growth",
        "notes": "Low asset growth sleeve using PIT total_assets.",
    },
]


def _base_config() -> dict[str, Any]:
    return {
        "start": START,
        "end": END,
        "universe": "liquid_us",
        "universe_mode": "dynamic",
        "top_n": 50,
        "rebalance": "weekly",
        "weighting": "equal",
        "costs_bps": 10.0,
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
    factor_params = spec.get("factor_params", {})
    if factor_params:
        cfg["factor_params"] = {factor_name: dict(factor_params)}
    return cfg


def _load_close_panel(start: str, end: str, max_tickers: int) -> pd.DataFrame:
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


def _build_special_factor_params(start: str, end: str, max_tickers: int) -> dict[str, dict[str, Any]]:
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
    return {
        "book_to_market": {"fundamentals_aligned": aligned},
        "asset_growth": {"fundamentals_aligned": aligned, "lag_days": 252},
    }


def _run_sleeves(
    special_factor_params: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_cache: dict[str, Any] = {}
    payloads: list[dict[str, Any]] = []
    manifest_runs: list[dict[str, Any]] = []

    for raw_spec in SLEEVE_SPECS:
        spec = dict(raw_spec)
        factor_name = str(spec["factor_name"])
        if factor_name in special_factor_params:
            spec["factor_params"] = special_factor_params[factor_name]
        print(f"Running {spec['strategy_name']}...")
        summary, run_outdir = run_backtest(**_run_config(spec), run_cache=run_cache)
        payload = composite._strategy_payload(name=str(spec["strategy_name"]), summary=summary, outdir=run_outdir)
        payloads.append(payload)
        manifest_runs.append(
            {
                "strategy_name": str(spec["strategy_name"]),
                "factor_name": factor_name,
                "notes": str(spec["notes"]),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
                "annual_turnover_from_summary": extract_annual_turnover(summary=summary, outdir=run_outdir),
            }
        )
    return payloads, manifest_runs


def _combined_payload(sleeve_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    return composite.weighted_sleeve_portfolio(
        sleeve_payloads=sleeve_payloads,
        weights=np.full(len(sleeve_payloads), 1.0 / len(sleeve_payloads), dtype=float),
        name="factor_benchmark_equal_weight",
    )


def _build_manifest(
    outdir: Path,
    sleeve_manifest: list[dict[str, Any]],
    max_tickers: int,
) -> dict[str, Any]:
    return {
        "timestamp_utc": outdir.name,
        "results_dir": str(outdir),
        "date_range": {"start": START, "end": END},
        "universe": "liquid_us",
        "universe_mode": "dynamic",
        "rebalance": "weekly",
        "top_n": 50,
        "weighting": "equal",
        "costs_bps": 10.0,
        "max_tickers": max_tickers,
        "factors": FACTOR_NAMES,
        "portfolio_definition": {
            "name": "factor_benchmark_equal_weight",
            "combination_method": "return-level equal-weight combination of sleeve daily returns",
            "weights": {factor: float(1.0 / len(FACTOR_NAMES)) for factor in FACTOR_NAMES},
        },
        "notes": [
            "All sleeves are run as standalone weekly Top-N backtests with equal weighting.",
            "book_to_market and asset_growth receive PIT-aligned fundamentals explicitly via factor_params.",
            "Portfolio turnover is the equal-weight average of sleeve turnover series.",
        ],
        "sleeve_runs": sleeve_manifest,
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

    special_factor_params = _build_special_factor_params(start=START, end=END, max_tickers=MAX_TICKERS)
    sleeve_payloads, sleeve_manifest = _run_sleeves(special_factor_params=special_factor_params)
    benchmark_payload = _combined_payload(sleeve_payloads=sleeve_payloads)

    benchmark_row = composite._portfolio_metrics(
        name=benchmark_payload["name"],
        daily_return=benchmark_payload["daily_return"],
        turnover=benchmark_payload["daily_turnover"],
    )
    results_df = pd.DataFrame(
        [
            {
                "Strategy": str(benchmark_row["Strategy"]),
                "CAGR": float(benchmark_row["CAGR"]),
                "Vol": float(benchmark_row["Vol"]),
                "Sharpe": float(benchmark_row["Sharpe"]),
                "MaxDD": float(benchmark_row["MaxDD"]),
                "Turnover": float(benchmark_row["Turnover"]),
                "TotalReturn": float(benchmark_row["TotalReturn"]),
                "HitRate": float(benchmark_row["HitRate"]),
            }
        ]
    )

    daily_returns_df = pd.concat(
        [pd.concat([p["daily_return"] for p in sleeve_payloads], axis=1), benchmark_payload["daily_return"]],
        axis=1,
    ).sort_index().fillna(0.0)

    results_path = outdir / "factor_benchmark_results.csv"
    daily_returns_path = outdir / "daily_returns.csv"
    manifest_path = outdir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index_label="date", float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                outdir=outdir,
                sleeve_manifest=sleeve_manifest,
                max_tickers=MAX_TICKERS,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    composite._copy_latest(
        files={
            "factor_benchmark_results.csv": results_path,
            "daily_returns.csv": daily_returns_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    table = results_df[["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]].copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]:
        table[col] = table[col].map(composite._format_float)
    print("FACTOR BENCHMARK PORTFOLIO")
    print("--------------------------")
    print("")
    print(table.to_string(index=False))
    print("")
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
