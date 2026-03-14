"""Compare standalone momentum against a quality-conditioned momentum sleeve."""

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


RESULTS_ROOT = Path("results") / "quality_momentum_sleeve"
START = "2000-01-01"
END = "2024-12-31"
MAX_TICKERS = 2000

STRATEGIES: list[dict[str, Any]] = [
    {
        "strategy_name": "momentum_sleeve",
        "factor_name": "momentum_12_1",
        "factor_params": {},
        "notes": "Standalone momentum sleeve using the existing momentum factor.",
    },
    {
        "strategy_name": "quality_momentum_sleeve",
        "factor_name": "quality_momentum_score",
        "factor_params": {},
        "notes": "Average of cross-sectional 0-1 momentum and profitability ranks.",
    },
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


def _build_quality_momentum_params(start: str, end: str, max_tickers: int) -> dict[str, Any]:
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
    return {"fundamentals_aligned": aligned}


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
            "HitRate": float(row["HitRate"]),
            "TotalReturn": float(row["TotalReturn"]),
        },
        payload["daily_return"],
    )


def _print_interpretation(results_df: pd.DataFrame) -> None:
    by_name = results_df.set_index("Strategy")
    mom = by_name.loc["momentum_sleeve"]
    qm = by_name.loc["quality_momentum_sleeve"]
    sharpe_note = (
        f"Quality momentum improved Sharpe ({qm['Sharpe']:.4f} vs {mom['Sharpe']:.4f})."
        if float(qm["Sharpe"]) > float(mom["Sharpe"])
        else f"Quality momentum did not improve Sharpe ({qm['Sharpe']:.4f} vs {mom['Sharpe']:.4f})."
    )
    dd_note = (
        f"Drawdown improved ({qm['MaxDD']:.4f} vs {mom['MaxDD']:.4f})."
        if float(qm["MaxDD"]) > float(mom["MaxDD"])
        else f"Drawdown did not improve ({qm['MaxDD']:.4f} vs {mom['MaxDD']:.4f})."
    )
    turnover_delta = float(qm["Turnover"] - mom["Turnover"])
    turnover_note = (
        f"Turnover changed materially ({turnover_delta:+.4f} annualized)."
        if abs(turnover_delta) >= 0.25
        else f"Turnover change was small ({turnover_delta:+.4f} annualized)."
    )
    print("INTERPRETATION")
    print("--------------")
    print(sharpe_note)
    print(dd_note)
    print(turnover_note)


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

    quality_params = _build_quality_momentum_params(start=START, end=END, max_tickers=MAX_TICKERS)
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    daily_returns: list[pd.Series] = []
    manifest_runs: list[dict[str, Any]] = []

    for spec in STRATEGIES:
        strategy_name = str(spec["strategy_name"])
        print(f"Running {strategy_name}...")
        if strategy_name == "quality_momentum_sleeve":
            spec = dict(spec)
            spec["factor_params"] = quality_params
        cfg = _run_config(spec)
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        row, daily_return = _result_row(summary=summary, outdir=run_outdir, strategy_name=strategy_name)
        rows.append(row)
        daily_returns.append(daily_return)
        manifest_runs.append(
            {
                "strategy_name": strategy_name,
                "factor_name": str(spec["factor_name"]),
                "notes": str(spec["notes"]),
                "use_factor_normalization": False,
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort")
    results_df = results_df.reset_index(drop=True)
    daily_returns_df = pd.concat(daily_returns, axis=1).sort_index().fillna(0.0)

    results_path = outdir / "quality_momentum_results.csv"
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
                "ranking_method": {
                    "momentum_sleeve": "raw momentum_12_1 ordering",
                    "quality_momentum_sleeve": "0.5 * momentum_rank_pct + 0.5 * profitability_rank_pct",
                },
                "notes": [
                    "Both runs use run_backtest with default causal execution behavior.",
                    "quality_momentum_score requires both momentum and profitability to be present on a date/ticker.",
                    "use_factor_normalization is disabled so the custom combined score is not re-normalized by the runner.",
                ],
                "runs": manifest_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    composite._copy_latest(
        files={
            "quality_momentum_results.csv": results_path,
            "daily_returns.csv": daily_returns_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    table = results_df[["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]].copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]:
        table[col] = table[col].map(composite._format_float)
    print("QUALITY MOMENTUM SLEEVE TEST")
    print("----------------------------")
    print("")
    print(table.to_string(index=False))
    print("")
    _print_interpretation(results_df)
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
