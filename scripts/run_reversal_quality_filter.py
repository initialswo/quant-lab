"""Test reversal performance under profitability filters."""

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


RESULTS_ROOT = Path("results") / "reversal_quality_filter"
START = "2000-01-01"
END = "2024-12-31"
MAX_TICKERS = 2000
QUANTILES = 5

FILTER_DEFS: list[dict[str, Any]] = [
    {"strategy_name": "baseline_reversal", "labels": None},
    {"strategy_name": "reversal_profit_q3plus", "labels": ["Q3", "Q4", "Q5"]},
    {"strategy_name": "reversal_profit_q4plus", "labels": ["Q4", "Q5"]},
    {"strategy_name": "reversal_profit_q5", "labels": ["Q5"]},
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
        "factor_name": ["reversal_1m"],
        "factor_names": ["reversal_1m"],
        "factor_weights": [1.0],
        "universe_min_tickers": 1,
        "universe_skip_below_min_tickers": False,
    }


def _load_price_volume_panels(start: str, end: str, max_tickers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _build_filter_memberships(start: str, end: str, max_tickers: int, outdir: Path) -> dict[str, Path]:
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
    bucket_labels = [f"Q{i}" for i in range(1, QUANTILES + 1)]
    bucket_frames = {
        label: pd.DataFrame(False, index=rb_dates, columns=close.columns, dtype=bool) for label in bucket_labels
    }

    for dt in rb_dates:
        gp_row = gp.loc[dt]
        elig_row = eligibility.loc[dt].fillna(False).astype(bool)
        valid = gp_row[elig_row & gp_row.notna()].astype(float)
        if int(valid.shape[0]) < QUANTILES:
            continue
        try:
            buckets = pd.qcut(valid.rank(method="first"), q=QUANTILES, labels=bucket_labels)
        except ValueError:
            continue
        for label in bucket_labels:
            picked = buckets.index[buckets.astype(str) == label]
            bucket_frames[label].loc[dt, picked] = True

    filter_masks: dict[str, pd.DataFrame] = {
        "Q3plus": bucket_frames["Q3"] | bucket_frames["Q4"] | bucket_frames["Q5"],
        "Q4plus": bucket_frames["Q4"] | bucket_frames["Q5"],
        "Q5only": bucket_frames["Q5"],
    }
    out: dict[str, Path] = {}
    for label, mask in filter_masks.items():
        path = outdir / f"profitability_{label.lower()}_membership.csv"
        mask.astype(int).to_csv(path, index_label="date")
        out[label] = path
    return out


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
        },
        payload["daily_return"],
    )


def _print_interpretation(results_df: pd.DataFrame) -> None:
    by_name = results_df.set_index("Strategy")
    baseline = by_name.loc["baseline_reversal"]
    filtered = results_df[results_df["Strategy"] != "baseline_reversal"].copy()
    best = filtered.iloc[0]
    sharpe_note = (
        f"Filtering improved Sharpe; best filtered variant is {best['Strategy']} ({best['Sharpe']:.4f} vs baseline {baseline['Sharpe']:.4f})."
        if float(best["Sharpe"]) > float(baseline["Sharpe"])
        else f"Filtering did not improve Sharpe; baseline remains best ({baseline['Sharpe']:.4f})."
    )
    drawdown_best = float(filtered["MaxDD"].max())
    dd_note = (
        f"At least one filter improved drawdown (best filtered MaxDD {drawdown_best:.4f} vs baseline {baseline['MaxDD']:.4f})."
        if drawdown_best > float(baseline["MaxDD"])
        else f"Drawdown did not improve versus baseline ({baseline['MaxDD']:.4f})."
    )
    print("INTERPRETATION")
    print("--------------")
    print(sharpe_note)
    print(f"Best threshold by Sharpe: {best['Strategy']}.")
    print(dd_note)


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

    membership_paths = _build_filter_memberships(
        start=START,
        end=END,
        max_tickers=MAX_TICKERS,
        outdir=outdir,
    )

    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    daily_returns: list[pd.Series] = []
    manifest_runs: list[dict[str, Any]] = []
    for spec in FILTER_DEFS:
        strategy_name = str(spec["strategy_name"])
        print(f"Running {strategy_name}...")
        cfg = dict(_base_config())
        labels = spec["labels"]
        if labels is not None:
            key = "Q5only" if labels == ["Q5"] else f"{labels[0]}plus"
            cfg["historical_membership_path"] = str(membership_paths[key])
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        row, daily_return = _result_row(summary=summary, outdir=run_outdir, strategy_name=strategy_name)
        rows.append(row)
        daily_returns.append(daily_return)
        manifest_runs.append(
            {
                "strategy_name": strategy_name,
                "profitability_filter": labels if labels is not None else "none",
                "membership_path": str(cfg.get("historical_membership_path", "")),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort")
    results_df = results_df.reset_index(drop=True)
    daily_returns_df = pd.concat(daily_returns, axis=1).sort_index().fillna(0.0)

    results_path = outdir / "reversal_quality_results.csv"
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
                "quantiles": QUANTILES,
                "filters": {
                    "Q3plus": ["Q3", "Q4", "Q5"],
                    "Q4plus": ["Q4", "Q5"],
                    "Q5only": ["Q5"],
                },
                "runs": manifest_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    composite._copy_latest(
        files={
            "reversal_quality_results.csv": results_path,
            "daily_returns.csv": daily_returns_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    table = results_df[["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]].copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]:
        table[col] = table[col].map(composite._format_float)
    print("REVERSAL QUALITY FILTER TEST")
    print("----------------------------")
    print("")
    print(table.to_string(index=False))
    print("")
    _print_interpretation(results_df)
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
