#!/usr/bin/env python3
"""Run the post-fix industry breadth filter test against the canonical baseline strategy."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.data.universe_dynamic import apply_universe_filter_to_scores
from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import (
    _apply_universe_rebalance_skip,
    _augment_factor_params_with_fundamentals,
    _collect_close_series,
    _collect_numeric_panel,
    _collect_price_series,
    _load_universe_seed_tickers,
    _prepare_close_panel,
    _price_panel_health_report,
)
from quant_lab.factors.combine import aggregate_factor_scores
from quant_lab.factors.normalize import percentile_rank_cs, robust_preprocess_base
from quant_lab.factors.registry import compute_factors
from quant_lab.strategies.topn import build_topn_weights, rebalance_mask, simulate_portfolio
from quant_lab.universe.liquid_us import build_liquid_us_universe


RESULTS_ROOT = Path("results") / "industry_breadth_experiment"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_TOP_N = 75
DEFAULT_MAX_TICKERS = 2000
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
DEFAULT_SECURITY_MASTER_PATH = "data/warehouse/security_master.parquet"
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
PRICE_FILL_MODE = "ffill"
FACTOR_NAMES = ["gross_profitability", "reversal_1m"]
FACTOR_WEIGHTS = {"gross_profitability": 0.7, "reversal_1m": 0.3}
BREADTH_FACTOR = "momentum_12_1"
MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 10_000_000.0
LIQUIDITY_LOOKBACK = 20
MIN_HISTORY_DAYS = 300
UNIVERSE_MIN_TICKERS = 20
STRATEGIES = [
    ("baseline", None),
    ("high_breadth", "median"),
    ("top10_breadth", 10),
    ("top5_breadth", 5),
]
RESULT_COLUMNS = [
    "strategy_name",
    "breadth_rule",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "Turnover",
    "hit_rate",
    "n_rebalance_dates",
    "median_selected_names",
    "avg_selected_industries",
]
SUMMARY_COLUMNS = ["strategy_name", "breadth_rule", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    parser.add_argument("--security-master-path", default=DEFAULT_SECURITY_MASTER_PATH)
    return parser.parse_args()



def _load_price_panels(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tickers = _load_universe_seed_tickers(
        universe=DEFAULT_UNIVERSE,
        max_tickers=int(DEFAULT_MAX_TICKERS),
        data_cache_dir=DATA_CACHE_DIR,
    )
    ohlcv_map, data_source_summary = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(start),
        end=str(end),
        cache_dir=DATA_CACHE_DIR,
        data_source=DATA_SOURCE,
        refresh=False,
        bulk_prepare=False,
    )
    close_cols, used_tickers, _, _, _ = _collect_close_series(ohlcv_map=ohlcv_map, requested_tickers=tickers)
    if len(used_tickers) < DEFAULT_TOP_N + 1:
        raise ValueError(
            f"Not enough tickers with data: got {len(used_tickers)}, need at least {DEFAULT_TOP_N + 1}."
        )

    close_raw = pd.concat(close_cols, axis=1, join="outer")
    adj_close_cols, _, _, _, _ = _collect_price_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Adj Close",
        fallback_field="Close",
    )
    adj_close_raw = pd.concat(adj_close_cols, axis=1, join="outer") if adj_close_cols else close_raw.copy()
    volume_raw = _collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Volume",
    )

    close = _prepare_close_panel(close_raw=close_raw, price_fill_mode=PRICE_FILL_MODE)
    adj_close = _prepare_close_panel(close_raw=adj_close_raw, price_fill_mode=PRICE_FILL_MODE)
    min_non_nan = int(0.8 * close.shape[1])
    close = close.dropna(thresh=min_non_nan)
    if close.empty:
        raise ValueError("No usable close-price history after alignment/fill.")

    _, broken_tickers, suspicious_tickers = _price_panel_health_report(
        close_raw=close_raw.reindex(index=close.index, columns=close.columns),
        close_filled=close,
    )
    if suspicious_tickers:
        print(
            "PRICE PANEL WARNING: long null runs detected "
            f"count={len(suspicious_tickers)} sample={suspicious_tickers[:10]}"
        )
    if broken_tickers:
        dropped = sorted(set(broken_tickers))
        print(
            "PRICE PANEL WARNING: dropping broken tickers "
            f"(close<=0 or max_abs_daily_return>200%) count={len(dropped)} sample={dropped[:10]}"
        )
        close = close.drop(columns=dropped, errors="ignore")
        adj_close = adj_close.drop(columns=dropped, errors="ignore")
        if close.shape[1] < DEFAULT_TOP_N + 1:
            raise ValueError(
                "Not enough tickers after dropping broken price series: "
                f"got {close.shape[1]}, need at least {DEFAULT_TOP_N + 1}."
            )

    volume = (
        volume_raw.reindex(index=close.index, columns=close.columns)
        if not volume_raw.empty
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    adj_close = adj_close.reindex(index=close.index, columns=close.columns).where(
        adj_close.reindex(index=close.index, columns=close.columns).notna(),
        close,
    )
    return close.astype(float), adj_close.astype(float), volume.astype(float), data_source_summary



def _build_industry_map(security_master_path: Path, tickers: list[str]) -> pd.Series:
    security = pd.read_parquet(security_master_path, columns=["canonical_symbol", "industry_ff48"])
    security["canonical_symbol"] = security["canonical_symbol"].astype(str).str.strip().str.upper()
    security["industry_ff48"] = security["industry_ff48"].astype("string")
    mapping = security.drop_duplicates(subset=["canonical_symbol"], keep="last").set_index("canonical_symbol")[
        "industry_ff48"
    ]
    return mapping.reindex(tickers)



def _compute_base_eligibility(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    eligibility = build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=MIN_PRICE,
        min_avg_dollar_volume=MIN_AVG_DOLLAR_VOLUME,
        adv_window=LIQUIDITY_LOOKBACK,
        min_history=MIN_HISTORY_DAYS,
    )
    return eligibility.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)



def _compute_factor_panels(
    close: pd.DataFrame,
    adj_close: pd.DataFrame,
    fundamentals_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_names = FACTOR_NAMES + [BREADTH_FACTOR]
    factor_params = _augment_factor_params_with_fundamentals(
        factor_names=factor_names,
        factor_params_map={},
        close=close,
        fundamentals_path=str(fundamentals_path),
        fundamentals_fallback_lag_days=60,
    )
    raw_scores = compute_factors(
        factor_names=factor_names,
        close=adj_close,
        factor_params=factor_params,
    )
    norm_scores = {
        name: percentile_rank_cs(robust_preprocess_base(raw_scores[name], winsor_p=0.05))
        for name in FACTOR_NAMES
    }
    composite_scores = aggregate_factor_scores(
        scores=norm_scores,
        weights=FACTOR_WEIGHTS,
        method="linear",
        require_all_factors=True,
    ).astype(float)
    breadth_source = raw_scores[BREADTH_FACTOR].astype(float)
    return composite_scores, breadth_source



def _build_breadth_filter(
    breadth_source: pd.DataFrame,
    eligibility: pd.DataFrame,
    industry_by_ticker: pd.Series,
    rule: str | int,
    rb_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    allowed = pd.DataFrame(False, index=breadth_source.index, columns=breadth_source.columns)
    selected_counts = pd.Series(np.nan, index=breadth_source.index, dtype=float)
    valid_industry = industry_by_ticker.notna() & industry_by_ticker.astype(str).str.strip().ne("")
    valid_tickers = [t for t in breadth_source.columns if bool(valid_industry.get(t, False))]
    industry_lookup = industry_by_ticker.loc[valid_tickers].astype(str)

    for dt in breadth_source.index[rb_mask]:
        row = breadth_source.loc[dt, valid_tickers]
        elig = eligibility.loc[dt, valid_tickers]
        valid = row.where(elig.astype(bool)).dropna()
        if valid.empty:
            continue
        industries = industry_lookup.reindex(valid.index)
        positive = (valid > 0.0).astype(float)
        breadth = positive.groupby(industries).mean().sort_values(ascending=False, kind="mergesort")
        if str(rule) == "median":
            threshold = float(breadth.median())
            selected_industries = breadth.index[breadth > threshold]
        else:
            selected_industries = breadth.index[: int(rule)]
        mask = industries.isin(selected_industries)
        allowed.loc[dt, valid.index[mask]] = True
        selected_counts.loc[dt] = float(len(selected_industries))

    return allowed, selected_counts



def _annual_turnover_from_sim(sim: pd.DataFrame) -> float:
    turnover = pd.to_numeric(sim.get("Turnover"), errors="coerce")
    if turnover is None or turnover.notna().sum() == 0:
        return float("nan")
    return float(turnover.mean() * 252.0)



def _artifact_stats_from_sim(sim: pd.DataFrame, holdings: pd.DataFrame, skipped_count: int) -> dict[str, float]:
    daily_return = pd.to_numeric(sim.get("DailyReturn"), errors="coerce").dropna()
    rb_mask = rebalance_mask(pd.DatetimeIndex(holdings.index), DEFAULT_REBALANCE).reindex(holdings.index).fillna(False)
    selected_counts = (holdings.loc[rb_mask].abs() > 0.0).sum(axis=1).astype(float)
    scheduled_rebalances = int(rb_mask.sum())
    n_rebalance_dates = max(0, scheduled_rebalances - int(skipped_count))
    return {
        "hit_rate": float((daily_return > 0.0).mean()) if not daily_return.empty else float("nan"),
        "median_selected_names": float(selected_counts.median()) if not selected_counts.empty else float("nan"),
        "n_rebalance_dates": int(n_rebalance_dates),
    }



def _simulate_strategy(
    strategy_name: str,
    breadth_rule: str | int | None,
    close: pd.DataFrame,
    adj_close: pd.DataFrame,
    composite_scores: pd.DataFrame,
    breadth_source: pd.DataFrame,
    eligibility: pd.DataFrame,
    industry_by_ticker: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    scores = apply_universe_filter_to_scores(composite_scores, eligibility, exempt={"SPY"}).astype(float)
    rb_mask = rebalance_mask(scores.index, DEFAULT_REBALANCE)
    avg_selected_industries = float("nan")

    if breadth_rule is not None:
        breadth_filter, breadth_counts = _build_breadth_filter(
            breadth_source=breadth_source,
            eligibility=eligibility,
            industry_by_ticker=industry_by_ticker,
            rule=breadth_rule,
            rb_mask=rb_mask,
        )
        filtered_eligibility = (
            eligibility.reindex(index=scores.index, columns=scores.columns).fillna(False).astype(bool)
            & breadth_filter.reindex(index=scores.index, columns=scores.columns).fillna(False).astype(bool)
        )
        scores = apply_universe_filter_to_scores(scores, filtered_eligibility, exempt=set()).astype(float)
        avg_selected_industries = float(breadth_counts.dropna().mean()) if breadth_counts.notna().any() else float("nan")

    scores_for_weights, skip_zero_count, skip_below_min_count = _apply_universe_rebalance_skip(
        scores=scores,
        rb_mask=rb_mask,
        universe_min_tickers=UNIVERSE_MIN_TICKERS,
        universe_skip_below_min_tickers=True,
    )
    skipped_count = int(skip_zero_count + skip_below_min_count)
    weights = build_topn_weights(
        scores=scores_for_weights,
        close=close,
        top_n=DEFAULT_TOP_N,
        rebalance=DEFAULT_REBALANCE,
        weighting="equal",
        vol_lookback=20,
        max_weight=0.15,
        score_clip=5.0,
        score_floor=0.0,
        sector_cap=0.0,
        sector_by_ticker=None,
        sector_neutral=False,
        rank_buffer=0,
        volatility_scaled_weights=False,
    )
    sim = simulate_portfolio(
        close=adj_close,
        weights=weights,
        costs_bps=DEFAULT_COSTS_BPS,
        rebalance_dates=pd.DatetimeIndex(scores.index[rb_mask]),
        execution_delay_days=0,
    )
    metrics = compute_metrics(sim["DailyReturn"])
    stats = _artifact_stats_from_sim(sim=sim, holdings=weights, skipped_count=skipped_count)
    row = {
        "strategy_name": str(strategy_name),
        "breadth_rule": pd.NA if breadth_rule is None else str(breadth_rule),
        "CAGR": float(metrics.get("CAGR", float("nan"))),
        "Vol": float(metrics.get("Vol", float("nan"))),
        "Sharpe": float(metrics.get("Sharpe", float("nan"))),
        "MaxDD": float(metrics.get("MaxDD", float("nan"))),
        "Turnover": _annual_turnover_from_sim(sim),
        "hit_rate": float(stats["hit_rate"]),
        "n_rebalance_dates": int(stats["n_rebalance_dates"]),
        "median_selected_names": float(stats["median_selected_names"]),
        "avg_selected_industries": float(avg_selected_industries),
    }
    return row, sim, weights



def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": [int(obj.shape[0]), int(obj.shape[1])],
            "columns_sample": [str(c) for c in list(obj.columns[:5])],
        }
    if isinstance(obj, pd.Series):
        return {"type": "Series", "length": int(obj.shape[0]), "name": str(obj.name)}
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)



def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"



def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)



def _write_strategy_artifacts(run_dir: Path, strategy_name: str, sim: pd.DataFrame, weights: pd.DataFrame) -> dict[str, str]:
    strat_dir = run_dir / strategy_name
    strat_dir.mkdir(parents=True, exist_ok=True)
    equity_path = strat_dir / "equity.csv"
    holdings_path = strat_dir / "holdings.csv"
    sim.to_csv(equity_path)
    weights.to_csv(holdings_path)
    return {"equity": str(equity_path), "holdings": str(holdings_path)}



def _print_summary(summary_df: pd.DataFrame) -> None:
    print("")
    print("INDUSTRY BREADTH SUMMARY")
    print("------------------------")
    print(f"{'Strategy':18s} {'Rule':>8s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Turnover':>10s}")
    for _, row in summary_df.iterrows():
        rule = "-" if pd.isna(row["breadth_rule"]) else str(row["breadth_rule"])
        print(
            f"{str(row['strategy_name']):18s} "
            f"{rule:>8s} "
            f"{_format_float(float(row['CAGR'])):>8s} "
            f"{_format_float(float(row['Vol'])):>8s} "
            f"{_format_float(float(row['Sharpe'])):>8s} "
            f"{_format_float(float(row['MaxDD'])):>8s} "
            f"{_format_float(float(row['Turnover'])):>10s}"
        )



def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("POST-FIX INDUSTRY BREADTH EXPERIMENT")
    print("-----------------------------------")
    print(
        "Config: "
        f"universe={DEFAULT_UNIVERSE} rebalance={DEFAULT_REBALANCE} costs_bps={DEFAULT_COSTS_BPS} "
        f"top_n={DEFAULT_TOP_N} factors={','.join(FACTOR_NAMES)} factor_weights={[0.7, 0.3]} "
        "factor_aggregation_method=linear"
    )
    print(
        "Note: industry breadth is the share of mapped eligible stocks in each industry_ff48 group "
        "with raw momentum_12_1 > 0 at each rebalance."
    )

    t0 = time.perf_counter()
    close, adj_close, volume, data_source_summary = _load_price_panels(start=str(args.start), end=str(args.end))
    industry_by_ticker = _build_industry_map(Path(args.security_master_path), list(close.columns))
    eligibility = _compute_base_eligibility(close=close, volume=volume)
    composite_scores, breadth_source = _compute_factor_panels(
        close=close,
        adj_close=adj_close,
        fundamentals_path=str(args.fundamentals_path),
    )

    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []
    for strategy_name, breadth_rule in STRATEGIES:
        row, sim, weights = _simulate_strategy(
            strategy_name=strategy_name,
            breadth_rule=breadth_rule,
            close=close,
            adj_close=adj_close,
            composite_scores=composite_scores,
            breadth_source=breadth_source,
            eligibility=eligibility,
            industry_by_ticker=industry_by_ticker,
        )
        artifact_paths = _write_strategy_artifacts(run_dir=run_dir, strategy_name=str(strategy_name), sim=sim, weights=weights)
        rows.append(row)
        run_manifest.append(
            {
                "strategy_name": str(strategy_name),
                "breadth_rule": None if breadth_rule is None else str(breadth_rule),
                "artifact_paths": artifact_paths,
            }
        )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(
        by=["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort"
    ).reset_index(drop=True)
    summary_df = results_df.loc[:, SUMMARY_COLUMNS].copy()

    results_path = run_dir / "industry_breadth_results.csv"
    summary_path = run_dir / "industry_breadth_summary.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    baseline_sharpe = float(results_df.loc[results_df["strategy_name"] == "baseline", "Sharpe"].iloc[0])
    breadth_df = results_df.loc[results_df["strategy_name"] != "baseline"].copy()
    best_breadth = breadth_df.sort_values(by=["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").iloc[0]
    best_breadth_sharpe = float(best_breadth["Sharpe"])

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_industry_breadth_experiment.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "rebalance": DEFAULT_REBALANCE,
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "top_n": int(DEFAULT_TOP_N),
        "factor_names": list(FACTOR_NAMES),
        "factor_weights": [0.7, 0.3],
        "factor_aggregation_method": "linear",
        "security_master_path": str(args.security_master_path),
        "fundamentals_path": str(args.fundamentals_path),
        "data_source_summary": _to_serializable(data_source_summary),
        "output_dir": str(run_dir),
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "baseline_sharpe": baseline_sharpe,
        "best_breadth_strategy": str(best_breadth["strategy_name"]),
        "best_breadth_sharpe": best_breadth_sharpe,
        "runtime_seconds": float(time.perf_counter() - t0),
        "notes": [
            "This is the post-fix industry breadth experiment against the canonical corrected baseline.",
            "Factors and portfolio PnL use adjusted prices consistently with the corrected runtime.",
            "Industry breadth is defined as the fraction of mapped eligible stocks in each industry_ff48 group with raw momentum_12_1 > 0.",
        ],
        "runs": _to_serializable(run_manifest),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    latest_root = base_output_dir / "latest"
    _copy_latest(
        files={
            results_path.name: results_path,
            summary_path.name: summary_path,
            manifest_path.name: manifest_path,
        },
        latest_root=latest_root,
    )

    _print_summary(summary_df)
    print("")
    print(f"Baseline Sharpe: {_format_float(baseline_sharpe)}")
    print(f"Best breadth strategy Sharpe: {_format_float(best_breadth_sharpe)} ({best_breadth['strategy_name']})")
    if best_breadth_sharpe > baseline_sharpe:
        print("Breadth filtering improves performance versus the unfiltered baseline.")
    else:
        print("Breadth filtering does not improve performance versus the unfiltered baseline.")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_root}")


if __name__ == "__main__":
    main()
