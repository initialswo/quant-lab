#!/usr/bin/env python3
"""Compare the canonical baseline with a momentum-neutralized variant."""

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

from quant_lab.data.universe_dynamic import apply_universe_filter_to_scores
from quant_lab.engine import runner
from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest
from quant_lab.factors.combine import aggregate_factor_scores
from quant_lab.factors.momentum import compute as compute_momentum_12_1
from quant_lab.factors.normalize import percentile_rank_cs, robust_preprocess_base
from quant_lab.factors.registry import compute_factors
from quant_lab.strategies.topn import build_topn_weights, rebalance_mask, simulate_portfolio
from quant_lab.universe.liquid_us import build_liquid_us_universe


RESULTS_ROOT = Path("results") / "momentum_neutralized_baseline_test"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_TOP_N = 75
DEFAULT_MAX_TICKERS = 2000
DEFAULT_DATA_SOURCE = "parquet"
DEFAULT_DATA_CACHE_DIR = "data/equities"
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
DEFAULT_PRICE_FILL_MODE = "ffill"
DEFAULT_UNIVERSE_MIN_HISTORY_DAYS = 300
DEFAULT_UNIVERSE_MIN_PRICE = 5.0
DEFAULT_MIN_AVG_DOLLAR_VOLUME = 10_000_000.0
DEFAULT_LIQUIDITY_LOOKBACK = 20
DEFAULT_UNIVERSE_MIN_TICKERS = 20
FACTOR_NAMES = ["gross_profitability", "reversal_1m"]
FACTOR_WEIGHTS = [0.7, 0.3]
FACTOR_AGGREGATION_METHOD = "linear"
BASELINE_STRATEGY = "baseline"
NEUTRALIZED_STRATEGY = "momentum_neutralized_baseline"
PERFORMANCE_SHARPE_MATERIAL_DELTA = 0.10
PERFORMANCE_CAGR_MATERIAL_DELTA = 0.02
EXPOSURE_ABS_MATERIAL_REDUCTION = 0.01
POSITIVE_EXPOSURE_MATERIAL_REDUCTION_PCT = 10.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--data_source", default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--data_cache_dir", default=DEFAULT_DATA_CACHE_DIR)
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    parser.add_argument("--max_tickers", type=int, default=DEFAULT_MAX_TICKERS)
    return parser.parse_args()


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "start": str(args.start),
        "end": str(args.end),
        "universe": DEFAULT_UNIVERSE,
        "universe_mode": "dynamic",
        "top_n": int(DEFAULT_TOP_N),
        "rebalance": DEFAULT_REBALANCE,
        "weighting": "equal",
        "costs_bps": float(DEFAULT_COSTS_BPS),
        "max_tickers": int(args.max_tickers),
        "data_source": str(args.data_source),
        "data_cache_dir": str(args.data_cache_dir),
        "factor_name": list(FACTOR_NAMES),
        "factor_names": list(FACTOR_NAMES),
        "factor_weights": list(FACTOR_WEIGHTS),
        "portfolio_mode": "composite",
        "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
        "use_factor_normalization": True,
        "use_sector_neutralization": False,
        "use_size_neutralization": False,
        "orthogonalize_factors": False,
        "fundamentals_path": str(args.fundamentals_path),
        "save_artifacts": True,
    }


def _read_matrix_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def _load_runtime_price_panels(
    *,
    start: str,
    end: str,
    data_source: str,
    data_cache_dir: str,
    max_tickers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tickers = runner._load_universe_seed_tickers(
        universe=DEFAULT_UNIVERSE,
        max_tickers=int(max_tickers),
        data_cache_dir=str(data_cache_dir),
    )
    ohlcv_map, data_source_summary = runner.fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(start),
        end=str(end),
        cache_dir=str(data_cache_dir),
        data_source=str(data_source),
        refresh=False,
        bulk_prepare=False,
    )

    close_cols, used_tickers, missing_tickers, rejected_tickers, _ = runner._collect_close_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=tickers,
    )
    if not close_cols:
        raise ValueError(
            "No valid OHLCV data found for the experiment. "
            f"missing={len(missing_tickers)} rejected={len(rejected_tickers)}"
        )

    close_raw = pd.concat(close_cols, axis=1, join="outer")
    adj_close_cols, _, _, _, _ = runner._collect_price_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Adj Close",
        fallback_field="Close",
    )
    adj_close_raw = pd.concat(adj_close_cols, axis=1, join="outer") if adj_close_cols else close_raw.copy()
    volume_raw = runner._collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Volume",
    )

    close = runner._prepare_close_panel(close_raw=close_raw, price_fill_mode=DEFAULT_PRICE_FILL_MODE)
    adj_close = runner._prepare_close_panel(close_raw=adj_close_raw, price_fill_mode=DEFAULT_PRICE_FILL_MODE)
    min_non_nan = int(0.8 * close.shape[1])
    close = close.dropna(thresh=min_non_nan)
    if close.empty:
        raise ValueError("No usable close-price history remained after alignment and fill.")

    _, broken_tickers, _ = runner._price_panel_health_report(
        close_raw=close_raw.reindex(index=close.index, columns=close.columns),
        close_filled=close,
    )
    if broken_tickers:
        close = close.drop(columns=broken_tickers, errors="ignore")
        adj_close = adj_close.drop(columns=broken_tickers, errors="ignore")

    volume = (
        volume_raw.reindex(index=close.index, columns=close.columns)
        if not volume_raw.empty
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    adj_close = adj_close.reindex(index=close.index, columns=close.columns)
    adj_close = adj_close.where(adj_close.notna(), close)

    panel_summary = {
        "requested_tickers": int(len(tickers)),
        "loaded_tickers": int(len(used_tickers)),
        "missing_tickers": int(len(missing_tickers)),
        "rejected_tickers": int(len(rejected_tickers)),
        "dropped_broken_tickers": int(len(broken_tickers)),
        "final_tickers": int(close.shape[1]),
        "final_dates": int(close.shape[0]),
    }
    return close.astype(float), adj_close.astype(float), volume.astype(float), {
        "data_source_summary": data_source_summary,
        "panel_summary": panel_summary,
    }


def _build_liquid_us_eligibility(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    effective_min_history = max(
        int(DEFAULT_UNIVERSE_MIN_HISTORY_DAYS),
        int(runner._factor_required_history_days(factor_names=list(FACTOR_NAMES), factor_params_map={})),
    )
    return build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=float(DEFAULT_UNIVERSE_MIN_PRICE),
        min_avg_dollar_volume=float(DEFAULT_MIN_AVG_DOLLAR_VOLUME),
        adv_window=int(DEFAULT_LIQUIDITY_LOOKBACK),
        min_history=int(effective_min_history),
    )


def build_baseline_composite_scores(
    *,
    close: pd.DataFrame,
    adj_close: pd.DataFrame,
    eligibility: pd.DataFrame,
    fundamentals_path: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    factor_params_map = runner._augment_factor_params_with_fundamentals(
        factor_names=list(FACTOR_NAMES),
        factor_params_map={},
        close=close,
        fundamentals_path=str(fundamentals_path),
        fundamentals_fallback_lag_days=60,
    )
    raw_scores = compute_factors(
        factor_names=list(FACTOR_NAMES),
        close=adj_close,
        factor_params=factor_params_map,
    )
    norm_scores = {
        name: percentile_rank_cs(robust_preprocess_base(raw_scores[name], winsor_p=0.05))
        for name in FACTOR_NAMES
    }
    weights = {name: float(w) for name, w in zip(FACTOR_NAMES, FACTOR_WEIGHTS)}
    composite = aggregate_factor_scores(
        scores=norm_scores,
        weights=weights,
        method=FACTOR_AGGREGATION_METHOD,
        require_all_factors=True,
    )
    composite = apply_universe_filter_to_scores(composite, eligibility, exempt={"SPY"})
    diag = {
        "factor_param_keys": {name: sorted(list((factor_params_map.get(name) or {}).keys())) for name in FACTOR_NAMES},
    }
    return composite.astype(float), raw_scores, diag


def neutralize_scores_against_momentum(scores: pd.DataFrame, momentum: pd.DataFrame) -> pd.DataFrame:
    residuals = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns, dtype=float)
    aligned_momentum = momentum.reindex(index=scores.index, columns=scores.columns).astype(float)

    for dt in scores.index:
        y = pd.to_numeric(scores.loc[dt], errors="coerce")
        x = pd.to_numeric(aligned_momentum.loc[dt], errors="coerce")
        valid = y.notna() & x.notna()
        if not bool(valid.any()):
            continue
        x_valid = x.loc[valid].astype(float)
        y_valid = y.loc[valid].astype(float)
        if len(x_valid) == 1:
            residuals.loc[dt, valid] = 0.0
            continue
        x_centered = x_valid - float(x_valid.mean())
        if float((x_centered.pow(2)).sum()) <= 1e-12:
            residuals.loc[dt, valid] = y_valid - float(y_valid.mean())
            continue
        design = np.column_stack([np.ones(len(x_valid), dtype=float), x_valid.to_numpy(dtype=float)])
        beta, *_ = np.linalg.lstsq(design, y_valid.to_numpy(dtype=float), rcond=None)
        fitted = design @ beta
        residuals.loc[dt, valid] = y_valid.to_numpy(dtype=float) - fitted

    return residuals.astype(float)


def compute_momentum_exposure_timeseries(
    *,
    holdings: pd.DataFrame,
    eligibility: pd.DataFrame,
    momentum: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    strategy: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dt in pd.DatetimeIndex(rebalance_dates):
        if dt not in holdings.index or dt not in eligibility.index or dt not in momentum.index:
            continue
        weights = pd.to_numeric(holdings.loc[dt], errors="coerce").fillna(0.0)
        held_tickers = weights[weights.abs() > 0.0].index.tolist()
        eligible_row = eligibility.loc[dt].fillna(False).astype(bool)
        eligible_tickers = eligible_row[eligible_row].index.tolist()

        portfolio_momentum = pd.to_numeric(momentum.loc[dt, held_tickers], errors="coerce").dropna()
        universe_momentum = pd.to_numeric(momentum.loc[dt, eligible_tickers], errors="coerce").dropna()

        portfolio_mean = float(portfolio_momentum.mean()) if not portfolio_momentum.empty else float("nan")
        universe_mean = float(universe_momentum.mean()) if not universe_momentum.empty else float("nan")
        exposure = (
            float(portfolio_mean - universe_mean)
            if pd.notna(portfolio_mean) and pd.notna(universe_mean)
            else float("nan")
        )
        rows.append(
            {
                "Strategy": str(strategy),
                "date": pd.Timestamp(dt),
                "portfolio_momentum": portfolio_mean,
                "universe_momentum": universe_mean,
                "momentum_exposure": exposure,
                "portfolio_count": int(portfolio_momentum.shape[0]),
                "universe_count": int(universe_momentum.shape[0]),
            }
        )

    return pd.DataFrame(rows)


def summarize_momentum_exposure(exposure_df: pd.DataFrame, strategy: str) -> dict[str, Any]:
    strategy_df = exposure_df.loc[exposure_df["Strategy"] == strategy].copy()
    portfolio = pd.to_numeric(strategy_df.get("portfolio_momentum"), errors="coerce").dropna()
    universe = pd.to_numeric(strategy_df.get("universe_momentum"), errors="coerce").dropna()
    exposure = pd.to_numeric(strategy_df.get("momentum_exposure"), errors="coerce").dropna()
    return {
        "Strategy": str(strategy),
        "Mean Portfolio Momentum": float(portfolio.mean()) if not portfolio.empty else float("nan"),
        "Mean Universe Momentum": float(universe.mean()) if not universe.empty else float("nan"),
        "Mean Momentum Exposure": float(exposure.mean()) if not exposure.empty else float("nan"),
        "Percent Positive Exposure": float((exposure > 0.0).mean() * 100.0) if not exposure.empty else float("nan"),
    }


def annual_turnover_from_holdings(holdings: pd.DataFrame) -> float:
    if holdings.empty:
        return float("nan")
    turnover = 0.5 * holdings.astype(float).diff().abs().sum(axis=1).fillna(0.0)
    return float(turnover.mean() * 252.0)


def build_results_row(strategy: str, metrics: dict[str, float], turnover: float) -> dict[str, Any]:
    return {
        "Strategy": str(strategy),
        "CAGR": float(metrics.get("CAGR", float("nan"))),
        "Vol": float(metrics.get("Vol", float("nan"))),
        "Sharpe": float(metrics.get("Sharpe", float("nan"))),
        "MaxDD": float(metrics.get("MaxDD", float("nan"))),
        "Turnover": float(turnover),
    }


def assess_dependency(
    results_df: pd.DataFrame,
    exposure_summary_df: pd.DataFrame,
) -> dict[str, Any]:
    perf = results_df.set_index("Strategy")
    exp = exposure_summary_df.set_index("Strategy")
    base_perf = perf.loc[BASELINE_STRATEGY]
    neu_perf = perf.loc[NEUTRALIZED_STRATEGY]
    base_exp = exp.loc[BASELINE_STRATEGY]
    neu_exp = exp.loc[NEUTRALIZED_STRATEGY]

    sharpe_delta = float(neu_perf["Sharpe"] - base_perf["Sharpe"])
    cagr_delta = float(neu_perf["CAGR"] - base_perf["CAGR"])
    exposure_delta = float(neu_exp["Mean Momentum Exposure"] - base_exp["Mean Momentum Exposure"])
    exposure_abs_reduction = float(abs(base_exp["Mean Momentum Exposure"]) - abs(neu_exp["Mean Momentum Exposure"]))
    positive_exposure_delta = float(
        neu_exp["Percent Positive Exposure"] - base_exp["Percent Positive Exposure"]
    )

    performance_materially_reduced = bool(
        sharpe_delta <= -float(PERFORMANCE_SHARPE_MATERIAL_DELTA)
        or cagr_delta <= -float(PERFORMANCE_CAGR_MATERIAL_DELTA)
    )
    exposure_materially_reduced = bool(
        exposure_abs_reduction >= float(EXPOSURE_ABS_MATERIAL_REDUCTION)
        or positive_exposure_delta <= -float(POSITIVE_EXPOSURE_MATERIAL_REDUCTION_PCT)
    )

    if performance_materially_reduced:
        performance_text = (
            "Neutralizing momentum materially reduces performance relative to the canonical baseline."
        )
    else:
        performance_text = (
            "Neutralizing momentum does not materially reduce performance relative to the canonical baseline."
        )

    if performance_materially_reduced and exposure_materially_reduced:
        dependency_text = (
            "Baseline alpha depends meaningfully on hidden momentum exposure."
        )
    elif (not performance_materially_reduced) and exposure_materially_reduced:
        dependency_text = (
            "Baseline alpha does not appear to depend meaningfully on hidden momentum exposure."
        )
    else:
        dependency_text = (
            "Evidence is inconclusive: performance changed without a clearly material drop in momentum exposure."
        )

    return {
        "baseline_sharpe": float(base_perf["Sharpe"]),
        "neutralized_sharpe": float(neu_perf["Sharpe"]),
        "sharpe_delta_neutralized_minus_baseline": sharpe_delta,
        "baseline_cagr": float(base_perf["CAGR"]),
        "neutralized_cagr": float(neu_perf["CAGR"]),
        "cagr_delta_neutralized_minus_baseline": cagr_delta,
        "baseline_mean_momentum_exposure": float(base_exp["Mean Momentum Exposure"]),
        "neutralized_mean_momentum_exposure": float(neu_exp["Mean Momentum Exposure"]),
        "mean_momentum_exposure_delta_neutralized_minus_baseline": exposure_delta,
        "mean_momentum_exposure_abs_reduction": exposure_abs_reduction,
        "percent_positive_exposure_delta_neutralized_minus_baseline": positive_exposure_delta,
        "performance_materially_reduced": performance_materially_reduced,
        "momentum_exposure_materially_reduced": exposure_materially_reduced,
        "performance_assessment": performance_text,
        "dependency_assessment": dependency_text,
    }


def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)


def _format_metric(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def _format_exposure_metric(metric: str, value: float) -> str:
    if pd.isna(value):
        return "-"
    if metric == "Percent Positive Exposure":
        return f"{float(value):.2f}%"
    return f"{float(value):.6f}"


def _score_alignment_diagnostic(rebuilt_scores: pd.DataFrame, backtest_dir: Path) -> dict[str, Any]:
    score_path = backtest_dir / "composite_scores_snapshot.csv"
    if not score_path.exists():
        return {"available": False}
    saved = _read_matrix_csv(score_path).astype(float)
    rebuilt_rb = rebuilt_scores.reindex(index=saved.index, columns=saved.columns)
    diff = (rebuilt_rb - saved).abs().where(rebuilt_rb.notna() & saved.notna()).stack()
    return {
        "available": True,
        "count": int(diff.shape[0]),
        "mean_abs_diff": float(diff.mean()) if not diff.empty else float("nan"),
        "max_abs_diff": float(diff.max()) if not diff.empty else float("nan"),
    }


def main() -> None:
    args = _parse_args()
    base_output_dir = Path(str(args.output_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    run_config = build_run_config(args)

    baseline_summary, baseline_outdir = run_backtest(**run_config, run_cache=run_cache)
    baseline_dir = Path(baseline_outdir)
    baseline_holdings = _read_matrix_csv(baseline_dir / "holdings.csv").astype(float)

    close, adj_close, volume, load_diag = _load_runtime_price_panels(
        start=str(args.start),
        end=str(args.end),
        data_source=str(args.data_source),
        data_cache_dir=str(args.data_cache_dir),
        max_tickers=int(args.max_tickers),
    )
    eligibility = _build_liquid_us_eligibility(close=close, volume=volume)
    baseline_scores, raw_scores, factor_diag = build_baseline_composite_scores(
        close=close,
        adj_close=adj_close,
        eligibility=eligibility,
        fundamentals_path=str(args.fundamentals_path),
    )
    momentum = compute_momentum_12_1(close=adj_close)
    neutralized_scores = neutralize_scores_against_momentum(baseline_scores, momentum)

    rb_mask = rebalance_mask(baseline_scores.index, DEFAULT_REBALANCE)
    rb_dates = pd.DatetimeIndex(baseline_scores.index[rb_mask])
    neutralized_scores_for_weights, skip_zero_count, skip_below_count = runner._apply_universe_rebalance_skip(
        scores=neutralized_scores,
        rb_mask=rb_mask,
        universe_min_tickers=int(DEFAULT_UNIVERSE_MIN_TICKERS),
        universe_skip_below_min_tickers=True,
    )
    neutralized_weights = build_topn_weights(
        scores=neutralized_scores_for_weights,
        close=close,
        top_n=int(DEFAULT_TOP_N),
        rebalance=DEFAULT_REBALANCE,
        weighting="equal",
    )
    neutralized_sim = simulate_portfolio(
        close=adj_close,
        weights=neutralized_weights,
        costs_bps=float(DEFAULT_COSTS_BPS),
        rebalance_dates=rb_dates,
    )
    neutralized_metrics = compute_metrics(neutralized_sim["DailyReturn"])

    results_rows = [
        build_results_row(
            strategy=BASELINE_STRATEGY,
            metrics=baseline_summary,
            turnover=annual_turnover_from_holdings(baseline_holdings),
        ),
        build_results_row(
            strategy=NEUTRALIZED_STRATEGY,
            metrics=neutralized_metrics,
            turnover=float(neutralized_sim["Turnover"].mean() * 252.0),
        ),
    ]
    results_df = pd.DataFrame(
        results_rows,
        columns=["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"],
    )

    exposure_ts = pd.concat(
        [
            compute_momentum_exposure_timeseries(
                holdings=baseline_holdings,
                eligibility=eligibility,
                momentum=momentum,
                rebalance_dates=rb_dates,
                strategy=BASELINE_STRATEGY,
            ),
            compute_momentum_exposure_timeseries(
                holdings=neutralized_weights,
                eligibility=eligibility,
                momentum=momentum,
                rebalance_dates=rb_dates,
                strategy=NEUTRALIZED_STRATEGY,
            ),
        ],
        axis=0,
        ignore_index=True,
    )
    exposure_summary_df = pd.DataFrame(
        [
            summarize_momentum_exposure(exposure_ts, BASELINE_STRATEGY),
            summarize_momentum_exposure(exposure_ts, NEUTRALIZED_STRATEGY),
        ],
        columns=[
            "Strategy",
            "Mean Portfolio Momentum",
            "Mean Universe Momentum",
            "Mean Momentum Exposure",
            "Percent Positive Exposure",
        ],
    )
    assessment = assess_dependency(results_df=results_df, exposure_summary_df=exposure_summary_df)
    summary_df = pd.DataFrame([assessment])

    results_path = run_dir / "results.csv"
    summary_path = run_dir / "summary.csv"
    exposure_path = run_dir / "momentum_exposure_comparison.csv"
    manifest_path = run_dir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")
    exposure_summary_df.to_csv(exposure_path, index=False, float_format="%.10g")

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_momentum_neutralized_baseline_test.py",
        "runtime_seconds": float(time.perf_counter() - t0),
        "start": str(args.start),
        "end": str(args.end),
        "baseline_strategy": {
            "factors": list(FACTOR_NAMES),
            "factor_weights": list(FACTOR_WEIGHTS),
            "factor_aggregation_method": FACTOR_AGGREGATION_METHOD,
            "top_n": int(DEFAULT_TOP_N),
            "rebalance": DEFAULT_REBALANCE,
            "universe": DEFAULT_UNIVERSE,
            "costs_bps": float(DEFAULT_COSTS_BPS),
            "baseline_backtest_outdir": str(baseline_dir),
            "baseline_summary": baseline_summary,
        },
        "neutralization": {
            "method": "cross_sectional_linear_regression",
            "target_factor": "momentum_12_1",
            "price_field_for_momentum": "adj_close",
            "score_input": "baseline composite score after liquid_us eligibility filtering",
            "residual_score_name": NEUTRALIZED_STRATEGY,
        },
        "load_diagnostics": load_diag,
        "factor_diagnostics": factor_diag,
        "baseline_score_alignment": _score_alignment_diagnostic(
            rebuilt_scores=baseline_scores.loc[rb_dates],
            backtest_dir=baseline_dir,
        ),
        "neutralized_rebalance_skip": {
            "skip_zero_count": int(skip_zero_count),
            "skip_below_min_count": int(skip_below_count),
        },
        "assessment_thresholds": {
            "sharpe_material_drop": float(PERFORMANCE_SHARPE_MATERIAL_DELTA),
            "cagr_material_drop": float(PERFORMANCE_CAGR_MATERIAL_DELTA),
            "mean_exposure_abs_reduction": float(EXPOSURE_ABS_MATERIAL_REDUCTION),
            "positive_exposure_pct_drop": float(POSITIVE_EXPOSURE_MATERIAL_REDUCTION_PCT),
        },
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "momentum_exposure_comparison": str(exposure_path),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    latest_root = base_output_dir / "latest"
    _copy_latest(
        files={
            "results.csv": results_path,
            "summary.csv": summary_path,
            "momentum_exposure_comparison.csv": exposure_path,
            "manifest.json": manifest_path,
        },
        latest_root=latest_root,
    )

    print("| Strategy | CAGR | Vol | Sharpe | MaxDD | Turnover |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in results_df.to_dict(orient="records"):
        print(
            f"| {row['Strategy']} | {_format_metric(float(row['CAGR']))} | {_format_metric(float(row['Vol']))} | "
            f"{_format_metric(float(row['Sharpe']))} | {_format_metric(float(row['MaxDD']))} | "
            f"{_format_metric(float(row['Turnover']))} |"
        )

    print("")
    print("| Strategy | Mean Portfolio Momentum | Mean Universe Momentum | Mean Momentum Exposure | Percent Positive Exposure |")
    print("|---|---:|---:|---:|---:|")
    for row in exposure_summary_df.to_dict(orient="records"):
        print(
            f"| {row['Strategy']} | {_format_exposure_metric('Mean Portfolio Momentum', float(row['Mean Portfolio Momentum']))} | "
            f"{_format_exposure_metric('Mean Universe Momentum', float(row['Mean Universe Momentum']))} | "
            f"{_format_exposure_metric('Mean Momentum Exposure', float(row['Mean Momentum Exposure']))} | "
            f"{_format_exposure_metric('Percent Positive Exposure', float(row['Percent Positive Exposure']))} |"
        )

    print("")
    print(assessment["performance_assessment"])
    print(assessment["dependency_assessment"])
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {exposure_path}")
    print(f"Saved: {manifest_path}")
    print(f"Latest: {latest_root}")


if __name__ == "__main__":
    main()
