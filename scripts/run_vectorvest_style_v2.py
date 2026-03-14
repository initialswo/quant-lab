"""Second-pass VectorVest-inspired screened stock backtest."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel, load_fundamentals_file
from quant_lab.data.universe_dynamic import apply_universe_filter_to_scores
from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import (
    _collect_close_series,
    _collect_numeric_panel,
    _factor_required_history_days,
    _load_universe_seed_tickers,
    _prepare_close_panel,
    _price_panel_health_report,
    _resolve_universe_eligibility,
)
from quant_lab.factors.combine import aggregate_factor_scores
from quant_lab.factors.normalize import preprocess_factor_scores
from quant_lab.factors.registry import compute_factors
from quant_lab.results.registry import append_registry_row
from quant_lab.strategies.topn import rebalance_mask, simulate_portfolio
from quant_lab.universe.liquid_us import build_liquid_us_universe


OUT_ROOT = Path("results") / "vectorvest_style_v2"
RUN_CONFIG: dict[str, Any] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
    "universe": "liquid_us",
    "universe_mode": "dynamic",
    "top_n": 50,
    "costs_bps": 10.0,
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "price_fill_mode": "ffill",
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
    "universe_min_history_days": 300,
    "universe_min_price": 1.0,
    "min_price": 0.0,
    "min_avg_dollar_volume": 0.0,
    "liquidity_lookback": 20,
    "universe_eligibility_source": "price",
    "quarterly_reconstitution": "first weekly evaluation date of each calendar quarter",
    "weekly_evaluation": True,
    "ma_window": 50,
}
V1_SUMMARY_PATH = Path("results/vectorvest_style_v1/20260312_025001/summary.json")


def _load_price_volume() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tickers = _load_universe_seed_tickers(
        universe=str(RUN_CONFIG["universe"]),
        max_tickers=int(RUN_CONFIG["max_tickers"]),
        data_cache_dir=str(RUN_CONFIG["data_cache_dir"]),
    )
    ohlcv_map, data_source_summary = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(RUN_CONFIG["start"]),
        end=str(RUN_CONFIG["end"]),
        cache_dir=str(RUN_CONFIG["data_cache_dir"]),
        data_source=str(RUN_CONFIG["data_source"]),
        refresh=False,
        bulk_prepare=False,
    )
    close_cols, used_tickers, _, _, _ = _collect_close_series(ohlcv_map=ohlcv_map, requested_tickers=tickers)
    if len(used_tickers) < int(RUN_CONFIG["top_n"]) + 1:
        raise ValueError(
            f"Not enough tickers with data: got {len(used_tickers)}, need at least {int(RUN_CONFIG['top_n']) + 1}."
        )
    close_raw = pd.concat(close_cols, axis=1, join="outer")
    volume_raw = _collect_numeric_panel(ohlcv_map=ohlcv_map, requested_tickers=used_tickers, field="Volume")
    close = _prepare_close_panel(close_raw=close_raw, price_fill_mode=str(RUN_CONFIG["price_fill_mode"]))
    close = close.dropna(thresh=int(0.8 * close.shape[1]))
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
    volume = (
        volume_raw.reindex(index=close.index, columns=close.columns)
        if not volume_raw.empty
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    return close.astype(float), volume.astype(float), data_source_summary


def _load_fundamentals_aligned(close: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
    path = Path(str(RUN_CONFIG["fundamentals_path"]))
    if not path.exists():
        return None
    fundamentals = load_fundamentals_file(
        path=path,
        fallback_lag_days=int(RUN_CONFIG["fundamentals_fallback_lag_days"]),
    )
    return align_fundamentals_to_daily_panel(
        fundamentals=fundamentals,
        price_index=pd.DatetimeIndex(close.index),
        price_columns=close.columns,
    )


def _build_dynamic_eligibility(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    factor_params_map: dict[str, dict[str, Any]] = {}
    effective_min_history_days = max(
        int(RUN_CONFIG["universe_min_history_days"]),
        _factor_required_history_days(
            factor_names=["gross_profitability", "momentum_12_1"],
            factor_params_map=factor_params_map,
        ),
    )
    eligibility_price = build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=max(float(RUN_CONFIG["universe_min_price"]), float(RUN_CONFIG["min_price"])),
        min_avg_dollar_volume=float(RUN_CONFIG["min_avg_dollar_volume"]),
        adv_window=int(RUN_CONFIG["liquidity_lookback"]),
        min_history=int(effective_min_history_days),
    )
    scores_stub = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    return _resolve_universe_eligibility(
        eligibility_price=eligibility_price,
        scores=scores_stub,
        source=str(RUN_CONFIG["universe_eligibility_source"]),
    )


def _compute_revenue_growth(
    close: pd.DataFrame,
    fundamentals_aligned: dict[str, pd.DataFrame] | None,
) -> tuple[pd.DataFrame | None, str | None]:
    if not fundamentals_aligned or "revenue" not in fundamentals_aligned:
        return None, "revenue not available in aligned PIT fundamentals"
    revenue = fundamentals_aligned["revenue"].reindex(index=close.index, columns=close.columns).astype(float)
    lagged = revenue.shift(252)
    growth = revenue.divide(lagged) - 1.0
    growth = growth.where(lagged > 0.0)
    valid = int(growth.notna().sum().sum())
    if valid <= 0:
        return None, "revenue growth panel had no valid PIT observations"
    return growth.astype(float), None


def _build_screen_components(
    close: pd.DataFrame,
    eligibility: pd.DataFrame,
    fundamentals_aligned: dict[str, pd.DataFrame] | None,
) -> tuple[dict[str, pd.DataFrame], list[str], str | None]:
    factor_params: dict[str, dict[str, Any]] = {}
    if fundamentals_aligned is not None:
        factor_params["gross_profitability"] = {"fundamentals_aligned": fundamentals_aligned}
    raw = compute_factors(
        factor_names=["gross_profitability", "momentum_12_1"],
        close=close,
        factor_params=factor_params,
    )
    components = {
        "gross_profitability": preprocess_factor_scores(raw["gross_profitability"]),
        "momentum_12_1": preprocess_factor_scores(raw["momentum_12_1"]),
    }
    implemented = ["gross_profitability", "momentum_12_1"]
    revenue_growth, revenue_skip_reason = _compute_revenue_growth(
        close=close,
        fundamentals_aligned=fundamentals_aligned,
    )
    if revenue_growth is not None:
        components["revenue_growth"] = preprocess_factor_scores(revenue_growth)
        implemented.append("revenue_growth")
        revenue_skip_reason = None
    components = {
        name: apply_universe_filter_to_scores(panel, eligibility, exempt=set()).astype(float)
        for name, panel in components.items()
    }
    return components, implemented, revenue_skip_reason


def _quarterly_reconstitution_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    weekly_mask = rebalance_mask(index, "weekly")
    weekly_dates = pd.DatetimeIndex(index[weekly_mask])
    out: list[pd.Timestamp] = []
    prev_period: tuple[int, int] | None = None
    for dt in weekly_dates:
        ts = pd.Timestamp(dt)
        period = (int(ts.year), int(ts.quarter))
        if prev_period != period:
            out.append(ts)
            prev_period = period
    return pd.DatetimeIndex(out)


def _build_quarterly_rankings(
    screen_scores: pd.DataFrame,
    quarter_dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, list[str]]:
    rankings: dict[pd.Timestamp, list[str]] = {}
    lagged_scores = screen_scores.shift(1)
    for dt in quarter_dates:
        row = lagged_scores.loc[dt].dropna().sort_values(ascending=False, kind="mergesort")
        rankings[pd.Timestamp(dt)] = row.index.tolist()
    return rankings


def _passes_timing(
    ts: pd.Timestamp,
    ticker: str,
    prev_close: pd.DataFrame,
    prev_sma: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> bool:
    if ticker not in prev_close.columns or ticker not in prev_sma.columns:
        return False
    if ticker not in eligibility.columns:
        return False
    if not bool(eligibility.loc[ts, ticker]):
        return False
    px = prev_close.loc[ts, ticker]
    sma = prev_sma.loc[ts, ticker]
    return bool(pd.notna(px) and pd.notna(sma) and float(px) > float(sma))


def _build_weekly_weights_v2(
    close: pd.DataFrame,
    eligibility: pd.DataFrame,
    quarter_dates: pd.DatetimeIndex,
    quarter_rankings: dict[pd.Timestamp, list[str]],
    top_n: int,
    ma_window: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
    weekly_eval = rebalance_mask(close.index, "weekly")
    sma = close.rolling(int(ma_window), min_periods=int(ma_window)).mean()
    prev_close = close.shift(1)
    prev_sma = sma.shift(1)

    current_ranking: list[str] = []
    current_holdings: list[str] = []
    quarter_set = set(pd.DatetimeIndex(quarter_dates))
    selection_history: dict[str, list[str]] = {}
    replacement_log: list[dict[str, Any]] = []

    for dt in close.index:
        ts = pd.Timestamp(dt)
        if not bool(weekly_eval.loc[ts]):
            continue

        if ts in quarter_set:
            current_ranking = list(quarter_rankings.get(ts, []))
            current_holdings = []
            selection_history[str(ts.date())] = current_ranking[: int(top_n)]

        surviving = [t for t in current_holdings if _passes_timing(ts, t, prev_close, prev_sma, eligibility)]
        held_set = set(surviving)
        replacements_added: list[str] = []
        for ticker in current_ranking:
            if len(surviving) >= int(top_n):
                break
            if ticker in held_set:
                continue
            if not _passes_timing(ts, ticker, prev_close, prev_sma, eligibility):
                continue
            surviving.append(str(ticker))
            held_set.add(str(ticker))
            replacements_added.append(str(ticker))

        current_holdings = surviving
        row = pd.Series(0.0, index=close.columns, dtype=float)
        if current_holdings:
            row.loc[current_holdings] = 1.0 / float(len(current_holdings))
        weights.loc[ts] = row
        replacement_log.append(
            {
                "date": str(ts.date()),
                "quarter_reconstitution": bool(ts in quarter_set),
                "active_holdings": int(len(current_holdings)),
                "new_additions": replacements_added,
            }
        )

    weights = weights.ffill().fillna(0.0)
    return weights.astype(float), {
        "selection_history": selection_history,
        "replacement_log": replacement_log,
    }


def _annual_turnover(sim: pd.DataFrame, index: pd.DatetimeIndex) -> float:
    weekly_mask = rebalance_mask(index, "weekly")
    turnover_rb = sim.loc[weekly_mask, "Turnover"] if "Turnover" in sim.columns else pd.Series(dtype=float)
    return float(turnover_rb.mean() * 52.0) if not turnover_rb.empty else float("nan")


def _load_v1_summary() -> dict[str, Any] | None:
    if not V1_SUMMARY_PATH.exists():
        return None
    return json.loads(V1_SUMMARY_PATH.read_text(encoding="utf-8"))


def _strategy_notes(
    implemented_components: list[str],
    revenue_skip_reason: str | None,
    quarter_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    return {
        "strategy_name": "vectorvest_style_v2",
        "implemented_screen_fields": implemented_components,
        "revenue_growth_definition": (
            "PIT-aligned revenue / PIT-aligned revenue from 252 trading days earlier - 1"
            if "revenue_growth" in implemented_components
            else None
        ),
        "skipped_candidate_fields": (
            []
            if revenue_skip_reason is None
            else [{"field": "revenue_growth", "reason": str(revenue_skip_reason)}]
        ),
        "screen_weighting": "equal weight across implemented screen components",
        "screen_definition": "composite screen score is the simple average of normalized component scores",
        "screen_score_timing": "quarterly reconstitution uses prior trading day's composite screen values",
        "reconstitution_schedule": {
            "rule": "first weekly evaluation date of each calendar quarter",
            "dates": [str(d.date()) for d in quarter_dates],
        },
        "timing_rule": {
            "window": 50,
            "evaluation_frequency": "weekly",
            "timing": "hold only if prior close > prior 50-day SMA on weekly evaluation dates",
            "lookahead": "none; lagged values only",
        },
        "replacement_rule": {
            "rule": "on each weekly evaluation date, after removing held names that fail timing or eligibility, refill vacancies from the frozen quarterly ranking list",
            "candidate_constraints": [
                "not already held",
                "currently eligible",
                "passes lagged 50-day SMA timing rule",
            ],
            "selection_priority": "highest-ranked available quarterly screen candidate first",
            "cash_behavior": "slots remain cash if no valid replacement exists",
        },
        "limitations": [
            "screen remains intentionally simple",
            "revenue_growth uses a simple PIT-aligned 252-trading-day proxy",
            "quarterly ranking is frozen between reconstitutions",
            "replacement logic uses the frozen quarterly rank list rather than a continuously refreshed score",
        ],
        "universe": "liquid_us dynamic",
        "shorting": False,
    }


def main() -> None:
    t0 = time.perf_counter()
    close, volume, data_source_summary = _load_price_volume()
    fundamentals_aligned = _load_fundamentals_aligned(close=close)
    eligibility = _build_dynamic_eligibility(close=close, volume=volume)
    components, implemented_components, revenue_skip_reason = _build_screen_components(
        close=close,
        eligibility=eligibility,
        fundamentals_aligned=fundamentals_aligned,
    )
    comp_weights = {name: 1.0 for name in implemented_components}
    screen_scores = aggregate_factor_scores(
        scores={name: components[name] for name in implemented_components},
        weights=comp_weights,
        method="linear",
        require_all_factors=True,
    ).astype(float)
    screen_scores = apply_universe_filter_to_scores(screen_scores, eligibility, exempt=set()).astype(float)

    quarter_dates = _quarterly_reconstitution_dates(close.index)
    quarter_rankings = _build_quarterly_rankings(screen_scores=screen_scores, quarter_dates=quarter_dates)
    weights, state_artifacts = _build_weekly_weights_v2(
        close=close,
        eligibility=eligibility,
        quarter_dates=quarter_dates,
        quarter_rankings=quarter_rankings,
        top_n=int(RUN_CONFIG["top_n"]),
        ma_window=int(RUN_CONFIG["ma_window"]),
    )
    weekly_dates = pd.DatetimeIndex(close.index[rebalance_mask(close.index, "weekly")])
    sim = simulate_portfolio(
        close=close,
        weights=weights,
        costs_bps=float(RUN_CONFIG["costs_bps"]),
        rebalance_dates=weekly_dates,
    )

    metrics = compute_metrics(sim["DailyReturn"])
    active_holdings = (weights > 0.0).sum(axis=1).astype(int)
    cash_weight = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
    equity = pd.DataFrame(
        {
            "date": close.index,
            "equity": sim["Equity"].to_numpy(dtype=float),
            "daily_return": sim["DailyReturn"].to_numpy(dtype=float),
            "turnover": sim["Turnover"].to_numpy(dtype=float),
            "active_holdings": active_holdings.to_numpy(dtype=int),
            "cash_weight": cash_weight.to_numpy(dtype=float),
        }
    )

    runtime_seconds = time.perf_counter() - t0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = OUT_ROOT / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    notes = _strategy_notes(
        implemented_components=implemented_components,
        revenue_skip_reason=revenue_skip_reason,
        quarter_dates=quarter_dates,
    )
    v1_summary = _load_v1_summary()
    summary = {
        "Strategy": "vectorvest_style_v2",
        "Universe": str(RUN_CONFIG["universe"]),
        "UniverseMode": str(RUN_CONFIG["universe_mode"]),
        "Start": str(RUN_CONFIG["start"]),
        "End": str(RUN_CONFIG["end"]),
        "TopN": int(RUN_CONFIG["top_n"]),
        "CostsBps": float(RUN_CONFIG["costs_bps"]),
        "ReconstitutionRule": str(RUN_CONFIG["quarterly_reconstitution"]),
        "MAWindow": int(RUN_CONFIG["ma_window"]),
        "ImplementedScreenFields": implemented_components,
        "RevenueGrowthImplemented": bool("revenue_growth" in implemented_components),
        "RevenueGrowthSkipReason": revenue_skip_reason,
        "AnnualTurnover": _annual_turnover(sim=sim, index=close.index),
        "AvgActiveHoldings": float(active_holdings.mean()),
        "AvgCashWeight": float(cash_weight.mean()),
        "RuntimeSeconds": float(runtime_seconds),
        "ReferenceV1SummaryPath": str(V1_SUMMARY_PATH) if v1_summary is not None else None,
        **metrics,
    }

    (outdir / "run_config.json").write_text(json.dumps(RUN_CONFIG, indent=2), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "strategy_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    equity.to_csv(outdir / "equity.csv", index=False, float_format="%.10g")
    (outdir / "selection_history.json").write_text(
        json.dumps(state_artifacts["selection_history"], indent=2),
        encoding="utf-8",
    )
    (outdir / "replacement_log.json").write_text(
        json.dumps(state_artifacts["replacement_log"], indent=2),
        encoding="utf-8",
    )

    append_registry_row(
        {
            "RunTag": timestamp,
            "Start": str(RUN_CONFIG["start"]),
            "End": str(RUN_CONFIG["end"]),
            "TickersUsed": int(close.shape[1]),
            "TopN": int(RUN_CONFIG["top_n"]),
            "Rebalance": "weekly",
            "CostsBps": float(RUN_CONFIG["costs_bps"]),
            "TrendFilter": False,
            "SaveArtifacts": True,
            "WindowsJsonPath": "",
            "CAGR": float(metrics.get("CAGR", float("nan"))),
            "Vol": float(metrics.get("Vol", float("nan"))),
            "Sharpe": float(metrics.get("Sharpe", float("nan"))),
            "MaxDD": float(metrics.get("MaxDD", float("nan"))),
            "Outdir": str(outdir),
            "TimestampUTC": timestamp,
            "strategy_name": "vectorvest_style_v2",
            "screen_fields": ";".join(implemented_components),
        }
    )

    print("")
    print("VECTORVEST STYLE V2")
    print("-------------------")
    print(json.dumps(summary, indent=2))
    print("")
    print(f"Saved: {outdir / 'summary.json'}")
    print(f"Saved: {outdir / 'equity.csv'}")
    print(f"Saved: {outdir / 'run_config.json'}")
    print(f"Saved: {outdir / 'strategy_notes.json'}")


if __name__ == "__main__":
    main()
