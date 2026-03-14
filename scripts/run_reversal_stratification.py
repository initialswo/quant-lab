"""Stratification study for reversal_1m inside the liquid_us dynamic universe."""

from __future__ import annotations

import json
import shutil
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
    _apply_universe_rebalance_skip,
    _collect_close_series,
    _collect_numeric_panel,
    _factor_required_history_days,
    _load_sector_map,
    _load_universe_seed_tickers,
    _prepare_close_panel,
    _price_panel_health_report,
    _resolve_universe_eligibility,
)
from quant_lab.factors.normalize import percentile_rank_cs, robust_preprocess_base
from quant_lab.factors.neutralize import neutralize_scores_cs
from quant_lab.factors.registry import compute_factors
from quant_lab.strategies.topn import build_topn_weights, rebalance_mask, simulate_portfolio
from quant_lab.universe.liquid_us import build_liquid_us_universe


UNIVERSE = "liquid_us"
FACTOR = "reversal_1m"
REBALANCE = "weekly"
TOP_N = 50
RESULT_COLUMNS: list[str] = [
    "bucket",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "AnnualTurnover",
    "AvgEligibleStocks",
    "RebalanceSkipped",
]
BASE_CONFIG: dict[str, Any] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
    "factor": FACTOR,
    "top_n": TOP_N,
    "rebalance": REBALANCE,
    "costs_bps": 10.0,
    "max_tickers": 2000,
    "universe": UNIVERSE,
    "universe_mode": "dynamic",
    "weighting": "equal",
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
    "universe_min_tickers": 20,
    "universe_skip_below_min_tickers": True,
    "universe_eligibility_source": "price",
    "use_sector_neutralization": True,
    "use_size_neutralization": True,
}
BUCKET_SPECS: dict[str, list[str]] = {
    "size": ["small", "mid", "large"],
    "volatility": ["low_vol", "mid_vol", "high_vol"],
    "liquidity": ["low_liq", "mid_liq", "high_liq"],
}


def _load_price_panels() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tickers = _load_universe_seed_tickers(
        universe=UNIVERSE,
        max_tickers=int(BASE_CONFIG["max_tickers"]),
        data_cache_dir=str(BASE_CONFIG["data_cache_dir"]),
    )
    ohlcv_map, data_source_summary = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(BASE_CONFIG["start"]),
        end=str(BASE_CONFIG["end"]),
        cache_dir=str(BASE_CONFIG["data_cache_dir"]),
        data_source=str(BASE_CONFIG["data_source"]),
        refresh=False,
        bulk_prepare=False,
    )
    close_cols, used_tickers, _, _, _ = _collect_close_series(ohlcv_map=ohlcv_map, requested_tickers=tickers)
    if len(used_tickers) < TOP_N + 1:
        raise ValueError(
            f"Not enough tickers with data: got {len(used_tickers)}, need at least {TOP_N + 1}."
        )

    close_raw = pd.concat(close_cols, axis=1, join="outer")
    volume_raw = _collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Volume",
    )
    close = _prepare_close_panel(close_raw=close_raw, price_fill_mode=str(BASE_CONFIG["price_fill_mode"]))
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
        print(
            "PRICE PANEL WARNING: dropping broken tickers "
            f"(close<=0 or max_abs_daily_return>200%) count={len(broken_tickers)} "
            f"sample={sorted(set(broken_tickers))[:10]}"
        )
        close = close.drop(columns=sorted(set(broken_tickers)), errors="ignore")
        if close.shape[1] < TOP_N + 1:
            raise ValueError(
                "Not enough tickers after dropping broken price series: "
                f"got {close.shape[1]}, need at least {TOP_N + 1}."
            )

    volume = (
        volume_raw.reindex(index=close.index, columns=close.columns)
        if not volume_raw.empty
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    volume = volume.astype(float)
    return close.astype(float), volume, data_source_summary


def _build_base_eligibility(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    effective_min_history_days = max(
        int(BASE_CONFIG["universe_min_history_days"]),
        _factor_required_history_days(
            factor_names=[FACTOR],
            factor_params_map={},
        ),
    )
    eligibility_price = build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=max(float(BASE_CONFIG["universe_min_price"]), float(BASE_CONFIG["min_price"])),
        min_avg_dollar_volume=float(BASE_CONFIG["min_avg_dollar_volume"]),
        adv_window=int(BASE_CONFIG["liquidity_lookback"]),
        min_history=int(effective_min_history_days),
    )
    scores_stub = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    return _resolve_universe_eligibility(
        eligibility_price=eligibility_price,
        scores=scores_stub,
        source=str(BASE_CONFIG["universe_eligibility_source"]),
    )


def _compute_reversal_scores(close: pd.DataFrame) -> pd.DataFrame:
    raw_scores = compute_factors(
        factor_names=[FACTOR],
        close=close,
        factor_params={},
    )[FACTOR]
    base_scores = robust_preprocess_base(raw_scores, winsor_p=0.05)
    sector_by_ticker = _load_sector_map("", list(close.columns))
    norm_scores = neutralize_scores_cs(
        base_scores,
        sector_by_ticker=sector_by_ticker if bool(BASE_CONFIG["use_sector_neutralization"]) else None,
        log_market_cap_by_ticker=None,
        use_sector_neutralization=bool(BASE_CONFIG["use_sector_neutralization"]),
        use_size_neutralization=bool(BASE_CONFIG["use_size_neutralization"]),
    )
    return percentile_rank_cs(norm_scores)


def _load_shares_outstanding(close: pd.DataFrame) -> pd.DataFrame | None:
    path = Path(str(BASE_CONFIG["fundamentals_path"]))
    if not path.exists():
        return None
    fundamentals = load_fundamentals_file(
        path=path,
        fallback_lag_days=int(BASE_CONFIG["fundamentals_fallback_lag_days"]),
    )
    aligned = align_fundamentals_to_daily_panel(
        fundamentals=fundamentals,
        price_index=pd.DatetimeIndex(close.index),
        price_columns=close.columns,
        value_columns=["shares_outstanding"],
    )
    shares = aligned.get("shares_outstanding")
    if shares is None:
        return None
    return shares.reindex(index=close.index, columns=close.columns).astype(float)


def _make_bucket_mask(metric_panel: pd.DataFrame, base_eligibility: pd.DataFrame, labels: list[str]) -> dict[str, pd.DataFrame]:
    out = {
        label: pd.DataFrame(False, index=metric_panel.index, columns=metric_panel.columns)
        for label in labels
    }
    for dt in metric_panel.index:
        eligible_row = base_eligibility.loc[dt]
        values = metric_panel.loc[dt].where(eligible_row.astype(bool))
        valid = values[np.isfinite(values.to_numpy(dtype=float))]
        if valid.empty:
            continue
        pct = valid.rank(method="average", pct=True)
        masks = {
            labels[0]: pct <= (1.0 / 3.0),
            labels[1]: (pct > (1.0 / 3.0)) & (pct <= (2.0 / 3.0)),
            labels[2]: pct > (2.0 / 3.0),
        }
        for label, mask in masks.items():
            out[label].loc[dt, mask.index] = mask.reindex(mask.index).fillna(False).astype(bool)
    return out


def _annual_turnover(sim: pd.DataFrame, index: pd.DatetimeIndex) -> float:
    rb_mask = rebalance_mask(index, REBALANCE).reindex(index).fillna(False)
    turnover_rb = sim.loc[rb_mask, "Turnover"] if "Turnover" in sim.columns else pd.Series(dtype=float)
    return float(turnover_rb.mean() * 52.0) if not turnover_rb.empty else float("nan")


def _simulate_bucket(
    bucket_name: str,
    bucket_mask: pd.DataFrame,
    close: pd.DataFrame,
    scores: pd.DataFrame,
    base_eligibility: pd.DataFrame,
) -> dict[str, Any]:
    bucket_elig = (
        base_eligibility.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
        & bucket_mask.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    )
    bucket_scores = apply_universe_filter_to_scores(scores, bucket_elig, exempt=set())
    rb_mask = rebalance_mask(bucket_scores.index, REBALANCE)
    bucket_scores_for_weights, skip_zero_count, skip_below_min_count = _apply_universe_rebalance_skip(
        scores=bucket_scores,
        rb_mask=rb_mask,
        universe_min_tickers=int(BASE_CONFIG["universe_min_tickers"]),
        universe_skip_below_min_tickers=bool(BASE_CONFIG["universe_skip_below_min_tickers"]),
    )
    weights = build_topn_weights(
        scores=bucket_scores_for_weights,
        close=close,
        top_n=TOP_N,
        rebalance=REBALANCE,
        weighting=str(BASE_CONFIG["weighting"]),
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
        close=close,
        weights=weights,
        costs_bps=float(BASE_CONFIG["costs_bps"]),
        rebalance_dates=pd.DatetimeIndex(close.index[rb_mask]),
    )
    metrics = compute_metrics(sim["DailyReturn"])
    eligible_counts = bucket_elig.loc[rb_mask].sum(axis=1).astype(int) if bool(rb_mask.any()) else pd.Series(dtype=int)
    return {
        "bucket": str(bucket_name),
        "CAGR": float(metrics.get("CAGR", float("nan"))),
        "Vol": float(metrics.get("Vol", float("nan"))),
        "Sharpe": float(metrics.get("Sharpe", float("nan"))),
        "MaxDD": float(metrics.get("MaxDD", float("nan"))),
        "AnnualTurnover": _annual_turnover(sim=sim, index=close.index),
        "AvgEligibleStocks": float(eligible_counts.mean()) if not eligible_counts.empty else float("nan"),
        "RebalanceSkipped": int(skip_zero_count + skip_below_min_count),
    }


def _metric_panels(close: pd.DataFrame, volume: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = close.pct_change()
    dollar_volume = close * volume
    adv20 = dollar_volume.rolling(int(BASE_CONFIG["liquidity_lookback"]), min_periods=int(BASE_CONFIG["liquidity_lookback"])).mean()
    vol20 = returns.rolling(20, min_periods=20).std(ddof=0)
    shares = _load_shares_outstanding(close)
    if shares is not None:
        market_cap = (close * shares).where((close * shares) > 0.0)
    else:
        market_cap = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    market_cap = market_cap.where(np.isfinite(market_cap), adv20 * close)
    market_cap = market_cap.where(market_cap > 0.0)
    return market_cap.astype(float), vol20.astype(float), adv20.astype(float)


def _run_bucket_family(
    metric_name: str,
    labels: list[str],
    metric_panel: pd.DataFrame,
    close: pd.DataFrame,
    scores: pd.DataFrame,
    base_eligibility: pd.DataFrame,
) -> pd.DataFrame:
    bucket_masks = _make_bucket_mask(metric_panel=metric_panel, base_eligibility=base_eligibility, labels=labels)
    rows = [
        _simulate_bucket(
            bucket_name=label,
            bucket_mask=bucket_masks[label],
            close=close,
            scores=scores,
            base_eligibility=base_eligibility,
        )
        for label in labels
    ]
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    print("")
    print(f"{metric_name.upper()} BUCKETS")
    print("-" * (len(metric_name) + 8))
    print(df.to_string(index=False))
    return df


def write_outputs(
    size_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    liq_df: pd.DataFrame,
    runtime_seconds: float,
    data_source_summary: dict[str, Any],
    results_root: Path = Path("results") / "reversal_stratification",
) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "reversal_size_buckets.csv": size_df,
        "reversal_volatility_buckets.csv": vol_df,
        "reversal_liquidity_buckets.csv": liq_df,
    }
    latest_map = {
        "reversal_size_buckets.csv": results_root / "reversal_size_buckets_latest.csv",
        "reversal_volatility_buckets.csv": results_root / "reversal_volatility_buckets_latest.csv",
        "reversal_liquidity_buckets.csv": results_root / "reversal_liquidity_buckets_latest.csv",
    }

    summary_paths: dict[str, str] = {}
    for name, df in file_map.items():
        path = outdir / name
        df.to_csv(path, index=False, float_format="%.10g")
        shutil.copy2(path, latest_map[name])
        summary_name = name.replace(".csv", "_summary.csv")
        summary_df = df.set_index("bucket")
        summary_path = outdir / summary_name
        summary_df.to_csv(summary_path, float_format="%.10g")
        summary_paths[summary_name] = str(summary_path)

    manifest = {
        "timestamp_utc": timestamp,
        "results_dir": str(outdir),
        "runtime_seconds": float(runtime_seconds),
        "base_config": BASE_CONFIG,
        "data_source_summary": data_source_summary,
        "files": {
            "size": str(outdir / "reversal_size_buckets.csv"),
            "volatility": str(outdir / "reversal_volatility_buckets.csv"),
            "liquidity": str(outdir / "reversal_liquidity_buckets.csv"),
            "summary_tables": summary_paths,
        },
    }
    (outdir / "reversal_stratification_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return outdir


def main() -> None:
    t0 = time.perf_counter()
    close, volume, data_source_summary = _load_price_panels()
    base_eligibility = _build_base_eligibility(close=close, volume=volume)
    scores = _compute_reversal_scores(close=close)
    scores = apply_universe_filter_to_scores(scores, base_eligibility, exempt=set())
    market_cap, vol20, adv20 = _metric_panels(close=close, volume=volume)

    size_df = _run_bucket_family(
        metric_name="size",
        labels=BUCKET_SPECS["size"],
        metric_panel=market_cap,
        close=close,
        scores=scores,
        base_eligibility=base_eligibility,
    )
    vol_df = _run_bucket_family(
        metric_name="volatility",
        labels=BUCKET_SPECS["volatility"],
        metric_panel=vol20,
        close=close,
        scores=scores,
        base_eligibility=base_eligibility,
    )
    liq_df = _run_bucket_family(
        metric_name="liquidity",
        labels=BUCKET_SPECS["liquidity"],
        metric_panel=adv20,
        close=close,
        scores=scores,
        base_eligibility=base_eligibility,
    )

    runtime_seconds = time.perf_counter() - t0
    outdir = write_outputs(
        size_df=size_df,
        vol_df=vol_df,
        liq_df=liq_df,
        runtime_seconds=runtime_seconds,
        data_source_summary=data_source_summary,
    )
    print("")
    print(f"Saved results to {outdir}")


if __name__ == "__main__":
    main()
