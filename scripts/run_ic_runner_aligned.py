"""Run runner-aligned IC diagnostics for a single factor on liquid_us."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel, load_fundamentals_file
from quant_lab.engine.runner import (
    _augment_factor_params_with_fundamentals,
    _collect_close_series,
    _collect_numeric_panel,
    _collect_price_series,
    _factor_required_history_days,
    _load_universe_seed_tickers,
    _prepare_close_panel,
)
from quant_lab.factors.normalize import percentile_rank_cs, robust_preprocess_base
from quant_lab.factors.registry import compute_factors
from quant_lab.research.cs_factor_diagnostics import (
    compute_forward_returns,
    run_cross_sectional_factor_diagnostics,
)
from quant_lab.strategies.topn import rebalance_mask
from quant_lab.universe.liquid_us import build_liquid_us_universe

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_ROOT = Path("results") / "ic_runner_aligned"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_HORIZON = 21
DEFAULT_QUANTILES = 5
DEFAULT_FACTOR = "momentum_12_1"
DEFAULT_MAX_TICKERS = 2000
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
PRICE_FILL_MODE = "ffill"
UNIVERSE_MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 10_000_000.0
LIQUIDITY_LOOKBACK = 20
UNIVERSE_MIN_HISTORY_DAYS = 252
FUNDAMENTALS_FALLBACK_LAG_DAYS = 60
SUBPERIODS: list[tuple[str, str, str]] = [
    ("2005-2009", "2005-01-01", "2009-12-31"),
    ("2010-2014", "2010-01-01", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2024", "2020-01-01", "2024-12-31"),
]
SUPPORTED_FACTORS = {"momentum_12_1", "reversal_1m", "gross_profitability", "book_to_market"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", default=DEFAULT_FACTOR)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--rebalance", choices=["daily", "weekly", "biweekly", "monthly"], default=DEFAULT_REBALANCE)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--fundamentals-path", default="")
    parser.add_argument("--quantiles", type=int, default=DEFAULT_QUANTILES)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--max_tickers", type=int, default=DEFAULT_MAX_TICKERS)
    parser.add_argument("--data_cache_dir", default=DATA_CACHE_DIR)
    return parser.parse_args()


def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)



def _load_price_panels(
    start: str,
    end: str,
    universe: str,
    data_cache_dir: str,
    max_tickers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tickers = _load_universe_seed_tickers(
        universe=str(universe),
        max_tickers=int(max_tickers),
        data_cache_dir=str(data_cache_dir),
    )
    ohlcv_map, data_source_summary = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(start),
        end=str(end),
        cache_dir=str(data_cache_dir),
        data_source=DATA_SOURCE,
        refresh=False,
        bulk_prepare=False,
    )
    close_cols, used_tickers, missing_tickers, rejected_tickers, _ = _collect_close_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=tickers,
    )
    if not close_cols:
        raise ValueError(
            "No valid OHLCV data found. "
            f"missing={len(missing_tickers)} rejected={len(rejected_tickers)}"
        )
    close_raw = pd.concat(close_cols, axis=1, join="outer")
    adj_close_cols, _, _, _, _ = _collect_price_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Adj Close",
        fallback_field="Close",
    )
    adj_close_raw = pd.concat(adj_close_cols, axis=1, join="outer") if adj_close_cols else close_raw.copy()
    close = _prepare_close_panel(close_raw=close_raw, price_fill_mode=PRICE_FILL_MODE).astype(float)
    adj_close = _prepare_close_panel(close_raw=adj_close_raw, price_fill_mode=PRICE_FILL_MODE).astype(float)
    adj_close = adj_close.reindex(index=close.index, columns=close.columns).where(
        adj_close.reindex(index=close.index, columns=close.columns).notna(),
        close,
    )
    volume = _collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Volume",
    ).reindex(index=close.index, columns=close.columns).astype(float)
    summary = dict(data_source_summary)
    summary["tickers_requested"] = int(len(tickers))
    summary["tickers_loaded"] = int(close.shape[1])
    return close, adj_close, volume, summary



def _aligned_fundamentals(close: pd.DataFrame, fundamentals_path: str) -> dict[str, pd.DataFrame]:
    if not str(fundamentals_path).strip():
        raise ValueError("This factor requires --fundamentals-path for PIT-aligned fundamentals.")
    fundamentals = load_fundamentals_file(
        path=str(fundamentals_path),
        fallback_lag_days=int(FUNDAMENTALS_FALLBACK_LAG_DAYS),
    )
    return align_fundamentals_to_daily_panel(
        fundamentals=fundamentals,
        price_index=pd.DatetimeIndex(close.index),
        price_columns=close.columns,
    )



def _factor_params_map(
    factor_name: str,
    close: pd.DataFrame,
    fundamentals_path: str,
) -> dict[str, dict[str, Any]]:
    factor = str(factor_name).strip()
    params = _augment_factor_params_with_fundamentals(
        factor_names=[factor],
        factor_params_map={},
        close=close,
        fundamentals_path=str(fundamentals_path),
        fundamentals_fallback_lag_days=int(FUNDAMENTALS_FALLBACK_LAG_DAYS),
    )
    if factor == "book_to_market":
        params[factor] = {"fundamentals_aligned": _aligned_fundamentals(close, fundamentals_path)}
    return params



def _build_factor_scores(
    factor_name: str,
    close: pd.DataFrame,
    fundamentals_path: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    factor = str(factor_name).strip()
    params_map = _factor_params_map(factor_name=factor, close=close, fundamentals_path=fundamentals_path)
    raw_scores = compute_factors(
        factor_names=[factor],
        close=close,
        factor_params=params_map,
    )[factor].astype(float)
    # Match the runner's standard factor-normalization path: robust base preprocessing,
    # then cross-sectional percentile ranking.
    prepared = percentile_rank_cs(robust_preprocess_base(raw_scores, winsor_p=0.05)).astype(float)
    return prepared, params_map



def _build_liquid_us_eligibility(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    factor_name: str,
    factor_params_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    effective_min_history_days = max(
        UNIVERSE_MIN_HISTORY_DAYS,
        _factor_required_history_days(
            factor_names=[str(factor_name)],
            factor_params_map=factor_params_map,
        ),
    )
    return build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=UNIVERSE_MIN_PRICE,
        min_avg_dollar_volume=MIN_AVG_DOLLAR_VOLUME,
        adv_window=LIQUIDITY_LOOKBACK,
        min_history=effective_min_history_days,
    )



def _summary_frame(summary: dict[str, Any], prefix: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    overall = dict(summary.get("overall", {}))
    overall["period"] = "overall"
    rows.append(overall)
    for period, payload in (summary.get("by_subperiod", {}) or {}).items():
        row = dict(payload)
        row["period"] = str(period)
        rows.append(row)
    frame = pd.DataFrame(rows)
    cols = ["period"] + [c for c in frame.columns if c != "period"]
    frame = frame.reindex(columns=cols)
    if prefix:
        rename = {c: f"{prefix}_{c}" for c in frame.columns if c != "period"}
        frame = frame.rename(columns=rename)
    return frame



def _plot_cumulative_ic(ic_by_date: pd.DataFrame, path: Path) -> bool:
    if ic_by_date.empty or "IC" not in ic_by_date.columns:
        return False
    frame = ic_by_date.copy().sort_index()
    frame["IC"] = pd.to_numeric(frame["IC"], errors="coerce")
    frame = frame.dropna(subset=["IC"])
    if frame.empty:
        return False
    frame["cumulative_ic"] = frame["IC"].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(frame.index, frame["cumulative_ic"], linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Cumulative IC")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative IC")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True



def _validate_args(args: argparse.Namespace) -> None:
    factor = str(args.factor).strip()
    if factor not in SUPPORTED_FACTORS:
        raise ValueError(f"Unsupported factor '{factor}'. Supported: {sorted(SUPPORTED_FACTORS)}")
    if str(args.universe).strip().lower() != "liquid_us":
        raise ValueError("This runner-aligned IC experiment currently supports only --universe liquid_us.")
    if factor == "book_to_market" and not str(args.fundamentals_path).strip():
        raise ValueError("book_to_market requires --fundamentals-path.")
    if int(args.horizon) <= 0:
        raise ValueError("--horizon must be > 0")
    if int(args.quantiles) < 2:
        raise ValueError("--quantiles must be >= 2")



def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(x) for x in obj]
    return obj



def main() -> None:
    args = _parse_args()
    _validate_args(args)
    start_ts = time.perf_counter()

    output_dir = Path(str(args.output_dir)).expanduser() if str(args.output_dir).strip() else None
    if output_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        outdir = RESULTS_ROOT / ts
    else:
        outdir = output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    close, adj_close, volume, data_source_summary = _load_price_panels(
        start=str(args.start),
        end=str(args.end),
        universe=str(args.universe),
        data_cache_dir=str(args.data_cache_dir),
        max_tickers=int(args.max_tickers),
    )
    factor_scores, factor_params_map = _build_factor_scores(
        factor_name=str(args.factor),
        close=adj_close,
        fundamentals_path=str(args.fundamentals_path),
    )
    eligibility = _build_liquid_us_eligibility(
        close=close,
        volume=volume,
        factor_name=str(args.factor),
        factor_params_map=factor_params_map,
    )
    factor_scores = factor_scores.where(eligibility, np.nan)
    rb = rebalance_mask(pd.DatetimeIndex(close.index), str(args.rebalance))
    forward_returns = compute_forward_returns(close=adj_close, horizon=int(args.horizon))

    report = run_cross_sectional_factor_diagnostics(
        factor_scores=factor_scores,
        close=adj_close,
        eligibility_mask=eligibility,
        rebalance=str(args.rebalance),
        quantiles=int(args.quantiles),
        horizon=int(args.horizon),
        subperiods=SUBPERIODS,
    )

    ic_by_date = pd.DataFrame(report["ic_by_date"]).copy()
    ic_by_date.index.name = "date"
    quantile_returns = pd.DataFrame(report["quantile_returns_by_date"]).copy()
    quantile_returns.index.name = "date"
    coverage_summary = _summary_frame(report.get("coverage_summary", {}), prefix="coverage")
    ic_summary = _summary_frame(report.get("ic_summary", {}), prefix="")

    ic_path = outdir / "ic_by_date.csv"
    ic_summary_path = outdir / "ic_summary.csv"
    quantile_path = outdir / "quantile_returns.csv"
    coverage_path = outdir / "coverage_summary.csv"
    manifest_path = outdir / "manifest.json"
    plot_path = outdir / "cumulative_ic.png"

    ic_by_date.to_csv(ic_path, float_format="%.10g")
    ic_summary.to_csv(ic_summary_path, index=False, float_format="%.10g")
    quantile_returns.to_csv(quantile_path, float_format="%.10g")
    coverage_summary.to_csv(coverage_path, index=False, float_format="%.10g")
    wrote_plot = _plot_cumulative_ic(ic_by_date=ic_by_date, path=plot_path)

    runtime_seconds = time.perf_counter() - start_ts
    rb_dates = pd.DatetimeIndex(close.index[rb])
    manifest = {
        "timestamp_utc": outdir.name,
        "script_name": "run_ic_runner_aligned.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": str(args.universe),
        "factor": str(args.factor),
        "rebalance": str(args.rebalance),
        "horizon": int(args.horizon),
        "quantiles": int(args.quantiles),
        "fundamentals_path": str(args.fundamentals_path),
        "data_cache_dir": str(args.data_cache_dir),
        "max_tickers": int(args.max_tickers),
        "runner_alignment": {
            "load_universe_seed_tickers": True,
            "collect_close_series": True,
            "prepare_close_panel": True,
            "augment_factor_params_with_fundamentals": True,
            "build_liquid_us_universe": True,
            "runner_style_preprocessing": "robust_preprocess_base -> percentile_rank_cs",
            "price_panel_for_signals": "adj_close",
            "price_panel_for_forward_returns": "adj_close",
            "price_panel_for_liquid_us_eligibility": "close",
        },
        "factor_params_keys": sorted((factor_params_map.get(str(args.factor), {}) or {}).keys()),
        "data_source_summary": data_source_summary,
        "n_dates": int(len(adj_close.index)),
        "n_tickers": int(adj_close.shape[1]),
        "n_rebalance_dates": int(len(rb_dates)),
        "n_forward_return_observations": int(forward_returns.notna().sum().sum()),
        "median_eligible_names": float(eligibility.loc[rb].sum(axis=1).median()) if bool(rb.any()) else float("nan"),
        "median_scored_names": float(factor_scores.loc[rb].notna().sum(axis=1).median()) if bool(rb.any()) else float("nan"),
        "outputs": {
            "ic_by_date": str(ic_path),
            "ic_summary": str(ic_summary_path),
            "quantile_returns": str(quantile_path),
            "coverage_summary": str(coverage_path),
            "manifest": str(manifest_path),
            "cumulative_ic_plot": str(plot_path) if wrote_plot else "",
        },
        "runtime_seconds": float(runtime_seconds),
        "notes": [
            "IC diagnostics reuse runner data loading, PIT fundamentals injection, liquid_us universe construction, and runner-style percentile-rank preprocessing.",
            "Forward returns are close[t+h] / close[t] - 1 and are evaluated only on rebalance dates inside run_cross_sectional_factor_diagnostics.",
            "book_to_market is supported only when --fundamentals-path is provided explicitly.",
        ],
    }
    manifest_path.write_text(json.dumps(_to_serializable(manifest), indent=2, sort_keys=True), encoding="utf-8")

    if output_dir is None:
        _copy_latest(
            files={
                "ic_by_date.csv": ic_path,
                "ic_summary.csv": ic_summary_path,
                "quantile_returns.csv": quantile_path,
                "coverage_summary.csv": coverage_path,
                "manifest.json": manifest_path,
                **({"cumulative_ic.png": plot_path} if wrote_plot else {}),
            },
            latest_root=RESULTS_ROOT / "latest",
        )

    overall = dict(report.get("ic_summary", {}).get("overall", {}))
    print("NOTE: IC diagnostics use adjusted prices for signal construction and forward returns; raw close remains the liquid_us eligibility basis to match run_backtest.")
    print("IC SUMMARY")
    print("----------")
    print(f"{'factor':22s} {'mean_ic':>10s} {'ic_ir':>10s} {'hit_rate':>10s}")
    print(
        f"{str(args.factor):22s} "
        f"{float(overall.get('mean_ic', np.nan)):10.4f} "
        f"{float(overall.get('ic_ir', np.nan)):10.4f} "
        f"{float(overall.get('ic_hit_rate', np.nan)):10.4f}"
    )
    print("")
    print(f"Saved: {ic_path}")
    print(f"Saved: {ic_summary_path}")
    print(f"Saved: {quantile_path}")
    print(f"Saved: {coverage_path}")
    print(f"Saved: {manifest_path}")
    if wrote_plot:
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
