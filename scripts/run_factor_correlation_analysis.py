"""Compute average cross-sectional factor score correlations on liquid_us."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.loaders import load_ohlcv_for_research
from quant_lab.engine.runner import (
    _augment_factor_params_with_fundamentals,
    _factor_required_history_days,
    _load_universe_seed_tickers,
    _prepare_close_panel,
)
from quant_lab.factors.normalize import percentile_rank_cs, robust_preprocess_base
from quant_lab.factors.registry import compute_factors
from quant_lab.research.signal_correlation import run_signal_correlation
from quant_lab.strategies.topn import rebalance_mask
from quant_lab.universe.liquid_us import build_liquid_us_universe


RESULTS_ROOT = Path("results") / "factor_correlation_analysis"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_REBALANCE = "weekly"
DEFAULT_MAX_TICKERS = 2000
DEFAULT_FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
DATA_CACHE_DIR = "data/equities"
PRICE_FILL_MODE = "ffill"
FACTOR_NAMES = ["gross_profitability", "roa"]
DISPLAY_NAME_MAP = {
    "gross_profitability": "profitability",
    "roa": "roa",
}
FUNDAMENTALS_FALLBACK_LAG_DAYS = 60
UNIVERSE_MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 10_000_000.0
LIQUIDITY_LOOKBACK = 20
UNIVERSE_MIN_HISTORY_DAYS = 252
CORRELATION_METHOD = "pearson"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--rebalance", choices=["daily", "weekly", "biweekly", "monthly"], default=DEFAULT_REBALANCE)
    parser.add_argument("--fundamentals-path", default=DEFAULT_FUNDAMENTALS_PATH)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--max_tickers", type=int, default=DEFAULT_MAX_TICKERS)
    return parser.parse_args()



def _load_price_panels(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tickers = _load_universe_seed_tickers(
        universe=str(args.universe),
        max_tickers=int(args.max_tickers),
        data_cache_dir=DATA_CACHE_DIR,
    )
    data = load_ohlcv_for_research(
        start=str(args.start),
        end=str(args.end),
        universe=None,
        tickers=tickers,
        max_tickers=None,
        store_root=DATA_CACHE_DIR,
    )
    close_raw = data.panels.get("close", pd.DataFrame()).astype(float)
    if close_raw.empty:
        raise ValueError("Empty close panel returned by load_ohlcv_for_research().")
    close = _prepare_close_panel(close_raw=close_raw, price_fill_mode=PRICE_FILL_MODE).astype(float)
    volume = data.panels.get("volume", pd.DataFrame()).astype(float)
    volume = volume.reindex(index=close.index, columns=close.columns)
    diagnostics = dict(data.diagnostics)
    diagnostics["price_fill_mode"] = PRICE_FILL_MODE
    diagnostics["loader_mode"] = "ticker_list"
    diagnostics["tickers_requested"] = int(len(tickers))
    diagnostics["universe"] = str(args.universe)
    return close, volume, diagnostics



def _build_factor_scores(
    close: pd.DataFrame,
    fundamentals_path: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    factor_params_map = _augment_factor_params_with_fundamentals(
        factor_names=list(FACTOR_NAMES),
        factor_params_map={},
        close=close,
        fundamentals_path=str(fundamentals_path),
        fundamentals_fallback_lag_days=int(FUNDAMENTALS_FALLBACK_LAG_DAYS),
    )
    raw_scores = compute_factors(
        factor_names=list(FACTOR_NAMES),
        close=close,
        factor_params=factor_params_map,
    )
    prepared_scores = {
        name: percentile_rank_cs(robust_preprocess_base(raw_scores[name], winsor_p=0.05)).astype(float)
        for name in FACTOR_NAMES
    }
    return prepared_scores, factor_params_map



def _build_eligibility(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    factor_params_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    effective_min_history = max(
        int(UNIVERSE_MIN_HISTORY_DAYS),
        _factor_required_history_days(
            factor_names=list(FACTOR_NAMES),
            factor_params_map=factor_params_map,
        ),
    )
    return build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=float(UNIVERSE_MIN_PRICE),
        min_avg_dollar_volume=float(MIN_AVG_DOLLAR_VOLUME),
        adv_window=int(LIQUIDITY_LOOKBACK),
        min_history=int(effective_min_history),
    )



def _masked_rebalance_panels(
    factor_scores: dict[str, pd.DataFrame],
    eligibility: pd.DataFrame,
    rebalance: str,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    rb_mask = rebalance_mask(pd.DatetimeIndex(eligibility.index), str(rebalance)).reindex(eligibility.index).fillna(False)
    rb_dates = pd.DatetimeIndex(eligibility.index[rb_mask])
    masked: dict[str, pd.DataFrame] = {}
    elig = eligibility.astype(bool)
    for name, panel in factor_scores.items():
        aligned = panel.reindex(index=elig.index, columns=elig.columns).where(elig).astype(float)
        masked[name] = aligned.reindex(index=rb_dates)
    return masked, rb_dates



def _rename_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    labels = [DISPLAY_NAME_MAP[name] for name in FACTOR_NAMES]
    return matrix.rename(index=DISPLAY_NAME_MAP, columns=DISPLAY_NAME_MAP).reindex(index=labels, columns=labels)



def _format_display(matrix: pd.DataFrame) -> pd.DataFrame:
    display = matrix.copy().astype(float)
    for col in display.columns:
        display[col] = display[col].map(lambda x: "-" if pd.isna(x) else f"{float(x):.4f}")
    return display



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



def main() -> None:
    args = _parse_args()
    if str(args.universe).strip().lower() != "liquid_us":
        raise ValueError("This factor correlation analysis currently supports only --universe liquid_us.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(str(args.output_dir)) if str(args.output_dir).strip() else RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    close, volume, loader_diagnostics = _load_price_panels(args=args)
    factor_scores, factor_params_map = _build_factor_scores(
        close=close,
        fundamentals_path=str(args.fundamentals_path),
    )
    eligibility = _build_eligibility(
        close=close,
        volume=volume,
        factor_params_map=factor_params_map,
    )
    masked_panels, rb_dates = _masked_rebalance_panels(
        factor_scores=factor_scores,
        eligibility=eligibility,
        rebalance=str(args.rebalance),
    )
    report = run_signal_correlation(signal_panels=masked_panels, method=CORRELATION_METHOD)
    matrix_df = _rename_matrix(pd.DataFrame(report["average_correlation_matrix"]))

    matrix_path = output_dir / "factor_score_correlation_matrix.csv"
    manifest_path = output_dir / "manifest.json"

    matrix_df.to_csv(matrix_path, float_format="%.10g")

    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_factor_correlation_analysis.py",
        "factor_names": list(FACTOR_NAMES),
        "display_names": DISPLAY_NAME_MAP,
        "correlation_method": CORRELATION_METHOD,
        "backtest_configuration": {
            "start": str(args.start),
            "end": str(args.end),
            "universe": str(args.universe),
            "rebalance": str(args.rebalance),
            "max_tickers": int(args.max_tickers),
            "fundamentals_path": str(args.fundamentals_path),
            "data_cache_dir": DATA_CACHE_DIR,
            "price_fill_mode": PRICE_FILL_MODE,
        },
        "liquid_us_thresholds": {
            "min_price": float(UNIVERSE_MIN_PRICE),
            "min_avg_dollar_volume": float(MIN_AVG_DOLLAR_VOLUME),
            "liquidity_lookback": int(LIQUIDITY_LOOKBACK),
            "min_history_days": int(max(UNIVERSE_MIN_HISTORY_DAYS, _factor_required_history_days(FACTOR_NAMES, factor_params_map))),
        },
        "loader_diagnostics": loader_diagnostics,
        "rebalance_dates_evaluated": int(len(rb_dates)),
        "output_paths": {
            "factor_score_correlation_matrix": str(matrix_path),
            "manifest": str(manifest_path),
        },
        "runtime_seconds": float(time.perf_counter() - t0),
        "coverage_summary": report.get("coverage_summary", pd.DataFrame()),
    }
    manifest_path.write_text(json.dumps(_to_serializable(manifest), indent=2), encoding="utf-8")

    print("## Factor Correlation Matrix")
    print("")
    print(_format_display(matrix_df).to_string())


if __name__ == "__main__":
    main()
