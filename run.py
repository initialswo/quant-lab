"""CLI entrypoint for minimal quant_lab backtests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# Avoid matplotlib cache warnings in non-interactive CLI runs.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# Make local src/ importable when running from repo root.
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.data.historical_membership import load_historical_membership
from quant_lab.data.ingest import ingest_equity_cache, ingest_universe_membership_csv
from quant_lab.data.stooq_downloader import download_stooq_universe, load_symbol_file
from quant_lab.data.tiingo_downloader import (
    download_tiingo_universe,
    load_symbol_file as load_tiingo_symbol_file,
)
from quant_lab.data.tiingo_universe import DEFAULT_TIINGO_US_COMMON_EQUITY_MANIFEST
from quant_lab.data.universe import load_sp500_tickers
from quant_lab.engine.runner import _collect_close_series, _prepare_close_panel, run_backtest, run_walkforward
from quant_lab.factors.normalize import preprocess_factor_scores
from quant_lab.factors.registry import compute_factors, list_factors
from quant_lab.research.factor_diagnostics import print_factor_diagnostics, run_factor_diagnostics
from quant_lab.research.factor_returns import (
    print_factor_return_analysis,
    print_factor_seasonality,
    plot_factor_seasonality,
    run_factor_return_analysis,
    run_factor_return_correlation,
    run_factor_seasonality,
)
from quant_lab.research.factor_heatmap import compute_momentum_sweep_matrix, plot_heatmap
from quant_lab.research.signal_correlation import print_signal_correlation, run_signal_correlation
from quant_lab.data.sp500_membership import save_sp500_historical_membership_csv


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    items = _parse_csv_list(raw)
    if not items:
        raise ValueError("Expected at least one integer value.")
    return [int(x) for x in items]


def _parse_csv_floats(raw: str) -> list[float]:
    items = _parse_csv_list(raw)
    if not items:
        raise ValueError("Expected at least one float value.")
    return [float(x) for x in items]


def _parse_csv_lookbacks(raw: str) -> list[int]:
    vals = _parse_csv_ints(raw)
    if any(v <= 0 for v in vals):
        raise ValueError("lookbacks must be positive integers.")
    return vals


def _parse_csv_rebalance(raw: str) -> list[str]:
    valid = {"daily", "weekly", "monthly"}
    items = [x.lower() for x in _parse_csv_list(raw)]
    if not items:
        raise ValueError("Expected at least one rebalance frequency.")
    bad = [x for x in items if x not in valid]
    if bad:
        raise ValueError(f"Invalid rebalance values: {bad}. Allowed: daily,weekly,monthly")
    return items


def _parse_csv_weighting(raw: str) -> list[str]:
    valid = {"equal", "inv_vol", "score", "score_inv_vol"}
    items = [x.lower() for x in _parse_csv_list(raw)]
    if not items:
        raise ValueError("Expected at least one weighting value.")
    bad = [x for x in items if x not in valid]
    if bad:
        raise ValueError(
            f"Invalid weighting values: {bad}. Allowed: equal,inv_vol,score,score_inv_vol"
        )
    return items


def _parse_csv_bool01(raw: str) -> list[bool]:
    items = _parse_csv_list(raw)
    if not items:
        raise ValueError("Expected at least one trend_filter value (0 or 1).")
    out: list[bool] = []
    for x in items:
        if x not in {"0", "1"}:
            raise ValueError(f"Invalid trend_filter value '{x}', expected 0 or 1.")
        out.append(x == "1")
    return out


def _parse_scalar(value: str):
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_factor_params(raw: str | None) -> dict:
    if raw is None or not raw.strip():
        return {}
    params: dict[str, object] = {}
    for item in _parse_csv_list(raw):
        if "=" not in item:
            raise ValueError(
                f"Invalid factor param '{item}'. Expected k=v format (comma-separated)."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid factor param '{item}': empty key.")
        params[key] = _parse_scalar(value.strip())
    return params


def _parse_csv_float_list_optional(raw: str | None) -> list[float] | None:
    if raw is None or not raw.strip():
        return None
    return _parse_csv_floats(raw)


def _parse_factor_sets(raw: str | None) -> list[list[str]]:
    if raw is None or not raw.strip():
        return []
    out: list[list[str]] = []
    for block in [x.strip() for x in raw.split(";") if x.strip()]:
        names = [x.strip() for x in block.split("|") if x.strip()]
        if not names:
            raise ValueError(f"Invalid factor set block '{block}'.")
        out.append(names)
    return out


def _parse_factor_weight_sets(raw: str | None) -> list[list[float]]:
    if raw is None or not raw.strip():
        return []
    out: list[list[float]] = []
    for block in [x.strip() for x in raw.split(";") if x.strip()]:
        vals = [x.strip() for x in block.split("|") if x.strip()]
        if not vals:
            raise ValueError(f"Invalid factor weight set block '{block}'.")
        out.append([float(x) for x in vals])
    return out


def _resolve_data_root(args) -> str:
    """Resolve canonical local data root for selected source."""
    if hasattr(args, "data_root") and str(args.data_root).strip():
        return str(args.data_root).strip()
    return str(getattr(args, "data_cache_dir", "data/equities")).strip()


def _require_parquet_for_research(data_source: str) -> None:
    """Block legacy network-backed sources for research/backtest commands."""
    if str(data_source).strip().lower() != "parquet":
        raise ValueError(
            "Research/backtest commands require local parquet data. "
            "Use --data_source parquet --data_root data/equities."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="quant_lab runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest", help="Run Top-N momentum backtest")
    backtest.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    backtest.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    backtest.add_argument("--universe", default="sp500", help="Universe name (supported: sp500, liquid_us)")
    backtest.add_argument("--max_tickers", type=int, default=50)
    backtest.add_argument("--top_n", type=int, default=10)
    backtest.add_argument("--rank_buffer", type=int, default=0)
    backtest.add_argument("--volatility_scaled_weights", type=int, choices=[0, 1], default=0)
    backtest.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default="weekly")
    backtest.add_argument("--costs_bps", type=float, default=10.0)
    backtest.add_argument("--seed", type=int, default=42)
    backtest.add_argument("--data_source", default="parquet", choices=["parquet", "default"])
    backtest.add_argument("--data_root", default="data/equities")
    backtest.add_argument("--data_cache_dir", default="data/equities")
    backtest.add_argument("--data_refresh", type=int, choices=[0, 1], default=0)
    backtest.add_argument("--data_bulk_prepare", type=int, choices=[0, 1], default=0)
    backtest.add_argument("--factor", default="momentum_12_1", help="Factor name or comma-separated list")
    backtest.add_argument("--factor_weights", default="", help="Optional comma-separated weights")
    backtest.add_argument("--factor_params", default="", help='Optional "k=v,k2=v2"')
    backtest.add_argument("--normalize", choices=["none", "zscore", "winsor_zscore"], default="zscore")
    backtest.add_argument("--winsor_p", type=float, default=0.01)
    backtest.add_argument("--use_factor_normalization", type=int, choices=[0, 1], default=1)
    backtest.add_argument("--use_sector_neutralization", type=int, choices=[0, 1], default=1)
    backtest.add_argument("--use_size_neutralization", type=int, choices=[0, 1], default=1)
    backtest.add_argument("--orthogonalize_factors", type=int, choices=[0, 1], default=0)
    backtest.add_argument(
        "--weighting",
        choices=["equal", "inv_vol", "score", "score_inv_vol"],
        default="equal",
    )
    backtest.add_argument("--vol_lookback", type=int, default=20)
    backtest.add_argument("--max_weight", type=float, default=0.15)
    backtest.add_argument("--score_clip", type=float, default=5.0)
    backtest.add_argument("--score_floor", type=float, default=0.0)
    backtest.add_argument("--sector_cap", type=float, default=0.0)
    backtest.add_argument("--sector_map", default="")
    backtest.add_argument("--sector_neutral", type=int, choices=[0, 1], default=0)
    backtest.add_argument("--target_vol", type=float, default=0.0)
    backtest.add_argument("--port_vol_lookback", type=int, default=20)
    backtest.add_argument("--max_leverage", type=float, default=1.0)
    backtest.add_argument("--slippage_bps", type=float, default=0.0)
    backtest.add_argument("--slippage_vol_mult", type=float, default=0.0)
    backtest.add_argument("--slippage_vol_lookback", type=int, default=20)
    backtest.add_argument("--execution_delay_days", type=int, default=0)
    backtest.add_argument("--regime_filter", type=int, choices=[0, 1], default=0)
    backtest.add_argument("--dynamic_factor_weights", type=int, choices=[0, 1], default=0)
    backtest.add_argument("--regime_benchmark", default="SPY")
    backtest.add_argument("--regime_trend_sma", type=int, default=200)
    backtest.add_argument("--regime_vol_lookback", type=int, default=20)
    backtest.add_argument("--regime_vol_median_lookback", type=int, default=252)
    backtest.add_argument(
        "--regime_bull_weights",
        default="momentum_12_1:0.7,low_vol_20:0.3",
    )
    backtest.add_argument(
        "--regime_bear_weights",
        default="momentum_12_1:0.3,low_vol_20:0.7",
    )
    backtest.add_argument("--bear_exposure_scale", type=float, default=1.0)
    backtest.add_argument("--trend_filter", type=int, choices=[0, 1], default=0)
    backtest.add_argument("--trend_sma_window", type=int, default=200)
    backtest.add_argument("--save_artifacts", type=int, choices=[0, 1], default=0)
    backtest.add_argument("--print_run_summary", type=int, choices=[0, 1], default=0)
    backtest.add_argument("--price_quality_check", type=int, choices=[0, 1], default=0)
    backtest.add_argument("--price_quality_mode", choices=["warn", "fail"], default="warn")
    backtest.add_argument("--price_quality_zero_ret_thresh", type=float, default=0.95)
    backtest.add_argument("--price_quality_min_valid_frac", type=float, default=0.98)
    backtest.add_argument("--price_quality_max_bad_tickers", type=int, default=0)
    backtest.add_argument("--price_quality_report_topk", type=int, default=20)
    backtest.add_argument("--price_fill_mode", choices=["ffill", "none"], default="ffill")
    backtest.add_argument("--drop_bad_tickers", type=int, choices=[0, 1], default=0)
    backtest.add_argument(
        "--drop_bad_tickers_scope",
        choices=["test", "train_and_test", "full"],
        default="test",
    )
    backtest.add_argument("--drop_bad_tickers_max_drop", type=int, default=0)
    backtest.add_argument("--drop_bad_tickers_exempt", default="SPY")
    backtest.add_argument("--universe_mode", choices=["static", "dynamic"], default="static")
    backtest.add_argument("--universe_min_history_days", type=int, default=300)
    backtest.add_argument("--universe_min_valid_frac", type=float, default=0.98)
    backtest.add_argument("--universe_valid_lookback", type=int, default=252)
    backtest.add_argument("--universe_min_price", type=float, default=1.0)
    backtest.add_argument("--min_price", type=float, default=0.0)
    backtest.add_argument("--min_avg_dollar_volume", type=float, default=0.0)
    backtest.add_argument("--liquidity_lookback", type=int, default=20)
    backtest.add_argument("--universe_min_tickers", type=int, default=20)
    backtest.add_argument("--universe_skip_below_min_tickers", type=int, choices=[0, 1], default=1)
    backtest.add_argument("--universe_eligibility_source", choices=["price", "score"], default="price")
    backtest.add_argument("--universe_exempt", default="SPY")
    backtest.add_argument("--universe_dataset_mode", choices=["off", "build", "use"], default="off")
    backtest.add_argument("--universe_dataset_freq", choices=["daily", "rebalance"], default="rebalance")
    backtest.add_argument("--universe_dataset_path", default="")
    backtest.add_argument("--universe_dataset_save", type=int, choices=[0, 1], default=1)
    backtest.add_argument("--universe_dataset_require", type=int, choices=[0, 1], default=0)
    backtest.add_argument("--historical_membership_path", default="")

    sweep = subparsers.add_parser("sweep", help="Run parameter sweep over Top-N settings")
    sweep.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    sweep.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    sweep.add_argument("--universe", default="sp500", help="Universe name (supported: sp500, liquid_us)")
    sweep.add_argument("--max_tickers", type=int, default=50)
    sweep.add_argument("--data_source", default="parquet", choices=["parquet", "default"])
    sweep.add_argument("--data_root", default="data/equities")
    sweep.add_argument("--data_cache_dir", default="data/equities")
    sweep.add_argument("--top_n", required=True, help="Comma-separated list, e.g. 5,10,20")
    sweep.add_argument("--rank_buffer", default="0", help="Comma-separated integer list")
    sweep.add_argument("--volatility_scaled_weights", default="0", help="Comma-separated 0/1 list")
    sweep.add_argument("--rebalance", required=True, help="Comma-separated list: daily,weekly,monthly")
    sweep.add_argument("--costs_bps", required=True, help="Comma-separated list, e.g. 5,10,20")
    sweep.add_argument(
        "--factor",
        default="momentum_12_1",
        help="Optional comma-separated factor names, e.g. momentum_12_1,momentum_6_1",
    )
    sweep.add_argument(
        "--factor_sets",
        default="",
        help='Optional sets, e.g. "momentum_12_1|low_vol_20;momentum_6_1|low_vol_20"',
    )
    sweep.add_argument(
        "--factor_weights_sets",
        default="",
        help='Optional weight sets, e.g. "0.7|0.3;0.6|0.4"',
    )
    sweep.add_argument("--factor_weights", default="", help="Optional comma-separated weights for --factor list")
    sweep.add_argument(
        "--factor_params",
        default="",
        help='Optional "k=v,k2=v2" applied to all factors',
    )
    sweep.add_argument("--normalize", choices=["none", "zscore", "winsor_zscore"], default="zscore")
    sweep.add_argument("--winsor_p", type=float, default=0.01)
    sweep.add_argument("--use_factor_normalization", type=int, choices=[0, 1], default=1)
    sweep.add_argument("--use_sector_neutralization", type=int, choices=[0, 1], default=1)
    sweep.add_argument("--use_size_neutralization", type=int, choices=[0, 1], default=1)
    sweep.add_argument("--orthogonalize_factors", default="0", help="Comma-separated 0/1 list, e.g. 0,1")
    sweep.add_argument(
        "--weighting",
        default="equal",
        help="Comma-separated list: equal,inv_vol,score,score_inv_vol",
    )
    sweep.add_argument("--vol_lookback", default="20", help="Comma-separated integer list")
    sweep.add_argument("--max_weight", default="0.15", help="Comma-separated float list")
    sweep.add_argument("--score_clip", default="5.0", help="Comma-separated float list")
    sweep.add_argument("--score_floor", default="0", help="Comma-separated float list")
    sweep.add_argument("--sector_cap", default="0", help="Comma-separated float list")
    sweep.add_argument("--sector_map", default="")
    sweep.add_argument("--sector_neutral", default="0", help="Comma-separated 0/1 list, e.g. 0,1")
    sweep.add_argument("--target_vol", default="0", help="Comma-separated float list")
    sweep.add_argument("--port_vol_lookback", default="20", help="Comma-separated integer list")
    sweep.add_argument("--max_leverage", default="1.0", help="Comma-separated float list")
    sweep.add_argument("--slippage_bps", default="0", help="Comma-separated float list")
    sweep.add_argument("--slippage_vol_mult", default="0", help="Comma-separated float list")
    sweep.add_argument("--slippage_vol_lookback", default="20", help="Comma-separated integer list")
    sweep.add_argument("--execution_delay_days", default="0", help="Comma-separated integer list")
    sweep.add_argument("--trend_filter", default="0", help="Comma-separated 0/1 list, e.g. 0,1")
    sweep.add_argument("--trend_sma_window", type=int, default=200)

    walkforward = subparsers.add_parser("walkforward", help="Run walk-forward out-of-sample backtest")
    walkforward.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    walkforward.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    walkforward.add_argument("--train_years", type=int, default=5)
    walkforward.add_argument("--test_years", type=int, default=2)
    walkforward.add_argument("--universe", default="sp500", help="Universe name (supported: sp500, liquid_us)")
    walkforward.add_argument("--max_tickers", type=int, default=50)
    walkforward.add_argument("--top_n", type=int, default=10)
    walkforward.add_argument("--rank_buffer", type=int, default=0)
    walkforward.add_argument("--volatility_scaled_weights", type=int, choices=[0, 1], default=0)
    walkforward.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default="weekly")
    walkforward.add_argument("--costs_bps", type=float, default=10.0)
    walkforward.add_argument("--seed", type=int, default=42)
    walkforward.add_argument("--data_source", default="parquet", choices=["parquet", "default"])
    walkforward.add_argument("--data_root", default="data/equities")
    walkforward.add_argument("--data_cache_dir", default="data/equities")
    walkforward.add_argument("--data_refresh", type=int, choices=[0, 1], default=0)
    walkforward.add_argument("--data_bulk_prepare", type=int, choices=[0, 1], default=0)
    walkforward.add_argument("--factor", default="momentum_12_1", help="Factor name or comma-separated list")
    walkforward.add_argument("--factor_weights", default="", help="Optional comma-separated weights")
    walkforward.add_argument("--factor_params", default="", help='Optional "k=v,k2=v2"')
    walkforward.add_argument("--normalize", choices=["none", "zscore", "winsor_zscore"], default="zscore")
    walkforward.add_argument("--winsor_p", type=float, default=0.01)
    walkforward.add_argument("--use_factor_normalization", type=int, choices=[0, 1], default=1)
    walkforward.add_argument("--use_sector_neutralization", type=int, choices=[0, 1], default=1)
    walkforward.add_argument("--use_size_neutralization", type=int, choices=[0, 1], default=1)
    walkforward.add_argument("--orthogonalize_factors", type=int, choices=[0, 1], default=0)
    walkforward.add_argument(
        "--weighting",
        choices=["equal", "inv_vol", "score", "score_inv_vol"],
        default="equal",
    )
    walkforward.add_argument("--vol_lookback", type=int, default=20)
    walkforward.add_argument("--max_weight", type=float, default=0.15)
    walkforward.add_argument("--score_clip", type=float, default=5.0)
    walkforward.add_argument("--score_floor", type=float, default=0.0)
    walkforward.add_argument("--sector_cap", type=float, default=0.0)
    walkforward.add_argument("--sector_map", default="")
    walkforward.add_argument("--sector_neutral", type=int, choices=[0, 1], default=0)
    walkforward.add_argument("--target_vol", type=float, default=0.0)
    walkforward.add_argument("--port_vol_lookback", type=int, default=20)
    walkforward.add_argument("--max_leverage", type=float, default=1.0)
    walkforward.add_argument("--slippage_bps", type=float, default=0.0)
    walkforward.add_argument("--slippage_vol_mult", type=float, default=0.0)
    walkforward.add_argument("--slippage_vol_lookback", type=int, default=20)
    walkforward.add_argument("--execution_delay_days", type=int, default=0)
    walkforward.add_argument("--regime_filter", type=int, choices=[0, 1], default=0)
    walkforward.add_argument("--dynamic_factor_weights", type=int, choices=[0, 1], default=0)
    walkforward.add_argument("--regime_benchmark", default="SPY")
    walkforward.add_argument("--regime_trend_sma", type=int, default=200)
    walkforward.add_argument("--regime_vol_lookback", type=int, default=20)
    walkforward.add_argument("--regime_vol_median_lookback", type=int, default=252)
    walkforward.add_argument(
        "--regime_bull_weights",
        default="momentum_12_1:0.7,low_vol_20:0.3",
    )
    walkforward.add_argument(
        "--regime_bear_weights",
        default="momentum_12_1:0.3,low_vol_20:0.7",
    )
    walkforward.add_argument("--bear_exposure_scale", type=float, default=1.0)
    walkforward.add_argument("--trend_filter", type=int, choices=[0, 1], default=0)
    walkforward.add_argument("--trend_sma_window", type=int, default=200)
    walkforward.add_argument("--save_artifacts", type=int, choices=[0, 1], default=0)
    walkforward.add_argument("--print_run_summary", type=int, choices=[0, 1], default=0)
    walkforward.add_argument("--price_quality_check", type=int, choices=[0, 1], default=0)
    walkforward.add_argument("--price_quality_mode", choices=["warn", "fail"], default="warn")
    walkforward.add_argument("--price_quality_zero_ret_thresh", type=float, default=0.95)
    walkforward.add_argument("--price_quality_min_valid_frac", type=float, default=0.98)
    walkforward.add_argument("--price_quality_max_bad_tickers", type=int, default=0)
    walkforward.add_argument("--price_quality_report_topk", type=int, default=20)
    walkforward.add_argument("--price_fill_mode", choices=["ffill", "none"], default="ffill")
    walkforward.add_argument("--drop_bad_tickers", type=int, choices=[0, 1], default=0)
    walkforward.add_argument(
        "--drop_bad_tickers_scope",
        choices=["test", "train_and_test", "full"],
        default="test",
    )
    walkforward.add_argument("--drop_bad_tickers_max_drop", type=int, default=0)
    walkforward.add_argument("--drop_bad_tickers_exempt", default="SPY")
    walkforward.add_argument("--universe_mode", choices=["static", "dynamic"], default="static")
    walkforward.add_argument("--universe_min_history_days", type=int, default=300)
    walkforward.add_argument("--universe_min_valid_frac", type=float, default=0.98)
    walkforward.add_argument("--universe_valid_lookback", type=int, default=252)
    walkforward.add_argument("--universe_min_price", type=float, default=1.0)
    walkforward.add_argument("--min_price", type=float, default=0.0)
    walkforward.add_argument("--min_avg_dollar_volume", type=float, default=0.0)
    walkforward.add_argument("--liquidity_lookback", type=int, default=20)
    walkforward.add_argument("--universe_min_tickers", type=int, default=20)
    walkforward.add_argument("--universe_skip_below_min_tickers", type=int, choices=[0, 1], default=1)
    walkforward.add_argument("--universe_eligibility_source", choices=["price", "score"], default="price")
    walkforward.add_argument("--universe_exempt", default="SPY")
    walkforward.add_argument("--universe_dataset_mode", choices=["off", "build", "use"], default="off")
    walkforward.add_argument("--universe_dataset_freq", choices=["daily", "rebalance"], default="rebalance")
    walkforward.add_argument("--universe_dataset_path", default="")
    walkforward.add_argument("--universe_dataset_save", type=int, choices=[0, 1], default=1)
    walkforward.add_argument("--universe_dataset_require", type=int, choices=[0, 1], default=0)
    walkforward.add_argument("--historical_membership_path", default="")
    subparsers.add_parser("list-factors", help="List available factor plugins")
    factor_diag = subparsers.add_parser("factor-diagnostics", help="Run cross-sectional factor diagnostics")
    factor_diag.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    factor_diag.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    factor_diag.add_argument("--universe", default="sp500", help="Universe name (currently: sp500)")
    factor_diag.add_argument("--max_tickers", type=int, default=50)
    factor_diag.add_argument("--data_source", default="parquet", choices=["parquet", "default"])
    factor_diag.add_argument("--data_root", default="data/equities")
    factor_diag.add_argument("--data_cache_dir", default="data/equities")
    factor_diag.add_argument("--data_refresh", type=int, choices=[0, 1], default=0)
    factor_diag.add_argument("--data_bulk_prepare", type=int, choices=[0, 1], default=0)
    factor_diag.add_argument("--factor", default="momentum_12_1")
    factor_diag.add_argument("--factor_params", default="", help='Optional "k=v,k2=v2"')
    factor_diag.add_argument("--quantiles", type=int, default=5)
    factor_diag.add_argument("--method", choices=["spearman", "pearson"], default="spearman")
    factor_diag.add_argument("--horizons", default="1,5,21,63", help="Comma-separated horizons")
    factor_diag.add_argument("--use_factor_normalization", type=int, choices=[0, 1], default=1)
    factor_diag.add_argument("--use_sector_neutralization", type=int, choices=[0, 1], default=1)
    factor_diag.add_argument("--use_size_neutralization", type=int, choices=[0, 1], default=1)
    factor_diag.add_argument("--price_fill_mode", choices=["ffill", "none"], default="ffill")
    factor_diag.add_argument("--historical_membership_path", default="")
    factor_diag.add_argument("--print_run_summary", type=int, choices=[0, 1], default=0)

    signal_corr = subparsers.add_parser("signal-correlation", help="Run signal correlation diagnostics")
    signal_corr.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    signal_corr.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    signal_corr.add_argument("--universe", default="sp500", help="Universe name (currently: sp500)")
    signal_corr.add_argument("--max_tickers", type=int, default=50)
    signal_corr.add_argument("--data_source", default="parquet", choices=["parquet", "default"])
    signal_corr.add_argument("--data_root", default="data/equities")
    signal_corr.add_argument("--data_cache_dir", default="data/equities")
    signal_corr.add_argument("--data_refresh", type=int, choices=[0, 1], default=0)
    signal_corr.add_argument("--data_bulk_prepare", type=int, choices=[0, 1], default=0)
    signal_corr.add_argument("--factors", required=True, help="Comma-separated factor names")
    signal_corr.add_argument("--factor_params", default="", help='Optional "k=v,k2=v2" applied to all factors')
    signal_corr.add_argument("--method", choices=["spearman", "pearson"], default="spearman")
    signal_corr.add_argument("--use_factor_normalization", type=int, choices=[0, 1], default=1)
    signal_corr.add_argument("--use_sector_neutralization", type=int, choices=[0, 1], default=1)
    signal_corr.add_argument("--use_size_neutralization", type=int, choices=[0, 1], default=1)
    signal_corr.add_argument("--price_fill_mode", choices=["ffill", "none"], default="ffill")
    signal_corr.add_argument("--historical_membership_path", default="")
    signal_corr.add_argument("--print_run_summary", type=int, choices=[0, 1], default=0)

    factor_returns = subparsers.add_parser("factor-returns", help="Run factor return-series diagnostics")
    factor_returns.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    factor_returns.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    factor_returns.add_argument("--universe", default="sp500", help="Universe name (currently: sp500)")
    factor_returns.add_argument("--max_tickers", type=int, default=50)
    factor_returns.add_argument("--data_source", default="parquet", choices=["parquet", "default"])
    factor_returns.add_argument("--data_root", default="data/equities")
    factor_returns.add_argument("--data_cache_dir", default="data/equities")
    factor_returns.add_argument("--data_refresh", type=int, choices=[0, 1], default=0)
    factor_returns.add_argument("--data_bulk_prepare", type=int, choices=[0, 1], default=0)
    factor_returns.add_argument("--factor", default="momentum_12_1")
    factor_returns.add_argument("--factors", default="", help="Optional comma-separated factor names")
    factor_returns.add_argument("--factor_params", default="", help='Optional "k=v,k2=v2" applied to all factors')
    factor_returns.add_argument("--quantiles", type=int, default=5)
    factor_returns.add_argument("--rolling_window", type=int, default=52)
    factor_returns.add_argument("--plot_seasonality", type=int, choices=[0, 1], default=0)
    factor_returns.add_argument("--seasonality_plot_dir", default="results")
    factor_returns.add_argument("--use_factor_normalization", type=int, choices=[0, 1], default=1)
    factor_returns.add_argument("--use_sector_neutralization", type=int, choices=[0, 1], default=1)
    factor_returns.add_argument("--use_size_neutralization", type=int, choices=[0, 1], default=1)
    factor_returns.add_argument("--price_fill_mode", choices=["ffill", "none"], default="ffill")
    factor_returns.add_argument("--historical_membership_path", default="")
    factor_returns.add_argument("--print_run_summary", type=int, choices=[0, 1], default=0)

    heatmap = subparsers.add_parser("factor-heatmap", help="Run momentum parameter sweep heatmap")
    heatmap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    heatmap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    heatmap.add_argument("--universe", default="sp500", help="Universe name (currently: sp500)")
    heatmap.add_argument("--max_tickers", type=int, default=50)
    heatmap.add_argument("--data_source", default="parquet", choices=["parquet", "default"])
    heatmap.add_argument("--data_root", default="data/equities")
    heatmap.add_argument("--data_cache_dir", default="data/equities")
    heatmap.add_argument("--data_refresh", type=int, choices=[0, 1], default=0)
    heatmap.add_argument("--data_bulk_prepare", type=int, choices=[0, 1], default=0)
    heatmap.add_argument("--lookbacks", default="21,63,126,189,252")
    heatmap.add_argument("--metric", choices=["sharpe", "ic"], default="sharpe")
    heatmap.add_argument("--period", choices=["year", "quarter"], default="year")
    heatmap.add_argument("--historical_membership_path", default="")
    heatmap.add_argument("--print_run_summary", type=int, choices=[0, 1], default=1)

    build_membership = subparsers.add_parser(
        "build-sp500-membership",
        help="Build historical S&P 500 membership CSV from Wikipedia",
    )
    build_membership.add_argument("--start", default="2000-01-01")
    build_membership.add_argument("--end", default="")
    build_membership.add_argument(
        "--output",
        default="data/universe/sp500_historical_membership.csv",
    )
    ingest_cache = subparsers.add_parser(
        "ingest-equity-cache",
        help="Ingest one-file-per-ticker cache into centralized parquet equity store",
    )
    ingest_cache.add_argument("--cache_dir", default="data/cache/stooq")
    ingest_cache.add_argument("--source", default="stooq_cache")
    ingest_cache.add_argument("--store_root", default="data/equities")

    ingest_membership = subparsers.add_parser(
        "ingest-universe-membership",
        help="Ingest universe membership CSV into centralized parquet membership table",
    )
    ingest_membership.add_argument("--csv_path", required=True)
    ingest_membership.add_argument("--universe", required=True)
    ingest_membership.add_argument("--store_root", default="data/equities")

    dl_stooq = subparsers.add_parser(
        "download-stooq-universe",
        help="Download large Stooq universe CSVs into cache directory",
    )
    dl_stooq.add_argument("--symbol_file", default="data/universe/us_equities.txt")
    dl_stooq.add_argument("--out_dir", default="data/cache/stooq")

    dl_tiingo = subparsers.add_parser(
        "download-tiingo-universe",
        help="Download large Tiingo universe CSVs into cache directory",
    )
    dl_tiingo.add_argument("--symbol_file", default=str(DEFAULT_TIINGO_US_COMMON_EQUITY_MANIFEST))
    dl_tiingo.add_argument("--out_dir", default="data/cache/tiingo")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "backtest":
        _require_parquet_for_research(args.data_source)
        if args.universe.lower() not in {"sp500", "liquid_us"}:
            raise ValueError("Only --universe sp500 or liquid_us is supported in this minimal runner.")
        factor_names = _parse_csv_list(args.factor) if args.factor else ["momentum_12_1"]
        factor_weights = _parse_csv_float_list_optional(args.factor_weights)
        factor_params_shared = _parse_factor_params(args.factor_params)
        factor_params = {name: dict(factor_params_shared) for name in factor_names}
        summary, outdir = run_backtest(
            start=args.start,
            end=args.end,
            universe=args.universe,
            max_tickers=args.max_tickers,
            data_source=args.data_source,
            data_cache_dir=_resolve_data_root(args),
            data_refresh=bool(args.data_refresh),
            data_bulk_prepare=bool(args.data_bulk_prepare),
            top_n=args.top_n,
            rank_buffer=args.rank_buffer,
            volatility_scaled_weights=bool(args.volatility_scaled_weights),
            rebalance=args.rebalance,
            costs_bps=args.costs_bps,
            seed=args.seed,
            factor_name=factor_names,
            factor_names=factor_names,
            factor_weights=factor_weights,
            factor_params=factor_params,
            normalize=args.normalize,
            winsor_p=args.winsor_p,
            use_factor_normalization=bool(args.use_factor_normalization),
            use_sector_neutralization=bool(args.use_sector_neutralization),
            use_size_neutralization=bool(args.use_size_neutralization),
            orthogonalize_factors=bool(args.orthogonalize_factors),
            weighting=args.weighting,
            vol_lookback=args.vol_lookback,
            max_weight=args.max_weight,
            score_clip=args.score_clip,
            score_floor=args.score_floor,
            sector_cap=args.sector_cap,
            sector_map=args.sector_map,
            sector_neutral=bool(args.sector_neutral),
            target_vol=args.target_vol,
            port_vol_lookback=args.port_vol_lookback,
            max_leverage=args.max_leverage,
            slippage_bps=args.slippage_bps,
            slippage_vol_mult=args.slippage_vol_mult,
            slippage_vol_lookback=args.slippage_vol_lookback,
            execution_delay_days=args.execution_delay_days,
            regime_filter=bool(args.regime_filter),
            dynamic_factor_weights=bool(args.dynamic_factor_weights),
            regime_benchmark=args.regime_benchmark,
            regime_trend_sma=args.regime_trend_sma,
            regime_vol_lookback=args.regime_vol_lookback,
            regime_vol_median_lookback=args.regime_vol_median_lookback,
            regime_bull_weights=args.regime_bull_weights,
            regime_bear_weights=args.regime_bear_weights,
            bear_exposure_scale=args.bear_exposure_scale,
            trend_filter=bool(args.trend_filter),
            trend_sma_window=args.trend_sma_window,
            save_artifacts=bool(args.save_artifacts),
            print_run_summary_flag=bool(args.print_run_summary),
            price_quality_check=bool(args.price_quality_check),
            price_quality_mode=args.price_quality_mode,
            price_quality_zero_ret_thresh=args.price_quality_zero_ret_thresh,
            price_quality_min_valid_frac=args.price_quality_min_valid_frac,
            price_quality_max_bad_tickers=args.price_quality_max_bad_tickers,
            price_quality_report_topk=args.price_quality_report_topk,
            price_fill_mode=args.price_fill_mode,
            drop_bad_tickers=bool(args.drop_bad_tickers),
            drop_bad_tickers_scope=args.drop_bad_tickers_scope,
            drop_bad_tickers_max_drop=args.drop_bad_tickers_max_drop,
            drop_bad_tickers_exempt=args.drop_bad_tickers_exempt,
            universe_mode=args.universe_mode,
            universe_min_history_days=args.universe_min_history_days,
            universe_min_valid_frac=args.universe_min_valid_frac,
            universe_valid_lookback=args.universe_valid_lookback,
            universe_min_price=args.universe_min_price,
            min_price=args.min_price,
            min_avg_dollar_volume=args.min_avg_dollar_volume,
            liquidity_lookback=args.liquidity_lookback,
            universe_min_tickers=args.universe_min_tickers,
            universe_skip_below_min_tickers=bool(args.universe_skip_below_min_tickers),
            universe_eligibility_source=args.universe_eligibility_source,
            universe_exempt=args.universe_exempt,
            universe_dataset_mode=args.universe_dataset_mode,
            universe_dataset_freq=args.universe_dataset_freq,
            universe_dataset_path=args.universe_dataset_path,
            universe_dataset_save=bool(args.universe_dataset_save),
            universe_dataset_require=bool(args.universe_dataset_require),
            historical_membership_path=args.historical_membership_path,
        )
        print(f"Results written to: {outdir}")
        print(json.dumps(summary, indent=2))
        return

    if args.command == "sweep":
        _require_parquet_for_research(args.data_source)
        if args.universe.lower() not in {"sp500", "liquid_us"}:
            raise ValueError("Only --universe sp500 or liquid_us is supported in this minimal runner.")

        top_n_values = _parse_csv_ints(args.top_n)
        rank_buffer_values = _parse_csv_ints(args.rank_buffer)
        volatility_scaled_values = _parse_csv_bool01(args.volatility_scaled_weights)
        rebalance_values = _parse_csv_rebalance(args.rebalance)
        costs_values = _parse_csv_floats(args.costs_bps)
        factor_params_shared = _parse_factor_params(args.factor_params)
        factor_sets = _parse_factor_sets(args.factor_sets)
        factor_weight_sets = _parse_factor_weight_sets(args.factor_weights_sets)
        weighting_values = _parse_csv_weighting(args.weighting)
        vol_lookback_values = _parse_csv_ints(args.vol_lookback)
        max_weight_values = _parse_csv_floats(args.max_weight)
        score_clip_values = _parse_csv_floats(args.score_clip)
        score_floor_values = _parse_csv_floats(args.score_floor)
        sector_cap_values = _parse_csv_floats(args.sector_cap)
        target_vol_values = _parse_csv_floats(args.target_vol)
        port_vol_lookback_values = _parse_csv_ints(args.port_vol_lookback)
        max_leverage_values = _parse_csv_floats(args.max_leverage)
        slippage_bps_values = _parse_csv_floats(args.slippage_bps)
        slippage_vol_mult_values = _parse_csv_floats(args.slippage_vol_mult)
        slippage_vol_lookback_values = _parse_csv_ints(args.slippage_vol_lookback)
        execution_delay_days_values = _parse_csv_ints(args.execution_delay_days)
        factor_configs: list[tuple[list[str], list[float] | None, str]] = []
        if factor_sets:
            if factor_weight_sets and len(factor_weight_sets) != len(factor_sets):
                raise ValueError("factor_weights_sets must align 1:1 with factor_sets.")
            for i, names in enumerate(factor_sets):
                ws = factor_weight_sets[i] if factor_weight_sets else None
                if ws is not None and len(ws) != len(names):
                    raise ValueError(
                        f"factor_weights_sets[{i}] length {len(ws)} != factor_sets[{i}] length {len(names)}"
                    )
                factor_configs.append((names, ws, "|".join(names)))
        else:
            factor_values = _parse_csv_list(args.factor) if args.factor else ["momentum_12_1"]
            default_weights = _parse_csv_float_list_optional(args.factor_weights)
            for name in factor_values:
                factor_configs.append(([name], default_weights, name))
        trend_values = _parse_csv_bool01(args.trend_filter)
        sector_neutral_values = _parse_csv_bool01(args.sector_neutral)
        orthogonalize_values = _parse_csv_bool01(args.orthogonalize_factors)

        combos = list(
            product(
                top_n_values,
                rank_buffer_values,
                volatility_scaled_values,
                rebalance_values,
                costs_values,
                factor_configs,
                weighting_values,
                vol_lookback_values,
                max_weight_values,
                score_clip_values,
                score_floor_values,
                sector_cap_values,
                target_vol_values,
                port_vol_lookback_values,
                max_leverage_values,
                slippage_bps_values,
                slippage_vol_mult_values,
                slippage_vol_lookback_values,
                execution_delay_days_values,
                trend_values,
                sector_neutral_values,
                orthogonalize_values,
            )
        )
        if not combos:
            raise ValueError("No sweep combinations generated.")

        sweep_rows: list[dict] = []
        total = len(combos)

        for i, (
            top_n,
            rank_buffer,
            volatility_scaled_weights,
            reb,
            costs,
            factor_cfg,
            weighting,
            vol_lookback,
            max_weight,
            score_clip,
            score_floor,
            sector_cap,
            target_vol,
            port_vol_lookback,
            max_leverage,
            slippage_bps,
            slippage_vol_mult,
            slippage_vol_lookback,
            execution_delay_days,
            trend,
            sector_neutral,
            orthogonalize_factors,
        ) in enumerate(
            combos, start=1
        ):
            factor_names, factor_weights, factor_label = factor_cfg
            factor_params = {name: dict(factor_params_shared) for name in factor_names}
            summary, outdir = run_backtest(
                start=args.start,
                end=args.end,
                universe=args.universe,
                max_tickers=args.max_tickers,
                data_source=args.data_source,
                data_cache_dir=_resolve_data_root(args),
                top_n=top_n,
                rank_buffer=rank_buffer,
                volatility_scaled_weights=bool(volatility_scaled_weights),
                rebalance=reb,
                costs_bps=costs,
                factor_name=factor_names,
                factor_names=factor_names,
                factor_weights=factor_weights,
                factor_params=factor_params,
                normalize=args.normalize,
                winsor_p=args.winsor_p,
                use_factor_normalization=bool(args.use_factor_normalization),
                use_sector_neutralization=bool(args.use_sector_neutralization),
                use_size_neutralization=bool(args.use_size_neutralization),
                orthogonalize_factors=bool(orthogonalize_factors),
                weighting=weighting,
                vol_lookback=vol_lookback,
                max_weight=max_weight,
                score_clip=score_clip,
                score_floor=score_floor,
                sector_cap=sector_cap,
                sector_map=args.sector_map,
                sector_neutral=sector_neutral,
                target_vol=target_vol,
                port_vol_lookback=port_vol_lookback,
                max_leverage=max_leverage,
                slippage_bps=slippage_bps,
                slippage_vol_mult=slippage_vol_mult,
                slippage_vol_lookback=slippage_vol_lookback,
                execution_delay_days=execution_delay_days,
                trend_filter=trend,
                trend_sma_window=args.trend_sma_window,
            )
            sweep_rows.append(summary)
            print(
                f"[{i}/{total}] top_n={top_n} rank_buffer={rank_buffer} "
                f"vol_scaled={int(volatility_scaled_weights)} rebalance={reb} costs={costs} "
                f"factor={factor_label} weighting={weighting} vol_lookback={vol_lookback} "
                f"max_weight={max_weight} score_clip={score_clip} score_floor={score_floor} "
                f"sector_cap={sector_cap} target_vol={target_vol} "
                f"trend={int(trend)} sector_neutral={int(sector_neutral)} "
                f"orthogonalize={int(orthogonalize_factors)} "
                f"delay={int(execution_delay_days)} "
                f"Sharpe={summary.get('Sharpe')} MaxDD={summary.get('MaxDD')}"
            )

        ranked = sorted(
            sweep_rows,
            key=lambda r: float("-inf") if pd.isna(r.get("Sharpe")) else float(r["Sharpe"]),
            reverse=True,
        )
        print("\nTop 10 by Sharpe (this sweep):")
        for j, row in enumerate(ranked[:10], start=1):
            print(
                f"{j:>2}. Sharpe={row.get('Sharpe')} MaxDD={row.get('MaxDD')} "
                f"top_n={row.get('TopN')} rank_buffer={row.get('RankBuffer')} "
                f"vol_scaled={int(bool(row.get('VolatilityScaledWeights')))} rebalance={row.get('Rebalance')} "
                f"costs={row.get('CostsBps')} factor={row.get('FactorNames')} "
                f"weighting={row.get('Weighting')} vol_lookback={row.get('VolLookback')} "
                f"max_weight={row.get('MaxWeight')} score_clip={row.get('ScoreClip')} "
                f"score_floor={row.get('ScoreFloor')} "
                f"sector_cap={row.get('SectorCap')} sector_neutral={int(bool(row.get('SectorNeutral')))} "
                f"orthogonalize={int(bool(row.get('OrthogonalizeFactors')))} "
                f"delay={int(row.get('ExecutionDelayDays', 0))} "
                f"target_vol={row.get('TargetVol')} "
                f"trend={int(bool(row.get('TrendFilter')))} "
                f"run={row.get('RunTag')}"
            )
        return

    if args.command == "walkforward":
        _require_parquet_for_research(args.data_source)
        if args.universe.lower() not in {"sp500", "liquid_us"}:
            raise ValueError("Only --universe sp500 or liquid_us is supported in this minimal runner.")
        factor_names = _parse_csv_list(args.factor) if args.factor else ["momentum_12_1"]
        factor_weights = _parse_csv_float_list_optional(args.factor_weights)
        factor_params_shared = _parse_factor_params(args.factor_params)
        factor_params = {name: dict(factor_params_shared) for name in factor_names}
        summary, outdir = run_walkforward(
            start=args.start,
            end=args.end,
            universe=args.universe,
            train_years=args.train_years,
            test_years=args.test_years,
            max_tickers=args.max_tickers,
            data_source=args.data_source,
            data_cache_dir=_resolve_data_root(args),
            data_refresh=bool(args.data_refresh),
            data_bulk_prepare=bool(args.data_bulk_prepare),
            top_n=args.top_n,
            rank_buffer=args.rank_buffer,
            volatility_scaled_weights=bool(args.volatility_scaled_weights),
            rebalance=args.rebalance,
            costs_bps=args.costs_bps,
            seed=args.seed,
            factor_name=factor_names,
            factor_names=factor_names,
            factor_weights=factor_weights,
            factor_params=factor_params,
            normalize=args.normalize,
            winsor_p=args.winsor_p,
            use_factor_normalization=bool(args.use_factor_normalization),
            use_sector_neutralization=bool(args.use_sector_neutralization),
            use_size_neutralization=bool(args.use_size_neutralization),
            orthogonalize_factors=bool(args.orthogonalize_factors),
            weighting=args.weighting,
            vol_lookback=args.vol_lookback,
            max_weight=args.max_weight,
            score_clip=args.score_clip,
            score_floor=args.score_floor,
            sector_cap=args.sector_cap,
            sector_map=args.sector_map,
            sector_neutral=bool(args.sector_neutral),
            target_vol=args.target_vol,
            port_vol_lookback=args.port_vol_lookback,
            max_leverage=args.max_leverage,
            slippage_bps=args.slippage_bps,
            slippage_vol_mult=args.slippage_vol_mult,
            slippage_vol_lookback=args.slippage_vol_lookback,
            execution_delay_days=args.execution_delay_days,
            regime_filter=bool(args.regime_filter),
            dynamic_factor_weights=bool(args.dynamic_factor_weights),
            regime_benchmark=args.regime_benchmark,
            regime_trend_sma=args.regime_trend_sma,
            regime_vol_lookback=args.regime_vol_lookback,
            regime_vol_median_lookback=args.regime_vol_median_lookback,
            regime_bull_weights=args.regime_bull_weights,
            regime_bear_weights=args.regime_bear_weights,
            bear_exposure_scale=args.bear_exposure_scale,
            trend_filter=bool(args.trend_filter),
            trend_sma_window=args.trend_sma_window,
            save_artifacts=bool(args.save_artifacts),
            print_run_summary_flag=bool(args.print_run_summary),
            price_quality_check=bool(args.price_quality_check),
            price_quality_mode=args.price_quality_mode,
            price_quality_zero_ret_thresh=args.price_quality_zero_ret_thresh,
            price_quality_min_valid_frac=args.price_quality_min_valid_frac,
            price_quality_max_bad_tickers=args.price_quality_max_bad_tickers,
            price_quality_report_topk=args.price_quality_report_topk,
            price_fill_mode=args.price_fill_mode,
            drop_bad_tickers=bool(args.drop_bad_tickers),
            drop_bad_tickers_scope=args.drop_bad_tickers_scope,
            drop_bad_tickers_max_drop=args.drop_bad_tickers_max_drop,
            drop_bad_tickers_exempt=args.drop_bad_tickers_exempt,
            universe_mode=args.universe_mode,
            universe_min_history_days=args.universe_min_history_days,
            universe_min_valid_frac=args.universe_min_valid_frac,
            universe_valid_lookback=args.universe_valid_lookback,
            universe_min_price=args.universe_min_price,
            min_price=args.min_price,
            min_avg_dollar_volume=args.min_avg_dollar_volume,
            liquidity_lookback=args.liquidity_lookback,
            universe_min_tickers=args.universe_min_tickers,
            universe_skip_below_min_tickers=bool(args.universe_skip_below_min_tickers),
            universe_eligibility_source=args.universe_eligibility_source,
            universe_exempt=args.universe_exempt,
            universe_dataset_mode=args.universe_dataset_mode,
            universe_dataset_freq=args.universe_dataset_freq,
            universe_dataset_path=args.universe_dataset_path,
            universe_dataset_save=bool(args.universe_dataset_save),
            universe_dataset_require=bool(args.universe_dataset_require),
            historical_membership_path=args.historical_membership_path,
        )
        print(f"Results written to: {outdir}")
        print(json.dumps(summary, indent=2))
        return

    if args.command == "list-factors":
        for name in list_factors():
            print(name)
        return

    if args.command == "factor-diagnostics":
        _require_parquet_for_research(args.data_source)
        if args.universe.lower() != "sp500":
            raise ValueError("Only --universe sp500 is supported in this minimal runner.")
        factor_params = _parse_factor_params(args.factor_params)
        horizons = _parse_csv_ints(args.horizons)

        tickers = sorted(load_sp500_tickers())[: args.max_tickers]
        ohlcv_map, data_source_summary = fetch_ohlcv_with_summary(
            tickers=tickers,
            start=args.start,
            end=args.end,
            cache_dir=_resolve_data_root(args),
            data_source=args.data_source,
            refresh=bool(args.data_refresh),
            bulk_prepare=bool(args.data_bulk_prepare),
        )
        close_cols, used_tickers, missing_tickers, rejected_tickers, _ = _collect_close_series(
            ohlcv_map=ohlcv_map,
            requested_tickers=tickers,
        )
        if not close_cols:
            raise ValueError("No valid OHLCV data available for factor diagnostics run.")

        close = pd.concat(close_cols, axis=1, join="outer")
        close = _prepare_close_panel(close_raw=close, price_fill_mode=args.price_fill_mode)
        if close.empty:
            raise ValueError("No close-price panel available after alignment.")

        score_map = compute_factors(
            factor_names=[args.factor],
            close=close,
            factor_params={args.factor: factor_params},
        )
        factor_scores = score_map[args.factor].astype(float)
        factor_scores = preprocess_factor_scores(
            factor_scores,
            use_factor_normalization=bool(args.use_factor_normalization),
            winsor_p=0.05,
        )
        one_period_fwd = close.pct_change().shift(-1).astype(float)

        if str(args.historical_membership_path).strip():
            membership = load_historical_membership(
                path=str(args.historical_membership_path).strip(),
                index=pd.DatetimeIndex(close.index),
                columns=list(close.columns),
            )
            factor_scores = factor_scores.where(membership, np.nan)
            one_period_fwd = one_period_fwd.where(membership, np.nan)

        report = run_factor_diagnostics(
            factor_scores=factor_scores,
            future_returns=one_period_fwd,
            quantiles=args.quantiles,
            method=args.method,
            horizons=horizons,
        )
        print_factor_diagnostics(report, factor_name=args.factor)
        if bool(args.print_run_summary):
            print("")
            print("DATA SUMMARY")
            print(f"RequestedTickers: {len(tickers)}")
            print(f"LoadedTickers: {len(used_tickers)}")
            print(f"MissingTickers: {len(missing_tickers)}")
            print(f"RejectedTickers: {len(rejected_tickers)}")
            print(f"DataSource: {data_source_summary.get('source', '')}")
            print(f"EarliestDate: {data_source_summary.get('earliest_date', '')}")
            print(f"LatestDate: {data_source_summary.get('latest_date', '')}")
        return

    if args.command == "signal-correlation":
        _require_parquet_for_research(args.data_source)
        if args.universe.lower() != "sp500":
            raise ValueError("Only --universe sp500 is supported in this minimal runner.")
        factors = _parse_csv_list(args.factors)
        if len(factors) < 2:
            raise ValueError("--factors must contain at least two factor names.")
        shared_params = _parse_factor_params(args.factor_params)

        tickers = sorted(load_sp500_tickers())[: args.max_tickers]
        ohlcv_map, data_source_summary = fetch_ohlcv_with_summary(
            tickers=tickers,
            start=args.start,
            end=args.end,
            cache_dir=_resolve_data_root(args),
            data_source=args.data_source,
            refresh=bool(args.data_refresh),
            bulk_prepare=bool(args.data_bulk_prepare),
        )
        close_cols, used_tickers, missing_tickers, rejected_tickers, _ = _collect_close_series(
            ohlcv_map=ohlcv_map,
            requested_tickers=tickers,
        )
        if not close_cols:
            raise ValueError("No valid OHLCV data available for signal-correlation run.")

        close = pd.concat(close_cols, axis=1, join="outer")
        close = _prepare_close_panel(close_raw=close, price_fill_mode=args.price_fill_mode)
        if close.empty:
            raise ValueError("No close-price panel available after alignment.")

        factor_params = {name: dict(shared_params) for name in factors}
        signal_panels = compute_factors(
            factor_names=factors,
            close=close,
            factor_params=factor_params,
        )
        for name in list(signal_panels.keys()):
            signal_panels[name] = preprocess_factor_scores(
                signal_panels[name],
                use_factor_normalization=bool(args.use_factor_normalization),
                winsor_p=0.05,
            )

        if str(args.historical_membership_path).strip():
            membership = load_historical_membership(
                path=str(args.historical_membership_path).strip(),
                index=pd.DatetimeIndex(close.index),
                columns=list(close.columns),
            )
            for name in list(signal_panels.keys()):
                signal_panels[name] = signal_panels[name].where(membership, np.nan)

        report = run_signal_correlation(signal_panels=signal_panels, method=args.method)
        print_signal_correlation(report)
        if bool(args.print_run_summary):
            print("")
            print("DATA SUMMARY")
            print(f"RequestedTickers: {len(tickers)}")
            print(f"LoadedTickers: {len(used_tickers)}")
            print(f"MissingTickers: {len(missing_tickers)}")
            print(f"RejectedTickers: {len(rejected_tickers)}")
            print(f"DataSource: {data_source_summary.get('source', '')}")
            print(f"EarliestDate: {data_source_summary.get('earliest_date', '')}")
            print(f"LatestDate: {data_source_summary.get('latest_date', '')}")
        return

    if args.command == "factor-returns":
        _require_parquet_for_research(args.data_source)
        if args.universe.lower() != "sp500":
            raise ValueError("Only --universe sp500 is supported in this minimal runner.")
        factors = _parse_csv_list(args.factors) if args.factors.strip() else [args.factor]
        if not factors:
            raise ValueError("At least one factor is required.")
        shared_params = _parse_factor_params(args.factor_params)
        factor_params = {name: dict(shared_params) for name in factors}

        tickers = sorted(load_sp500_tickers())[: args.max_tickers]
        ohlcv_map, data_source_summary = fetch_ohlcv_with_summary(
            tickers=tickers,
            start=args.start,
            end=args.end,
            cache_dir=_resolve_data_root(args),
            data_source=args.data_source,
            refresh=bool(args.data_refresh),
            bulk_prepare=bool(args.data_bulk_prepare),
        )
        close_cols, used_tickers, missing_tickers, rejected_tickers, _ = _collect_close_series(
            ohlcv_map=ohlcv_map,
            requested_tickers=tickers,
        )
        if not close_cols:
            raise ValueError("No valid OHLCV data available for factor-returns run.")

        close = pd.concat(close_cols, axis=1, join="outer")
        close = _prepare_close_panel(close_raw=close, price_fill_mode=args.price_fill_mode)
        if close.empty:
            raise ValueError("No close-price panel available after alignment.")

        score_panels = compute_factors(
            factor_names=factors,
            close=close,
            factor_params=factor_params,
        )
        for name in list(score_panels.keys()):
            score_panels[name] = preprocess_factor_scores(
                score_panels[name],
                use_factor_normalization=bool(args.use_factor_normalization),
                winsor_p=0.05,
            )
        one_period_fwd = close.pct_change().shift(-1).astype(float)
        if str(args.historical_membership_path).strip():
            membership = load_historical_membership(
                path=str(args.historical_membership_path).strip(),
                index=pd.DatetimeIndex(close.index),
                columns=list(close.columns),
            )
            one_period_fwd = one_period_fwd.where(membership, np.nan)
            for name in list(score_panels.keys()):
                score_panels[name] = score_panels[name].where(membership, np.nan)

        spread_map: dict[str, pd.Series] = {}
        for name in factors:
            rep = run_factor_return_analysis(
                factor_scores=score_panels[name],
                future_returns=one_period_fwd,
                quantiles=args.quantiles,
                rolling_window=args.rolling_window,
            )
            spread_map[name] = rep["spread_returns_by_date"]
            print_factor_return_analysis(rep, factor_name=name)
            seasonality = run_factor_seasonality(rep["spread_returns_by_date"])
            print("")
            print_factor_seasonality(seasonality, factor_name=name)
            if bool(args.plot_seasonality):
                plot_dir = Path(args.seasonality_plot_dir)
                plot_dir.mkdir(parents=True, exist_ok=True)
                safe_name = name.replace("/", "_")
                outpath = plot_dir / f"factor_seasonality_{safe_name}.png"
                plot_factor_seasonality(
                    seasonality=seasonality,
                    outpath=str(outpath),
                    title=f"Factor Seasonality ({name})",
                )
                print(f"Saved seasonality plot: {outpath}")
            print("")

        if len(factors) > 1:
            corr = run_factor_return_correlation(spread_map)
            print("FACTOR RETURN CORRELATION")
            print("-------------------------")
            print(corr.round(4).to_string())

        if bool(args.print_run_summary):
            print("")
            print("DATA SUMMARY")
            print(f"RequestedTickers: {len(tickers)}")
            print(f"LoadedTickers: {len(used_tickers)}")
            print(f"MissingTickers: {len(missing_tickers)}")
            print(f"RejectedTickers: {len(rejected_tickers)}")
            print(f"DataSource: {data_source_summary.get('source', '')}")
            print(f"EarliestDate: {data_source_summary.get('earliest_date', '')}")
            print(f"LatestDate: {data_source_summary.get('latest_date', '')}")
        return

    if args.command == "factor-heatmap":
        _require_parquet_for_research(args.data_source)
        if args.universe.lower() != "sp500":
            raise ValueError("Only --universe sp500 is supported in this minimal runner.")
        lookbacks = _parse_csv_lookbacks(args.lookbacks)
        tickers = sorted(load_sp500_tickers())[: args.max_tickers]
        ohlcv_map, data_source_summary = fetch_ohlcv_with_summary(
            tickers=tickers,
            start=args.start,
            end=args.end,
            cache_dir=_resolve_data_root(args),
            data_source=args.data_source,
            refresh=bool(args.data_refresh),
            bulk_prepare=bool(args.data_bulk_prepare),
        )
        close_cols, used_tickers, missing_tickers, rejected_tickers, _ = _collect_close_series(
            ohlcv_map=ohlcv_map,
            requested_tickers=tickers,
        )
        if not close_cols:
            raise ValueError("No valid OHLCV data available for factor-heatmap run.")
        close = pd.concat(close_cols, axis=1, join="outer")
        close = _prepare_close_panel(close_raw=close, price_fill_mode="ffill")
        if close.empty:
            raise ValueError("No close-price panel available after alignment.")

        future_returns = close.pct_change().shift(-1).astype(float)
        if str(args.historical_membership_path).strip():
            membership = load_historical_membership(
                path=str(args.historical_membership_path).strip(),
                index=pd.DatetimeIndex(close.index),
                columns=list(close.columns),
            )
            close = close.where(membership, np.nan)
            future_returns = future_returns.where(membership, np.nan)

        matrix = compute_momentum_sweep_matrix(
            close=close,
            future_returns=future_returns,
            lookbacks=lookbacks,
            metric=args.metric,
            period=args.period,
        )
        run_tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        outdir = ROOT / "results" / run_tag
        outdir.mkdir(parents=True, exist_ok=True)
        matrix_path = outdir / f"momentum_sweep_{args.metric}_{args.period}.csv"
        matrix.to_csv(matrix_path, float_format="%.10g")
        png_path = outdir / f"momentum_sweep_{args.metric}_{args.period}.png"
        plot_heatmap(
            matrix=matrix,
            title=f"Momentum Sweep ({args.metric}, by {args.period})",
            outpath=png_path,
        )
        print(f"Saved matrix: {matrix_path}")
        print(f"Saved heatmap: {png_path}")
        if bool(args.print_run_summary):
            print("")
            print("DATA SUMMARY")
            print(f"RequestedTickers: {len(tickers)}")
            print(f"LoadedTickers: {len(used_tickers)}")
            print(f"MissingTickers: {len(missing_tickers)}")
            print(f"RejectedTickers: {len(rejected_tickers)}")
            print(f"DataSource: {data_source_summary.get('source', '')}")
            print(f"EarliestDate: {data_source_summary.get('earliest_date', '')}")
            print(f"LatestDate: {data_source_summary.get('latest_date', '')}")
            print(f"Metric: {args.metric}")
            print(f"Period: {args.period}")
            print(f"Lookbacks: {lookbacks}")
        return

    if args.command == "build-sp500-membership":
        out = save_sp500_historical_membership_csv(
            output_path=args.output,
            start_date=args.start,
            end_date=(args.end if args.end.strip() else None),
        )
        print(str(out))
        return

    if args.command == "ingest-equity-cache":
        report = ingest_equity_cache(
            cache_dir=args.cache_dir,
            source=args.source,
            store_root=args.store_root,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "ingest-universe-membership":
        report = ingest_universe_membership_csv(
            csv_path=args.csv_path,
            universe=args.universe,
            store_root=args.store_root,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "download-stooq-universe":
        symbols = load_symbol_file(Path(args.symbol_file))
        summary = download_stooq_universe(symbols=symbols, out_dir=Path(args.out_dir))
        print("\nDownload summary:")
        print(f"downloads attempted: {summary['attempted']}")
        print(f"downloads successful: {summary['successful']}")
        print(f"downloads skipped: {summary['skipped']}")
        print(f"downloads failed: {summary['failed']}")
        return

    if args.command == "download-tiingo-universe":
        symbols = load_tiingo_symbol_file(Path(args.symbol_file))
        summary = download_tiingo_universe(symbols=symbols, out_dir=Path(args.out_dir))
        print("\nDownload Summary")
        print(f"attempted: {summary['attempted']}")
        print(f"successful: {summary['successful']}")
        print(f"skipped: {summary['skipped']}")
        print(f"failed: {summary['failed']}")
        return


if __name__ == "__main__":
    main()
