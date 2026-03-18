"""Backtest runner."""

from __future__ import annotations

import json
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.data.fundamentals import normalize_ticker_symbol
from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel, load_fundamentals_file
from quant_lab.data.historical_membership import (
    load_historical_membership,
    load_historical_membership_from_parquet,
)
from quant_lab.data.quality import (
    compute_price_quality,
    flag_bad_tickers,
    format_price_quality_message,
    summarize_price_quality,
)
from quant_lab.data.universe_dynamic import (
    apply_universe_filter_to_scores,
    compute_eligibility_components,
)
from quant_lab.data.universe_dataset import (
    build_point_in_time_universe,
    load_universe_dataset,
    save_universe_dataset,
    summarize_universe_membership,
)
from quant_lab.data.universe import load_sp500_tickers
from quant_lab.engine.metrics import compute_metrics
from quant_lab.factors.combine import aggregate_factor_scores, combine_factor_scores
from quant_lab.factors.neutralize import neutralize_scores_cs
from quant_lab.factors.normalize import (
    normalize_scores,
    percentile_rank_cs,
    preprocess_factor_scores,
    robust_preprocess_base,
)
from quant_lab.factors.orthogonalize import maybe_orthogonalize_factor_scores
from quant_lab.factors.registry import compute_factors
from quant_lab.results.artifacts import (
    write_composite_scores_snapshot,
    write_equity_curve,
    write_holdings,
    write_price_quality,
    write_regime,
    write_windows,
)
from quant_lab.results.registry import append_registry_row
from quant_lab.risk.regime import (
    build_regime_weight_series,
    compute_regime_label,
    parse_weight_map,
    should_apply_dynamic_factor_weights,
    weight_map_to_json,
)
from quant_lab.risk.trend_filter import apply_trend_filter
from quant_lab.strategies.topn import (
    rebalance_mask,
    build_multi_sleeve_weights,
    build_topn_weights,
    simulate_portfolio,
)
from quant_lab.universe.liquid_us import build_liquid_us_universe

_WARNED_SECTOR_MAP = False
_WARNED_MARKET_CAP_MAP = False
_FACTOR_DEFAULT_HISTORY = {
    "momentum_12_1": 252,
    "momentum_6_1": 126,
    "momentum_3_1": 63,
    "reversal_1m": 21,
    "mean_reversion_5": 5,
    "low_vol_20": 20,
    "low_vol_60": 60,
}


def _default_multi_sleeve_config() -> dict[str, Any]:
    return {
        "sleeves": [
            {
                "name": "momentum",
                "factors": ["momentum_12_1", "reversal_1m"],
                "factor_weights": [0.65, 0.35],
                "allocation": 0.50,
                "top_n": 25,
            },
            {
                "name": "defensive",
                "factors": ["low_vol_20"],
                "factor_weights": [1.0],
                "allocation": 0.25,
                "top_n": 15,
            },
            {
                "name": "quality",
                "factors": ["gross_profitability"],
                "factor_weights": [1.0],
                "allocation": 0.25,
                "top_n": 15,
            },
        ]
    }


def _resolve_multi_sleeve_config(
    config: dict[str, Any] | None,
    available_factors: list[str],
) -> list[dict[str, Any]]:
    raw = _default_multi_sleeve_config() if config is None else dict(config)
    sleeves = raw.get("sleeves", [])
    if not isinstance(sleeves, list) or not sleeves:
        raise ValueError("multi_sleeve_config must include a non-empty 'sleeves' list")
    out: list[dict[str, Any]] = []
    factor_set = set(available_factors)
    for i, sleeve in enumerate(sleeves):
        if not isinstance(sleeve, dict):
            raise ValueError("each sleeve in multi_sleeve_config must be a dict")
        name = str(sleeve.get("name", f"sleeve_{i+1}")).strip().lower()
        factors = [str(x).strip() for x in list(sleeve.get("factors", []))]
        if not factors:
            raise ValueError(f"multi_sleeve_config[{name}] factors must be non-empty")
        missing = [f for f in factors if f not in factor_set]
        if missing:
            raise ValueError(f"multi_sleeve_config[{name}] factors missing from run factors: {missing}")
        weights = sleeve.get("factor_weights", [1.0] * len(factors))
        w = np.asarray(weights, dtype=float)
        if len(w) != len(factors):
            raise ValueError(f"multi_sleeve_config[{name}] factor_weights length mismatch")
        if not np.isfinite(w).all():
            raise ValueError(f"multi_sleeve_config[{name}] factor_weights must be finite")
        if abs(float(w.sum())) < 1e-12:
            raise ValueError(f"multi_sleeve_config[{name}] factor_weights sum must be non-zero")
        w = w / float(w.sum())
        allocation = float(sleeve.get("allocation", 0.0))
        top_n = int(sleeve.get("top_n", 0))
        if allocation < 0.0:
            raise ValueError(f"multi_sleeve_config[{name}] allocation must be >= 0")
        if top_n <= 0:
            raise ValueError(f"multi_sleeve_config[{name}] top_n must be > 0")
        out.append(
            {
                "name": name,
                "factors": factors,
                "factor_weights": w.tolist(),
                "allocation": allocation,
                "top_n": top_n,
            }
        )
    alloc_sum = float(sum(float(s["allocation"]) for s in out))
    if alloc_sum <= 0.0:
        raise ValueError("multi_sleeve_config allocations must sum to > 0")
    return out


def _default_sector_map_path() -> Path:
    candidates = [
        Path("projects/dashboard_legacy/data/sp500_tickers.csv"),
        Path("data/sp500_tickers.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return Path("")


def _load_universe_seed_tickers(universe: str, max_tickers: int, data_cache_dir: str) -> list[str]:
    """Resolve initial ticker seed set for a backtest run."""
    name = str(universe).strip().lower()
    if name in {"", "sp500"}:
        root = Path(data_cache_dir)
        membership_path = root / "universe_membership.parquet"
        if membership_path.exists():
            try:
                mem = pd.read_parquet(membership_path, columns=["universe", "ticker", "in_universe"])
                mem.columns = [str(c).strip().lower() for c in mem.columns]
                mem = mem.loc[mem["universe"].astype(str).str.strip().str.lower().eq("sp500")].copy()
                if not mem.empty:
                    in_universe = mem["in_universe"]
                    if in_universe.dtype != bool:
                        if pd.api.types.is_numeric_dtype(in_universe):
                            in_universe = in_universe.fillna(0).astype(float) > 0.0
                        else:
                            txt = in_universe.astype(str).str.strip().str.lower()
                            in_universe = txt.isin({"1", "true", "t", "yes", "y"})
                    tickers = (
                        mem.loc[in_universe, "ticker"]
                        .astype(str)
                        .map(normalize_ticker_symbol)
                        .replace("", np.nan)
                        .dropna()
                        .drop_duplicates()
                        .sort_values()
                        .tolist()
                    )
                    if tickers:
                        return tickers[: int(max_tickers)]
            except Exception as exc:
                warnings.warn(
                    f"Failed to read sp500 tickers from {membership_path}; falling back to CSV seed ({exc}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return sorted(load_sp500_tickers())[: int(max_tickers)]
    if name in {"all", "all_us", "all-us", "us", "us_equities", "liquid_us"}:
        root = Path(data_cache_dir)
        meta_path = root / "metadata.parquet"
        daily_path = root / "daily_ohlcv.parquet"
        tickers: list[str] = []
        if meta_path.exists():
            meta = pd.read_parquet(meta_path, columns=["ticker"])
            tickers = (
                meta["ticker"]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace("", np.nan)
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
        elif daily_path.exists():
            daily = pd.read_parquet(daily_path, columns=["ticker"])
            tickers = (
                daily["ticker"]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace("", np.nan)
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
        if not tickers:
            raise ValueError(
                f"No tickers found for universe='{universe}' in data_cache_dir='{data_cache_dir}'."
            )
        return tickers[: int(max_tickers)]
    raise ValueError(
        "Unsupported universe seed. Use 'sp500', 'liquid_us', or 'all' "
        "(aliases: all_us, all-us, us, us_equities)."
    )


def _parse_ticker_csv(raw: str) -> set[str]:
    return {x.strip().upper() for x in str(raw).split(",") if x.strip()}


def _parse_factor_neutralization_mode(raw: str | None) -> str | None:
    if raw is None:
        return None
    txt = str(raw).strip().lower()
    if txt in {"", "none", "null"}:
        return None
    allowed = {"beta", "sector", "beta_sector", "beta_sector_size"}
    if txt not in allowed:
        raise ValueError(
            "factor_neutralization must be one of: None, beta, sector, beta_sector, beta_sector_size"
        )
    return txt


def _parse_factor_aggregation_method(raw: str | None) -> str:
    txt = str(raw).strip().lower() if raw is not None else "linear"
    txt = txt or "linear"
    if txt not in {"linear", "mean_rank", "geometric_rank"}:
        raise ValueError("factor_aggregation_method must be one of: linear, mean_rank, geometric_rank")
    return txt


def _compute_beta_exposure(
    close: pd.DataFrame,
    spy_close: pd.Series,
    lookback: int = 60,
) -> pd.DataFrame:
    if int(lookback) <= 1:
        raise ValueError("beta lookback must be > 1")
    rets = close.astype(float).pct_change()
    spy = pd.Series(spy_close).astype(float).reindex(close.index)
    spy_ret = spy.pct_change()
    cov = rets.rolling(int(lookback)).cov(spy_ret)
    var = spy_ret.rolling(int(lookback)).var(ddof=0)
    beta = cov.div(var, axis=0).shift(1)
    return beta.replace([np.inf, -np.inf], np.nan).astype(float)


def _factor_required_history_days(
    factor_names: list[str],
    factor_params_map: dict[str, dict],
) -> int:
    """Estimate max historical requirement across selected factors."""
    req = 0
    for name in factor_names:
        base = int(_FACTOR_DEFAULT_HISTORY.get(name, 0))
        params = factor_params_map.get(name, {}) or {}
        lookbacks = [
            int(v)
            for k, v in params.items()
            if isinstance(v, (int, float)) and "lookback" in str(k).lower()
        ]
        req = max(req, base, max(lookbacks) if lookbacks else 0)
    return int(req)


def _augment_factor_params_with_fundamentals(
    factor_names: list[str],
    factor_params_map: dict[str, dict],
    close: pd.DataFrame,
    fundamentals_path: str,
    fundamentals_fallback_lag_days: int,
) -> dict[str, dict]:
    """Inject PIT-aligned fundamentals panels for fundamentals-dependent factors."""
    fundamentals_factors = {"gross_profitability", "earnings_yield", "roa", "asset_turnover", "book_to_market"}
    needed = set(factor_names).intersection(fundamentals_factors)
    if not needed:
        return {k: dict(v) for k, v in factor_params_map.items()}
    path = str(fundamentals_path).strip()
    if not path:
        raise ValueError(
            "fundamentals-dependent factors require fundamentals data. Set fundamentals_path "
            "(for example: data/fundamentals/fundamentals_fmp.parquet)."
        )
    fundamentals = load_fundamentals_file(path=path, fallback_lag_days=int(fundamentals_fallback_lag_days))
    aligned = align_fundamentals_to_daily_panel(
        fundamentals=fundamentals,
        price_index=pd.DatetimeIndex(close.index),
        price_columns=close.columns,
    )
    out = {k: dict(v) for k, v in factor_params_map.items()}
    for factor_name in needed:
        f_params = dict(out.get(factor_name, {}))
        f_params["fundamentals_aligned"] = aligned
        out[factor_name] = f_params
    return out


def _extract_log_market_cap_exposure(
    close: pd.DataFrame,
    factor_params_map: dict[str, dict],
) -> pd.DataFrame | None:
    """Build PIT-safe log market-cap exposure from close * shares_outstanding when available."""
    aligned_obj: dict[str, pd.DataFrame] | None = None
    for params in factor_params_map.values():
        if isinstance(params, dict) and isinstance(params.get("fundamentals_aligned"), dict):
            aligned_obj = params.get("fundamentals_aligned")  # type: ignore[assignment]
            break
    if not aligned_obj:
        return None
    shares = aligned_obj.get("shares_outstanding")
    if not isinstance(shares, pd.DataFrame):
        return None
    sh = shares.reindex(index=close.index, columns=close.columns).astype(float)
    mc = close.astype(float) * sh
    mc = mc.where(mc > 0.0)
    return np.log(mc).replace([np.inf, -np.inf], np.nan).astype(float)


def _default_universe_dataset_path() -> Path:
    return Path("results") / "universe_dataset" / "universe_membership.csv"


def _align_universe_membership(
    membership: pd.DataFrame,
    close: pd.DataFrame,
    exempt: set[str],
) -> pd.DataFrame:
    aligned = (
        membership.reindex(index=close.index)
        .ffill()
        .reindex(columns=close.columns)
        .fillna(False)
        .astype(bool)
    )
    for ticker in exempt:
        if ticker in aligned.columns:
            aligned.loc[:, ticker] = True
    return aligned


def print_run_summary(summary: dict) -> None:
    """Print concise terminal diagnostics for a completed run."""
    def _fmt(v) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            if pd.isna(v):
                return "-"
            return f"{v:.6g}"
        return str(v)

    print("\nRUN SUMMARY")
    print(f"Outdir: {_fmt(summary.get('Outdir'))}")
    print(f"Mode: {_fmt(summary.get('Mode'))}")

    print("\nUNIVERSE LOADING")
    print(f"RequestedTickers: {_fmt(summary.get('RequestedTickersCount'))}")
    print(f"LoadedTickers: {_fmt(summary.get('LoadedTickersCount'))}")
    print(f"MissingTickers: {_fmt(summary.get('MissingTickersCount'))}")
    print(f"RejectedTickers: {_fmt(summary.get('RejectedTickersCount'))}")
    if summary.get("FilteringStages"):
        print(f"FilteringStages: {_fmt(summary.get('FilteringStages'))}")
    missing_examples = summary.get("MissingTickersSample") or []
    if isinstance(missing_examples, list) and missing_examples:
        print("\nMissing examples:")
        for sym in missing_examples[:10]:
            print(str(sym))
    rejected_examples = summary.get("RejectedTickersSample") or []
    if isinstance(rejected_examples, list) and rejected_examples:
        print("\nRejected examples:")
        for sym in rejected_examples[:10]:
            print(str(sym))

    print("\nDATA")
    print(f"DataSource: {_fmt(summary.get('DataSource'))}")
    print(f"UniverseDatasetStartDate: {_fmt(summary.get('UniverseDatasetStartDate'))}")
    print(f"UniverseDatasetEndDate: {_fmt(summary.get('UniverseDatasetEndDate'))}")
    print(f"TickersUsed: {_fmt(summary.get('TickersUsed'))}")
    if "DataCachedFilesUsed" in summary or "DataFetchedOrRefreshed" in summary:
        print(f"cached_files_used: {_fmt(summary.get('DataCachedFilesUsed'))}")
        print(f"fetched_or_refreshed: {_fmt(summary.get('DataFetchedOrRefreshed'))}")

    print("\nUNIVERSE")
    print(f"EligibleTickersMin: {_fmt(summary.get('EligibleTickersMin'))}")
    print(f"EligibleTickersMedian: {_fmt(summary.get('EligibleTickersMedian'))}")
    print(f"EligibleTickersMax: {_fmt(summary.get('EligibleTickersMax'))}")

    print("\nPORTFOLIO")
    print(f"FactorNames: {_fmt(summary.get('FactorNames'))}")
    print(f"FactorWeights: {_fmt(summary.get('FactorWeights'))}")
    print(f"TopN: {_fmt(summary.get('TopN'))}")
    print(f"RankBuffer: {_fmt(summary.get('RankBuffer'))}")
    print(f"VolatilityScaledWeights: {_fmt(summary.get('VolatilityScaledWeights'))}")
    print(f"Rebalance: {_fmt(summary.get('Rebalance'))}")
    print(f"CostsBps: {_fmt(summary.get('CostsBps'))}")

    print("\nPERFORMANCE")
    print(f"CAGR: {_fmt(summary.get('CAGR'))}")
    print(f"Sharpe: {_fmt(summary.get('Sharpe'))}")
    print(f"Vol: {_fmt(summary.get('Vol'))}")
    print(f"MaxDD: {_fmt(summary.get('MaxDD'))}")
    if int(summary.get("SanityWarningsCount") or 0) > 0:
        print(f"SanityWarnings: {_fmt(summary.get('SanityWarningsCount'))}")

    windows = summary.get("Windows")
    if isinstance(windows, list) and windows:
        print("\nWINDOW RESULTS")
        for w in windows:
            print(
                f"{_fmt(w.get('window_id'))} | "
                f"{_fmt(w.get('test_start'))} -> {_fmt(w.get('test_end'))} | "
                f"Sharpe {_fmt(w.get('Sharpe'))} | CAGR {_fmt(w.get('CAGR'))}"
            )


def _prepare_close_panel(close_raw: pd.DataFrame, price_fill_mode: str) -> pd.DataFrame:
    """Align close panel with configurable price fill behavior."""
    mode = str(price_fill_mode).lower()
    close = close_raw.astype(float).sort_index()
    if mode == "ffill":
        close = close.ffill()
    elif mode == "none":
        close = close.loc[close.notna().any(axis=1)]
    else:
        raise ValueError("price_fill_mode must be one of: ffill, none")
    return close


def _lookup_ohlcv_frame(ohlcv_map: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame | None:
    """Lookup OHLCV frame by ticker with conservative key normalization."""
    if ticker in ohlcv_map:
        return ohlcv_map[ticker]
    lut = {str(k).upper(): k for k in ohlcv_map.keys()}
    raw = str(ticker).upper()
    candidates = [
        raw,
        raw.replace(".", "-"),
        raw.replace("-", "."),
        raw + ".US",
        raw.replace(".", "-") + ".US",
    ]
    if raw.endswith(".US"):
        base = raw[:-3]
        candidates.extend([base, base.replace(".", "-"), base.replace("-", ".")])
    for c in candidates:
        key = lut.get(c)
        if key is not None:
            return ohlcv_map.get(key)
    return None


def _collect_price_series(
    ohlcv_map: dict[str, pd.DataFrame],
    requested_tickers: list[str],
    field: str,
    fallback_field: str | None = None,
) -> tuple[list[pd.Series], list[str], list[str], list[str], list[str]]:
    price_cols: list[pd.Series] = []
    used_tickers: list[str] = []
    missing_tickers: list[str] = []
    rejected_tickers: list[str] = []
    primary = str(field)
    fallback = str(fallback_field) if fallback_field is not None else None
    for ticker in requested_tickers:
        df = _lookup_ohlcv_frame(ohlcv_map, ticker)
        if df is None:
            missing_tickers.append(ticker)
            continue
        if df.empty:
            rejected_tickers.append(ticker)
            continue
        series = None
        if primary in df.columns and pd.Series(df[primary]).notna().any():
            series = pd.Series(df[primary])
        elif fallback is not None and fallback in df.columns and pd.Series(df[fallback]).notna().any():
            series = pd.Series(df[fallback])
        if series is None:
            rejected_tickers.append(ticker)
            continue
        price_cols.append(series.rename(ticker))
        used_tickers.append(ticker)
    key_sample = sorted([str(k) for k in ohlcv_map.keys()])[:10]
    return price_cols, used_tickers, sorted(set(missing_tickers)), sorted(set(rejected_tickers)), key_sample


def _collect_close_series(
    ohlcv_map: dict[str, pd.DataFrame],
    requested_tickers: list[str],
) -> tuple[list[pd.Series], list[str], list[str], list[str], list[str]]:
    return _collect_price_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=requested_tickers,
        field="Close",
    )


def _collect_numeric_panel(
    ohlcv_map: dict[str, pd.DataFrame],
    requested_tickers: list[str],
    field: str,
) -> pd.DataFrame:
    """Collect one numeric OHLCV field into a wide panel aligned by date."""
    cols: list[pd.Series] = []
    for ticker in requested_tickers:
        df = _lookup_ohlcv_frame(ohlcv_map, ticker)
        if df is None or df.empty or field not in df.columns:
            continue
        s = pd.to_numeric(df[field], errors="coerce")
        if not s.notna().any():
            continue
        cols.append(pd.Series(s, index=df.index, name=ticker))
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1, join="outer")


def _max_true_run(mask: pd.Series) -> int:
    vals = mask.fillna(False).to_numpy(dtype=bool)
    best = 0
    cur = 0
    for v in vals:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _price_panel_health_report(
    close_raw: pd.DataFrame,
    close_filled: pd.DataFrame,
    max_abs_ret_thresh: float = 2.0,
    max_null_run_thresh: int = 252,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rows: list[dict] = []
    for ticker in close_filled.columns:
        s_raw = close_raw[ticker] if ticker in close_raw.columns else pd.Series(index=close_raw.index, dtype=float)
        s = close_filled[ticker]
        ret = s.pct_change()
        rows.append(
            {
                "ticker": str(ticker),
                "min_close": float(s.min(skipna=True)) if s.notna().any() else float("nan"),
                "max_close": float(s.max(skipna=True)) if s.notna().any() else float("nan"),
                "non_null_frac_raw": float(s_raw.notna().mean()) if len(s_raw) > 0 else 0.0,
                "max_abs_daily_return": float(ret.abs().max(skipna=True)) if ret.notna().any() else float("nan"),
                "max_null_run_raw": int(_max_true_run(s_raw.isna())) if len(s_raw) > 0 else 0,
            }
        )
    report = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    report["bad_nonpositive_close"] = report["min_close"] <= 0.0
    report["bad_extreme_return"] = report["max_abs_daily_return"] > float(max_abs_ret_thresh)
    report["bad_long_null_run"] = report["max_null_run_raw"] > int(max_null_run_thresh)
    broken = report.loc[report["bad_nonpositive_close"] | report["bad_extreme_return"], "ticker"].astype(str).tolist()
    suspicious = report.loc[report["bad_long_null_run"], "ticker"].astype(str).tolist()
    return report, broken, suspicious


def _print_portfolio_sanity_warnings(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    sim: pd.DataFrame,
    cagr: float,
    vol: float,
) -> list[str]:
    warnings_out: list[str] = []
    daily = pd.to_numeric(sim["DailyReturn"], errors="coerce")
    extreme_mask = (daily > 1.0) | (daily < -0.95)
    if bool(extreme_mask.any()):
        aligned_w = (
            weights.reindex(close.index)
            .ffill()
            .reindex(columns=close.columns, fill_value=0.0)
            .fillna(0.0)
            .astype(float)
        )
        contrib = aligned_w.shift(1).fillna(0.0) * close.pct_change().fillna(0.0)
        extreme_dates = list(daily.index[extreme_mask])[:10]
        for dt in extreme_dates:
            top = contrib.loc[dt].sort_values(key=lambda x: x.abs(), ascending=False).head(5)
            pairs = [(str(k), float(v)) for k, v in top.items() if np.isfinite(v)]
            msg = (
                f"SANITY WARNING: extreme portfolio return date={str(pd.Timestamp(dt).date())} "
                f"ret={float(daily.loc[dt]):.6g} top_contributors={pairs}"
            )
            print(msg)
            warnings_out.append(msg)
    if np.isfinite(cagr) and cagr > 1.0:
        msg = f"SANITY WARNING: CAGR unusually high ({float(cagr):.6g} > 1.0)."
        print(msg)
        warnings_out.append(msg)
    if np.isfinite(vol) and vol > 1.0:
        msg = f"SANITY WARNING: annualized vol unusually high ({float(vol):.6g} > 1.0)."
        print(msg)
        warnings_out.append(msg)
    return warnings_out


def _stats_min_med_max(counts: pd.Series) -> tuple[float, int, int]:
    if counts.empty:
        return float("nan"), 0, 0
    return float(counts.median()), int(counts.min()), int(counts.max())


def _rebalance_score_counts(scores: pd.DataFrame, rb_mask: pd.Series) -> pd.Series:
    """Count finite scores on rebalance dates only."""
    rb_scores = scores.loc[rb_mask]
    if rb_scores.empty:
        return pd.Series(dtype=int, index=rb_scores.index)
    counts = np.isfinite(rb_scores.to_numpy()).sum(axis=1)
    return pd.Series(counts.astype(int), index=rb_scores.index)


def _apply_universe_rebalance_skip(
    scores: pd.DataFrame,
    rb_mask: pd.Series,
    universe_min_tickers: int,
    universe_skip_below_min_tickers: bool,
) -> tuple[pd.DataFrame, int, int]:
    """Optionally skip rebalance rows by blanking scores on those dates."""
    rb_score_counts = _rebalance_score_counts(scores, rb_mask)
    skip_zero_mask = rb_score_counts <= 0
    if bool(universe_skip_below_min_tickers):
        skip_below_mask = (rb_score_counts > 0) & (rb_score_counts < int(universe_min_tickers))
    else:
        skip_below_mask = pd.Series(False, index=rb_score_counts.index)
    skip_zero_count = int(skip_zero_mask.sum())
    skip_below_count = int(skip_below_mask.sum())
    if skip_zero_count + skip_below_count == 0:
        return scores, skip_zero_count, skip_below_count
    out = scores.copy()
    skip_dates = rb_score_counts.index[skip_zero_mask | skip_below_mask]
    out.loc[skip_dates] = np.nan
    return out, skip_zero_count, skip_below_count


def _resolve_universe_eligibility(
    eligibility_price: pd.DataFrame,
    scores: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    """Resolve final eligibility source: price-only or price & score-available."""
    src = str(source).lower()
    if src not in {"price", "score"}:
        raise ValueError("universe_eligibility_source must be one of: price, score")
    elig_price = (
        eligibility_price.reindex(index=scores.index, columns=scores.columns)
        .fillna(False)
        .astype(bool)
    )
    if src == "price":
        return elig_price
    return (elig_price & scores.notna()).astype(bool)


def _build_zero_eligible_debug_frame(
    rebalance_dates: pd.DatetimeIndex,
    close: pd.DataFrame,
    eligibility_price: pd.DataFrame,
    price_ok: pd.DataFrame,
    history_ok: pd.DataFrame,
    valid_ok: pd.DataFrame,
    factor_scores: dict[str, pd.DataFrame],
    composite_scores: pd.DataFrame,
    eligibility_final: pd.DataFrame,
    factor_names: list[str],
    effective_min_history_days: int,
    valid_lookback: int,
    min_valid_frac: float,
    window_id: int | None = None,
) -> pd.DataFrame:
    """Build debug rows for rebalance dates where final eligible count is zero."""
    rb_dates = pd.DatetimeIndex(rebalance_dates)
    if rb_dates.empty:
        return pd.DataFrame()
    rb_dates = rb_dates.intersection(close.index).intersection(composite_scores.index)
    if rb_dates.empty:
        return pd.DataFrame()

    close_valid_count = close.reindex(rb_dates).notna().sum(axis=1).astype(int)
    eligibility_price_count = (
        eligibility_price.reindex(index=rb_dates, columns=composite_scores.columns)
        .fillna(False)
        .sum(axis=1)
        .astype(int)
    )
    price_ok_count = (
        price_ok.reindex(index=rb_dates, columns=composite_scores.columns)
        .fillna(False)
        .sum(axis=1)
        .astype(int)
    )
    history_ok_count = (
        history_ok.reindex(index=rb_dates, columns=composite_scores.columns)
        .fillna(False)
        .sum(axis=1)
        .astype(int)
    )
    valid_ok_count = (
        valid_ok.reindex(index=rb_dates, columns=composite_scores.columns)
        .fillna(False)
        .sum(axis=1)
        .astype(int)
    )
    composite_valid_count = composite_scores.reindex(rb_dates).notna().sum(axis=1).astype(int)
    eligibility_final_count = (
        eligibility_final.reindex(index=rb_dates, columns=composite_scores.columns)
        .fillna(False)
        .sum(axis=1)
        .astype(int)
    )
    zero_mask = eligibility_final_count == 0

    frame = pd.DataFrame(
        {
            "date": rb_dates.astype(str),
            "close_valid_count": close_valid_count.to_numpy(dtype=int),
            "eligibility_price_count": eligibility_price_count.to_numpy(dtype=int),
            "price_ok_count": price_ok_count.to_numpy(dtype=int),
            "history_ok_count": history_ok_count.to_numpy(dtype=int),
            "valid_ok_count": valid_ok_count.to_numpy(dtype=int),
            "composite_valid_count": composite_valid_count.to_numpy(dtype=int),
            "eligibility_final_count": eligibility_final_count.to_numpy(dtype=int),
            "effective_min_history_days": int(effective_min_history_days),
            "valid_lookback": int(valid_lookback),
            "min_valid_frac": float(min_valid_frac),
        }
    )
    for name in factor_names:
        panel = factor_scores.get(name)
        if panel is None:
            vals = pd.Series(0, index=rb_dates, dtype=int)
        else:
            vals = panel.reindex(rb_dates).notna().sum(axis=1).astype(int)
        frame[f"factor_valid_count_{name}"] = vals.to_numpy(dtype=int)

    frame = frame.loc[zero_mask.to_numpy()].copy()
    if window_id is not None and not frame.empty:
        frame.insert(0, "window_id", int(window_id))
    return frame


def filter_bad_tickers(
    close: pd.DataFrame,
    window_name: str,
    zero_ret_thresh: float,
    min_valid_frac: float,
    exempt: set[str],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Drop bad tickers (except exempt) based on existing quality definitions."""
    q = compute_price_quality(
        close=close,
        window_name=window_name,
        zero_ret_thresh=zero_ret_thresh,
        min_valid_frac=min_valid_frac,
    )
    flagged = flag_bad_tickers(
        quality_df=q,
        zero_ret_thresh=zero_ret_thresh,
        min_valid_frac=min_valid_frac,
    )
    tickers_upper = pd.Series(flagged["ticker"].astype(str).str.upper().to_numpy(), index=flagged.index)
    bad_drop_mask = flagged["is_bad"] & (~tickers_upper.isin({x.upper() for x in exempt}))
    dropped = sorted(flagged.loc[bad_drop_mask, "ticker"].astype(str).tolist())
    filtered = close.drop(columns=dropped, errors="ignore")
    return filtered, dropped, flagged


def _git_commit() -> str:
    """Best-effort git commit hash for provenance."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return ""


def _write_summary_artifacts(outdir: Path, run_config: dict, summary: dict) -> None:
    """Write canonical JSON summary while keeping legacy text compatibility."""
    (outdir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (outdir / "summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n",
        encoding="utf-8",
    )


def _load_sector_map(sector_map_path: str, tickers: list[str]) -> dict[str, str] | None:
    """Load ticker->sector map; unknowns become UNKNOWN."""
    global _WARNED_SECTOR_MAP
    map_path = str(sector_map_path).strip()
    if not map_path:
        default_path = _default_sector_map_path()
        map_path = str(default_path) if default_path else ""
    if not map_path:
        return None
    try:
        df = pd.read_csv(map_path)
    except Exception as exc:
        if not _WARNED_SECTOR_MAP:
            warnings.warn(
                f"Could not read sector_map '{map_path}': {exc}. Sector-aware features disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_SECTOR_MAP = True
        return None

    ticker_col = next(
        (c for c in ["Ticker", "ticker", "Symbol", "symbol"] if c in df.columns),
        None,
    )
    sector_col = next(
        (c for c in ["Sector", "sector", "GICS Sector", "gics sector"] if c in df.columns),
        None,
    )
    if ticker_col is None or sector_col is None:
        if not _WARNED_SECTOR_MAP:
            warnings.warn(
                f"sector_map '{map_path}' missing required ticker/sector columns. Sector-aware features disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_SECTOR_MAP = True
        return None

    mapping = {
        normalize_ticker_symbol(str(t).strip()): str(s).strip() if str(s).strip() else "UNKNOWN"
        for t, s in zip(df[ticker_col], df[sector_col])
    }
    return {t: mapping.get(normalize_ticker_symbol(t), "UNKNOWN") for t in tickers}


def _load_market_cap_map(sector_map_path: str, tickers: list[str]) -> dict[str, float] | None:
    """Load ticker->log_market_cap map from optional MarketCap column."""
    global _WARNED_MARKET_CAP_MAP
    if not sector_map_path.strip():
        return None
    try:
        df = pd.read_csv(sector_map_path)
    except Exception as exc:
        if not _WARNED_MARKET_CAP_MAP:
            warnings.warn(
                f"Could not read market-cap map from '{sector_map_path}': {exc}. Size neutralization disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_MARKET_CAP_MAP = True
        return None
    if "Ticker" not in df.columns or "MarketCap" not in df.columns:
        if not _WARNED_MARKET_CAP_MAP:
            warnings.warn(
                f"sector_map '{sector_map_path}' missing required MarketCap column for size neutralization.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_MARKET_CAP_MAP = True
        return None
    tmp = df[["Ticker", "MarketCap"]].copy()
    tmp["Ticker"] = tmp["Ticker"].astype(str).str.strip()
    tmp["MarketCap"] = pd.to_numeric(tmp["MarketCap"], errors="coerce")
    tmp = tmp.loc[tmp["MarketCap"] > 0.0]
    if tmp.empty:
        if not _WARNED_MARKET_CAP_MAP:
            warnings.warn(
                f"sector_map '{sector_map_path}' has no positive MarketCap values; size neutralization disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_MARKET_CAP_MAP = True
        return None
    mp = {str(t): float(np.log(v)) for t, v in zip(tmp["Ticker"], tmp["MarketCap"])}
    return {t: mp.get(t, np.nan) for t in tickers}


def _apply_vol_target(
    sim: pd.DataFrame,
    target_vol: float,
    lookback: int,
    max_leverage: float,
) -> tuple[pd.DataFrame, pd.Series, float, float]:
    """
    Apply lagged portfolio-level vol targeting on net daily returns.

    Costs are applied first in `simulate_portfolio`; this overlay then scales
    the net return stream using lagged realized volatility to avoid lookahead.
    """
    if "DailyReturn" not in sim.columns:
        raise ValueError("simulate_portfolio output must include DailyReturn.")
    out = sim.copy()
    raw_ret = pd.to_numeric(out["DailyReturn"], errors="coerce").fillna(0.0)
    raw_vol = float(raw_ret.std(ddof=0) * np.sqrt(252.0))
    lookback_i = int(lookback)
    if target_vol > 0.0 and lookback_i <= 0:
        raise ValueError("port_vol_lookback must be > 0")
    roll_win = max(1, lookback_i)
    realized_raw = raw_ret.rolling(roll_win).std(ddof=0) * np.sqrt(252.0)

    if target_vol <= 0.0:
        lev_unlagged = pd.Series(1.0, index=out.index, dtype=float)
        lev_applied = pd.Series(1.0, index=out.index, dtype=float)
        vt_ret = raw_ret.copy()
    else:
        if max_leverage <= 0.0:
            raise ValueError("max_leverage must be > 0")
        lev_unlagged = (float(target_vol) / realized_raw.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        lev_unlagged = lev_unlagged.fillna(1.0).clip(lower=0.0, upper=float(max_leverage))
        # Lag by one day so scale at t uses information available through t-1 only.
        lev_applied = lev_unlagged.shift(1).fillna(1.0).clip(lower=0.0, upper=float(max_leverage))
        vt_ret = raw_ret * lev_applied

    out["RawDailyReturn"] = raw_ret
    out["VolTargetScaleUnlagged"] = lev_unlagged
    out["VolTargetScaleApplied"] = lev_applied
    out["PortRealizedVolAnnRaw"] = realized_raw
    out["DailyReturn"] = vt_ret
    out["PortRealizedVolAnnVT"] = vt_ret.rolling(roll_win).std(ddof=0) * np.sqrt(252.0)
    out["Equity"] = (1.0 + out["DailyReturn"]).cumprod()
    final_realized_vt = (
        float(out["PortRealizedVolAnnVT"].dropna().iloc[-1])
        if out["PortRealizedVolAnnVT"].notna().any()
        else float("nan")
    )
    return out, lev_applied, raw_vol, final_realized_vt


def _count_rebalance_dates(index: pd.DatetimeIndex, rebalance: str) -> int:
    """Diagnostic rebalance count independent of trading implementation internals."""
    return int(_diagnostic_rebalance_mask(index=index, rebalance=rebalance).sum())


def _apply_bear_exposure_overlay(
    sim: pd.DataFrame,
    regime_label: pd.Series | None,
    regime_filter: bool,
    bear_exposure_scale: float,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Scale returns on bear/volatile dates using lagged regime labels."""
    scale_val = float(bear_exposure_scale)
    if scale_val <= 0.0:
        raise ValueError("bear_exposure_scale must be > 0")
    scale_series = pd.Series(1.0, index=sim.index, dtype=float)
    if not bool(regime_filter) or regime_label is None or abs(scale_val - 1.0) < 1e-12:
        sim_out = sim.copy()
        sim_out["BearExposureScaleApplied"] = scale_series
        sim_out["Equity"] = (
            1.0 + pd.to_numeric(sim_out["DailyReturn"], errors="coerce").fillna(0.0)
        ).cumprod()
        return sim_out, scale_series, 1.0

    lbl = pd.Series(regime_label).reindex(sim.index).ffill().shift(1)
    bear_mask = lbl.eq("bear_or_volatile")
    scale_series.loc[bear_mask] = scale_val
    sim_out = sim.copy()
    for col in ["DailyReturn", "RawDailyReturn"]:
        if col in sim_out.columns:
            sim_out[col] = pd.to_numeric(sim_out[col], errors="coerce").fillna(0.0) * scale_series
    sim_out["BearExposureScaleApplied"] = scale_series
    sim_out["Equity"] = (
        1.0 + pd.to_numeric(sim_out["DailyReturn"], errors="coerce").fillna(0.0)
    ).cumprod()
    bear_scale_avg = float(scale_series.loc[bear_mask].mean()) if bool(bear_mask.any()) else 1.0
    return sim_out, scale_series, bear_scale_avg


def _diagnostic_rebalance_mask(index: pd.DatetimeIndex, rebalance: str) -> pd.Series:
    """Calendar-based rebalance mask for diagnostics/summaries."""
    dates = pd.DatetimeIndex(index)
    if len(dates) == 0:
        return pd.Series(dtype=bool, index=dates)
    reb = rebalance.lower()
    if reb == "daily":
        return pd.Series(True, index=dates)
    if reb == "weekly":
        periods = pd.Series(dates.to_period("W-FRI"), index=dates)
        return (periods != periods.shift(1)).fillna(True)
    if reb == "biweekly":
        periods = pd.Series(dates.to_period("W-FRI"), index=dates)
        week_codes = pd.Series(pd.factorize(periods)[0], index=dates)
        period_start = (periods != periods.shift(1)).fillna(True)
        return (period_start & ((week_codes % 2) == 0)).astype(bool)
    if reb == "monthly":
        periods = pd.Series(dates.to_period("M"), index=dates)
        return (periods != periods.shift(1)).fillna(True)
    raise ValueError(f"Unsupported rebalance frequency: {rebalance}")


def _get_benchmark_close_from_map(
    ohlcv_map: dict[str, pd.DataFrame],
    score_index: pd.DatetimeIndex,
    benchmark: str,
    source: str,
) -> pd.Series:
    """Get benchmark proxy from already-loaded OHLCV data."""
    symbol = str(benchmark).strip().upper()
    bench_df = _lookup_ohlcv_frame(ohlcv_map, symbol)
    if bench_df is None or bench_df.empty:
        if str(source).lower().strip() == "parquet":
            raise ValueError(
                f"Regime benchmark {symbol} missing from parquet store / loaded OHLCV map"
            )
        raise ValueError(f"Regime benchmark {symbol} data is required but missing from loaded OHLCV map.")
    col = "Adj Close" if "Adj Close" in bench_df.columns else "Close"
    if col not in bench_df.columns:
        raise ValueError(f"Benchmark {symbol} frame must contain 'Adj Close' or 'Close'.")
    bench_close = pd.Series(bench_df[col]).astype(float).reindex(score_index).ffill()
    if bench_close.isna().all():
        raise ValueError(f"Benchmark {symbol} proxy series is empty after alignment.")
    return bench_close


def _cache_get_bucket(run_cache: dict[str, Any] | None, name: str) -> dict[Any, Any] | None:
    if run_cache is None:
        return None
    cache_store = run_cache.setdefault("__store", {})
    return cache_store.setdefault(name, {})


def _cache_get(
    run_cache: dict[str, Any] | None,
    bucket: str,
    key: Any,
    *,
    debug: bool = False,
) -> Any:
    b = _cache_get_bucket(run_cache, bucket)
    if b is None:
        return None
    if key in b:
        if debug:
            print(f"CACHE HIT  [{bucket}]")
        return b[key]
    if debug:
        print(f"CACHE MISS [{bucket}]")
    return None


def _cache_put(run_cache: dict[str, Any] | None, bucket: str, key: Any, value: Any) -> None:
    b = _cache_get_bucket(run_cache, bucket)
    if b is None:
        return
    b[key] = value


def _factor_raw_cache_key(
    *,
    factor_names: list[str],
    raw_factor_params: dict[str, Any],
    start: str,
    end: str,
    data_source: str,
    data_cache_dir: str,
    price_fill_mode: str,
    drop_bad_tickers: bool,
    universe_mode: str,
    max_tickers: int,
    close_columns: pd.Index | list[str],
) -> tuple[Any, ...]:
    """Build the raw-factor cache key from inputs that affect factor values."""
    return (
        tuple(str(x) for x in factor_names),
        json.dumps(raw_factor_params, sort_keys=True, default=str),
        str(start),
        str(end),
        str(data_source),
        str(data_cache_dir),
        str(price_fill_mode),
        bool(drop_bad_tickers),
        str(universe_mode),
        int(max_tickers),
        tuple(str(x) for x in close_columns),
    )


def run_backtest(
    start: str,
    end: str,
    max_tickers: int,
    top_n: int,
    rebalance: str,
    costs_bps: float,
    universe: str = "sp500",
    rank_buffer: int = 0,
    volatility_scaled_weights: bool = False,
    data_source: str = "parquet",
    data_cache_dir: str = "data/equities",
    data_refresh: bool = False,
    data_bulk_prepare: bool = False,
    seed: int = 42,
    factor_name: str | list[str] = "momentum_12_1",
    factor_names: list[str] | None = None,
    factor_params: dict | None = None,
    factor_weights: list[float] | None = None,
    portfolio_mode: str = "composite",
    multi_sleeve_config: dict[str, Any] | None = None,
    factor_aggregation_method: str = "linear",
    normalize: str = "zscore",
    winsor_p: float = 0.01,
    use_factor_normalization: bool = True,
    use_sector_neutralization: bool = True,
    use_size_neutralization: bool = True,
    factor_neutralization: str | None = None,
    orthogonalize_factors: bool = False,
    sector_neutral: bool = False,
    weighting: str = "equal",
    vol_lookback: int = 20,
    max_weight: float = 0.15,
    score_clip: float = 5.0,
    score_floor: float = 0.0,
    sector_cap: float = 0.0,
    sector_map: str = "",
    target_vol: float = 0.0,
    port_vol_lookback: int = 20,
    max_leverage: float = 1.0,
    slippage_bps: float = 0.0,
    slippage_vol_mult: float = 0.0,
    slippage_vol_lookback: int = 20,
    execution_delay_days: int = 0,
    regime_filter: bool = False,
    dynamic_factor_weights: bool = False,
    regime_benchmark: str = "SPY",
    regime_trend_sma: int = 200,
    regime_vol_lookback: int = 20,
    regime_vol_median_lookback: int = 252,
    regime_bull_weights: str = "momentum_12_1:0.7,low_vol_20:0.3",
    regime_bear_weights: str = "momentum_12_1:0.3,low_vol_20:0.7",
    bear_exposure_scale: float = 1.0,
    trend_filter: bool = False,
    trend_sma_window: int = 200,
    save_artifacts: bool = False,
    print_run_summary_flag: bool = False,
    price_quality_check: bool = False,
    price_quality_mode: str = "warn",
    price_quality_zero_ret_thresh: float = 0.95,
    price_quality_min_valid_frac: float = 0.98,
    price_quality_max_bad_tickers: int = 0,
    price_quality_report_topk: int = 20,
    price_fill_mode: str = "ffill",
    drop_bad_tickers: bool = False,
    drop_bad_tickers_scope: str = "test",
    drop_bad_tickers_max_drop: int = 0,
    drop_bad_tickers_exempt: str = "SPY",
    universe_mode: str = "static",
    universe_min_history_days: int = 300,
    universe_min_valid_frac: float = 0.98,
    universe_valid_lookback: int = 252,
    universe_min_price: float = 1.0,
    min_price: float = 0.0,
    min_avg_dollar_volume: float = 0.0,
    liquidity_lookback: int = 20,
    universe_min_tickers: int = 20,
    universe_skip_below_min_tickers: bool = True,
    universe_eligibility_source: str = "price",
    universe_exempt: str = "SPY",
    universe_dataset_mode: str = "off",
    universe_dataset_freq: str = "rebalance",
    universe_dataset_path: str = "",
    universe_dataset_save: bool = True,
    universe_dataset_require: bool = False,
    historical_membership_path: str = "",
    fundamentals_path: str = "data/fundamentals/fundamentals_fmp.parquet",
    fundamentals_fallback_lag_days: int = 60,
    run_cache: dict[str, Any] | None = None,
    cache_debug: bool = False,
) -> tuple[dict, str]:
    """Run a complete Top-N momentum backtest and persist run artifacts."""
    del seed  # Placeholder for future deterministic extensions.

    if str(universe).strip().lower() == "liquid_us" and str(universe_mode).strip().lower() == "dynamic":
        # Canonical research thresholds for the corrected liquid_us universe.
        universe_min_history_days = max(int(universe_min_history_days), 252)
        universe_min_price = max(float(universe_min_price), 5.0)
        min_price = max(float(min_price), 5.0)
        min_avg_dollar_volume = max(float(min_avg_dollar_volume), 10_000_000.0)
        liquidity_lookback = max(int(liquidity_lookback), 20)
    t_all_start = time.perf_counter()
    timing_sections: dict[str, float] = {
        "data_load_seconds": 0.0,
        "factor_compute_seconds": 0.0,
        "portfolio_backtest_seconds": 0.0,
        "report_write_seconds": 0.0,
    }
    cache_stats = {
        "fetch_hit": 0,
        "factor_raw_hit": 0,
        "factor_norm_hit": 0,
        "fundamentals_hit": 0,
    }
    portfolio_mode_normalized = str(portfolio_mode).strip().lower() or "composite"
    if portfolio_mode_normalized not in {"composite", "multi_sleeve"}:
        raise ValueError("portfolio_mode must be one of: composite, multi_sleeve")
    factor_neutralization_mode = _parse_factor_neutralization_mode(factor_neutralization)
    factor_aggregation_method_normalized = _parse_factor_aggregation_method(factor_aggregation_method)

    t_data_start = time.perf_counter()
    tickers = _load_universe_seed_tickers(
        universe=universe,
        max_tickers=max_tickers,
        data_cache_dir=data_cache_dir,
    )
    if not tickers:
        raise ValueError("No tickers were loaded.")

    regime_benchmark = str(regime_benchmark).strip().upper() or "SPY"
    extra_symbols: list[str] = []
    if trend_filter and "SPY" not in tickers:
        extra_symbols.append("SPY")
    if (regime_filter or dynamic_factor_weights) and regime_benchmark not in tickers and regime_benchmark not in extra_symbols:
        extra_symbols.append(regime_benchmark)
    fetch_tickers = tickers + extra_symbols
    fetch_request_tickers = list(fetch_tickers)
    fetch_cache_key = (
        tuple(fetch_tickers),
        str(start),
        str(end),
        str(data_cache_dir),
        str(data_source),
        bool(data_refresh),
        bool(data_bulk_prepare),
    )
    fetch_cached = _cache_get(run_cache, "fetch_ohlcv", fetch_cache_key, debug=cache_debug)
    if fetch_cached is not None:
        cache_stats["fetch_hit"] = 1
        ohlcv_map, data_source_summary = fetch_cached
    else:
        ohlcv_map, data_source_summary = fetch_ohlcv_with_summary(
            tickers=fetch_tickers,
            start=start,
            end=end,
            cache_dir=data_cache_dir,
            data_source=data_source,
            refresh=bool(data_refresh),
            bulk_prepare=bool(data_bulk_prepare),
        )
        _cache_put(run_cache, "fetch_ohlcv", fetch_cache_key, (ohlcv_map, data_source_summary))
    close_cols, used_tickers, missing_tickers, rejected_tickers, ohlcv_map_key_sample = (
        _collect_close_series(ohlcv_map=ohlcv_map, requested_tickers=tickers)
    )
    requested_tickers_count = int(len(tickers))
    loaded_tickers_count = int(len(used_tickers))
    missing_tickers_count = int(len(missing_tickers))
    rejected_tickers_count = int(len(rejected_tickers))
    missing_tickers_sample = missing_tickers[:10]
    rejected_tickers_sample = rejected_tickers[:10]
    fetched_keys_count = int(len(ohlcv_map))
    fetched_keys_sample = sorted(list(ohlcv_map.keys()))[:10]
    post_fetch_filtered = [t for t in tickers if _lookup_ohlcv_frame(ohlcv_map, t) is not None]
    post_fetch_filtered_keys_count = int(len(post_fetch_filtered))
    post_fetch_filtered_keys_sample = post_fetch_filtered[:10]

    if not close_cols:
        raise ValueError(
            "No valid OHLCV data found for any ticker. "
            f"requested={len(tickers)} fetch_request={len(fetch_request_tickers)} "
            f"fetched_keys={fetched_keys_count} key_sample={fetched_keys_sample} "
            f"data_source={data_source} cache_dir={data_cache_dir} "
            f"historical_membership_path={str(historical_membership_path)}"
        )

    if len(used_tickers) < (top_n + 1):
        raise ValueError(
            f"Not enough tickers with data: got {len(used_tickers)}, "
            f"need at least top_n + 1 = {top_n + 1}."
        )
    if sector_cap > 0.0 and not sector_map.strip():
        warnings.warn("sector_cap>0 but no sector_map provided; sector caps disabled.", RuntimeWarning)

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
    close = _prepare_close_panel(close_raw=close_raw, price_fill_mode=price_fill_mode)
    adj_close = _prepare_close_panel(close_raw=adj_close_raw, price_fill_mode=price_fill_mode)
    close_full = close.copy()
    min_non_nan = int(0.8 * close.shape[1])
    close = close.dropna(thresh=min_non_nan)
    if close.empty:
        raise ValueError(
            "No usable close-price history after alignment/fill; "
            "check date range and ticker coverage."
        )
    price_health_report, broken_tickers, suspicious_tickers = _price_panel_health_report(
        close_raw=close_raw.reindex(index=close.index, columns=close.columns),
        close_filled=close,
    )
    if suspicious_tickers:
        print(
            "PRICE PANEL WARNING: long null runs detected "
            f"count={len(suspicious_tickers)} sample={suspicious_tickers[:10]}"
        )
    dropped_broken_tickers: list[str] = []
    if broken_tickers:
        dropped_broken_tickers = sorted(set(broken_tickers))
        print(
            "PRICE PANEL WARNING: dropping broken tickers "
            f"(close<=0 or max_abs_daily_return>200%) count={len(dropped_broken_tickers)} "
            f"sample={dropped_broken_tickers[:10]}"
        )
        close = close.drop(columns=dropped_broken_tickers, errors="ignore")
        adj_close = adj_close.drop(columns=dropped_broken_tickers, errors="ignore")
        if close.shape[1] < (top_n + 1):
            raise ValueError(
                "Not enough tickers after dropping broken price series: "
                f"got {close.shape[1]}, need at least {top_n + 1}."
            )
    volume = (
        volume_raw.reindex(index=close.index, columns=close.columns)
        if not volume_raw.empty
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    exempt_tickers = _parse_ticker_csv(drop_bad_tickers_exempt)
    dropped_rows: list[dict] = []
    dropped_tickers_count = 0
    if drop_bad_tickers:
        window_name = f"backtest_{start}_{end}"
        close, dropped, flagged_drop = filter_bad_tickers(
            close=close,
            window_name=window_name,
            zero_ret_thresh=price_quality_zero_ret_thresh,
            min_valid_frac=price_quality_min_valid_frac,
            exempt=exempt_tickers,
        )
        dropped_tickers_count = len(dropped)
        if dropped_tickers_count > 0:
            dropped_view = flagged_drop.loc[flagged_drop["ticker"].isin(dropped)].copy()
            dropped_view["reason"] = (
                dropped_view["bad_valid"].map({True: "bad_valid", False: ""})
                + dropped_view["bad_zero"].map({True: "|bad_zero", False: ""})
            ).str.strip("|")
            dropped_rows = dropped_view[
                ["window_name", "ticker", "reason", "zero_ret_frac", "valid_frac_close"]
            ].to_dict(orient="records")
        mode = str(price_quality_mode).lower()
        if drop_bad_tickers_max_drop > 0 and dropped_tickers_count > int(drop_bad_tickers_max_drop):
            msg = (
                f"PRICE QUALITY DROP LIMIT: window=backtest_{start}..{end} "
                f"dropped={dropped_tickers_count} max_drop={int(drop_bad_tickers_max_drop)} mode={mode}"
            )
            if mode == "fail":
                raise ValueError(msg)
            print(msg)
        if close.shape[1] < (top_n + 1):
            raise ValueError(
                f"Not enough tickers after drop_bad_tickers: got {close.shape[1]}, need at least {top_n + 1}."
            )
    adj_close = adj_close.reindex(index=close.index, columns=close.columns)
    adj_close = adj_close.where(adj_close.notna(), close)
    quality_flagged = pd.DataFrame()
    quality_summary: dict = {"num_bad": 0, "worst_by_zero_ret": [], "worst_by_valid_frac": []}
    if price_quality_check:
        mode = str(price_quality_mode).lower()
        if mode not in {"warn", "fail"}:
            raise ValueError("price_quality_mode must be one of: warn, fail")
        quality_raw = compute_price_quality(
            close=close,
            window_name=f"backtest_{start}_{end}",
            zero_ret_thresh=price_quality_zero_ret_thresh,
            min_valid_frac=price_quality_min_valid_frac,
        )
        quality_flagged = flag_bad_tickers(
            quality_df=quality_raw,
            zero_ret_thresh=price_quality_zero_ret_thresh,
            min_valid_frac=price_quality_min_valid_frac,
        )
        quality_summary = summarize_price_quality(quality_flagged, topk=price_quality_report_topk)
        if int(quality_summary["num_bad"]) > int(price_quality_max_bad_tickers):
            msg = format_price_quality_message(
                window_name=f"backtest_{start}..{end}",
                summary=quality_summary,
                max_bad_tickers=price_quality_max_bad_tickers,
                mode=mode,
                preview_k=min(3, int(price_quality_report_topk)),
            )
            if mode == "fail":
                raise ValueError(msg)
            print(msg)
    u_mode = str(universe_mode).lower()
    if u_mode not in {"static", "dynamic"}:
        raise ValueError("universe_mode must be one of: static, dynamic")
    uds_mode = str(universe_dataset_mode).lower()
    if uds_mode not in {"off", "build", "use"}:
        raise ValueError("universe_dataset_mode must be one of: off, build, use")
    uds_freq = str(universe_dataset_freq).lower()
    if uds_freq not in {"daily", "rebalance"}:
        raise ValueError("universe_dataset_freq must be one of: daily, rebalance")
    historical_membership = None
    historical_membership_source = "none"
    membership_alignment_diag: dict[str, Any] = {}
    membership_covered_tickers_count = loaded_tickers_count
    membership_covered_tickers_sample = used_tickers[:10]
    if str(historical_membership_path).strip():
        historical_membership = load_historical_membership(
            path=str(historical_membership_path).strip(),
            index=pd.DatetimeIndex(close.index),
            columns=list(close.columns),
        )
        historical_membership_source = "file"
        aligned_counts = historical_membership.sum(axis=1).astype(int) if not historical_membership.empty else pd.Series(dtype=int)
        membership_alignment_diag = {
            "MembershipSource": "file",
            "MembershipPath": str(historical_membership_path).strip(),
            "MembershipUniqueTickers": int(historical_membership.columns.size),
            "MembershipAlignedPerDateMin": int(aligned_counts.min()) if not aligned_counts.empty else 0,
            "MembershipAlignedPerDateMedian": float(aligned_counts.median()) if not aligned_counts.empty else 0.0,
            "MembershipAlignedPerDateMax": int(aligned_counts.max()) if not aligned_counts.empty else 0,
        }
    elif str(universe).strip().lower() == "sp500":
        try:
            historical_membership, membership_alignment_diag = load_historical_membership_from_parquet(
                store_root=data_cache_dir,
                universe="sp500",
                start=start,
                end=end,
                index=pd.DatetimeIndex(close.index),
                columns=list(close.columns),
            )
            historical_membership_source = "parquet_auto"
            print(
                "SP500 MEMBERSHIP ALIGNMENT: "
                f"unique={int(membership_alignment_diag.get('MembershipUniqueTickers', 0))} "
                f"overlap_norm={int(membership_alignment_diag.get('MembershipPriceOverlapNormalized', 0))} "
                f"unmatched_norm={int(membership_alignment_diag.get('MembershipUnmatchedNormalized', 0))} "
                f"aligned_med={float(membership_alignment_diag.get('MembershipAlignedPerDateMedian', 0.0)):.1f}"
            )
            unmatched_sample = membership_alignment_diag.get("MembershipUnmatchedSample", [])
            if isinstance(unmatched_sample, list) and unmatched_sample:
                print(f"SP500 MEMBERSHIP UNMATCHED SAMPLE: {unmatched_sample[:10]}")
        except Exception as exc:
            warnings.warn(
                "Could not auto-load SP500 historical membership from parquet; "
                f"proceeding without membership filter ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            historical_membership_source = "none"
            membership_alignment_diag = {}
    if historical_membership is not None:
        covered_mask = historical_membership.any(axis=0).reindex(close.columns).fillna(False).astype(bool)
        covered = [c for c in close.columns if bool(covered_mask.get(c, False))]
        membership_covered_tickers_count = int(len(covered))
        membership_covered_tickers_sample = covered[:10]
    universe_exempt_set = _parse_ticker_csv(universe_exempt) | {"SPY"}
    eligibility_price: pd.DataFrame | None = None
    eligibility_price_ok: pd.DataFrame | None = None
    eligibility_history_ok: pd.DataFrame | None = None
    eligibility_valid_ok: pd.DataFrame | None = None
    eligibility_liquidity_price_ok: pd.DataFrame | None = None
    eligibility_liquidity_adv_ok: pd.DataFrame | None = None
    universe_membership: pd.DataFrame | None = None
    universe_membership_summary: pd.DataFrame | None = None
    universe_dataset_saved = False
    universe_dataset_save_paths: dict[str, str] = {}
    universe_dataset_path_used = str(universe_dataset_path).strip() if uds_mode == "build" else ""
    universe_dataset_membership_path = ""
    universe_dataset_summary_path = ""
    universe_dataset_fallback = False
    need_sector_map = bool(
        sector_cap > 0.0
        or use_sector_neutralization
        or sector_neutral
        or factor_neutralization_mode in {"sector", "beta_sector", "beta_sector_size"}
    )
    sector_by_ticker = _load_sector_map(sector_map, list(close.columns)) if need_sector_map else None
    market_cap_by_ticker = (
        _load_market_cap_map(sector_map, list(close.columns))
        if bool(use_size_neutralization or factor_neutralization_mode == "beta_sector_size")
        else None
    )

    if factor_names is not None and len(factor_names) > 0:
        resolved_factor_names = [str(x).strip() for x in factor_names if str(x).strip()]
    elif isinstance(factor_name, list):
        resolved_factor_names = [str(x).strip() for x in factor_name if str(x).strip()]
    else:
        resolved_factor_names = [str(factor_name).strip()]
    if not resolved_factor_names:
        raise ValueError("At least one factor must be provided.")

    raw_factor_params = dict(factor_params or {})
    if raw_factor_params and all(isinstance(v, dict) for v in raw_factor_params.values()):
        factor_params_map = {k: dict(v) for k, v in raw_factor_params.items()}
    elif raw_factor_params:
        if len(resolved_factor_names) == 1:
            factor_params_map = {resolved_factor_names[0]: raw_factor_params}
        else:
            factor_params_map = {name: dict(raw_factor_params) for name in resolved_factor_names}
    else:
        factor_params_map = {}
    effective_min_history_days = max(
        int(universe_min_history_days),
        _factor_required_history_days(
            factor_names=resolved_factor_names,
            factor_params_map=factor_params_map,
        ),
    )
    if u_mode == "dynamic":
        (
            eligibility_price_ok,
            eligibility_history_ok,
            eligibility_valid_ok,
            eligibility_liquidity_price_ok,
            eligibility_liquidity_adv_ok,
            eligibility_price,
        ) = compute_eligibility_components(
            close=close,
            min_history_days=effective_min_history_days,
            valid_lookback=universe_valid_lookback,
            min_valid_frac=universe_min_valid_frac,
            min_price=max(float(universe_min_price), float(min_price)),
            volume=volume,
            min_avg_dollar_volume=float(min_avg_dollar_volume),
            liquidity_lookback=int(liquidity_lookback),
        )
        assert set(eligibility_price.columns) == set(close.columns)
    if str(universe).strip().lower() == "liquid_us":
        eligibility_price = build_liquid_us_universe(
            prices=close,
            volumes=volume,
            min_price=max(float(universe_min_price), float(min_price)),
            min_avg_dollar_volume=float(min_avg_dollar_volume),
            adv_window=int(liquidity_lookback),
            min_history=int(effective_min_history_days),
        )
    if uds_mode in {"build", "use"}:
        if uds_mode == "build":
            dataset_dates = (
                pd.DatetimeIndex(close_full.index[rebalance_mask(close_full.index, rebalance)])
                if uds_freq == "rebalance"
                else None
            )
            universe_membership = build_point_in_time_universe(
                close=close_full,
                min_history_days=effective_min_history_days,
                valid_lookback=universe_valid_lookback,
                min_valid_frac=universe_min_valid_frac,
                min_price=max(float(universe_min_price), float(min_price)),
                exempt=universe_exempt_set,
                dates=dataset_dates,
            )
            universe_membership_summary = summarize_universe_membership(universe_membership)
        else:
            in_path = str(universe_dataset_path).strip() or str(_default_universe_dataset_path())
            try:
                loaded = load_universe_dataset(in_path)
                universe_membership = _align_universe_membership(
                    membership=loaded,
                    close=close_full,
                    exempt=universe_exempt_set,
                )
                universe_membership_summary = summarize_universe_membership(universe_membership)
                universe_dataset_path_used = in_path
            except FileNotFoundError:
                if bool(universe_dataset_require):
                    raise ValueError(f"Universe dataset required but missing: {in_path}")
                warnings.warn(
                    f"Universe dataset not found at {in_path}; falling back to runtime dynamic eligibility.",
                    RuntimeWarning,
                )
                universe_dataset_fallback = True
        if universe_membership is not None and u_mode == "dynamic":
            eligibility_price = _align_universe_membership(
                membership=universe_membership,
                close=close,
                exempt=universe_exempt_set,
            )
            assert set(eligibility_price.columns) == set(close.columns)

    if factor_weights is None:
        weights = np.repeat(1.0 / len(resolved_factor_names), len(resolved_factor_names))
    else:
        if len(factor_weights) != len(resolved_factor_names):
            raise ValueError(
                "factor_weights length must match number of factors: "
                f"{len(factor_weights)} vs {len(resolved_factor_names)}"
            )
        weights = np.asarray(factor_weights, dtype=float)
        if not np.isfinite(weights).all():
            raise ValueError("factor_weights must be finite numbers.")
        total_w = float(weights.sum())
        if abs(total_w) < 1e-12:
            raise ValueError("factor_weights sum must be non-zero.")
        weights = weights / total_w

    fundamentals_cache_key = (
        tuple(resolved_factor_names),
        str(fundamentals_path),
        int(fundamentals_fallback_lag_days),
        tuple(str(x) for x in close.index),
        tuple(str(x) for x in close.columns),
    )
    cached_factor_params = _cache_get(
        run_cache, "factor_params_augmented", fundamentals_cache_key, debug=cache_debug
    )
    if cached_factor_params is not None:
        cache_stats["fundamentals_hit"] = 1
        factor_params_map = cached_factor_params
    else:
        factor_params_map = _augment_factor_params_with_fundamentals(
            factor_names=resolved_factor_names,
            factor_params_map=factor_params_map,
            close=close,
            fundamentals_path=fundamentals_path,
            fundamentals_fallback_lag_days=int(fundamentals_fallback_lag_days),
        )
        _cache_put(run_cache, "factor_params_augmented", fundamentals_cache_key, factor_params_map)

    timing_sections["data_load_seconds"] = time.perf_counter() - t_data_start
    t_factor_start = time.perf_counter()
    factor_raw_cache_key = _factor_raw_cache_key(
        factor_names=resolved_factor_names,
        raw_factor_params=raw_factor_params,
        start=str(start),
        end=str(end),
        data_source=str(data_source),
        data_cache_dir=str(data_cache_dir),
        price_fill_mode=str(price_fill_mode),
        drop_bad_tickers=bool(drop_bad_tickers),
        universe_mode=str(universe_mode),
        max_tickers=int(max_tickers),
        close_columns=close.columns,
    )
    raw_scores = _cache_get(run_cache, "factor_raw_scores", factor_raw_cache_key, debug=cache_debug)
    if raw_scores is not None:
        cache_stats["factor_raw_hit"] = 1
    else:
        raw_scores = compute_factors(
            factor_names=resolved_factor_names,
            close=adj_close,
            factor_params=factor_params_map,
        )
        _cache_put(run_cache, "factor_raw_scores", factor_raw_cache_key, raw_scores)
    if bool(use_factor_normalization):
        norm_cache_key = (
            "factor_normalized",
            factor_raw_cache_key,
            bool(use_sector_neutralization),
            bool(use_size_neutralization),
            bool(orthogonalize_factors),
            tuple(sorted((sector_by_ticker or {}).items())) if sector_by_ticker is not None else (),
            tuple(sorted((market_cap_by_ticker or {}).items())) if market_cap_by_ticker is not None else (),
        )
        norm_scores = _cache_get(run_cache, "factor_norm_scores", norm_cache_key, debug=cache_debug)
        if norm_scores is not None:
            cache_stats["factor_norm_hit"] = 1
        else:
            base_scores = {
                name: robust_preprocess_base(raw_scores[name], winsor_p=0.05)
                for name in resolved_factor_names
            }
            if bool(use_sector_neutralization or use_size_neutralization):
                base_scores = {
                    name: neutralize_scores_cs(
                        base_scores[name],
                        sector_by_ticker=sector_by_ticker,
                        log_market_cap_by_ticker=market_cap_by_ticker,
                        use_sector_neutralization=use_sector_neutralization,
                        use_size_neutralization=use_size_neutralization,
                    )
                    for name in resolved_factor_names
                }
            norm_scores = {name: percentile_rank_cs(base_scores[name]) for name in resolved_factor_names}
            norm_scores = maybe_orthogonalize_factor_scores(
                factor_scores=norm_scores,
                enabled=bool(orthogonalize_factors),
                factor_order=resolved_factor_names,
            )
            _cache_put(run_cache, "factor_norm_scores", norm_cache_key, norm_scores)
    else:
        norm_scores = {
            name: normalize_scores(raw_scores[name], method=normalize, winsor_p=winsor_p)
            for name in resolved_factor_names
        }
        norm_scores = maybe_orthogonalize_factor_scores(
            factor_scores=norm_scores,
            enabled=bool(orthogonalize_factors),
            factor_order=resolved_factor_names,
        )
    timing_sections["factor_compute_seconds"] = time.perf_counter() - t_factor_start
    static_weights = {
        name: float(w) for name, w in zip(resolved_factor_names, np.asarray(weights, dtype=float))
    }
    sleeve_specs = (
        _resolve_multi_sleeve_config(
            config=multi_sleeve_config,
            available_factors=resolved_factor_names,
        )
        if portfolio_mode_normalized == "multi_sleeve"
        else []
    )
    require_all_factor_scores = u_mode == "dynamic"
    regime_pct_bull = float("nan")
    regime_pct_bear_or_volatile = float("nan")
    regime_last_label: str | None = None
    regime_label_used: pd.Series | None = None
    tv_weights_used: dict[str, pd.Series] | None = None
    use_dynamic_weights = should_apply_dynamic_factor_weights(
        regime_filter=bool(regime_filter),
        dynamic_factor_weights=bool(dynamic_factor_weights),
    )
    apply_dynamic_for_composite = bool(
        use_dynamic_weights and portfolio_mode_normalized == "composite"
    )
    bull_map = parse_weight_map(regime_bull_weights) if use_dynamic_weights else {}
    bear_map = parse_weight_map(regime_bear_weights) if use_dynamic_weights else {}
    if apply_dynamic_for_composite:
        spy_close = _get_benchmark_close_from_map(
            ohlcv_map=ohlcv_map,
            score_index=close.index,
            benchmark=regime_benchmark,
            source=data_source,
        )
        label = compute_regime_label(
            spy_close=spy_close,
            score_index=close.index,
            trend_sma=regime_trend_sma,
            vol_lookback=regime_vol_lookback,
            vol_median_lookback=regime_vol_median_lookback,
        )
        tv_weights = build_regime_weight_series(
            factor_names=resolved_factor_names,
            static_weights=static_weights,
            label=label,
            bull_weights=bull_map,
            bear_weights=bear_map,
        )
        scores = aggregate_factor_scores(
            norm_scores,
            tv_weights,
            method=factor_aggregation_method_normalized,
            require_all_factors=require_all_factor_scores,
        )
        if regime_filter:
            regime_label_used = label
            tv_weights_used = tv_weights
            valid = label.notna()
            if bool(valid.any()):
                regime_pct_bull = float((label[valid] == "bull").mean())
                regime_pct_bear_or_volatile = float((label[valid] == "bear_or_volatile").mean())
            else:
                regime_pct_bull = 0.0
                regime_pct_bear_or_volatile = 0.0
            non_na_labels = label.dropna()
            regime_last_label = str(non_na_labels.iloc[-1]) if not non_na_labels.empty else None
    else:
        scores = aggregate_factor_scores(
            norm_scores,
            static_weights,
            method=factor_aggregation_method_normalized,
            require_all_factors=require_all_factor_scores,
        )
    if portfolio_mode_normalized == "multi_sleeve" and bool(dynamic_factor_weights):
        warnings.warn(
            "dynamic_factor_weights is ignored when portfolio_mode='multi_sleeve'.",
            RuntimeWarning,
        )
    if historical_membership is not None:
        scores = apply_universe_filter_to_scores(scores, historical_membership, exempt=set())  # type: ignore[assignment]
    scores_pre_universe = scores.copy()
    if float(min_price) > 0.0:
        liquidity_price_ok = (close.reindex(index=scores.index, columns=scores.columns) >= float(min_price)).fillna(
            False
        )
    else:
        liquidity_price_ok = pd.DataFrame(True, index=scores.index, columns=scores.columns)
    if float(min_avg_dollar_volume) > 0.0:
        dv = (
            close.reindex(index=scores.index, columns=scores.columns)
            * volume.reindex(index=scores.index, columns=scores.columns)
        )
        adv = dv.rolling(int(liquidity_lookback), min_periods=int(liquidity_lookback)).mean()
        liquidity_adv_ok = (adv >= float(min_avg_dollar_volume)).fillna(False)
    else:
        liquidity_adv_ok = pd.DataFrame(True, index=scores.index, columns=scores.columns)
    liquidity_elig = (liquidity_price_ok & liquidity_adv_ok).astype(bool)
    liquidity_filter_enabled = bool(float(min_price) > 0.0 or float(min_avg_dollar_volume) > 0.0)
    if eligibility_price is not None:
        base_elig = _resolve_universe_eligibility(
            eligibility_price=eligibility_price,
            scores=scores_pre_universe,
            source=universe_eligibility_source,
        )
        if historical_membership is not None:
            hist_aligned = (
                historical_membership.reindex(index=scores.index, columns=scores.columns)
                .fillna(False)
                .astype(bool)
            )
            elig_scores = (base_elig & hist_aligned).astype(bool)
        else:
            elig_scores = base_elig
        if liquidity_filter_enabled:
            elig_scores = (elig_scores & liquidity_elig).astype(bool)
        scores = apply_universe_filter_to_scores(scores, elig_scores, exempt=universe_exempt_set)  # type: ignore[assignment]
    elif historical_membership is not None:
        elig_scores = (
            historical_membership.reindex(index=scores.index, columns=scores.columns)
            .fillna(False)
            .astype(bool)
        )
        if liquidity_filter_enabled:
            elig_scores = (elig_scores & liquidity_elig).astype(bool)
        scores = apply_universe_filter_to_scores(scores, elig_scores, exempt=set())  # type: ignore[assignment]
    else:
        if liquidity_filter_enabled:
            elig_scores = liquidity_elig
            scores = apply_universe_filter_to_scores(scores, elig_scores, exempt=universe_exempt_set)  # type: ignore[assignment]
        else:
            elig_scores = None
    if factor_neutralization_mode is not None:
        use_beta = factor_neutralization_mode in {"beta", "beta_sector", "beta_sector_size"}
        use_sector = factor_neutralization_mode in {"sector", "beta_sector", "beta_sector_size"}
        use_size = factor_neutralization_mode in {"beta_sector_size"}
        beta_panel: pd.DataFrame | None = None
        log_mc_panel: pd.DataFrame | None = None
        if use_beta:
            spy_close = _get_benchmark_close_from_map(
                ohlcv_map=ohlcv_map,
                score_index=close.index,
                benchmark="SPY",
                source=data_source,
            )
            beta_panel = _compute_beta_exposure(
                close=close.reindex(index=scores.index, columns=scores.columns),
                spy_close=spy_close.reindex(scores.index),
                lookback=60,
            )
        if use_size:
            log_mc_panel = _extract_log_market_cap_exposure(
                close=close.reindex(index=scores.index, columns=scores.columns),
                factor_params_map=factor_params_map,
            )
        scores = neutralize_scores_cs(
            scores=scores,
            sector_by_ticker=sector_by_ticker if use_sector else None,
            log_market_cap_by_ticker=market_cap_by_ticker if use_size else None,
            log_market_cap_exposure=log_mc_panel if use_size else None,
            beta_exposure=beta_panel if use_beta else None,
            use_beta_neutralization=bool(use_beta),
            use_sector_neutralization=bool(use_sector),
            use_size_neutralization=bool(use_size),
        )

    factor_names_str = ";".join(resolved_factor_names)
    factor_weights_str = ";".join(f"{float(w):.10g}" for w in weights)
    if len(resolved_factor_names) == 1:
        factor_name_legacy = resolved_factor_names[0]
        factor_params_legacy = json.dumps(
            factor_params_map.get(factor_name_legacy, {}),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    else:
        factor_name_legacy = "MULTI"
        factor_params_legacy = "{}"

    rb_mask = rebalance_mask(scores.index, rebalance)
    rb_dates = pd.DatetimeIndex(scores.index[rb_mask])
    source_counts_rb = close.reindex(rb_dates).notna().sum(axis=1).astype(int)
    membership_counts_rb = (
        historical_membership.reindex(rb_dates).fillna(False).sum(axis=1).astype(int)
        if historical_membership is not None
        else source_counts_rb
    )
    scores_for_weights = scores
    if elig_scores is not None:
        elig_rb_counts = elig_scores.loc[rb_dates].sum(axis=1).astype(int)
        if len(elig_rb_counts) > len(rb_dates):
            raise ValueError(
                f"Backtest dynamic universe count mismatch: counts={len(elig_rb_counts)} rebalances={len(rb_dates)}"
            )
        eligible_median = float(elig_rb_counts.median()) if not elig_rb_counts.empty else 0.0
        eligible_min = int(elig_rb_counts.min()) if not elig_rb_counts.empty else 0
        eligible_max = int(elig_rb_counts.max()) if not elig_rb_counts.empty else 0
        low_elig_count = int((elig_rb_counts < int(universe_min_tickers)).sum())
        zero_elig_count = int((elig_rb_counts == 0).sum())
        if zero_elig_count > len(rb_dates) or low_elig_count > len(rb_dates):
            raise ValueError(
                "Backtest dynamic universe sanity failed: "
                f"zero={zero_elig_count} under_min={low_elig_count} rebalances={len(rb_dates)}"
            )
        if low_elig_count > 0:
            print(
                f"UNIVERSE DYNAMIC: rebalances with eligible<{int(universe_min_tickers)} = {low_elig_count}"
            )
    else:
        elig_rb_counts = _rebalance_score_counts(scores, rb_mask)
        eligible_median = float(elig_rb_counts.median()) if not elig_rb_counts.empty else float("nan")
        eligible_min = int(elig_rb_counts.min()) if not elig_rb_counts.empty else 0
        eligible_max = int(elig_rb_counts.max()) if not elig_rb_counts.empty else 0
    if elig_scores is not None:
        scores_for_weights, rebalance_skip_zero_count, rebalance_skip_below_min_count = (
            _apply_universe_rebalance_skip(
                scores=scores,
                rb_mask=rb_mask,
                universe_min_tickers=universe_min_tickers,
                universe_skip_below_min_tickers=universe_skip_below_min_tickers,
            )
        )
        rebalance_skipped_count = rebalance_skip_zero_count + rebalance_skip_below_min_count
    else:
        rebalance_skip_zero_count = 0
        rebalance_skip_below_min_count = 0
        rebalance_skipped_count = 0
    rb_scores = scores_for_weights.loc[rb_mask]
    eligibility_counts_rb = (
        elig_scores.loc[rb_dates].sum(axis=1).astype(int)
        if elig_scores is not None
        else membership_counts_rb
    )
    final_tradable_counts_rb = scores.loc[rb_mask].notna().sum(axis=1).astype(int)
    membership_median, membership_min, membership_max = _stats_min_med_max(membership_counts_rb)
    source_median, source_min, source_max = _stats_min_med_max(source_counts_rb)
    eligibility_median, eligibility_min_r, eligibility_max_r = _stats_min_med_max(eligibility_counts_rb)
    final_median, final_min, final_max = _stats_min_med_max(final_tradable_counts_rb)
    liquidity_counts_rb = liquidity_elig.loc[rb_dates].sum(axis=1).astype(int) if not rb_dates.empty else pd.Series(dtype=int)
    liquidity_eligible_median = float(liquidity_counts_rb.median()) if not liquidity_counts_rb.empty else float("nan")
    liquidity_eligible_min = int(liquidity_counts_rb.min()) if not liquidity_counts_rb.empty else 0
    liquidity_eligible_max = int(liquidity_counts_rb.max()) if not liquidity_counts_rb.empty else 0
    if elig_scores is not None and not rb_dates.empty:
        pre_liq = (
            _resolve_universe_eligibility(
                eligibility_price=eligibility_price,
                scores=scores_pre_universe,
                source=universe_eligibility_source,
            ).loc[rb_dates]
            if eligibility_price is not None
            else (
                historical_membership.reindex(index=rb_dates, columns=scores.columns).fillna(False).astype(bool)
                if historical_membership is not None
                else pd.DataFrame(True, index=rb_dates, columns=scores.columns)
            )
        )
        filtered_out_rb = (pre_liq.sum(axis=1) - elig_scores.loc[rb_dates].sum(axis=1)).clip(lower=0).astype(int)
    else:
        filtered_out_rb = pd.Series(dtype=int)
    liquidity_filtered_out_median = (
        float(filtered_out_rb.median()) if not filtered_out_rb.empty else float("nan")
    )
    liquidity_filtered_out_max = int(filtered_out_rb.max()) if not filtered_out_rb.empty else 0

    finite_counts = np.isfinite(rb_scores.to_numpy()).sum(axis=1) if not rb_scores.empty else np.array([])
    num_rebalance_with_signal = int((finite_counts > 0).sum()) if finite_counts.size > 0 else 0
    if num_rebalance_with_signal == 0:
        raise ValueError(
            "No finite scores on any rebalance date; check factor lookback vs date range"
        )

    debug_sample_tickers = [t for t in used_tickers if t in close.columns][:5]
    ticker_debug_rows: list[dict] = []
    for ticker in debug_sample_tickers:
        s = close[ticker].dropna()
        if s.empty:
            continue
        if elig_scores is not None and ticker in elig_scores.columns:
            survives = bool(elig_scores[ticker].fillna(False).any())
        else:
            survives = bool(scores[ticker].notna().any()) if ticker in scores.columns else False
        r = s.pct_change()
        ticker_debug_rows.append(
            {
                "ticker": ticker,
                "first_dates": ";".join(str(d.date()) for d in s.index[:5]),
                "last_dates": ";".join(str(d.date()) for d in s.index[-5:]),
                "first_closes": ";".join(f"{float(v):.10g}" for v in s.iloc[:5].to_numpy(dtype=float)),
                "last_closes": ";".join(f"{float(v):.10g}" for v in s.iloc[-5:].to_numpy(dtype=float)),
                "max_abs_daily_return": float(r.abs().max(skipna=True)) if r.notna().any() else float("nan"),
                "survives_eligibility": bool(survives),
            }
        )

    t_portfolio_start = time.perf_counter()
    sleeve_weights_output: dict[str, pd.DataFrame] | None = None
    if portfolio_mode_normalized == "multi_sleeve":
        sleeve_scores: dict[str, pd.DataFrame] = {}
        sleeve_allocations: dict[str, float] = {}
        sleeve_top_n: dict[str, int] = {}
        for sleeve in sleeve_specs:
            sleeve_name = str(sleeve["name"])
            factors = [str(x) for x in sleeve["factors"]]
            factor_weights_cs = {
                f: float(w) for f, w in zip(factors, np.asarray(sleeve["factor_weights"], dtype=float))
            }
            sleeve_score = aggregate_factor_scores(
                {f: norm_scores[f] for f in factors},
                factor_weights_cs,
                method=factor_aggregation_method_normalized,
                require_all_factors=require_all_factor_scores,
            )
            if elig_scores is not None:
                sleeve_score = apply_universe_filter_to_scores(sleeve_score, elig_scores, exempt=set())
            sleeve_scores[sleeve_name] = sleeve_score
            sleeve_allocations[sleeve_name] = float(sleeve["allocation"])
            sleeve_top_n[sleeve_name] = int(sleeve["top_n"])
        weights, sleeve_weights_output = build_multi_sleeve_weights(
            sleeve_scores=sleeve_scores,
            sleeve_allocations=sleeve_allocations,
            sleeve_top_n=sleeve_top_n,
            close=close,
            rebalance=rebalance,
            weighting=weighting,
            vol_lookback=vol_lookback,
            max_weight=max_weight,
            score_clip=score_clip,
            score_floor=score_floor,
            sector_cap=sector_cap,
            sector_by_ticker=sector_by_ticker,
            sector_neutral=bool(sector_neutral),
            rank_buffer=int(rank_buffer),
            volatility_scaled_weights=bool(volatility_scaled_weights),
        )
    else:
        weights = build_topn_weights(
            scores=scores_for_weights,
            close=close,
            top_n=top_n,
            rank_buffer=int(rank_buffer),
            volatility_scaled_weights=bool(volatility_scaled_weights),
            rebalance=rebalance,
            weighting=weighting,
            vol_lookback=vol_lookback,
            max_weight=max_weight,
            score_clip=score_clip,
            score_floor=score_floor,
            sector_cap=sector_cap,
            sector_by_ticker=sector_by_ticker,
            sector_neutral=bool(sector_neutral),
        )
    if trend_filter:
        market_close = _get_benchmark_close_from_map(
            ohlcv_map=ohlcv_map,
            score_index=close.index,
            benchmark="SPY",
            source=data_source,
        )
        weights = apply_trend_filter(
            weights=weights,
            market_close=market_close,
            sma_window=trend_sma_window,
        )

    sim = simulate_portfolio(
        # Use adjusted prices for PnL so simulation stays aligned with factor inputs
        # and avoids split-related distortions in the active research runtime.
        close=adj_close,
        weights=weights,
        costs_bps=costs_bps,
        slippage_bps=slippage_bps,
        slippage_vol_mult=slippage_vol_mult,
        slippage_vol_lookback=slippage_vol_lookback,
        rebalance_dates=rb_dates,
        execution_delay_days=int(execution_delay_days),
    )
    sim, leverage, raw_vol, final_realized_vol = _apply_vol_target(
        sim=sim,
        target_vol=target_vol,
        lookback=port_vol_lookback,
        max_leverage=max_leverage,
    )
    sim, bear_scale_series, bear_scale_avg_bear = _apply_bear_exposure_overlay(
        sim=sim,
        regime_label=regime_label_used,
        regime_filter=bool(regime_filter),
        bear_exposure_scale=float(bear_exposure_scale),
    )
    metrics = compute_metrics(sim["DailyReturn"])
    sanity_warnings = _print_portfolio_sanity_warnings(
        close=close,
        weights=weights,
        sim=sim,
        cagr=float(metrics.get("CAGR", np.nan)),
        vol=float(metrics.get("Vol", np.nan)),
    )
    invested_mask = weights.abs().sum(axis=1) > 0
    if invested_mask.any():
        eff_holdings = 1.0 / (weights.loc[invested_mask].pow(2).sum(axis=1).replace(0.0, np.nan))
        effective_holdings_avg = float(eff_holdings.mean())
    else:
        effective_holdings_avg = float("nan")
    leverage_avg = float(leverage.mean())
    leverage_max = float(leverage.max())
    effective_cost_bps_avg = (
        float(sim["EffectiveCostBps"].mean()) if "EffectiveCostBps" in sim.columns else float(costs_bps)
    )
    timing_sections["portfolio_backtest_seconds"] = time.perf_counter() - t_portfolio_start

    t_report_start = time.perf_counter()
    run_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    outdir = Path("results") / run_tag
    outdir.mkdir(parents=True, exist_ok=True)
    if uds_mode == "build" and universe_membership is not None and bool(universe_dataset_save):
        build_target = str(outdir)
        if universe_membership_summary is None:
            universe_membership_summary = summarize_universe_membership(universe_membership)
        universe_dataset_save_paths = save_universe_dataset(
            membership=universe_membership,
            summary=universe_membership_summary,
            outdir_or_path=build_target,
        )
        universe_dataset_saved = True
        universe_dataset_path_used = universe_dataset_save_paths.get("membership_path", "")
        universe_dataset_membership_path = universe_dataset_save_paths.get("membership_path", "")
        universe_dataset_summary_path = universe_dataset_save_paths.get("summary_path", "")
    if save_artifacts and uds_mode in {"build", "use"} and universe_membership is not None:
        if universe_membership_summary is None:
            universe_membership_summary = summarize_universe_membership(universe_membership)
        out_paths = save_universe_dataset(
            membership=universe_membership,
            summary=universe_membership_summary,
            outdir_or_path=str(outdir),
        )
        universe_dataset_save_paths = out_paths
        universe_dataset_saved = True
        if not universe_dataset_path_used:
            universe_dataset_path_used = out_paths.get("membership_path", "")
        if not universe_dataset_membership_path:
            universe_dataset_membership_path = out_paths.get("membership_path", "")
        if not universe_dataset_summary_path:
            universe_dataset_summary_path = out_paths.get("summary_path", "")
    universe_dataset_start_date = (
        str(pd.Timestamp(universe_membership.index.min()).date())
        if universe_membership is not None and not universe_membership.empty
        else ""
    )
    universe_dataset_end_date = (
        str(pd.Timestamp(universe_membership.index.max()).date())
        if universe_membership is not None and not universe_membership.empty
        else ""
    )
    universe_dataset_rows = int(len(universe_membership)) if universe_membership is not None else 0

    run_config = {
        "mode": "backtest",
        "GitCommit": _git_commit(),
        "start": start,
        "end": end,
        "max_tickers": max_tickers,
        "requested_tickers_count": requested_tickers_count,
        "requested_tickers_sample": tickers[:10],
        "fetch_request_tickers_count": int(len(fetch_request_tickers)),
        "fetch_request_tickers_sample": fetch_request_tickers[:10],
        "fetched_keys_count": fetched_keys_count,
        "fetched_keys_sample": fetched_keys_sample,
        "post_fetch_filtered_keys_count": post_fetch_filtered_keys_count,
        "post_fetch_filtered_keys_sample": post_fetch_filtered_keys_sample,
        "loaded_tickers_count": loaded_tickers_count,
        "loaded_tickers_sample": used_tickers[:10],
        "missing_tickers_count": missing_tickers_count,
        "missing_tickers_sample": missing_tickers_sample,
        "rejected_tickers_count": rejected_tickers_count,
        "rejected_tickers_sample": rejected_tickers_sample,
        "ohlcv_map_key_sample": ohlcv_map_key_sample,
        "historical_membership_path": str(historical_membership_path),
        "historical_membership_source": str(historical_membership_source),
        "membership_covered_tickers_count": membership_covered_tickers_count,
        "membership_covered_tickers_sample": membership_covered_tickers_sample,
        "membership_unique_tickers": int(membership_alignment_diag.get("MembershipUniqueTickers", 0)),
        "membership_price_overlap_normalized": int(
            membership_alignment_diag.get("MembershipPriceOverlapNormalized", 0)
        ),
        "membership_unmatched_normalized": int(
            membership_alignment_diag.get("MembershipUnmatchedNormalized", 0)
        ),
        "membership_unmatched_sample": membership_alignment_diag.get("MembershipUnmatchedSample", []),
        "membership_aligned_per_date_min": int(
            membership_alignment_diag.get("MembershipAlignedPerDateMin", 0)
        ),
        "membership_aligned_per_date_median": float(
            membership_alignment_diag.get("MembershipAlignedPerDateMedian", 0.0)
        ),
        "membership_aligned_per_date_max": int(
            membership_alignment_diag.get("MembershipAlignedPerDateMax", 0)
        ),
        "selected_tickers": used_tickers,
        "top_n": top_n,
        "rebalance": rebalance,
        "costs_bps": costs_bps,
        "portfolio_mode": portfolio_mode_normalized,
        "multi_sleeve_config": sleeve_specs,
        "FactorName": factor_name_legacy,
        "FactorParams": factor_params_legacy,
        "FactorNames": factor_names_str,
        "FactorWeights": factor_weights_str,
        "FactorAggregationMethod": factor_aggregation_method_normalized,
        "Normalize": normalize,
        "WinsorP": float(winsor_p),
        "UseFactorNormalization": bool(use_factor_normalization),
        "UseSectorNeutralization": bool(use_sector_neutralization),
        "UseSizeNeutralization": bool(use_size_neutralization),
        "FactorNeutralization": factor_neutralization_mode or "none",
        "DynamicFactorWeights": bool(dynamic_factor_weights),
        "OrthogonalizeFactors": bool(orthogonalize_factors),
        "SectorNeutral": bool(sector_neutral),
        "Weighting": weighting,
        "VolLookback": int(vol_lookback),
        "MaxWeight": float(max_weight),
        "ScoreClip": float(score_clip),
        "ScoreFloor": float(score_floor),
        "SectorCap": float(sector_cap),
        "SectorMap": sector_map,
        "TargetVol": float(target_vol),
        "PortVolLookback": int(port_vol_lookback),
        "MaxLeverage": float(max_leverage),
        "VolTargetingEnabled": bool(target_vol > 0.0),
        "RawVol": raw_vol,
        "FinalRealizedVol": final_realized_vol,
        "SlippageBps": float(slippage_bps),
        "SlippageVolMult": float(slippage_vol_mult),
        "SlippageVolLookback": int(slippage_vol_lookback),
        "ExecutionDelayDays": int(execution_delay_days),
        "RegimeFilter": bool(regime_filter),
        "RegimeBenchmark": str(regime_benchmark),
        "RegimeTrendSMA": int(regime_trend_sma),
        "RegimeVolLookback": int(regime_vol_lookback),
        "RegimeVolMedianLookback": int(regime_vol_median_lookback),
        "RegimeBullWeights": weight_map_to_json(bull_map),
        "RegimeBearWeights": weight_map_to_json(bear_map),
        "BearExposureScale": float(bear_exposure_scale),
        "BearExposureScaleAvgBearPeriods": float(bear_scale_avg_bear),
        "LeverageAvg": leverage_avg,
        "LeverageMax": leverage_max,
        "trend_filter": bool(trend_filter),
        "trend_sma_window": int(trend_sma_window),
        "DataSource": str(data_source),
        "DataCacheDir": str(data_cache_dir),
        "DataRoot": str(data_cache_dir),
        "DataRefresh": bool(data_refresh),
        "DataBulkPrepare": bool(data_bulk_prepare),
        "DataCachedFilesUsed": int(data_source_summary.get("cached_files_used", 0)),
        "DataFetchedOrRefreshed": int(data_source_summary.get("fetched_or_refreshed", 0)),
        "ParquetRawUniqueTickers": int(data_source_summary.get("parquet_raw_unique_tickers", 0)),
        "ParquetNormalizedUniqueTickers": int(
            data_source_summary.get("parquet_normalized_unique_tickers", 0)
        ),
        "ParquetNormCollisionCount": int(data_source_summary.get("parquet_norm_collision_count", 0)),
        "SaveArtifacts": bool(save_artifacts),
        "PriceQualityCheck": bool(price_quality_check),
        "PriceQualityMode": str(price_quality_mode),
        "PriceQualityZeroRetThresh": float(price_quality_zero_ret_thresh),
        "PriceQualityMinValidFrac": float(price_quality_min_valid_frac),
        "PriceQualityMaxBadTickers": int(price_quality_max_bad_tickers),
        "PriceQualityReportTopK": int(price_quality_report_topk),
        "PriceFillMode": str(price_fill_mode),
        "DropBadTickers": bool(drop_bad_tickers),
        "DropBadTickersScope": str(drop_bad_tickers_scope),
        "DropBadTickersMaxDrop": int(drop_bad_tickers_max_drop),
        "DropBadTickersExempt": str(drop_bad_tickers_exempt),
        "DroppedTickersCount": int(dropped_tickers_count),
        "DroppedBrokenTickersCount": int(len(dropped_broken_tickers)),
        "DroppedBrokenTickersSample": dropped_broken_tickers[:10],
        "UniverseMode": str(universe_mode),
        "UniverseMinHistoryDays": int(universe_min_history_days),
        "UniverseEffectiveMinHistoryDays": int(effective_min_history_days),
        "UniverseMinValidFrac": float(universe_min_valid_frac),
        "UniverseValidLookback": int(universe_valid_lookback),
        "UniverseMinPrice": float(universe_min_price),
        "MinPrice": float(min_price),
        "MinAvgDollarVolume": float(min_avg_dollar_volume),
        "LiquidityLookback": int(liquidity_lookback),
        "UniverseMinTickers": int(universe_min_tickers),
        "UniverseSkipBelowMinTickers": bool(universe_skip_below_min_tickers),
        "UniverseEligibilitySource": str(universe_eligibility_source),
        "UniverseExempt": str(universe_exempt),
        "UniverseDatasetMode": str(universe_dataset_mode),
        "UniverseDatasetFreq": str(universe_dataset_freq),
        "UniverseDatasetPath": str(universe_dataset_path_used),
        "UniverseDatasetMembershipPath": str(universe_dataset_membership_path),
        "UniverseDatasetSummaryPath": str(universe_dataset_summary_path),
        "UniverseDatasetSaved": bool(universe_dataset_saved),
        "UniverseDatasetFallback": bool(universe_dataset_fallback),
        "UniverseDatasetStartDate": universe_dataset_start_date,
        "UniverseDatasetEndDate": universe_dataset_end_date,
        "UniverseDatasetRows": universe_dataset_rows,
        "LiquidityEligibleMedian": liquidity_eligible_median,
        "LiquidityEligibleMin": liquidity_eligible_min,
        "LiquidityEligibleMax": liquidity_eligible_max,
        "LiquidityFilteredOutMedian": liquidity_filtered_out_median,
        "LiquidityFilteredOutMax": liquidity_filtered_out_max,
        "CacheFetchHit": int(cache_stats["fetch_hit"]),
        "CacheFactorRawHit": int(cache_stats["factor_raw_hit"]),
        "CacheFactorNormHit": int(cache_stats["factor_norm_hit"]),
        "CacheFundamentalsHit": int(cache_stats["fundamentals_hit"]),
        "TimingDataLoadSeconds": float(timing_sections["data_load_seconds"]),
        "TimingFactorComputeSeconds": float(timing_sections["factor_compute_seconds"]),
        "TimingPortfolioBacktestSeconds": float(timing_sections["portfolio_backtest_seconds"]),
        "TimingReportWriteSeconds": float(timing_sections["report_write_seconds"]),
    }
    summary = {
        "RunTag": run_tag,
        "Mode": "backtest",
        "Start": start,
        "End": end,
        "TickersUsed": len(used_tickers),
        "RequestedTickersCount": requested_tickers_count,
        "RequestedTickersSample": tickers[:10],
        "FetchRequestTickersCount": int(len(fetch_request_tickers)),
        "FetchRequestTickerSample": fetch_request_tickers[:10],
        "FetchedKeysCount": fetched_keys_count,
        "FetchedKeysSample": fetched_keys_sample,
        "PostFetchFilteredKeysCount": post_fetch_filtered_keys_count,
        "PostFetchFilteredKeysSample": post_fetch_filtered_keys_sample,
        "LoadedTickersCount": loaded_tickers_count,
        "LoadedTickersSample": used_tickers[:10],
        "MissingTickersCount": missing_tickers_count,
        "MissingTickersSample": missing_tickers_sample,
        "OHLCVMapKeySample": ohlcv_map_key_sample,
        "RejectedTickersCount": rejected_tickers_count,
        "RejectedTickersSample": rejected_tickers_sample,
        "HistoricalMembershipPath": str(historical_membership_path),
        "HistoricalMembershipSource": str(historical_membership_source),
        "MembershipCoveredTickersCount": membership_covered_tickers_count,
        "MembershipCoveredTickersSample": membership_covered_tickers_sample,
        "MembershipUniqueTickers": int(membership_alignment_diag.get("MembershipUniqueTickers", 0)),
        "MembershipPriceOverlapNormalized": int(
            membership_alignment_diag.get("MembershipPriceOverlapNormalized", 0)
        ),
        "MembershipUnmatchedNormalized": int(
            membership_alignment_diag.get("MembershipUnmatchedNormalized", 0)
        ),
        "MembershipUnmatchedSample": membership_alignment_diag.get("MembershipUnmatchedSample", []),
        "MembershipAlignedPerDateMin": int(
            membership_alignment_diag.get("MembershipAlignedPerDateMin", 0)
        ),
        "MembershipAlignedPerDateMedian": float(
            membership_alignment_diag.get("MembershipAlignedPerDateMedian", 0.0)
        ),
        "MembershipAlignedPerDateMax": int(
            membership_alignment_diag.get("MembershipAlignedPerDateMax", 0)
        ),
        "SourceAvailableOnRebalanceMedian": source_median,
        "SourceAvailableOnRebalanceMin": source_min,
        "SourceAvailableOnRebalanceMax": source_max,
        "MembershipOnRebalanceMedian": membership_median,
        "MembershipOnRebalanceMin": membership_min,
        "MembershipOnRebalanceMax": membership_max,
        "EligibilityFilteredOnRebalanceMedian": eligibility_median,
        "EligibilityFilteredOnRebalanceMin": eligibility_min_r,
        "EligibilityFilteredOnRebalanceMax": eligibility_max_r,
        "FinalTradableOnRebalanceMedian": final_median,
        "FinalTradableOnRebalanceMin": final_min,
        "FinalTradableOnRebalanceMax": final_max,
        "TopN": top_n,
        "PortfolioMode": portfolio_mode_normalized,
        "MultiSleeveConfig": json.dumps(sleeve_specs, separators=(",", ":")),
        "RankBuffer": int(rank_buffer),
        "VolatilityScaledWeights": bool(volatility_scaled_weights),
        "Rebalance": rebalance,
        "CostsBps": costs_bps,
        "FactorName": factor_name_legacy,
        "FactorParams": factor_params_legacy,
        "FactorNames": factor_names_str,
        "FactorWeights": factor_weights_str,
        "FactorAggregationMethod": factor_aggregation_method_normalized,
        "Normalize": normalize,
        "WinsorP": float(winsor_p),
        "UseFactorNormalization": bool(use_factor_normalization),
        "UseSectorNeutralization": bool(use_sector_neutralization),
        "UseSizeNeutralization": bool(use_size_neutralization),
        "FactorNeutralization": factor_neutralization_mode or "none",
        "DynamicFactorWeights": bool(dynamic_factor_weights),
        "OrthogonalizeFactors": bool(orthogonalize_factors),
        "SectorNeutral": bool(sector_neutral),
        "Weighting": weighting,
        "VolLookback": int(vol_lookback),
        "MaxWeight": float(max_weight),
        "ScoreClip": float(score_clip),
        "ScoreFloor": float(score_floor),
        "SectorCap": float(sector_cap),
        "SectorMap": sector_map,
        "TargetVol": float(target_vol),
        "PortVolLookback": int(port_vol_lookback),
        "MaxLeverage": float(max_leverage),
        "VolTargetingEnabled": bool(target_vol > 0.0),
        "RawVol": raw_vol,
        "FinalRealizedVol": final_realized_vol,
        "SlippageBps": float(slippage_bps),
        "SlippageVolMult": float(slippage_vol_mult),
        "SlippageVolLookback": int(slippage_vol_lookback),
        "ExecutionDelayDays": int(execution_delay_days),
        "RegimeFilter": bool(regime_filter),
        "RegimeBenchmark": str(regime_benchmark),
        "RegimeTrendSMA": int(regime_trend_sma),
        "RegimeVolLookback": int(regime_vol_lookback),
        "RegimeVolMedianLookback": int(regime_vol_median_lookback),
        "RegimeBullWeights": weight_map_to_json(bull_map),
        "RegimeBearWeights": weight_map_to_json(bear_map),
        "BearExposureScale": float(bear_exposure_scale),
        "BearExposureScaleAvgBearPeriods": float(bear_scale_avg_bear),
        "regime_pct_bull": regime_pct_bull,
        "regime_pct_bear_or_volatile": regime_pct_bear_or_volatile,
        "regime_last_label": regime_last_label,
        "EffectiveHoldingsAvg": effective_holdings_avg,
        "LeverageAvg": leverage_avg,
        "LeverageMax": leverage_max,
        "EffectiveCostBpsAvg": effective_cost_bps_avg,
        "TrendFilter": bool(trend_filter),
        "TrendSMA": int(trend_sma_window),
        "DataSource": str(data_source),
        "DataCacheDir": str(data_cache_dir),
        "DataRoot": str(data_cache_dir),
        "DataRefresh": bool(data_refresh),
        "DataBulkPrepare": bool(data_bulk_prepare),
        "DataCachedFilesUsed": int(data_source_summary.get("cached_files_used", 0)),
        "DataFetchedOrRefreshed": int(data_source_summary.get("fetched_or_refreshed", 0)),
        "ParquetRawUniqueTickers": int(data_source_summary.get("parquet_raw_unique_tickers", 0)),
        "ParquetNormalizedUniqueTickers": int(
            data_source_summary.get("parquet_normalized_unique_tickers", 0)
        ),
        "ParquetNormCollisionCount": int(data_source_summary.get("parquet_norm_collision_count", 0)),
        "SaveArtifacts": bool(save_artifacts),
        "PriceQualityCheck": bool(price_quality_check),
        "PriceQualityMode": str(price_quality_mode),
        "PriceQualityBadTickers": int(quality_summary.get("num_bad", 0)),
        "PriceFillMode": str(price_fill_mode),
        "DropBadTickers": bool(drop_bad_tickers),
        "DropBadTickersScope": str(drop_bad_tickers_scope),
        "DroppedTickersCount": int(dropped_tickers_count),
        "DroppedBrokenTickersCount": int(len(dropped_broken_tickers)),
        "DroppedBrokenTickersSample": dropped_broken_tickers[:10],
        "UniverseMode": str(universe_mode),
        "UniverseEligibilitySource": str(universe_eligibility_source),
        "UniverseEffectiveMinHistoryDays": int(effective_min_history_days),
        "UniverseDatasetMode": str(universe_dataset_mode),
        "UniverseDatasetFreq": str(universe_dataset_freq),
        "UniverseDatasetPath": str(universe_dataset_path_used),
        "UniverseDatasetMembershipPath": str(universe_dataset_membership_path),
        "UniverseDatasetSummaryPath": str(universe_dataset_summary_path),
        "UniverseDatasetSaved": bool(universe_dataset_saved),
        "UniverseDatasetFallback": bool(universe_dataset_fallback),
        "UniverseDatasetStartDate": universe_dataset_start_date,
        "UniverseDatasetEndDate": universe_dataset_end_date,
        "UniverseDatasetRows": universe_dataset_rows,
        "MinPrice": float(min_price),
        "MinAvgDollarVolume": float(min_avg_dollar_volume),
        "LiquidityLookback": int(liquidity_lookback),
        "LiquidityEligibleMedian": liquidity_eligible_median,
        "LiquidityEligibleMin": liquidity_eligible_min,
        "LiquidityEligibleMax": liquidity_eligible_max,
        "LiquidityFilteredOutMedian": liquidity_filtered_out_median,
        "LiquidityFilteredOutMax": liquidity_filtered_out_max,
        "CacheFetchHit": int(cache_stats["fetch_hit"]),
        "CacheFactorRawHit": int(cache_stats["factor_raw_hit"]),
        "CacheFactorNormHit": int(cache_stats["factor_norm_hit"]),
        "CacheFundamentalsHit": int(cache_stats["fundamentals_hit"]),
        "TimingDataLoadSeconds": float(timing_sections["data_load_seconds"]),
        "TimingFactorComputeSeconds": float(timing_sections["factor_compute_seconds"]),
        "TimingPortfolioBacktestSeconds": float(timing_sections["portfolio_backtest_seconds"]),
        "TimingReportWriteSeconds": float(timing_sections["report_write_seconds"]),
        "EligibleTickersMedian": eligible_median,
        "EligibleTickersMin": eligible_min,
        "EligibleTickersMax": eligible_max,
        "EligibleOnRebalanceMedian": eligible_median,
        "EligibleOnRebalanceMin": eligible_min,
        "EligibleOnRebalanceMax": eligible_max,
        "RebalanceSkippedCount": rebalance_skipped_count,
        "RebalanceSkipBelowMinCount": rebalance_skip_below_min_count,
        "RebalanceSkipZeroEligibleCount": rebalance_skip_zero_count,
        "FilteringStages": (
            f"requested={requested_tickers_count} "
            f"loaded={loaded_tickers_count} "
            f"membership_covered={membership_covered_tickers_count} "
            f"eligible_med={eligible_median} "
            f"final_tradable_med={final_median}"
        ),
        "SanityWarningsCount": int(len(sanity_warnings)),
        "SanityWarningsSample": sanity_warnings[:10],
        "Outdir": str(outdir),
        **metrics,
    }

    _write_summary_artifacts(outdir=outdir, run_config=run_config, summary=summary)
    price_health_report.to_csv(outdir / "price_panel_health.csv", index=False, float_format="%.10g")
    pd.DataFrame(ticker_debug_rows).to_csv(outdir / "ticker_debug_report.csv", index=False)
    if historical_membership is not None:
        membership_counts_daily = historical_membership.sum(axis=1).astype(int)
        pd.DataFrame(
            {
                "date": membership_counts_daily.index.astype(str),
                "membership_count": membership_counts_daily.to_numpy(dtype=int),
            }
        ).to_csv(outdir / "membership_alignment_counts.csv", index=False)
    if save_artifacts:
        write_equity_curve(
            outdir=outdir,
            equity=sim["Equity"],
            returns=sim["DailyReturn"],
        )
        write_holdings(outdir=outdir, holdings=weights)
        if regime_filter and regime_label_used is not None and tv_weights_used is not None:
            fw = pd.DataFrame(
                {name: tv_weights_used[name] for name in resolved_factor_names},
                index=scores.index,
            )
            write_regime(
                outdir=outdir,
                label=regime_label_used.reindex(scores.index),
                factor_weights=fw,
                factor_names=resolved_factor_names,
            )
        write_composite_scores_snapshot(outdir=outdir, scores=scores, rebalance_mask=rb_mask)
        if price_quality_check:
            write_price_quality(outdir=outdir, quality_df=quality_flagged, filename="price_quality.csv")
            write_price_quality(
                outdir=outdir,
                quality_df=quality_flagged.loc[quality_flagged["is_bad"]].copy(),
                filename="price_quality_bad.csv",
            )
        if drop_bad_tickers:
            pd.DataFrame(dropped_rows).to_csv(
                outdir / "dropped_tickers.csv",
                index=False,
                float_format="%.10g",
            )
        with (outdir / "data_source_summary.json").open("w", encoding="utf-8") as f:
            json.dump(data_source_summary, f, indent=2, sort_keys=True)
        if elig_scores is not None:
            pd.DataFrame(
                {
                    "date": elig_rb_counts.index.astype(str),
                    "eligible_count": elig_rb_counts.to_numpy(dtype=int),
                    "liquidity_eligible_count": liquidity_counts_rb.reindex(elig_rb_counts.index).to_numpy(
                        dtype=int
                    ),
                    "liquidity_filtered_out_count": filtered_out_rb.reindex(
                        elig_rb_counts.index, fill_value=0
                    ).to_numpy(dtype=int),
                }
            ).to_csv(outdir / "universe_eligibility_summary.csv", index=False)
            pd.DataFrame(
                {
                    "date": liquidity_counts_rb.index.astype(str),
                    "liquidity_eligible_count": liquidity_counts_rb.to_numpy(dtype=int),
                    "liquidity_filtered_out_count": filtered_out_rb.reindex(
                        liquidity_counts_rb.index, fill_value=0
                    ).to_numpy(dtype=int),
                }
            ).to_csv(outdir / "rebalance_liquidity_eligibility.csv", index=False)
            if u_mode == "dynamic" and eligibility_price is not None:
                dbg = _build_zero_eligible_debug_frame(
                    rebalance_dates=rb_dates,
                    close=close,
                    eligibility_price=eligibility_price,
                    price_ok=eligibility_price_ok if eligibility_price_ok is not None else eligibility_price,
                    history_ok=(
                        eligibility_history_ok if eligibility_history_ok is not None else eligibility_price
                    ),
                    valid_ok=eligibility_valid_ok if eligibility_valid_ok is not None else eligibility_price,
                    factor_scores=norm_scores,
                    composite_scores=scores_pre_universe,
                    eligibility_final=elig_scores,
                    factor_names=resolved_factor_names,
                    effective_min_history_days=effective_min_history_days,
                    valid_lookback=universe_valid_lookback,
                    min_valid_frac=universe_min_valid_frac,
                )
                dbg.to_csv(outdir / "universe_zero_eligible_debug.csv", index=False, float_format="%.10g")
    append_registry_row(
        {
            "RunTag": run_tag,
            "Mode": "backtest",
            "GitCommit": _git_commit(),
            "Start": start,
            "End": end,
            "TickersUsed": len(used_tickers),
            "RequestedTickersCount": requested_tickers_count,
            "RequestedTickersSample": ";".join(tickers[:10]),
            "FetchRequestTickersCount": int(len(fetch_request_tickers)),
            "FetchRequestTickerSample": ";".join(fetch_request_tickers[:10]),
            "FetchedKeysCount": fetched_keys_count,
            "FetchedKeysSample": ";".join(fetched_keys_sample),
            "PostFetchFilteredKeysCount": post_fetch_filtered_keys_count,
            "PostFetchFilteredKeysSample": ";".join(post_fetch_filtered_keys_sample),
            "LoadedTickersCount": loaded_tickers_count,
            "LoadedTickersSample": ";".join(used_tickers[:10]),
            "MissingTickersCount": missing_tickers_count,
            "MissingTickersSample": ";".join(missing_tickers_sample),
            "OHLCVMapKeySample": ";".join(ohlcv_map_key_sample),
            "RejectedTickersCount": rejected_tickers_count,
            "RejectedTickersSample": ";".join(rejected_tickers_sample),
            "HistoricalMembershipPath": str(historical_membership_path),
            "HistoricalMembershipSource": str(historical_membership_source),
            "MembershipCoveredTickersCount": membership_covered_tickers_count,
            "MembershipCoveredTickersSample": ";".join(membership_covered_tickers_sample),
            "MembershipUniqueTickers": int(membership_alignment_diag.get("MembershipUniqueTickers", 0)),
            "MembershipPriceOverlapNormalized": int(
                membership_alignment_diag.get("MembershipPriceOverlapNormalized", 0)
            ),
            "MembershipUnmatchedNormalized": int(
                membership_alignment_diag.get("MembershipUnmatchedNormalized", 0)
            ),
            "MembershipUnmatchedSample": ";".join(
                list(membership_alignment_diag.get("MembershipUnmatchedSample", []))[:25]
            ),
            "MembershipAlignedPerDateMin": int(
                membership_alignment_diag.get("MembershipAlignedPerDateMin", 0)
            ),
            "MembershipAlignedPerDateMedian": float(
                membership_alignment_diag.get("MembershipAlignedPerDateMedian", 0.0)
            ),
            "MembershipAlignedPerDateMax": int(
                membership_alignment_diag.get("MembershipAlignedPerDateMax", 0)
            ),
            "SourceAvailableOnRebalanceMedian": source_median,
            "SourceAvailableOnRebalanceMin": source_min,
            "SourceAvailableOnRebalanceMax": source_max,
            "MembershipOnRebalanceMedian": membership_median,
            "MembershipOnRebalanceMin": membership_min,
            "MembershipOnRebalanceMax": membership_max,
            "EligibilityFilteredOnRebalanceMedian": eligibility_median,
            "EligibilityFilteredOnRebalanceMin": eligibility_min_r,
            "EligibilityFilteredOnRebalanceMax": eligibility_max_r,
            "FinalTradableOnRebalanceMedian": final_median,
            "FinalTradableOnRebalanceMin": final_min,
            "FinalTradableOnRebalanceMax": final_max,
            "TopN": top_n,
            "RankBuffer": int(rank_buffer),
            "VolatilityScaledWeights": bool(volatility_scaled_weights),
            "Rebalance": rebalance,
            "CostsBps": costs_bps,
            "FactorName": factor_name_legacy,
            "FactorParams": factor_params_legacy,
            "FactorNames": factor_names_str,
            "FactorWeights": factor_weights_str,
            "Normalize": normalize,
            "WinsorP": float(winsor_p),
            "UseFactorNormalization": bool(use_factor_normalization),
            "UseSectorNeutralization": bool(use_sector_neutralization),
            "UseSizeNeutralization": bool(use_size_neutralization),
            "DynamicFactorWeights": bool(dynamic_factor_weights),
            "OrthogonalizeFactors": bool(orthogonalize_factors),
            "SectorNeutral": bool(sector_neutral),
            "Weighting": weighting,
            "VolLookback": int(vol_lookback),
            "MaxWeight": float(max_weight),
            "ScoreClip": float(score_clip),
            "ScoreFloor": float(score_floor),
            "SectorCap": float(sector_cap),
            "SectorMap": sector_map,
            "TargetVol": float(target_vol),
            "PortVolLookback": int(port_vol_lookback),
            "MaxLeverage": float(max_leverage),
            "VolTargetingEnabled": bool(target_vol > 0.0),
            "RawVol": raw_vol,
            "FinalRealizedVol": final_realized_vol,
            "SlippageBps": float(slippage_bps),
            "SlippageVolMult": float(slippage_vol_mult),
            "SlippageVolLookback": int(slippage_vol_lookback),
            "ExecutionDelayDays": int(execution_delay_days),
            "RegimeFilter": bool(regime_filter),
            "RegimeBenchmark": str(regime_benchmark),
            "RegimeTrendSMA": int(regime_trend_sma),
            "RegimeVolLookback": int(regime_vol_lookback),
            "RegimeVolMedianLookback": int(regime_vol_median_lookback),
            "RegimeBullWeights": weight_map_to_json(bull_map),
            "RegimeBearWeights": weight_map_to_json(bear_map),
            "BearExposureScale": float(bear_exposure_scale),
            "BearExposureScaleAvgBearPeriods": float(bear_scale_avg_bear),
            "regime_pct_bull": regime_pct_bull,
            "regime_pct_bear_or_volatile": regime_pct_bear_or_volatile,
            "regime_last_label": regime_last_label,
            "EffectiveHoldingsAvg": effective_holdings_avg,
            "LeverageAvg": leverage_avg,
            "LeverageMax": leverage_max,
            "EffectiveCostBpsAvg": effective_cost_bps_avg,
            "TrendFilter": bool(trend_filter),
            "DataSource": str(data_source),
            "DataCacheDir": str(data_cache_dir),
            "DataRoot": str(data_cache_dir),
            "DataRefresh": bool(data_refresh),
            "DataBulkPrepare": bool(data_bulk_prepare),
            "DataCachedFilesUsed": int(data_source_summary.get("cached_files_used", 0)),
            "DataFetchedOrRefreshed": int(data_source_summary.get("fetched_or_refreshed", 0)),
            "SaveArtifacts": bool(save_artifacts),
            "PriceQualityCheck": bool(price_quality_check),
            "PriceQualityMode": str(price_quality_mode),
            "PriceQualityBadTickers": int(quality_summary.get("num_bad", 0)),
            "PriceFillMode": str(price_fill_mode),
            "DropBadTickers": bool(drop_bad_tickers),
            "DropBadTickersScope": str(drop_bad_tickers_scope),
            "DroppedTickersCount": int(dropped_tickers_count),
            "UniverseMode": str(universe_mode),
            "UniverseEligibilitySource": str(universe_eligibility_source),
            "UniverseEffectiveMinHistoryDays": int(effective_min_history_days),
            "MinPrice": float(min_price),
            "MinAvgDollarVolume": float(min_avg_dollar_volume),
            "LiquidityLookback": int(liquidity_lookback),
            "UniverseDatasetMode": str(universe_dataset_mode),
            "UniverseDatasetFreq": str(universe_dataset_freq),
            "UniverseDatasetPath": str(universe_dataset_path_used),
            "UniverseDatasetMembershipPath": str(universe_dataset_membership_path),
            "UniverseDatasetSummaryPath": str(universe_dataset_summary_path),
            "UniverseDatasetSaved": bool(universe_dataset_saved),
            "UniverseDatasetFallback": bool(universe_dataset_fallback),
            "UniverseDatasetStartDate": universe_dataset_start_date,
            "UniverseDatasetEndDate": universe_dataset_end_date,
            "UniverseDatasetRows": universe_dataset_rows,
            "LiquidityEligibleMedian": liquidity_eligible_median,
            "LiquidityEligibleMin": liquidity_eligible_min,
            "LiquidityEligibleMax": liquidity_eligible_max,
            "LiquidityFilteredOutMedian": liquidity_filtered_out_median,
            "LiquidityFilteredOutMax": liquidity_filtered_out_max,
            "EligibleTickersMedian": eligible_median,
            "EligibleTickersMin": eligible_min,
            "EligibleTickersMax": eligible_max,
            "EligibleOnRebalanceMedian": eligible_median,
            "EligibleOnRebalanceMin": eligible_min,
            "EligibleOnRebalanceMax": eligible_max,
            "RebalanceSkippedCount": rebalance_skipped_count,
            "RebalanceSkipBelowMinCount": rebalance_skip_below_min_count,
            "RebalanceSkipZeroEligibleCount": rebalance_skip_zero_count,
            "WindowsJsonPath": "",
            "CAGR": metrics.get("CAGR"),
            "Vol": metrics.get("Vol"),
            "Sharpe": metrics.get("Sharpe"),
            "MaxDD": metrics.get("MaxDD"),
            "Outdir": str(outdir),
            "TimestampUTC": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    sim.to_csv(outdir / "equity.csv", index_label="Date")
    timing_sections["report_write_seconds"] = time.perf_counter() - t_report_start
    timing_sections["total_seconds"] = time.perf_counter() - t_all_start
    summary["Timing"] = {
        "data_load_seconds": float(timing_sections["data_load_seconds"]),
        "factor_compute_seconds": float(timing_sections["factor_compute_seconds"]),
        "portfolio_backtest_seconds": float(timing_sections["portfolio_backtest_seconds"]),
        "report_write_seconds": float(timing_sections["report_write_seconds"]),
        "total_seconds": float(timing_sections["total_seconds"]),
    }
    run_config["TimingReportWriteSeconds"] = float(timing_sections["report_write_seconds"])
    run_config["TimingTotalSeconds"] = float(timing_sections["total_seconds"])
    summary["TimingReportWriteSeconds"] = float(timing_sections["report_write_seconds"])
    summary["TimingTotalSeconds"] = float(timing_sections["total_seconds"])
    _write_summary_artifacts(outdir=outdir, run_config=run_config, summary=summary)

    if bool(print_run_summary_flag):
        print_run_summary(summary)
    return summary, str(outdir)


def run_walkforward(
    start: str,
    end: str,
    train_years: int = 5,
    test_years: int = 2,
    max_tickers: int = 50,
    data_source: str = "parquet",
    data_cache_dir: str = "data/equities",
    data_refresh: bool = False,
    data_bulk_prepare: bool = False,
    top_n: int = 10,
    rank_buffer: int = 0,
    volatility_scaled_weights: bool = False,
    rebalance: str = "weekly",
    costs_bps: float = 10.0,
    seed: int = 42,
    factor_name: str | list[str] = "momentum_12_1",
    factor_names: list[str] | None = None,
    factor_params: dict | None = None,
    factor_weights: list[float] | None = None,
    normalize: str = "zscore",
    winsor_p: float = 0.01,
    use_factor_normalization: bool = True,
    use_sector_neutralization: bool = True,
    use_size_neutralization: bool = True,
    orthogonalize_factors: bool = False,
    sector_neutral: bool = False,
    weighting: str = "equal",
    vol_lookback: int = 20,
    max_weight: float = 0.15,
    score_clip: float = 5.0,
    score_floor: float = 0.0,
    sector_cap: float = 0.0,
    sector_map: str = "",
    target_vol: float = 0.0,
    port_vol_lookback: int = 20,
    max_leverage: float = 1.0,
    slippage_bps: float = 0.0,
    slippage_vol_mult: float = 0.0,
    slippage_vol_lookback: int = 20,
    execution_delay_days: int = 0,
    regime_filter: bool = False,
    dynamic_factor_weights: bool = False,
    regime_benchmark: str = "SPY",
    regime_trend_sma: int = 200,
    regime_vol_lookback: int = 20,
    regime_vol_median_lookback: int = 252,
    regime_bull_weights: str = "momentum_12_1:0.7,low_vol_20:0.3",
    regime_bear_weights: str = "momentum_12_1:0.3,low_vol_20:0.7",
    bear_exposure_scale: float = 1.0,
    trend_filter: bool = False,
    trend_sma_window: int = 200,
    save_artifacts: bool = False,
    print_run_summary_flag: bool = False,
    price_quality_check: bool = False,
    price_quality_mode: str = "warn",
    price_quality_zero_ret_thresh: float = 0.95,
    price_quality_min_valid_frac: float = 0.98,
    price_quality_max_bad_tickers: int = 0,
    price_quality_report_topk: int = 20,
    price_fill_mode: str = "ffill",
    drop_bad_tickers: bool = False,
    drop_bad_tickers_scope: str = "test",
    drop_bad_tickers_max_drop: int = 0,
    drop_bad_tickers_exempt: str = "SPY",
    universe_mode: str = "static",
    universe_min_history_days: int = 300,
    universe_min_valid_frac: float = 0.98,
    universe_valid_lookback: int = 252,
    universe_min_price: float = 1.0,
    min_price: float = 0.0,
    min_avg_dollar_volume: float = 0.0,
    liquidity_lookback: int = 20,
    universe_min_tickers: int = 20,
    universe_skip_below_min_tickers: bool = True,
    universe_eligibility_source: str = "price",
    universe_exempt: str = "SPY",
    universe_dataset_mode: str = "off",
    universe_dataset_freq: str = "rebalance",
    universe_dataset_path: str = "",
    universe_dataset_save: bool = True,
    universe_dataset_require: bool = False,
    historical_membership_path: str = "",
    fundamentals_path: str = "data/fundamentals/fundamentals_fmp.parquet",
    fundamentals_fallback_lag_days: int = 60,
) -> tuple[dict, str]:
    """Run walk-forward OOS backtest and persist artifacts."""
    del seed  # Placeholder for future deterministic extensions.

    if str(universe).strip().lower() == "liquid_us" and str(universe_mode).strip().lower() == "dynamic":
        # Canonical research thresholds for the corrected liquid_us universe.
        universe_min_history_days = max(int(universe_min_history_days), 252)
        universe_min_price = max(float(universe_min_price), 5.0)
        min_price = max(float(min_price), 5.0)
        min_avg_dollar_volume = max(float(min_avg_dollar_volume), 10_000_000.0)
        liquidity_lookback = max(int(liquidity_lookback), 20)

    if train_years <= 0:
        raise ValueError("train_years must be > 0")
    if test_years <= 0:
        raise ValueError("test_years must be > 0")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts >= end_ts:
        raise ValueError("start must be before end")

    windows: list[dict[str, pd.Timestamp]] = []
    t0 = start_ts
    while True:
        train_start = t0
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        if test_start >= end_ts:
            break
        test_end = min(test_start + pd.DateOffset(years=test_years), end_ts)
        if test_end <= test_start:
            break
        windows.append(
            {
                "TrainStart": train_start,
                "TrainEnd": train_end,
                "TestStart": test_start,
                "TestEnd": test_end,
            }
        )
        t0 = t0 + pd.DateOffset(years=test_years)
        if t0 >= end_ts:
            break

    if not windows:
        raise ValueError("No valid walk-forward windows generated for the given date range/settings.")

    tickers = sorted(load_sp500_tickers())[:max_tickers]
    if not tickers:
        raise ValueError("No tickers were loaded.")

    regime_benchmark = str(regime_benchmark).strip().upper() or "SPY"
    extra_symbols: list[str] = []
    if trend_filter and "SPY" not in tickers:
        extra_symbols.append("SPY")
    if (regime_filter or dynamic_factor_weights) and regime_benchmark not in tickers and regime_benchmark not in extra_symbols:
        extra_symbols.append(regime_benchmark)
    fetch_tickers = tickers + extra_symbols
    fetch_request_tickers = list(fetch_tickers)
    ohlcv_map, data_source_summary = fetch_ohlcv_with_summary(
        tickers=fetch_tickers,
        start=start,
        end=end,
        cache_dir=data_cache_dir,
        data_source=data_source,
        refresh=bool(data_refresh),
        bulk_prepare=bool(data_bulk_prepare),
    )
    close_cols, used_tickers, missing_tickers, rejected_tickers, ohlcv_map_key_sample = (
        _collect_close_series(ohlcv_map=ohlcv_map, requested_tickers=tickers)
    )
    requested_tickers_count = int(len(tickers))
    loaded_tickers_count = int(len(used_tickers))
    missing_tickers_count = int(len(missing_tickers))
    rejected_tickers_count = int(len(rejected_tickers))
    missing_tickers_sample = missing_tickers[:10]
    rejected_tickers_sample = rejected_tickers[:10]
    fetched_keys_count = int(len(ohlcv_map))
    fetched_keys_sample = sorted(list(ohlcv_map.keys()))[:10]
    post_fetch_filtered = [t for t in tickers if _lookup_ohlcv_frame(ohlcv_map, t) is not None]
    post_fetch_filtered_keys_count = int(len(post_fetch_filtered))
    post_fetch_filtered_keys_sample = post_fetch_filtered[:10]

    if not close_cols:
        raise ValueError(
            "No valid OHLCV data found for any ticker. "
            f"requested={len(tickers)} fetch_request={len(fetch_request_tickers)} "
            f"fetched_keys={fetched_keys_count} key_sample={fetched_keys_sample} "
            f"data_source={data_source} cache_dir={data_cache_dir} "
            f"historical_membership_path={str(historical_membership_path)}"
        )
    if len(used_tickers) < (top_n + 1):
        raise ValueError(
            f"Not enough tickers with data: got {len(used_tickers)}, "
            f"need at least top_n + 1 = {top_n + 1}."
        )

    close = pd.concat(close_cols, axis=1, join="outer")
    adj_close_cols, _, _, _, _ = _collect_price_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Adj Close",
        fallback_field="Close",
    )
    adj_close_raw = pd.concat(adj_close_cols, axis=1, join="outer") if adj_close_cols else close.copy()
    volume_raw = _collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Volume",
    )
    close = _prepare_close_panel(close_raw=close, price_fill_mode=price_fill_mode)
    adj_close = _prepare_close_panel(close_raw=adj_close_raw, price_fill_mode=price_fill_mode)
    close_full = close.copy()
    min_non_nan = int(0.8 * close.shape[1])
    close = close.dropna(thresh=min_non_nan)
    if close.empty:
        raise ValueError(
            "No usable close-price history after alignment/fill; "
            "check date range and ticker coverage."
        )
    if sector_cap > 0.0 and not sector_map.strip():
        warnings.warn("sector_cap>0 but no sector_map provided; sector caps disabled.", RuntimeWarning)
    volume_full = (
        volume_raw.reindex(index=close.index, columns=close.columns)
        if not volume_raw.empty
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    need_sector_map = bool(sector_cap > 0.0 or use_sector_neutralization or sector_neutral)
    sector_by_ticker = _load_sector_map(sector_map, list(close.columns)) if need_sector_map else None
    market_cap_by_ticker = (
        _load_market_cap_map(sector_map, list(close.columns)) if bool(use_size_neutralization) else None
    )

    if factor_names is not None and len(factor_names) > 0:
        resolved_factor_names = [str(x).strip() for x in factor_names if str(x).strip()]
    elif isinstance(factor_name, list):
        resolved_factor_names = [str(x).strip() for x in factor_name if str(x).strip()]
    else:
        resolved_factor_names = [str(factor_name).strip()]
    if not resolved_factor_names:
        raise ValueError("At least one factor must be provided.")

    raw_factor_params = dict(factor_params or {})
    if raw_factor_params and all(isinstance(v, dict) for v in raw_factor_params.values()):
        factor_params_map = {k: dict(v) for k, v in raw_factor_params.items()}
    elif raw_factor_params:
        if len(resolved_factor_names) == 1:
            factor_params_map = {resolved_factor_names[0]: raw_factor_params}
        else:
            factor_params_map = {name: dict(raw_factor_params) for name in resolved_factor_names}
    else:
        factor_params_map = {}
    u_mode = str(universe_mode).lower()
    if u_mode not in {"static", "dynamic"}:
        raise ValueError("universe_mode must be one of: static, dynamic")
    uds_mode = str(universe_dataset_mode).lower()
    if uds_mode not in {"off", "build", "use"}:
        raise ValueError("universe_dataset_mode must be one of: off, build, use")
    uds_freq = str(universe_dataset_freq).lower()
    if uds_freq not in {"daily", "rebalance"}:
        raise ValueError("universe_dataset_freq must be one of: daily, rebalance")
    effective_min_history_days = max(
        int(universe_min_history_days),
        _factor_required_history_days(
            factor_names=resolved_factor_names,
            factor_params_map=factor_params_map,
        ),
    )

    if factor_weights is None:
        blend_weights = np.repeat(1.0 / len(resolved_factor_names), len(resolved_factor_names))
    else:
        if len(factor_weights) != len(resolved_factor_names):
            raise ValueError(
                "factor_weights length must match number of factors: "
                f"{len(factor_weights)} vs {len(resolved_factor_names)}"
            )
        blend_weights = np.asarray(factor_weights, dtype=float)
        if not np.isfinite(blend_weights).all():
            raise ValueError("factor_weights must be finite numbers.")
        total_w = float(blend_weights.sum())
        if abs(total_w) < 1e-12:
            raise ValueError("factor_weights sum must be non-zero.")
        blend_weights = blend_weights / total_w

    factor_params_map = _augment_factor_params_with_fundamentals(
        factor_names=resolved_factor_names,
        factor_params_map=factor_params_map,
        close=close,
        fundamentals_path=fundamentals_path,
        fundamentals_fallback_lag_days=int(fundamentals_fallback_lag_days),
    )

    factor_names_str = ";".join(resolved_factor_names)
    factor_weights_str = ";".join(f"{float(w):.10g}" for w in blend_weights)
    static_weights = {
        name: float(w) for name, w in zip(resolved_factor_names, np.asarray(blend_weights, dtype=float))
    }
    require_all_factor_scores = u_mode == "dynamic"
    use_dynamic_weights = should_apply_dynamic_factor_weights(
        regime_filter=bool(regime_filter),
        dynamic_factor_weights=bool(dynamic_factor_weights),
    )
    bull_map = parse_weight_map(regime_bull_weights) if use_dynamic_weights else {}
    bear_map = parse_weight_map(regime_bear_weights) if use_dynamic_weights else {}
    regime_label_all: pd.Series | None = None
    if use_dynamic_weights:
        spy_close_all = _get_benchmark_close_from_map(
            ohlcv_map=ohlcv_map,
            score_index=close.index,
            benchmark=regime_benchmark,
            source=data_source,
        )
        regime_label_all = compute_regime_label(
            spy_close=spy_close_all,
            score_index=close.index,
            trend_sma=regime_trend_sma,
            vol_lookback=regime_vol_lookback,
            vol_median_lookback=regime_vol_median_lookback,
        )
    if len(resolved_factor_names) == 1:
        factor_name_legacy = resolved_factor_names[0]
        factor_params_legacy = json.dumps(
            factor_params_map.get(factor_name_legacy, {}),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    else:
        factor_name_legacy = "MULTI"
        factor_params_legacy = "{}"

    market_close_all = None
    if trend_filter:
        market_close_all = _get_benchmark_close_from_map(
            ohlcv_map=ohlcv_map,
            score_index=close.index,
            benchmark="SPY",
            source=data_source,
        )
    universe_exempt_set = _parse_ticker_csv(universe_exempt) | {"SPY"}
    universe_membership_full: pd.DataFrame | None = None
    universe_membership_summary_full: pd.DataFrame | None = None
    universe_dataset_saved = False
    universe_dataset_save_paths: dict[str, str] = {}
    universe_dataset_path_used = str(universe_dataset_path).strip() if uds_mode == "build" else ""
    universe_dataset_membership_path = ""
    universe_dataset_summary_path = ""
    universe_dataset_fallback = False
    membership_covered_tickers_count = loaded_tickers_count
    membership_covered_tickers_sample = used_tickers[:10]
    historical_membership_full: pd.DataFrame | None = None
    if str(historical_membership_path).strip():
        historical_membership_full = load_historical_membership(
            path=str(historical_membership_path).strip(),
            index=pd.DatetimeIndex(close.index),
            columns=list(close.columns),
        )
        covered_mask = (
            historical_membership_full.any(axis=0).reindex(close.columns).fillna(False).astype(bool)
        )
        covered = [c for c in close.columns if bool(covered_mask.get(c, False))]
        membership_covered_tickers_count = int(len(covered))
        membership_covered_tickers_sample = covered[:10]
    if u_mode == "dynamic":
        (
            price_ok_full,
            history_ok_full,
            valid_ok_full,
            liquidity_price_ok_full,
            liquidity_adv_ok_full,
            elig_full,
        ) = compute_eligibility_components(
            close=close,
            min_history_days=effective_min_history_days,
            valid_lookback=universe_valid_lookback,
            min_valid_frac=universe_min_valid_frac,
            min_price=max(float(universe_min_price), float(min_price)),
            volume=volume_full,
            min_avg_dollar_volume=float(min_avg_dollar_volume),
            liquidity_lookback=int(liquidity_lookback),
        )
        assert set(elig_full.columns) == set(close.columns)
        if str(universe).strip().lower() == "liquid_us":
            elig_full = build_liquid_us_universe(
                prices=close,
                volumes=volume_full,
                min_price=max(float(universe_min_price), float(min_price)),
                min_avg_dollar_volume=float(min_avg_dollar_volume),
                adv_window=int(liquidity_lookback),
                min_history=int(effective_min_history_days),
            )
    if uds_mode in {"build", "use"}:
        if uds_mode == "build":
            dataset_dates = (
                pd.DatetimeIndex(close_full.index[rebalance_mask(close_full.index, rebalance)])
                if uds_freq == "rebalance"
                else None
            )
            universe_membership_full = build_point_in_time_universe(
                close=close_full,
                min_history_days=effective_min_history_days,
                valid_lookback=universe_valid_lookback,
                min_valid_frac=universe_min_valid_frac,
                min_price=max(float(universe_min_price), float(min_price)),
                exempt=universe_exempt_set,
                dates=dataset_dates,
            )
            universe_membership_summary_full = summarize_universe_membership(universe_membership_full)
        else:
            in_path = str(universe_dataset_path).strip() or str(_default_universe_dataset_path())
            try:
                loaded = load_universe_dataset(in_path)
                universe_membership_full = _align_universe_membership(
                    membership=loaded,
                    close=close_full,
                    exempt=universe_exempt_set,
                )
                universe_membership_summary_full = summarize_universe_membership(
                    universe_membership_full
                )
                universe_dataset_path_used = in_path
            except FileNotFoundError:
                if bool(universe_dataset_require):
                    raise ValueError(f"Universe dataset required but missing: {in_path}")
                warnings.warn(
                    f"Universe dataset not found at {in_path}; falling back to runtime dynamic eligibility.",
                    RuntimeWarning,
                )
                universe_dataset_fallback = True
        if universe_membership_full is not None and u_mode == "dynamic":
            elig_full = _align_universe_membership(
                membership=universe_membership_full,
                close=close,
                exempt=universe_exempt_set,
            )
            assert set(elig_full.columns) == set(close.columns)
    else:
        price_ok_full = None
        history_ok_full = None
        valid_ok_full = None
        liquidity_price_ok_full = None
        liquidity_adv_ok_full = None
        elig_full = None

    window_rows: list[dict] = []
    windows_payload: list[dict] = []
    oos_parts: list[pd.DataFrame] = []
    holdings_parts: list[pd.DataFrame] = []
    leverage_parts: list[pd.Series] = []
    regime_parts: list[pd.Series] = []
    regime_weight_parts: list[pd.DataFrame] = []
    quality_parts: list[pd.DataFrame] = []
    dropped_rows_all: list[dict] = []
    dropped_tickers_total = 0
    exempt_tickers = _parse_ticker_csv(drop_bad_tickers_exempt)
    window_eligibility_parts: list[pd.DataFrame] = []
    eligibility_counts_all: list[pd.Series] = []
    universe_added_medians_all: list[float] = []
    universe_removed_medians_all: list[float] = []
    liquidity_eligible_counts_all: list[pd.Series] = []
    liquidity_filtered_out_counts_all: list[pd.Series] = []
    bear_scale_avg_bear_windows: list[float] = []
    rebalance_skip_below_min_total = 0
    rebalance_skip_zero_total = 0
    zero_eligible_debug_parts: list[pd.DataFrame] = []
    source_counts_all: list[pd.Series] = []
    membership_counts_all: list[pd.Series] = []
    final_tradable_counts_all: list[pd.Series] = []

    for i, w in enumerate(windows, start=1):
        test_start = w["TestStart"]
        test_end = w["TestEnd"]
        train_start = w["TrainStart"]

        hist_mask = close.index < test_end
        close_hist = close.loc[hist_mask]
        adj_close_hist = adj_close.loc[hist_mask].reindex(index=close_hist.index, columns=close_hist.columns)
        adj_close_hist = adj_close_hist.where(adj_close_hist.notna(), close_hist)
        test_mask = (close_hist.index >= test_start) & (close_hist.index < test_end)
        if close_hist.empty or not bool(test_mask.any()):
            continue
        dropped_in_window: list[str] = []
        if drop_bad_tickers:
            scope = str(drop_bad_tickers_scope).lower()
            if scope not in {"test", "train_and_test", "full"}:
                raise ValueError("drop_bad_tickers_scope must be one of: test, train_and_test, full")
            if scope == "test":
                close_for_drop = close_hist.loc[test_mask]
            elif scope == "train_and_test":
                train_test_mask = (close_hist.index >= train_start) & (close_hist.index < test_end)
                close_for_drop = close_hist.loc[train_test_mask]
            else:
                close_for_drop = close_hist
            win_name = f"wf_{i-1}_test_{test_start.strftime('%Y-%m-%d')}_{test_end.strftime('%Y-%m-%d')}"
            _, dropped_in_window, flagged_drop = filter_bad_tickers(
                close=close_for_drop,
                window_name=win_name,
                zero_ret_thresh=price_quality_zero_ret_thresh,
                min_valid_frac=price_quality_min_valid_frac,
                exempt=exempt_tickers,
            )
            close_hist = close_hist.drop(columns=dropped_in_window, errors="ignore")
            adj_close_hist = adj_close_hist.drop(columns=dropped_in_window, errors="ignore")
            dropped_tickers_total += len(dropped_in_window)
            if dropped_in_window:
                dropped_view = flagged_drop.loc[flagged_drop["ticker"].isin(dropped_in_window)].copy()
                dropped_view["reason"] = (
                    dropped_view["bad_valid"].map({True: "bad_valid", False: ""})
                    + dropped_view["bad_zero"].map({True: "|bad_zero", False: ""})
                ).str.strip("|")
                dropped_view["window_id"] = i - 1
                dropped_rows_all.extend(
                    dropped_view[
                        [
                            "window_id",
                            "window_name",
                            "ticker",
                            "reason",
                            "zero_ret_frac",
                            "valid_frac_close",
                        ]
                    ].to_dict(orient="records")
                )
            mode = str(price_quality_mode).lower()
            if drop_bad_tickers_max_drop > 0 and len(dropped_in_window) > int(drop_bad_tickers_max_drop):
                msg = (
                    f"PRICE QUALITY DROP LIMIT: window={win_name} dropped={len(dropped_in_window)} "
                    f"max_drop={int(drop_bad_tickers_max_drop)} mode={mode}"
                )
                if mode == "fail":
                    raise ValueError(msg)
                print(msg)
            if close_hist.shape[1] < (top_n + 1):
                raise ValueError(
                    f"Not enough tickers after drop_bad_tickers in {win_name}: "
                    f"got {close_hist.shape[1]}, need at least {top_n + 1}."
                )
            test_mask = (close_hist.index >= test_start) & (close_hist.index < test_end)
        q_summary: dict = {"num_bad": 0, "worst_by_zero_ret": [], "worst_by_valid_frac": []}
        if price_quality_check:
            mode = str(price_quality_mode).lower()
            if mode not in {"warn", "fail"}:
                raise ValueError("price_quality_mode must be one of: warn, fail")
            window_name = f"wf_{i-1}_test_{test_start.strftime('%Y-%m-%d')}_{test_end.strftime('%Y-%m-%d')}"
            q_raw = compute_price_quality(
                close=close_hist.loc[test_mask],
                window_name=window_name,
                zero_ret_thresh=price_quality_zero_ret_thresh,
                min_valid_frac=price_quality_min_valid_frac,
            )
            q_flagged = flag_bad_tickers(
                quality_df=q_raw,
                zero_ret_thresh=price_quality_zero_ret_thresh,
                min_valid_frac=price_quality_min_valid_frac,
            )
            q_summary = summarize_price_quality(q_flagged, topk=price_quality_report_topk)
            quality_parts.append(q_flagged)
            if int(q_summary["num_bad"]) > int(price_quality_max_bad_tickers):
                msg = format_price_quality_message(
                    window_name=f"wf_{i-1}_test_{test_start.strftime('%Y-%m-%d')}..{test_end.strftime('%Y-%m-%d')}",
                    summary=q_summary,
                    max_bad_tickers=price_quality_max_bad_tickers,
                    mode=mode,
                    preview_k=min(3, int(price_quality_report_topk)),
                )
                if mode == "fail":
                    raise ValueError(msg)
                print(msg)
        if u_mode == "dynamic":
            if elig_full is None:
                raise ValueError("Dynamic universe requested but full eligibility matrix is missing.")
            eligibility_price_hist = elig_full.loc[close_hist.index]
            assert set(eligibility_price_hist.columns) == set(close.columns)
            price_ok_hist = price_ok_full.loc[close_hist.index] if price_ok_full is not None else None
            history_ok_hist = (
                history_ok_full.loc[close_hist.index] if history_ok_full is not None else None
            )
            valid_ok_hist = valid_ok_full.loc[close_hist.index] if valid_ok_full is not None else None
            liquidity_price_ok_hist = (
                liquidity_price_ok_full.loc[close_hist.index]
                if liquidity_price_ok_full is not None
                else None
            )
            liquidity_adv_ok_hist = (
                liquidity_adv_ok_full.loc[close_hist.index] if liquidity_adv_ok_full is not None else None
            )
        else:
            eligibility_price_hist = None
            price_ok_hist = None
            history_ok_hist = None
            valid_ok_hist = None
            liquidity_price_ok_hist = None
            liquidity_adv_ok_hist = None
        historical_membership_hist = (
            historical_membership_full.reindex(index=close_hist.index, columns=close_hist.columns)
            .fillna(False)
            .astype(bool)
            if historical_membership_full is not None
            else None
        )

        factor_close_hist = adj_close_hist.reindex(index=close_hist.index, columns=close_hist.columns)
        factor_close_hist = factor_close_hist.where(factor_close_hist.notna(), close_hist)
        raw_scores = compute_factors(
            factor_names=resolved_factor_names,
            close=factor_close_hist,
            factor_params=factor_params_map,
        )
        if bool(use_factor_normalization):
            base_scores: dict[str, pd.DataFrame] = {
                name: robust_preprocess_base(raw_scores[name], winsor_p=0.05)
                for name in resolved_factor_names
            }
            if bool(use_sector_neutralization or use_size_neutralization):
                base_scores = {
                    name: neutralize_scores_cs(
                        base_scores[name],
                        sector_by_ticker=sector_by_ticker,
                        log_market_cap_by_ticker=market_cap_by_ticker,
                        use_sector_neutralization=use_sector_neutralization,
                        use_size_neutralization=use_size_neutralization,
                    )
                    for name in resolved_factor_names
                }
            norm_scores: dict[str, pd.DataFrame] = {
                name: percentile_rank_cs(base_scores[name]) for name in resolved_factor_names
            }
            norm_scores = maybe_orthogonalize_factor_scores(
                factor_scores=norm_scores,
                enabled=bool(orthogonalize_factors),
                factor_order=resolved_factor_names,
            )
        else:
            norm_scores = {
                name: normalize_scores(raw_scores[name], method=normalize, winsor_p=winsor_p)
                for name in resolved_factor_names
            }
            norm_scores = maybe_orthogonalize_factor_scores(
                factor_scores=norm_scores,
                enabled=bool(orthogonalize_factors),
                factor_order=resolved_factor_names,
            )
        if use_dynamic_weights:
            label_hist = regime_label_all.loc[close_hist.index] if regime_label_all is not None else None
            if label_hist is None:
                raise ValueError("Regime labels are missing while dynamic factor weighting is enabled.")
            tv_weights = build_regime_weight_series(
                factor_names=resolved_factor_names,
                static_weights=static_weights,
                label=label_hist,
                bull_weights=bull_map,
                bear_weights=bear_map,
            )
            scores = combine_factor_scores(
                norm_scores, tv_weights, require_all_factors=require_all_factor_scores
            )
            if regime_filter:
                regime_parts.append(label_hist.loc[test_mask].rename("RegimeLabel"))
                fw_test = pd.DataFrame(
                    {name: tv_weights[name] for name in resolved_factor_names},
                    index=close_hist.index,
                ).loc[test_mask]
                regime_weight_parts.append(fw_test)
        else:
            scores = combine_factor_scores(
                norm_scores, static_weights, require_all_factors=require_all_factor_scores
            )
        scores_pre_universe = scores.copy()
        volume_hist = volume_full.reindex(index=close_hist.index, columns=close_hist.columns)
        if float(min_price) > 0.0:
            liquidity_price_hist_calc = (
                close_hist.reindex(index=scores.index, columns=scores.columns) >= float(min_price)
            ).fillna(False)
        else:
            liquidity_price_hist_calc = pd.DataFrame(True, index=scores.index, columns=scores.columns)
        if float(min_avg_dollar_volume) > 0.0:
            dv_hist = (
                close_hist.reindex(index=scores.index, columns=scores.columns)
                * volume_hist.reindex(index=scores.index, columns=scores.columns)
            )
            adv_hist = dv_hist.rolling(
                int(liquidity_lookback),
                min_periods=int(liquidity_lookback),
            ).mean()
            liquidity_adv_hist_calc = (adv_hist >= float(min_avg_dollar_volume)).fillna(False)
        else:
            liquidity_adv_hist_calc = pd.DataFrame(True, index=scores.index, columns=scores.columns)
        liquidity_elig_hist = (liquidity_price_hist_calc & liquidity_adv_hist_calc).astype(bool)
        liquidity_filter_enabled = bool(float(min_price) > 0.0 or float(min_avg_dollar_volume) > 0.0)
        if historical_membership_hist is not None:
            scores = apply_universe_filter_to_scores(scores, historical_membership_hist, exempt=set())  # type: ignore[assignment]
        if eligibility_price_hist is not None:
            base_elig = _resolve_universe_eligibility(
                eligibility_price=eligibility_price_hist,
                scores=scores_pre_universe,
                source=universe_eligibility_source,
            )
            if historical_membership_hist is not None:
                hist_aligned = (
                    historical_membership_hist.reindex(index=scores.index, columns=scores.columns)
                    .fillna(False)
                    .astype(bool)
                )
                elig_for_scores = (base_elig & hist_aligned).astype(bool)
            else:
                elig_for_scores = base_elig
            if liquidity_filter_enabled:
                elig_for_scores = (elig_for_scores & liquidity_elig_hist).astype(bool)
            scores = apply_universe_filter_to_scores(
                scores, elig_for_scores, exempt=universe_exempt_set
            )  # type: ignore[assignment]
        elif historical_membership_hist is not None:
            elig_for_scores = (
                historical_membership_hist.reindex(index=scores.index, columns=scores.columns)
                .fillna(False)
                .astype(bool)
            )
            if liquidity_filter_enabled:
                elig_for_scores = (elig_for_scores & liquidity_elig_hist).astype(bool)
            scores = apply_universe_filter_to_scores(scores, elig_for_scores, exempt=set())  # type: ignore[assignment]
        else:
            if liquidity_filter_enabled:
                elig_for_scores = liquidity_elig_hist
                scores = apply_universe_filter_to_scores(
                    scores, elig_for_scores, exempt=universe_exempt_set
                )  # type: ignore[assignment]
            else:
                elig_for_scores = None
        test_dates = close_hist.index[test_mask]
        rebalance_count = _count_rebalance_dates(pd.DatetimeIndex(test_dates), rebalance)
        if rebalance.lower() == "weekly" and len(test_dates) >= 20:
            ratio = rebalance_count / float(len(test_dates))
            assert 0.10 <= ratio <= 0.35, (
                f"Weekly RebalanceCount sanity failed: count={rebalance_count} obs={len(test_dates)} ratio={ratio:.3f}"
            )
        rb_hist = rebalance_mask(close_hist.index, rebalance)
        rb_test_mask_full = rb_hist & test_mask
        rb_dates_win = pd.DatetimeIndex(scores.index[rb_test_mask_full])
        liquidity_counts_rb_test = (
            liquidity_elig_hist.loc[rb_test_mask_full].sum(axis=1).astype(int)
            if bool(rb_test_mask_full.any())
            else pd.Series(dtype=int)
        )
        source_counts_rb_test = (
            close_hist.reindex(rb_dates_win).notna().sum(axis=1).astype(int)
            if not rb_dates_win.empty
            else pd.Series(dtype=int)
        )
        membership_counts_rb_test = (
            historical_membership_hist.reindex(rb_dates_win).fillna(False).sum(axis=1).astype(int)
            if historical_membership_hist is not None and not rb_dates_win.empty
            else source_counts_rb_test
        )
        scores_for_weights = scores
        universe_added_median_win = float("nan")
        universe_removed_median_win = float("nan")

        if elig_for_scores is not None:
            elig_counts_rb_test = (
                elig_for_scores.loc[rb_test_mask_full].sum(axis=1).astype(int)
                if bool(rb_test_mask_full.any())
                else pd.Series(dtype=int)
            )
            if len(elig_counts_rb_test) != len(rb_dates_win):
                raise ValueError(
                    f"Window {i-1} dynamic universe count mismatch: "
                    f"counts={len(elig_counts_rb_test)} rebalances={len(rb_dates_win)}"
                )
            eligible_median_win = (
                float(elig_counts_rb_test.median()) if not elig_counts_rb_test.empty else 0.0
            )
            eligible_min_win = int(elig_counts_rb_test.min()) if not elig_counts_rb_test.empty else 0
            eligible_max_win = int(elig_counts_rb_test.max()) if not elig_counts_rb_test.empty else 0
            if not rb_dates_win.empty:
                membership_rb_summary = summarize_universe_membership(
                    elig_for_scores.loc[rb_dates_win].fillna(False).astype(bool)
                )
                if not membership_rb_summary.empty:
                    universe_added_median_win = float(membership_rb_summary["added_count"].median())
                    universe_removed_median_win = float(
                        membership_rb_summary["removed_count"].median()
                    )
                    universe_added_medians_all.append(universe_added_median_win)
                    universe_removed_medians_all.append(universe_removed_median_win)
            rebalance_skipped_count_win = int((elig_counts_rb_test <= 0).sum())
            low_elig_win = int((elig_counts_rb_test < int(universe_min_tickers)).sum())
            zero_elig_win = int((elig_counts_rb_test == 0).sum())
            if zero_elig_win > len(rb_dates_win) or low_elig_win > len(rb_dates_win):
                raise ValueError(
                    f"Window {i-1} dynamic universe sanity failed: "
                    f"zero={zero_elig_win} under_min={low_elig_win} rebalances={len(rb_dates_win)}"
                )
            if low_elig_win > 0:
                print(
                    f"UNIVERSE DYNAMIC: window={i-1} rebalances with eligible<{int(universe_min_tickers)} = {low_elig_win}"
                )
            scores_test = scores.loc[test_mask]
            rb_test_mask = rb_hist.loc[test_mask]
            scores_test_adj, rebalance_skip_zero_count_win, rebalance_skip_below_min_count_win = (
                _apply_universe_rebalance_skip(
                    scores=scores_test,
                    rb_mask=rb_test_mask,
                    universe_min_tickers=universe_min_tickers,
                    universe_skip_below_min_tickers=universe_skip_below_min_tickers,
                )
            )
            rebalance_skipped_count_win = (
                rebalance_skip_zero_count_win + rebalance_skip_below_min_count_win
            )
            rebalance_skip_zero_total += rebalance_skip_zero_count_win
            rebalance_skip_below_min_total += rebalance_skip_below_min_count_win
            if rebalance_skipped_count_win > 0:
                scores_for_weights = scores.copy()
                skip_dates = scores_test.index[rb_test_mask & scores_test_adj.isna().all(axis=1)]
                scores_for_weights.loc[skip_dates] = np.nan
            eligibility_counts_all.append(elig_counts_rb_test)
            if not elig_counts_rb_test.empty:
                window_eligibility_parts.append(
                    pd.DataFrame(
                        {
                            "window_id": i - 1,
                            "date": elig_counts_rb_test.index.astype(str),
                            "eligible_count": elig_counts_rb_test.to_numpy(dtype=int),
                            "liquidity_eligible_count": liquidity_counts_rb_test.reindex(
                                elig_counts_rb_test.index, fill_value=0
                            ).to_numpy(dtype=int),
                        }
                    )
                )
            if save_artifacts and eligibility_price_hist is not None:
                dbg_win = _build_zero_eligible_debug_frame(
                    rebalance_dates=pd.DatetimeIndex(scores.index[rb_test_mask_full]),
                    close=close_hist,
                    eligibility_price=eligibility_price_hist,
                    price_ok=price_ok_hist if price_ok_hist is not None else eligibility_price_hist,
                    history_ok=history_ok_hist if history_ok_hist is not None else eligibility_price_hist,
                    valid_ok=valid_ok_hist if valid_ok_hist is not None else eligibility_price_hist,
                    factor_scores=norm_scores,
                    composite_scores=scores_pre_universe,
                    eligibility_final=elig_for_scores,
                    factor_names=resolved_factor_names,
                    effective_min_history_days=effective_min_history_days,
                    valid_lookback=universe_valid_lookback,
                    min_valid_frac=universe_min_valid_frac,
                    window_id=i - 1,
                )
                zero_eligible_debug_parts.append(dbg_win)
            if not rb_dates_win.empty:
                pre_liq_hist = (
                    _resolve_universe_eligibility(
                        eligibility_price=eligibility_price_hist,
                        scores=scores_pre_universe,
                        source=universe_eligibility_source,
                    ).loc[rb_dates_win]
                    if eligibility_price_hist is not None
                    else (
                        historical_membership_hist.reindex(index=rb_dates_win, columns=scores.columns)
                        .fillna(False)
                        .astype(bool)
                        if historical_membership_hist is not None
                        else pd.DataFrame(True, index=rb_dates_win, columns=scores.columns)
                    )
                )
                liquidity_filtered_out_rb_test = (
                    pre_liq_hist.sum(axis=1) - elig_for_scores.loc[rb_dates_win].sum(axis=1)
                ).clip(lower=0).astype(int)
            else:
                liquidity_filtered_out_rb_test = pd.Series(dtype=int)
        else:
            fallback_counts = membership_counts_rb_test
            eligible_median_win = (
                float(fallback_counts.median()) if not fallback_counts.empty else float("nan")
            )
            eligible_min_win = int(fallback_counts.min()) if not fallback_counts.empty else 0
            eligible_max_win = int(fallback_counts.max()) if not fallback_counts.empty else 0
            if not fallback_counts.empty:
                eligibility_counts_all.append(fallback_counts)
            rebalance_skipped_count_win = 0
            rebalance_skip_zero_count_win = 0
            rebalance_skip_below_min_count_win = 0
            liquidity_filtered_out_rb_test = pd.Series(dtype=int)
        final_tradable_counts_rb_test = (
            scores.loc[rb_test_mask_full].notna().sum(axis=1).astype(int)
            if bool(rb_test_mask_full.any())
            else pd.Series(dtype=int)
        )
        source_median_win, source_min_win, source_max_win = _stats_min_med_max(source_counts_rb_test)
        membership_median_win, membership_min_win, membership_max_win = _stats_min_med_max(
            membership_counts_rb_test
        )
        eligibility_median_win, eligibility_min_r_win, eligibility_max_r_win = _stats_min_med_max(
            elig_counts_rb_test if elig_for_scores is not None else fallback_counts
        )
        final_median_win, final_min_win, final_max_win = _stats_min_med_max(final_tradable_counts_rb_test)
        if not source_counts_rb_test.empty:
            source_counts_all.append(source_counts_rb_test)
        if not membership_counts_rb_test.empty:
            membership_counts_all.append(membership_counts_rb_test)
        if not final_tradable_counts_rb_test.empty:
            final_tradable_counts_all.append(final_tradable_counts_rb_test)
        liquidity_eligible_median_win = (
            float(liquidity_counts_rb_test.median()) if not liquidity_counts_rb_test.empty else float("nan")
        )
        liquidity_eligible_min_win = int(liquidity_counts_rb_test.min()) if not liquidity_counts_rb_test.empty else 0
        liquidity_eligible_max_win = int(liquidity_counts_rb_test.max()) if not liquidity_counts_rb_test.empty else 0
        liquidity_filtered_out_median_win = (
            float(liquidity_filtered_out_rb_test.median())
            if not liquidity_filtered_out_rb_test.empty
            else float("nan")
        )
        liquidity_filtered_out_max_win = (
            int(liquidity_filtered_out_rb_test.max()) if not liquidity_filtered_out_rb_test.empty else 0
        )
        if not liquidity_counts_rb_test.empty:
            liquidity_eligible_counts_all.append(liquidity_counts_rb_test)
        if not liquidity_filtered_out_rb_test.empty:
            liquidity_filtered_out_counts_all.append(liquidity_filtered_out_rb_test)

        weights = build_topn_weights(
            scores=scores_for_weights,
            close=close_hist,
            top_n=top_n,
            rank_buffer=int(rank_buffer),
            volatility_scaled_weights=bool(volatility_scaled_weights),
            rebalance=rebalance,
            weighting=weighting,
            vol_lookback=vol_lookback,
            max_weight=max_weight,
            score_clip=score_clip,
            score_floor=score_floor,
            sector_cap=sector_cap,
            sector_by_ticker=sector_by_ticker,
            sector_neutral=bool(sector_neutral),
        )
        risk_on_pct = 1.0
        if trend_filter:
            market_close_hist = market_close_all.loc[close_hist.index]  # type: ignore[index]
            sma = market_close_hist.rolling(trend_sma_window).mean()
            risk_on = (market_close_hist > sma).shift(1)
            risk_on = risk_on.reindex(weights.index).fillna(False).astype(bool)
            risk_on_test = risk_on.loc[test_mask]
            risk_on_pct = float(risk_on_test.mean()) if not risk_on_test.empty else 0.0
            weights = apply_trend_filter(
                weights=weights,
                market_close=market_close_hist,
                sma_window=trend_sma_window,
            )

        sim_hist = simulate_portfolio(
            # Use adjusted prices for walkforward PnL so simulation stays aligned
            # with factor inputs and avoids split-related distortions.
            close=adj_close_hist,
            weights=weights,
            costs_bps=costs_bps,
            slippage_bps=slippage_bps,
            slippage_vol_mult=slippage_vol_mult,
            slippage_vol_lookback=slippage_vol_lookback,
            rebalance_dates=pd.DatetimeIndex(close_hist.index[rb_hist]),
            execution_delay_days=int(execution_delay_days),
        )
        sim_hist, leverage, raw_vol_win, final_realized_vol_win = _apply_vol_target(
            sim=sim_hist,
            target_vol=target_vol,
            lookback=port_vol_lookback,
            max_leverage=max_leverage,
        )
        sim_hist, _, bear_scale_avg_bear_win = _apply_bear_exposure_overlay(
            sim=sim_hist,
            regime_label=label_hist if regime_filter else None,
            regime_filter=bool(regime_filter),
            bear_exposure_scale=float(bear_exposure_scale),
        )
        bear_scale_avg_bear_windows.append(float(bear_scale_avg_bear_win))
        sim_test = sim_hist.loc[test_mask]
        if sim_test.empty:
            continue

        w_test = weights.loc[test_mask]
        holdings_parts.append(w_test)
        gross_exposure = w_test.abs().sum(axis=1)
        invested_pct = float((gross_exposure > 0.0).mean()) if not gross_exposure.empty else 0.0
        holdings_count = (w_test.abs() > 0.0).sum(axis=1)
        avg_holdings = float(holdings_count.mean()) if not holdings_count.empty else 0.0
        rb_test_mask = _diagnostic_rebalance_mask(w_test.index, rebalance)
        turnover_on_reb = sim_test.loc[rb_test_mask, "Turnover"] if "Turnover" in sim_test.columns else pd.Series(dtype=float)
        if not turnover_on_reb.empty:
            turnover_avg = float(turnover_on_reb.mean())
        else:
            turnover_rb = 0.5 * w_test.diff().abs().sum(axis=1)
            turnover_rb = turnover_rb.loc[rb_test_mask]
            turnover_avg = float(turnover_rb.mean()) if not turnover_rb.empty else 0.0
        max_name_weight = float(w_test.max(axis=1).max()) if not w_test.empty else 0.0
        if sector_cap > 0.0 and sector_by_ticker is not None:
            sectors = pd.Series({t: sector_by_ticker.get(t, "UNKNOWN") for t in w_test.columns}, dtype=object)
            sec_panel = w_test.T.groupby(sectors).sum().T
            max_sector_weight = float(sec_panel.max(axis=1).max()) if not sec_panel.empty else float("nan")
        else:
            max_sector_weight = float("nan")
        invested_rows = gross_exposure > 0.0
        if invested_rows.any():
            eff_holdings = 1.0 / (
                w_test.loc[invested_rows].pow(2).sum(axis=1).replace(0.0, np.nan)
            )
            effective_holdings_avg = float(eff_holdings.mean())
        else:
            effective_holdings_avg = float("nan")

        w_metrics = compute_metrics(sim_test["DailyReturn"])
        effective_cost_bps_avg_win = (
            float(sim_test["EffectiveCostBps"].mean())
            if "EffectiveCostBps" in sim_test.columns
            else float(costs_bps + slippage_bps)
        )
        lev_test = leverage.loc[test_mask]
        leverage_avg_win = float(lev_test.mean()) if not lev_test.empty else 1.0
        leverage_max_win = float(lev_test.max()) if not lev_test.empty else 1.0
        if regime_filter:
            lbl_test = label_hist.loc[test_mask]  # type: ignore[union-attr]
            lbl_valid = lbl_test.notna()
            if bool(lbl_valid.any()):
                regime_pct_bull_win = float((lbl_test[lbl_valid] == "bull").mean())
                regime_pct_bear_win = float((lbl_test[lbl_valid] == "bear_or_volatile").mean())
            else:
                regime_pct_bull_win = 0.0
                regime_pct_bear_win = 0.0
            lbl_non_na = lbl_test.dropna()
            regime_last_label_win = str(lbl_non_na.iloc[-1]) if not lbl_non_na.empty else None
        else:
            regime_pct_bull_win = float("nan")
            regime_pct_bear_win = float("nan")
            regime_last_label_win = None

        window_rows.append(
            {
                "Window": i,
                "TrainStart": w["TrainStart"].strftime("%Y-%m-%d"),
                "TrainEnd": w["TrainEnd"].strftime("%Y-%m-%d"),
                "TestStart": test_start.strftime("%Y-%m-%d"),
                "TestEnd": test_end.strftime("%Y-%m-%d"),
                "Obs": int(len(sim_test)),
                "RebalanceCount": rebalance_count,
                "RiskOnPct": risk_on_pct,
                "InvestedPct": invested_pct,
                "AvgHoldings": avg_holdings,
                "TurnoverAvg": turnover_avg,
                "MaxNameWeight": max_name_weight,
                "MaxSectorWeight": max_sector_weight,
                "EffectiveHoldingsAvg": effective_holdings_avg,
                "EffectiveCostBpsAvg": effective_cost_bps_avg_win,
                "LeverageAvg": leverage_avg_win,
                "LeverageMax": leverage_max_win,
                "VolTargetingEnabled": bool(target_vol > 0.0),
                "RawVol": raw_vol_win,
                "FinalRealizedVol": final_realized_vol_win,
                "regime_pct_bull": regime_pct_bull_win,
                "regime_pct_bear_or_volatile": regime_pct_bear_win,
                "regime_last_label": regime_last_label_win,
                "BearExposureScale": float(bear_exposure_scale),
                "BearExposureScaleAvgBearPeriods": float(bear_scale_avg_bear_win),
                "PriceQualityBadTickers": int(q_summary.get("num_bad", 0)),
                "DroppedTickersCount": int(len(dropped_in_window)),
                "HistoricalMembershipPath": str(historical_membership_path),
                "SourceAvailableOnRebalanceMedian": source_median_win,
                "SourceAvailableOnRebalanceMin": source_min_win,
                "SourceAvailableOnRebalanceMax": source_max_win,
                "MembershipOnRebalanceMedian": membership_median_win,
                "MembershipOnRebalanceMin": membership_min_win,
                "MembershipOnRebalanceMax": membership_max_win,
                "EligibilityFilteredOnRebalanceMedian": eligibility_median_win,
                "EligibilityFilteredOnRebalanceMin": eligibility_min_r_win,
                "EligibilityFilteredOnRebalanceMax": eligibility_max_r_win,
                "FinalTradableOnRebalanceMedian": final_median_win,
                "FinalTradableOnRebalanceMin": final_min_win,
                "FinalTradableOnRebalanceMax": final_max_win,
                "EligibleTickersMedian": eligible_median_win,
                "EligibleTickersMin": eligible_min_win,
                "EligibleTickersMax": eligible_max_win,
                "EligibleOnRebalanceMedian": eligible_median_win,
                "EligibleOnRebalanceMin": eligible_min_win,
                "EligibleOnRebalanceMax": eligible_max_win,
                "LiquidityEligibleMedian": liquidity_eligible_median_win,
                "LiquidityEligibleMin": liquidity_eligible_min_win,
                "LiquidityEligibleMax": liquidity_eligible_max_win,
                "LiquidityFilteredOutMedian": liquidity_filtered_out_median_win,
                "LiquidityFilteredOutMax": liquidity_filtered_out_max_win,
                "UniverseAddedMedian": universe_added_median_win,
                "UniverseRemovedMedian": universe_removed_median_win,
                "UniverseMembershipPath": universe_dataset_path_used,
                "RebalanceSkippedCount": rebalance_skipped_count_win,
                "RebalanceSkipBelowMinCount": rebalance_skip_below_min_count_win,
                "RebalanceSkipZeroEligibleCount": rebalance_skip_zero_count_win,
                **w_metrics,
            }
        )
        windows_payload.append(
            {
                "window_id": i - 1,
                "train_start": w["TrainStart"].strftime("%Y-%m-%d"),
                "train_end": w["TrainEnd"].strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "CAGR": w_metrics.get("CAGR"),
                "Vol": w_metrics.get("Vol"),
                "Sharpe": w_metrics.get("Sharpe"),
                "MaxDD": w_metrics.get("MaxDD"),
                "EffectiveCostBpsAvg": effective_cost_bps_avg_win,
                "TurnoverAvg": turnover_avg,
                "LeverageAvg": leverage_avg_win,
                "LeverageMax": leverage_max_win,
                "VolTargetingEnabled": bool(target_vol > 0.0),
                "RawVol": raw_vol_win,
                "FinalRealizedVol": final_realized_vol_win,
                "regime_pct_bull": regime_pct_bull_win,
                "regime_pct_bear_or_volatile": regime_pct_bear_win,
                "regime_last_label": regime_last_label_win,
                "BearExposureScale": float(bear_exposure_scale),
                "BearExposureScaleAvgBearPeriods": float(bear_scale_avg_bear_win),
                "PriceQualityBadTickers": int(q_summary.get("num_bad", 0)),
                "DroppedTickersCount": int(len(dropped_in_window)),
                "HistoricalMembershipPath": str(historical_membership_path),
                "SourceAvailableOnRebalanceMedian": source_median_win,
                "SourceAvailableOnRebalanceMin": source_min_win,
                "SourceAvailableOnRebalanceMax": source_max_win,
                "MembershipOnRebalanceMedian": membership_median_win,
                "MembershipOnRebalanceMin": membership_min_win,
                "MembershipOnRebalanceMax": membership_max_win,
                "EligibilityFilteredOnRebalanceMedian": eligibility_median_win,
                "EligibilityFilteredOnRebalanceMin": eligibility_min_r_win,
                "EligibilityFilteredOnRebalanceMax": eligibility_max_r_win,
                "FinalTradableOnRebalanceMedian": final_median_win,
                "FinalTradableOnRebalanceMin": final_min_win,
                "FinalTradableOnRebalanceMax": final_max_win,
                "EligibleTickersMedian": eligible_median_win,
                "EligibleTickersMin": eligible_min_win,
                "EligibleTickersMax": eligible_max_win,
                "EligibleOnRebalanceMedian": eligible_median_win,
                "EligibleOnRebalanceMin": eligible_min_win,
                "EligibleOnRebalanceMax": eligible_max_win,
                "LiquidityEligibleMedian": liquidity_eligible_median_win,
                "LiquidityEligibleMin": liquidity_eligible_min_win,
                "LiquidityEligibleMax": liquidity_eligible_max_win,
                "LiquidityFilteredOutMedian": liquidity_filtered_out_median_win,
                "LiquidityFilteredOutMax": liquidity_filtered_out_max_win,
                "UniverseAddedMedian": universe_added_median_win,
                "UniverseRemovedMedian": universe_removed_median_win,
                "UniverseMembershipPath": universe_dataset_path_used,
                "RebalanceSkippedCount": rebalance_skipped_count_win,
                "RebalanceSkipBelowMinCount": rebalance_skip_below_min_count_win,
                "RebalanceSkipZeroEligibleCount": rebalance_skip_zero_count_win,
            }
        )
        print(
            f"[WF {i}/{len(windows)}] "
            f"train={w['TrainStart'].strftime('%Y-%m-%d')}..{w['TrainEnd'].strftime('%Y-%m-%d')} "
            f"test={test_start.strftime('%Y-%m-%d')}..{test_end.strftime('%Y-%m-%d')} "
            f"rebal={rebalance_count} "
            f"risk_on={risk_on_pct * 100.0:.1f}% "
            f"invested={invested_pct * 100.0:.1f}% "
            f"hold={avg_holdings:.2f} "
            f"turnover={turnover_avg:.4f} "
            f"sharpe={w_metrics.get('Sharpe')} maxdd={w_metrics.get('MaxDD')}"
        )
        cols = [
            c
            for c in [
                "DailyReturn",
                "RawDailyReturn",
                "VolTargetScaleApplied",
                "VolTargetScaleUnlagged",
                "PortRealizedVolAnnRaw",
                "PortRealizedVolAnnVT",
                "Turnover",
                "EffectiveCostBps",
            ]
            if c in sim_test.columns
        ]
        oos_parts.append(sim_test[cols])
        leverage_parts.append(leverage.loc[test_mask].rename("Leverage"))

    if not oos_parts:
        raise ValueError("No OOS results produced for any walk-forward window.")

    oos = pd.concat(oos_parts, axis=0).sort_index()
    oos = oos.loc[~oos.index.duplicated(keep="first")]
    lev_oos = pd.concat(leverage_parts, axis=0).sort_index()
    lev_oos = lev_oos.loc[~lev_oos.index.duplicated(keep="first")]
    if regime_parts:
        regime_oos = pd.concat(regime_parts, axis=0).sort_index()
        regime_oos = regime_oos.loc[~regime_oos.index.duplicated(keep="first")]
        valid = regime_oos.notna()
        if bool(valid.any()):
            regime_pct_bull = float((regime_oos[valid] == "bull").mean())
            regime_pct_bear_or_volatile = float((regime_oos[valid] == "bear_or_volatile").mean())
        else:
            regime_pct_bull = 0.0
            regime_pct_bear_or_volatile = 0.0
        non_na_labels = regime_oos.dropna()
        regime_last_label = str(non_na_labels.iloc[-1]) if not non_na_labels.empty else None
    else:
        regime_pct_bull = float("nan")
        regime_pct_bear_or_volatile = float("nan")
        regime_last_label = None
    oos_equity = (1.0 + oos["DailyReturn"].fillna(0.0)).cumprod()
    oos_frame = pd.DataFrame(
        {"Equity": oos_equity, "DailyReturn": oos["DailyReturn"], "Turnover": oos["Turnover"]}
    )
    if "RawDailyReturn" in oos.columns:
        oos_frame["RawDailyReturn"] = oos["RawDailyReturn"]
    if "VolTargetScaleApplied" in oos.columns:
        oos_frame["VolTargetScaleApplied"] = oos["VolTargetScaleApplied"]
    if "VolTargetScaleUnlagged" in oos.columns:
        oos_frame["VolTargetScaleUnlagged"] = oos["VolTargetScaleUnlagged"]
    if "PortRealizedVolAnnRaw" in oos.columns:
        oos_frame["PortRealizedVolAnnRaw"] = oos["PortRealizedVolAnnRaw"]
    if "PortRealizedVolAnnVT" in oos.columns:
        oos_frame["PortRealizedVolAnnVT"] = oos["PortRealizedVolAnnVT"]
    if "EffectiveCostBps" in oos.columns:
        oos_frame["EffectiveCostBps"] = oos["EffectiveCostBps"]
    metrics = compute_metrics(oos_frame["DailyReturn"])
    effective_cost_bps_avg = (
        float(oos_frame["EffectiveCostBps"].mean())
        if "EffectiveCostBps" in oos_frame.columns
        else float(costs_bps + slippage_bps)
    )
    leverage_avg = float(lev_oos.mean()) if not lev_oos.empty else 1.0
    leverage_max = float(lev_oos.max()) if not lev_oos.empty else 1.0
    quality_bad_total = (
        int(pd.concat(quality_parts, axis=0, ignore_index=True)["is_bad"].sum())
        if quality_parts
        else 0
    )
    if eligibility_counts_all:
        elig_all = pd.concat(eligibility_counts_all, axis=0).astype(int)
        eligible_median_all = float(elig_all.median()) if not elig_all.empty else 0.0
        eligible_min_all = int(elig_all.min()) if not elig_all.empty else 0
        eligible_max_all = int(elig_all.max()) if not elig_all.empty else 0
    else:
        elig_all = pd.Series(dtype=int)
        eligible_median_all = float("nan")
        eligible_min_all = 0
        eligible_max_all = 0
    source_all = pd.concat(source_counts_all, axis=0).astype(int) if source_counts_all else pd.Series(dtype=int)
    membership_all = (
        pd.concat(membership_counts_all, axis=0).astype(int) if membership_counts_all else pd.Series(dtype=int)
    )
    final_tradable_all = (
        pd.concat(final_tradable_counts_all, axis=0).astype(int)
        if final_tradable_counts_all
        else pd.Series(dtype=int)
    )
    source_median_all, source_min_all, source_max_all = _stats_min_med_max(source_all)
    membership_median_all, membership_min_all, membership_max_all = _stats_min_med_max(membership_all)
    eligibility_median_all, eligibility_min_r_all, eligibility_max_r_all = _stats_min_med_max(elig_all)
    final_median_all, final_min_all, final_max_all = _stats_min_med_max(final_tradable_all)
    liquidity_counts_all = (
        pd.concat(liquidity_eligible_counts_all, axis=0).astype(int)
        if liquidity_eligible_counts_all
        else pd.Series(dtype=int)
    )
    liquidity_filtered_out_all = (
        pd.concat(liquidity_filtered_out_counts_all, axis=0).astype(int)
        if liquidity_filtered_out_counts_all
        else pd.Series(dtype=int)
    )
    liquidity_eligible_median_all = (
        float(liquidity_counts_all.median()) if not liquidity_counts_all.empty else float("nan")
    )
    liquidity_eligible_min_all = int(liquidity_counts_all.min()) if not liquidity_counts_all.empty else 0
    liquidity_eligible_max_all = int(liquidity_counts_all.max()) if not liquidity_counts_all.empty else 0
    liquidity_filtered_out_median_all = (
        float(liquidity_filtered_out_all.median()) if not liquidity_filtered_out_all.empty else float("nan")
    )
    liquidity_filtered_out_max_all = (
        int(liquidity_filtered_out_all.max()) if not liquidity_filtered_out_all.empty else 0
    )
    bear_scale_avg_bear_all = (
        float(pd.Series(bear_scale_avg_bear_windows, dtype=float).mean())
        if bear_scale_avg_bear_windows
        else 1.0
    )
    universe_added_median_all = (
        float(pd.Series(universe_added_medians_all, dtype=float).median())
        if universe_added_medians_all
        else float("nan")
    )
    universe_removed_median_all = (
        float(pd.Series(universe_removed_medians_all, dtype=float).median())
        if universe_removed_medians_all
        else float("nan")
    )

    run_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    outdir = Path("results") / run_tag
    outdir.mkdir(parents=True, exist_ok=True)
    if uds_mode == "build" and universe_membership_full is not None and bool(universe_dataset_save):
        build_target = str(outdir)
        if universe_membership_summary_full is None:
            universe_membership_summary_full = summarize_universe_membership(universe_membership_full)
        universe_dataset_save_paths = save_universe_dataset(
            membership=universe_membership_full,
            summary=universe_membership_summary_full,
            outdir_or_path=build_target,
        )
        universe_dataset_saved = True
        universe_dataset_path_used = universe_dataset_save_paths.get("membership_path", "")
        universe_dataset_membership_path = universe_dataset_save_paths.get("membership_path", "")
        universe_dataset_summary_path = universe_dataset_save_paths.get("summary_path", "")
    if save_artifacts and uds_mode in {"build", "use"} and universe_membership_full is not None:
        if universe_membership_summary_full is None:
            universe_membership_summary_full = summarize_universe_membership(universe_membership_full)
        out_paths = save_universe_dataset(
            membership=universe_membership_full,
            summary=universe_membership_summary_full,
            outdir_or_path=str(outdir),
        )
        universe_dataset_save_paths = out_paths
        universe_dataset_saved = True
        if not universe_dataset_path_used:
            universe_dataset_path_used = out_paths.get("membership_path", "")
        if not universe_dataset_membership_path:
            universe_dataset_membership_path = out_paths.get("membership_path", "")
        if not universe_dataset_summary_path:
            universe_dataset_summary_path = out_paths.get("summary_path", "")
    universe_dataset_start_date = (
        str(pd.Timestamp(universe_membership_full.index.min()).date())
        if universe_membership_full is not None and not universe_membership_full.empty
        else ""
    )
    universe_dataset_end_date = (
        str(pd.Timestamp(universe_membership_full.index.max()).date())
        if universe_membership_full is not None and not universe_membership_full.empty
        else ""
    )
    universe_dataset_rows = (
        int(len(universe_membership_full)) if universe_membership_full is not None else 0
    )

    run_config = {
        "mode": "walkforward",
        "GitCommit": _git_commit(),
        "start": start,
        "end": end,
        "train_years": int(train_years),
        "test_years": int(test_years),
        "num_windows": int(len(window_rows)),
        "max_tickers": max_tickers,
        "requested_tickers_count": requested_tickers_count,
        "requested_tickers_sample": tickers[:10],
        "fetch_request_tickers_count": int(len(fetch_request_tickers)),
        "fetch_request_tickers_sample": fetch_request_tickers[:10],
        "fetched_keys_count": fetched_keys_count,
        "fetched_keys_sample": fetched_keys_sample,
        "post_fetch_filtered_keys_count": post_fetch_filtered_keys_count,
        "post_fetch_filtered_keys_sample": post_fetch_filtered_keys_sample,
        "loaded_tickers_count": loaded_tickers_count,
        "loaded_tickers_sample": used_tickers[:10],
        "missing_tickers_count": missing_tickers_count,
        "missing_tickers_sample": missing_tickers_sample,
        "rejected_tickers_count": rejected_tickers_count,
        "rejected_tickers_sample": rejected_tickers_sample,
        "ohlcv_map_key_sample": ohlcv_map_key_sample,
        "historical_membership_path": str(historical_membership_path),
        "membership_covered_tickers_count": membership_covered_tickers_count,
        "membership_covered_tickers_sample": membership_covered_tickers_sample,
        "selected_tickers": used_tickers,
        "top_n": top_n,
        "rebalance": rebalance,
        "costs_bps": costs_bps,
        "FactorName": factor_name_legacy,
        "FactorParams": factor_params_legacy,
        "FactorNames": factor_names_str,
        "FactorWeights": factor_weights_str,
        "Normalize": normalize,
        "WinsorP": float(winsor_p),
        "UseFactorNormalization": bool(use_factor_normalization),
        "UseSectorNeutralization": bool(use_sector_neutralization),
        "UseSizeNeutralization": bool(use_size_neutralization),
        "DynamicFactorWeights": bool(dynamic_factor_weights),
        "OrthogonalizeFactors": bool(orthogonalize_factors),
        "SectorNeutral": bool(sector_neutral),
        "Weighting": weighting,
        "VolLookback": int(vol_lookback),
        "MaxWeight": float(max_weight),
        "ScoreClip": float(score_clip),
        "ScoreFloor": float(score_floor),
        "SectorCap": float(sector_cap),
        "SectorMap": sector_map,
        "TargetVol": float(target_vol),
        "PortVolLookback": int(port_vol_lookback),
        "MaxLeverage": float(max_leverage),
        "SlippageBps": float(slippage_bps),
        "SlippageVolMult": float(slippage_vol_mult),
        "SlippageVolLookback": int(slippage_vol_lookback),
        "ExecutionDelayDays": int(execution_delay_days),
        "RegimeFilter": bool(regime_filter),
        "RegimeTrendSMA": int(regime_trend_sma),
        "RegimeVolLookback": int(regime_vol_lookback),
        "RegimeVolMedianLookback": int(regime_vol_median_lookback),
        "RegimeBullWeights": weight_map_to_json(bull_map),
        "RegimeBearWeights": weight_map_to_json(bear_map),
        "BearExposureScale": float(bear_exposure_scale),
        "BearExposureScaleAvgBearPeriods": float(bear_scale_avg_bear_all),
        "trend_filter": bool(trend_filter),
        "trend_sma_window": int(trend_sma_window),
        "DataSource": str(data_source),
        "DataCacheDir": str(data_cache_dir),
        "DataRoot": str(data_cache_dir),
        "DataRefresh": bool(data_refresh),
        "DataBulkPrepare": bool(data_bulk_prepare),
        "DataCachedFilesUsed": int(data_source_summary.get("cached_files_used", 0)),
        "DataFetchedOrRefreshed": int(data_source_summary.get("fetched_or_refreshed", 0)),
        "SaveArtifacts": bool(save_artifacts),
        "PriceQualityCheck": bool(price_quality_check),
        "PriceQualityMode": str(price_quality_mode),
        "PriceQualityZeroRetThresh": float(price_quality_zero_ret_thresh),
        "PriceQualityMinValidFrac": float(price_quality_min_valid_frac),
        "PriceQualityMaxBadTickers": int(price_quality_max_bad_tickers),
        "PriceQualityReportTopK": int(price_quality_report_topk),
        "PriceFillMode": str(price_fill_mode),
        "DropBadTickers": bool(drop_bad_tickers),
        "DropBadTickersScope": str(drop_bad_tickers_scope),
        "DropBadTickersMaxDrop": int(drop_bad_tickers_max_drop),
        "DropBadTickersExempt": str(drop_bad_tickers_exempt),
        "DroppedTickersCount": int(dropped_tickers_total),
        "UniverseMode": str(universe_mode),
        "UniverseMinHistoryDays": int(universe_min_history_days),
        "UniverseEffectiveMinHistoryDays": int(effective_min_history_days),
        "UniverseMinValidFrac": float(universe_min_valid_frac),
        "UniverseValidLookback": int(universe_valid_lookback),
        "UniverseMinPrice": float(universe_min_price),
        "MinPrice": float(min_price),
        "MinAvgDollarVolume": float(min_avg_dollar_volume),
        "LiquidityLookback": int(liquidity_lookback),
        "UniverseMinTickers": int(universe_min_tickers),
        "UniverseSkipBelowMinTickers": bool(universe_skip_below_min_tickers),
        "UniverseEligibilitySource": str(universe_eligibility_source),
        "UniverseExempt": str(universe_exempt),
        "UniverseDatasetMode": str(universe_dataset_mode),
        "UniverseDatasetFreq": str(universe_dataset_freq),
        "UniverseDatasetPath": str(universe_dataset_path_used),
        "UniverseDatasetMembershipPath": str(universe_dataset_membership_path),
        "UniverseDatasetSummaryPath": str(universe_dataset_summary_path),
        "UniverseDatasetSaved": bool(universe_dataset_saved),
        "UniverseDatasetFallback": bool(universe_dataset_fallback),
        "UniverseDatasetStartDate": universe_dataset_start_date,
        "UniverseDatasetEndDate": universe_dataset_end_date,
        "UniverseDatasetRows": universe_dataset_rows,
        "SourceAvailableOnRebalanceMedian": source_median_all,
        "SourceAvailableOnRebalanceMin": source_min_all,
        "SourceAvailableOnRebalanceMax": source_max_all,
        "MembershipOnRebalanceMedian": membership_median_all,
        "MembershipOnRebalanceMin": membership_min_all,
        "MembershipOnRebalanceMax": membership_max_all,
        "EligibilityFilteredOnRebalanceMedian": eligibility_median_all,
        "EligibilityFilteredOnRebalanceMin": eligibility_min_r_all,
        "EligibilityFilteredOnRebalanceMax": eligibility_max_r_all,
        "FinalTradableOnRebalanceMedian": final_median_all,
        "FinalTradableOnRebalanceMin": final_min_all,
        "FinalTradableOnRebalanceMax": final_max_all,
        "LiquidityEligibleMedian": liquidity_eligible_median_all,
        "LiquidityEligibleMin": liquidity_eligible_min_all,
        "LiquidityEligibleMax": liquidity_eligible_max_all,
        "LiquidityFilteredOutMedian": liquidity_filtered_out_median_all,
        "LiquidityFilteredOutMax": liquidity_filtered_out_max_all,
    }
    windows_json_path = str(outdir / "windows.csv") if save_artifacts else ""
    summary = {
        "RunTag": run_tag,
        "Mode": "walkforward",
        "Start": start,
        "End": end,
        "TrainYears": int(train_years),
        "TestYears": int(test_years),
        "NumWindows": int(len(window_rows)),
        "TickersUsed": len(used_tickers),
        "RequestedTickersCount": requested_tickers_count,
        "RequestedTickersSample": tickers[:10],
        "FetchRequestTickersCount": int(len(fetch_request_tickers)),
        "FetchRequestTickerSample": fetch_request_tickers[:10],
        "FetchedKeysCount": fetched_keys_count,
        "FetchedKeysSample": fetched_keys_sample,
        "PostFetchFilteredKeysCount": post_fetch_filtered_keys_count,
        "PostFetchFilteredKeysSample": post_fetch_filtered_keys_sample,
        "LoadedTickersCount": loaded_tickers_count,
        "LoadedTickersSample": used_tickers[:10],
        "MissingTickersCount": missing_tickers_count,
        "MissingTickersSample": missing_tickers_sample,
        "OHLCVMapKeySample": ohlcv_map_key_sample,
        "RejectedTickersCount": rejected_tickers_count,
        "RejectedTickersSample": rejected_tickers_sample,
        "HistoricalMembershipPath": str(historical_membership_path),
        "MembershipCoveredTickersCount": membership_covered_tickers_count,
        "MembershipCoveredTickersSample": membership_covered_tickers_sample,
        "SourceAvailableOnRebalanceMedian": source_median_all,
        "SourceAvailableOnRebalanceMin": source_min_all,
        "SourceAvailableOnRebalanceMax": source_max_all,
        "MembershipOnRebalanceMedian": membership_median_all,
        "MembershipOnRebalanceMin": membership_min_all,
        "MembershipOnRebalanceMax": membership_max_all,
        "EligibilityFilteredOnRebalanceMedian": eligibility_median_all,
        "EligibilityFilteredOnRebalanceMin": eligibility_min_r_all,
        "EligibilityFilteredOnRebalanceMax": eligibility_max_r_all,
        "FinalTradableOnRebalanceMedian": final_median_all,
        "FinalTradableOnRebalanceMin": final_min_all,
        "FinalTradableOnRebalanceMax": final_max_all,
        "TopN": top_n,
        "RankBuffer": int(rank_buffer),
        "VolatilityScaledWeights": bool(volatility_scaled_weights),
        "Rebalance": rebalance,
        "CostsBps": costs_bps,
        "FactorName": factor_name_legacy,
        "FactorParams": factor_params_legacy,
        "FactorNames": factor_names_str,
        "FactorWeights": factor_weights_str,
        "Normalize": normalize,
        "WinsorP": float(winsor_p),
        "UseFactorNormalization": bool(use_factor_normalization),
        "UseSectorNeutralization": bool(use_sector_neutralization),
        "UseSizeNeutralization": bool(use_size_neutralization),
        "DynamicFactorWeights": bool(dynamic_factor_weights),
        "OrthogonalizeFactors": bool(orthogonalize_factors),
        "SectorNeutral": bool(sector_neutral),
        "Weighting": weighting,
        "VolLookback": int(vol_lookback),
        "MaxWeight": float(max_weight),
        "ScoreClip": float(score_clip),
        "ScoreFloor": float(score_floor),
        "SectorCap": float(sector_cap),
        "SectorMap": sector_map,
        "TargetVol": float(target_vol),
        "PortVolLookback": int(port_vol_lookback),
        "MaxLeverage": float(max_leverage),
        "SlippageBps": float(slippage_bps),
        "SlippageVolMult": float(slippage_vol_mult),
        "SlippageVolLookback": int(slippage_vol_lookback),
        "ExecutionDelayDays": int(execution_delay_days),
        "RegimeFilter": bool(regime_filter),
        "RegimeTrendSMA": int(regime_trend_sma),
        "RegimeVolLookback": int(regime_vol_lookback),
        "RegimeVolMedianLookback": int(regime_vol_median_lookback),
        "RegimeBullWeights": weight_map_to_json(bull_map),
        "RegimeBearWeights": weight_map_to_json(bear_map),
        "BearExposureScale": float(bear_exposure_scale),
        "BearExposureScaleAvgBearPeriods": float(bear_scale_avg_bear_all),
        "regime_pct_bull": regime_pct_bull,
        "regime_pct_bear_or_volatile": regime_pct_bear_or_volatile,
        "regime_last_label": regime_last_label,
        "LeverageAvg": leverage_avg,
        "LeverageMax": leverage_max,
        "EffectiveCostBpsAvg": effective_cost_bps_avg,
        "TrendFilter": bool(trend_filter),
        "TrendSMA": int(trend_sma_window),
        "DataSource": str(data_source),
        "DataCacheDir": str(data_cache_dir),
        "DataRoot": str(data_cache_dir),
        "DataRefresh": bool(data_refresh),
        "DataBulkPrepare": bool(data_bulk_prepare),
        "DataCachedFilesUsed": int(data_source_summary.get("cached_files_used", 0)),
        "DataFetchedOrRefreshed": int(data_source_summary.get("fetched_or_refreshed", 0)),
        "SaveArtifacts": bool(save_artifacts),
        "PriceQualityCheck": bool(price_quality_check),
        "PriceQualityMode": str(price_quality_mode),
        "PriceQualityBadTickers": quality_bad_total,
        "PriceFillMode": str(price_fill_mode),
        "DropBadTickers": bool(drop_bad_tickers),
        "DropBadTickersScope": str(drop_bad_tickers_scope),
        "DroppedTickersCount": int(dropped_tickers_total),
        "UniverseMode": str(universe_mode),
        "UniverseEffectiveMinHistoryDays": int(effective_min_history_days),
        "UniverseEligibilitySource": str(universe_eligibility_source),
        "MinPrice": float(min_price),
        "MinAvgDollarVolume": float(min_avg_dollar_volume),
        "LiquidityLookback": int(liquidity_lookback),
        "UniverseDatasetMode": str(universe_dataset_mode),
        "UniverseDatasetFreq": str(universe_dataset_freq),
        "UniverseDatasetPath": str(universe_dataset_path_used),
        "UniverseDatasetMembershipPath": str(universe_dataset_membership_path),
        "UniverseDatasetSummaryPath": str(universe_dataset_summary_path),
        "UniverseDatasetSaved": bool(universe_dataset_saved),
        "UniverseDatasetFallback": bool(universe_dataset_fallback),
        "UniverseDatasetStartDate": universe_dataset_start_date,
        "UniverseDatasetEndDate": universe_dataset_end_date,
        "UniverseDatasetRows": universe_dataset_rows,
        "LiquidityEligibleMedian": liquidity_eligible_median_all,
        "LiquidityEligibleMin": liquidity_eligible_min_all,
        "LiquidityEligibleMax": liquidity_eligible_max_all,
        "LiquidityFilteredOutMedian": liquidity_filtered_out_median_all,
        "LiquidityFilteredOutMax": liquidity_filtered_out_max_all,
        "EligibleTickersMedian": eligible_median_all,
        "EligibleTickersMin": eligible_min_all,
        "EligibleTickersMax": eligible_max_all,
        "EligibleOnRebalanceMedian": eligible_median_all,
        "EligibleOnRebalanceMin": eligible_min_all,
        "EligibleOnRebalanceMax": eligible_max_all,
        "UniverseAddedMedian": universe_added_median_all,
        "UniverseRemovedMedian": universe_removed_median_all,
        "RebalanceSkipBelowMinCount": int(rebalance_skip_below_min_total),
        "RebalanceSkipZeroEligibleCount": int(rebalance_skip_zero_total),
        "Windows": windows_payload,
        "WindowsJsonPath": windows_json_path,
        "Outdir": str(outdir),
        **metrics,
    }

    _write_summary_artifacts(outdir=outdir, run_config=run_config, summary=summary)
    pd.DataFrame(window_rows).to_csv(outdir / "walkforward_summary.csv", index=False)
    oos_frame.to_csv(outdir / "equity_oos.csv", index_label="Date")
    if save_artifacts:
        write_equity_curve(
            outdir=outdir,
            equity=oos_frame["Equity"],
            returns=oos_frame["DailyReturn"],
            cumulative_return=oos_frame["Equity"] - 1.0,
        )
        if holdings_parts:
            holdings_oos = pd.concat(holdings_parts, axis=0).sort_index()
            holdings_oos = holdings_oos.loc[~holdings_oos.index.duplicated(keep="first")]
            write_holdings(outdir=outdir, holdings=holdings_oos)
        if regime_filter and regime_parts and regime_weight_parts:
            regime_oos_full = pd.concat(regime_parts, axis=0).sort_index()
            regime_oos_full = regime_oos_full.loc[~regime_oos_full.index.duplicated(keep="first")]
            fw_oos = pd.concat(regime_weight_parts, axis=0).sort_index()
            fw_oos = fw_oos.loc[~fw_oos.index.duplicated(keep="first")]
            write_regime(
                outdir=outdir,
                label=regime_oos_full,
                factor_weights=fw_oos,
                factor_names=resolved_factor_names,
            )
        if price_quality_check and quality_parts:
            quality_all = pd.concat(quality_parts, axis=0, ignore_index=True)
            write_price_quality(outdir=outdir, quality_df=quality_all, filename="price_quality.csv")
            write_price_quality(
                outdir=outdir,
                quality_df=quality_all.loc[quality_all["is_bad"]].copy(),
                filename="price_quality_bad.csv",
            )
            for j, q in enumerate(quality_parts):
                if q.empty:
                    continue
                write_price_quality(
                    outdir=outdir,
                    quality_df=q,
                    filename=f"price_quality_wf_window_{j}.csv",
                )
        if drop_bad_tickers:
            dropped_df = pd.DataFrame(dropped_rows_all)
            dropped_df.to_csv(outdir / "dropped_tickers.csv", index=False, float_format="%.10g")
            if not dropped_df.empty and "window_id" in dropped_df.columns:
                for window_id, group in dropped_df.groupby("window_id"):
                    group.to_csv(
                        outdir / f"dropped_tickers_wf_window_{int(window_id)}.csv",
                        index=False,
                        float_format="%.10g",
                    )
        if u_mode == "dynamic":
            if window_eligibility_parts:
                elig_df = pd.concat(window_eligibility_parts, axis=0, ignore_index=True)
                elig_df.to_csv(outdir / "universe_eligibility_summary.csv", index=False)
                for window_id, group in elig_df.groupby("window_id"):
                    group.to_csv(
                        outdir / f"universe_eligibility_wf_window_{int(window_id)}.csv",
                        index=False,
                    )
            else:
                pd.DataFrame(
                    columns=["window_id", "date", "eligible_count", "liquidity_eligible_count"]
                ).to_csv(
                    outdir / "universe_eligibility_summary.csv",
                    index=False,
                )
            if zero_eligible_debug_parts:
                dbg_all = pd.concat(zero_eligible_debug_parts, axis=0, ignore_index=True)
                for window_id, group in dbg_all.groupby("window_id"):
                    group.to_csv(
                        outdir / f"universe_zero_eligible_debug_wf_window_{int(window_id)}.csv",
                        index=False,
                        float_format="%.10g",
                    )
        with (outdir / "data_source_summary.json").open("w", encoding="utf-8") as f:
            json.dump(data_source_summary, f, indent=2, sort_keys=True)
        write_windows(outdir=outdir, windows=windows_payload)
    append_registry_row(
        {
            "RunTag": run_tag,
            "Mode": "walkforward",
            "GitCommit": _git_commit(),
            "Start": start,
            "End": end,
            "TrainYears": int(train_years),
            "TestYears": int(test_years),
            "NumWindows": int(len(window_rows)),
            "TickersUsed": len(used_tickers),
            "RequestedTickersCount": requested_tickers_count,
            "RequestedTickersSample": ";".join(tickers[:10]),
            "FetchRequestTickersCount": int(len(fetch_request_tickers)),
            "FetchRequestTickerSample": ";".join(fetch_request_tickers[:10]),
            "FetchedKeysCount": fetched_keys_count,
            "FetchedKeysSample": ";".join(fetched_keys_sample),
            "PostFetchFilteredKeysCount": post_fetch_filtered_keys_count,
            "PostFetchFilteredKeysSample": ";".join(post_fetch_filtered_keys_sample),
            "LoadedTickersCount": loaded_tickers_count,
            "LoadedTickersSample": ";".join(used_tickers[:10]),
            "MissingTickersCount": missing_tickers_count,
            "MissingTickersSample": ";".join(missing_tickers_sample),
            "OHLCVMapKeySample": ";".join(ohlcv_map_key_sample),
            "RejectedTickersCount": rejected_tickers_count,
            "RejectedTickersSample": ";".join(rejected_tickers_sample),
            "HistoricalMembershipPath": str(historical_membership_path),
            "MembershipCoveredTickersCount": membership_covered_tickers_count,
            "MembershipCoveredTickersSample": ";".join(membership_covered_tickers_sample),
            "SourceAvailableOnRebalanceMedian": source_median_all,
            "SourceAvailableOnRebalanceMin": source_min_all,
            "SourceAvailableOnRebalanceMax": source_max_all,
            "MembershipOnRebalanceMedian": membership_median_all,
            "MembershipOnRebalanceMin": membership_min_all,
            "MembershipOnRebalanceMax": membership_max_all,
            "EligibilityFilteredOnRebalanceMedian": eligibility_median_all,
            "EligibilityFilteredOnRebalanceMin": eligibility_min_r_all,
            "EligibilityFilteredOnRebalanceMax": eligibility_max_r_all,
            "FinalTradableOnRebalanceMedian": final_median_all,
            "FinalTradableOnRebalanceMin": final_min_all,
            "FinalTradableOnRebalanceMax": final_max_all,
            "TopN": top_n,
            "RankBuffer": int(rank_buffer),
            "VolatilityScaledWeights": bool(volatility_scaled_weights),
            "Rebalance": rebalance,
            "CostsBps": costs_bps,
            "FactorName": factor_name_legacy,
            "FactorParams": factor_params_legacy,
            "FactorNames": factor_names_str,
            "FactorWeights": factor_weights_str,
            "Normalize": normalize,
            "WinsorP": float(winsor_p),
            "UseFactorNormalization": bool(use_factor_normalization),
            "UseSectorNeutralization": bool(use_sector_neutralization),
            "UseSizeNeutralization": bool(use_size_neutralization),
            "DynamicFactorWeights": bool(dynamic_factor_weights),
            "OrthogonalizeFactors": bool(orthogonalize_factors),
            "SectorNeutral": bool(sector_neutral),
            "Weighting": weighting,
            "VolLookback": int(vol_lookback),
            "MaxWeight": float(max_weight),
            "ScoreClip": float(score_clip),
            "ScoreFloor": float(score_floor),
            "SectorCap": float(sector_cap),
            "SectorMap": sector_map,
            "TargetVol": float(target_vol),
            "PortVolLookback": int(port_vol_lookback),
            "MaxLeverage": float(max_leverage),
            "SlippageBps": float(slippage_bps),
            "SlippageVolMult": float(slippage_vol_mult),
            "SlippageVolLookback": int(slippage_vol_lookback),
            "ExecutionDelayDays": int(execution_delay_days),
            "RegimeFilter": bool(regime_filter),
            "RegimeTrendSMA": int(regime_trend_sma),
            "RegimeVolLookback": int(regime_vol_lookback),
            "RegimeVolMedianLookback": int(regime_vol_median_lookback),
            "RegimeBullWeights": weight_map_to_json(bull_map),
            "RegimeBearWeights": weight_map_to_json(bear_map),
            "BearExposureScale": float(bear_exposure_scale),
            "BearExposureScaleAvgBearPeriods": float(bear_scale_avg_bear_all),
            "regime_pct_bull": regime_pct_bull,
            "regime_pct_bear_or_volatile": regime_pct_bear_or_volatile,
            "regime_last_label": regime_last_label,
            "LeverageAvg": leverage_avg,
            "LeverageMax": leverage_max,
            "EffectiveCostBpsAvg": effective_cost_bps_avg,
            "TrendFilter": bool(trend_filter),
            "DataSource": str(data_source),
            "DataCacheDir": str(data_cache_dir),
            "DataRoot": str(data_cache_dir),
            "DataRefresh": bool(data_refresh),
            "DataBulkPrepare": bool(data_bulk_prepare),
            "DataCachedFilesUsed": int(data_source_summary.get("cached_files_used", 0)),
            "DataFetchedOrRefreshed": int(data_source_summary.get("fetched_or_refreshed", 0)),
            "SaveArtifacts": bool(save_artifacts),
            "PriceQualityCheck": bool(price_quality_check),
            "PriceQualityMode": str(price_quality_mode),
            "PriceQualityBadTickers": quality_bad_total,
            "PriceFillMode": str(price_fill_mode),
            "DropBadTickers": bool(drop_bad_tickers),
            "DropBadTickersScope": str(drop_bad_tickers_scope),
            "DroppedTickersCount": int(dropped_tickers_total),
            "UniverseMode": str(universe_mode),
            "UniverseEffectiveMinHistoryDays": int(effective_min_history_days),
            "UniverseEligibilitySource": str(universe_eligibility_source),
            "MinPrice": float(min_price),
            "MinAvgDollarVolume": float(min_avg_dollar_volume),
            "LiquidityLookback": int(liquidity_lookback),
            "UniverseDatasetMode": str(universe_dataset_mode),
            "UniverseDatasetFreq": str(universe_dataset_freq),
            "UniverseDatasetPath": str(universe_dataset_path_used),
            "UniverseDatasetMembershipPath": str(universe_dataset_membership_path),
            "UniverseDatasetSummaryPath": str(universe_dataset_summary_path),
            "UniverseDatasetSaved": bool(universe_dataset_saved),
            "UniverseDatasetFallback": bool(universe_dataset_fallback),
            "UniverseDatasetStartDate": universe_dataset_start_date,
            "UniverseDatasetEndDate": universe_dataset_end_date,
            "UniverseDatasetRows": universe_dataset_rows,
            "LiquidityEligibleMedian": liquidity_eligible_median_all,
            "LiquidityEligibleMin": liquidity_eligible_min_all,
            "LiquidityEligibleMax": liquidity_eligible_max_all,
            "LiquidityFilteredOutMedian": liquidity_filtered_out_median_all,
            "LiquidityFilteredOutMax": liquidity_filtered_out_max_all,
            "EligibleTickersMedian": eligible_median_all,
            "EligibleTickersMin": eligible_min_all,
            "EligibleTickersMax": eligible_max_all,
            "EligibleOnRebalanceMedian": eligible_median_all,
            "EligibleOnRebalanceMin": eligible_min_all,
            "EligibleOnRebalanceMax": eligible_max_all,
            "UniverseAddedMedian": universe_added_median_all,
            "UniverseRemovedMedian": universe_removed_median_all,
            "RebalanceSkipBelowMinCount": int(rebalance_skip_below_min_total),
            "RebalanceSkipZeroEligibleCount": int(rebalance_skip_zero_total),
            "WindowsJsonPath": windows_json_path,
            "CAGR": metrics.get("CAGR"),
            "Vol": metrics.get("Vol"),
            "Sharpe": metrics.get("Sharpe"),
            "MaxDD": metrics.get("MaxDD"),
            "Outdir": str(outdir),
            "TimestampUTC": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )

    if bool(print_run_summary_flag):
        print_run_summary(summary)
    return summary, str(outdir)
