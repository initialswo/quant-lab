"""Run rank-decay / quantile testing for existing price factors."""

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
    _collect_close_series,
    _collect_numeric_panel,
    _factor_required_history_days,
    _load_universe_seed_tickers,
    _prepare_close_panel,
    _price_panel_health_report,
)
from quant_lab.factors.registry import compute_factors
from quant_lab.research.rank_decay import (
    SleeveBacktestResult,
    monotonicity_score_from_cagr,
    run_rank_decay_backtest,
)
from quant_lab.universe.liquid_us import build_liquid_us_universe


RESULTS_ROOT = Path("results") / "rank_decay"
DEFAULT_START = "2005-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_FACTORS = ["momentum_12_1", "reversal_1m", "low_vol_20"]
DEFAULT_QUANTILES = 5
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
PRICE_FILL_MODE = "ffill"
UNIVERSE_MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 10_000_000.0
LIQUIDITY_LOOKBACK = 20
UNIVERSE_MIN_HISTORY_DAYS = 252
RESULT_COLUMNS = [
    "factor",
    "sleeve",
    "start",
    "end",
    "universe",
    "rebalance",
    "quantiles",
    "CAGR",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "hit_rate",
    "avg_turnover",
    "annual_turnover",
    "n_rebalance_dates",
    "median_names",
]
SUMMARY_COLUMNS = [
    "factor",
    "q1_sharpe",
    "q2_sharpe",
    "q3_sharpe",
    "q4_sharpe",
    "q5_sharpe",
    "spread_sharpe",
    "q1_cagr",
    "q5_cagr",
    "spread_cagr",
    "monotonicity_score",
    "median_bucket_size",
    "notes",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--factors", default=",".join(DEFAULT_FACTORS))
    parser.add_argument("--quantiles", type=int, default=DEFAULT_QUANTILES)
    parser.add_argument("--rebalance", choices=["daily", "weekly", "biweekly", "monthly"], default=DEFAULT_REBALANCE)
    parser.add_argument("--costs_bps", type=float, default=DEFAULT_COSTS_BPS)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--max_tickers", type=int, default=0)
    parser.add_argument("--data_cache_dir", default=DATA_CACHE_DIR)
    return parser.parse_args()


def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)


def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def _load_price_panels(
    start: str,
    end: str,
    universe: str,
    data_cache_dir: str,
    max_tickers: int,
    factor_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tickers = _load_universe_seed_tickers(
        universe=universe,
        max_tickers=int(max_tickers) if int(max_tickers) > 0 else 10_000_000,
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
    close_cols, used_tickers, _, _, _ = _collect_close_series(ohlcv_map=ohlcv_map, requested_tickers=tickers)
    if not close_cols or not used_tickers:
        raise ValueError("No OHLCV close series were loaded for the requested run.")

    close_raw = pd.concat(close_cols, axis=1, join="outer")
    volume_raw = _collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Volume",
    )
    close = _prepare_close_panel(close_raw=close_raw, price_fill_mode=PRICE_FILL_MODE)
    min_non_nan = max(1, int(0.8 * close.shape[1]))
    close = close.dropna(thresh=min_non_nan)
    if close.empty:
        raise ValueError("No usable close-price history after alignment and fill.")

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
            f"count={len(broken_tickers)} sample={sorted(set(broken_tickers))[:10]}"
        )
        close = close.drop(columns=sorted(set(broken_tickers)), errors="ignore")

    volume = (
        volume_raw.reindex(index=close.index, columns=close.columns)
        if not volume_raw.empty
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    volume = volume.astype(float)
    data_source_summary = dict(data_source_summary)
    data_source_summary["tickers_requested"] = int(len(tickers))
    data_source_summary["tickers_loaded"] = int(close.shape[1])
    data_source_summary["factor_history_days"] = int(
        max(
            UNIVERSE_MIN_HISTORY_DAYS,
            _factor_required_history_days(factor_names=factor_names, factor_params_map={}),
        )
    )
    return close.astype(float), volume, data_source_summary


def _build_eligibility(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    factor_names: list[str],
    universe: str,
) -> pd.DataFrame:
    if str(universe).strip().lower() != "liquid_us":
        return close.notna().astype(bool)
    effective_min_history_days = max(
        UNIVERSE_MIN_HISTORY_DAYS,
        _factor_required_history_days(factor_names=factor_names, factor_params_map={}),
    )
    return build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=UNIVERSE_MIN_PRICE,
        min_avg_dollar_volume=MIN_AVG_DOLLAR_VOLUME,
        adv_window=LIQUIDITY_LOOKBACK,
        min_history=effective_min_history_days,
    )


def _spread_row(
    factor_name: str,
    spread: pd.Series,
    start: str,
    end: str,
    universe: str,
    rebalance: str,
    quantiles: int,
    median_names: float,
    n_rebalance_dates: int,
) -> dict[str, Any]:
    metrics = compute_metrics(spread)
    valid = pd.Series(spread, dtype=float).dropna()
    return {
        "factor": str(factor_name),
        "sleeve": "Spread",
        "start": str(start),
        "end": str(end),
        "universe": str(universe),
        "rebalance": str(rebalance),
        "quantiles": int(quantiles),
        "CAGR": float(metrics.get("CAGR", np.nan)),
        "ann_vol": float(metrics.get("Vol", np.nan)),
        "sharpe": float(metrics.get("Sharpe", np.nan)),
        "max_drawdown": float(metrics.get("MaxDD", np.nan)),
        "hit_rate": float((valid > 0.0).mean()) if not valid.empty else float("nan"),
        "avg_turnover": float("nan"),
        "annual_turnover": float("nan"),
        "n_rebalance_dates": int(n_rebalance_dates),
        "median_names": float(median_names),
    }


def _sleeve_row(
    factor_name: str,
    sleeve_result: SleeveBacktestResult,
    start: str,
    end: str,
    universe: str,
    rebalance: str,
    quantiles: int,
) -> dict[str, Any]:
    summary = sleeve_result.summary
    return {
        "factor": str(factor_name),
        "sleeve": str(sleeve_result.sleeve),
        "start": str(start),
        "end": str(end),
        "universe": str(universe),
        "rebalance": str(rebalance),
        "quantiles": int(quantiles),
        "CAGR": float(summary.get("CAGR", np.nan)),
        "ann_vol": float(summary.get("Vol", np.nan)),
        "sharpe": float(summary.get("Sharpe", np.nan)),
        "max_drawdown": float(summary.get("MaxDD", np.nan)),
        "hit_rate": float(summary.get("HitRate", np.nan)),
        "avg_turnover": float(summary.get("AvgTurnover", np.nan)),
        "annual_turnover": float(summary.get("AnnualTurnover", np.nan)),
        "n_rebalance_dates": int(summary.get("NRebalanceDates", 0.0)),
        "median_names": float(summary.get("MedianNames", np.nan)),
    }


def _value_for_sleeve(frame: pd.DataFrame, sleeve: str, column: str) -> float:
    vals = frame.loc[frame["sleeve"].eq(sleeve), column]
    if vals.empty:
        return float("nan")
    return float(vals.iloc[0])


def _summary_row(factor_name: str, factor_rows: pd.DataFrame, quantiles: int) -> dict[str, Any]:
    q_rows = factor_rows.loc[factor_rows["sleeve"].str.startswith("Q")].copy()
    q_rows["q_idx"] = q_rows["sleeve"].str.replace("Q", "", regex=False).astype(int)
    q_rows = q_rows.sort_values("q_idx", kind="mergesort")
    cagr_values = q_rows["CAGR"].astype(float).tolist()
    median_bucket_size = float(q_rows["median_names"].astype(float).median()) if not q_rows.empty else float("nan")
    top_label = f"Q{int(quantiles)}"
    out = {
        "factor": str(factor_name),
        "spread_sharpe": _value_for_sleeve(factor_rows, "Spread", "sharpe"),
        "q1_cagr": _value_for_sleeve(factor_rows, "Q1", "CAGR"),
        "q5_cagr": _value_for_sleeve(factor_rows, top_label, "CAGR"),
        "spread_cagr": _value_for_sleeve(factor_rows, "Spread", "CAGR"),
        "monotonicity_score": monotonicity_score_from_cagr(cagr_values),
        "median_bucket_size": median_bucket_size,
        "notes": "Spearman(bucket_index, bucket_CAGR) mapped from [-1,1] to [0,1].",
    }
    for q in range(1, 6):
        label = f"Q{q}"
        out[f"q{q}_sharpe"] = _value_for_sleeve(factor_rows, label, "sharpe")
    return {col: out.get(col, float("nan")) for col in SUMMARY_COLUMNS}


def _factor_daily_returns(
    factor_name: str,
    report: dict[str, Any],
    quantiles: int,
) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for q in range(1, int(quantiles) + 1):
        sleeve_result = report[f"Q{q}"]
        assert isinstance(sleeve_result, SleeveBacktestResult)
        data[f"Q{q}"] = pd.Series(sleeve_result.sim["DailyReturn"], dtype=float)
    data["Spread"] = pd.Series(report["Spread"], dtype=float)
    frame = pd.DataFrame(data).sort_index()
    frame.index.name = "date"
    frame.attrs["factor"] = str(factor_name)
    return frame


def _write_outputs(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    daily_paths: dict[str, str],
    manifest: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "rank_decay_results.csv"
    summary_path = output_dir / "rank_decay_summary.csv"
    manifest_path = output_dir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if output_dir.parent == RESULTS_ROOT:
        _copy_latest(
            files={
                "rank_decay_results.csv": results_path,
                "rank_decay_summary.csv": summary_path,
                "manifest.json": manifest_path,
            },
            latest_root=RESULTS_ROOT / "latest",
        )
        for factor_name, src in daily_paths.items():
            shutil.copy2(Path(src), RESULTS_ROOT / "latest" / Path(src).name)


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()
    factors = [x.strip() for x in str(args.factors).split(",") if x.strip()]
    if not factors:
        raise ValueError("At least one factor must be provided via --factors.")
    if int(args.quantiles) < 2:
        raise ValueError("--quantiles must be >= 2")

    close, volume, data_source_summary = _load_price_panels(
        start=str(args.start),
        end=str(args.end),
        universe=str(args.universe),
        data_cache_dir=str(args.data_cache_dir),
        max_tickers=int(args.max_tickers),
        factor_names=factors,
    )
    eligibility = _build_eligibility(
        close=close,
        volume=volume,
        factor_names=factors,
        universe=str(args.universe),
    )

    factor_scores = compute_factors(factor_names=factors, close=close, factor_params={})
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(str(args.output_dir)) if str(args.output_dir).strip() else RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    daily_paths: dict[str, str] = {}
    run_notes: list[str] = [
        "Quantile buckets are formed from worst to best score, so Q1 is the lowest-ranked sleeve and QN is the highest-ranked sleeve.",
        "Ties are broken deterministically by ticker symbol after sorting by factor score.",
        "If a rebalance date has fewer valid names than the requested quantile count, the rebalance is skipped and prior holdings are carried forward.",
        "Spread metrics are computed from the daily return spread series: QN daily return minus Q1 daily return.",
    ]

    print("")
    print("RANK DECAY SUMMARY")
    print("------------------")
    for factor_name in factors:
        masked_scores = apply_universe_filter_to_scores(
            factor_scores[factor_name],
            eligibility,
            exempt=set(),
        )
        report = run_rank_decay_backtest(
            scores=masked_scores,
            close=close,
            quantiles=int(args.quantiles),
            rebalance=str(args.rebalance),
            costs_bps=float(args.costs_bps),
            slippage_bps=0.0,
            execution_delay_days=0,
        )
        q1 = report["Q1"]
        qn = report[f"Q{int(args.quantiles)}"]
        assert isinstance(q1, SleeveBacktestResult)
        assert isinstance(qn, SleeveBacktestResult)
        median_names = float(q1.summary.get("MedianNames", np.nan))
        n_rebalance_dates = int(q1.summary.get("NRebalanceDates", 0.0))

        for q in range(1, int(args.quantiles) + 1):
            sleeve_result = report[f"Q{q}"]
            assert isinstance(sleeve_result, SleeveBacktestResult)
            rows.append(
                _sleeve_row(
                    factor_name=factor_name,
                    sleeve_result=sleeve_result,
                    start=str(args.start),
                    end=str(args.end),
                    universe=str(args.universe),
                    rebalance=str(args.rebalance),
                    quantiles=int(args.quantiles),
                )
            )
        rows.append(
            _spread_row(
                factor_name=factor_name,
                spread=pd.Series(report["Spread"], dtype=float),
                start=str(args.start),
                end=str(args.end),
                universe=str(args.universe),
                rebalance=str(args.rebalance),
                quantiles=int(args.quantiles),
                median_names=median_names,
                n_rebalance_dates=n_rebalance_dates,
            )
        )

        daily_df = _factor_daily_returns(factor_name=factor_name, report=report, quantiles=int(args.quantiles))
        daily_path = output_dir / f"{factor_name}_daily_returns.csv"
        daily_df.to_csv(daily_path, float_format="%.10g")
        daily_paths[factor_name] = str(daily_path)

        factor_result_df = pd.DataFrame([row for row in rows if row["factor"] == factor_name], columns=RESULT_COLUMNS)
        summary_rows.append(_summary_row(factor_name=factor_name, factor_rows=factor_result_df, quantiles=int(args.quantiles)))

        spread_row = factor_result_df.loc[factor_result_df["sleeve"].eq("Spread")].iloc[0]
        print(
            f"{factor_name:16s} "
            f"Q1 Sharpe={_format_float(float(factor_result_df.loc[factor_result_df['sleeve'].eq('Q1'), 'sharpe'].iloc[0]))} "
            f"Q{int(args.quantiles)} Sharpe={_format_float(float(factor_result_df.loc[factor_result_df['sleeve'].eq(f'Q{int(args.quantiles)}'), 'sharpe'].iloc[0]))} "
            f"Spread Sharpe={_format_float(float(spread_row['sharpe']))} "
            f"Spread CAGR={_format_float(float(spread_row['CAGR']))}"
        )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_df = results_df.sort_values(["factor", "sleeve"], kind="mergesort").reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_df = summary_df.sort_values(["spread_sharpe", "factor"], ascending=[False, True], kind="mergesort").reset_index(drop=True)

    runtime_seconds = time.perf_counter() - t0
    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_rank_decay.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": str(args.universe),
        "factors": factors,
        "quantiles": int(args.quantiles),
        "rebalance": str(args.rebalance),
        "costs_bps": float(args.costs_bps),
        "output_dir": str(output_dir),
        "output_paths": {
            "rank_decay_results": str(output_dir / "rank_decay_results.csv"),
            "rank_decay_summary": str(output_dir / "rank_decay_summary.csv"),
            "manifest": str(output_dir / "manifest.json"),
            "daily_returns": daily_paths,
        },
        "runtime_seconds": float(runtime_seconds),
        "data_source_summary": data_source_summary,
        "notes": run_notes,
        "monotonicity_definition": "Spearman correlation between bucket index [1..Q] and bucket CAGR, linearly mapped from [-1, 1] to [0, 1].",
    }
    _write_outputs(
        results_df=results_df,
        summary_df=summary_df,
        daily_paths=daily_paths,
        manifest=manifest,
        output_dir=output_dir,
    )

    print("")
    print(results_df[["factor", "sleeve", "CAGR", "sharpe", "median_names"]].to_string(index=False, float_format=_format_float))
    print("")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
