"""Compute a raw factor score correlation matrix for the current six-factor research set."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel, load_fundamentals_file
from quant_lab.engine import runner
from quant_lab.factors.registry import compute_factors
from quant_lab.research.signal_correlation import run_signal_correlation
from quant_lab.universe.liquid_us import build_liquid_us_universe


RESULTS_ROOT = Path("results") / "factor_score_correlation_matrix"
START = "2000-01-01"
END = "2024-12-31"
UNIVERSE = "liquid_us"
UNIVERSE_MODE = "dynamic"
MAX_TICKERS = 300
DATA_ROOT = "data/equities"
FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
FUNDAMENTALS_FALLBACK_LAG_DAYS = 60
CORRELATION_METHOD = "spearman"
TARGET_FACTORS: list[str] = [
    "momentum_12_1",
    "reversal_1m",
    "low_vol_20",
    "gross_profitability",
    "book_to_market",
    "asset_growth",
]


def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)


def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def _load_panels(start: str, end: str, max_tickers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = runner._load_universe_seed_tickers(
        universe=UNIVERSE,
        max_tickers=int(max_tickers),
        data_cache_dir=DATA_ROOT,
    )
    ohlcv_map, _ = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(start),
        end=str(end),
        cache_dir=DATA_ROOT,
        data_source="parquet",
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
    close = runner._prepare_close_panel(close_raw=close, price_fill_mode="ffill").astype(float).sort_index()
    volume = runner._collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Volume",
    ).reindex(index=close.index, columns=close.columns).astype(float)
    return close, volume


def _load_fundamentals_aligned(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fundamentals = load_fundamentals_file(
        path=FUNDAMENTALS_PATH,
        fallback_lag_days=int(FUNDAMENTALS_FALLBACK_LAG_DAYS),
    )
    return align_fundamentals_to_daily_panel(
        fundamentals=fundamentals,
        price_index=pd.DatetimeIndex(close.index),
        price_columns=close.columns,
    )


def _factor_params_map(fundamentals_aligned: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    return {
        "gross_profitability": {"fundamentals_aligned": fundamentals_aligned},
        "book_to_market": {"fundamentals_aligned": fundamentals_aligned},
        "asset_growth": {"fundamentals_aligned": fundamentals_aligned, "lag_days": 252},
    }


def _masked_signal_panels(
    factor_scores: dict[str, pd.DataFrame],
    eligibility: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    elig = eligibility.astype(bool)
    return {
        name: panel.reindex(index=elig.index, columns=elig.columns).where(elig).astype(float)
        for name, panel in factor_scores.items()
    }


def _pairwise_timeseries_frame(pairwise: dict[str, pd.Series]) -> pd.DataFrame:
    if not pairwise:
        return pd.DataFrame(columns=["date"])
    frame = pd.concat(pairwise, axis=1).sort_index()
    frame.index.name = "date"
    return frame.reset_index()


def _build_manifest(
    outdir: Path,
    method: str,
    max_tickers: int,
) -> dict[str, Any]:
    return {
        "timestamp_utc": outdir.name,
        "results_dir": str(outdir),
        "date_range": {"start": START, "end": END},
        "universe": UNIVERSE,
        "universe_mode": UNIVERSE_MODE,
        "max_tickers": max_tickers,
        "data_root": DATA_ROOT,
        "fundamentals_path": FUNDAMENTALS_PATH,
        "fundamentals_fallback_lag_days": FUNDAMENTALS_FALLBACK_LAG_DAYS,
        "correlation_method": method,
        "factors": TARGET_FACTORS,
        "notes": [
            "Correlation study uses raw factor score panels, not sleeve returns.",
            "Factors are computed on PIT-aligned fundamentals where required.",
            "Signals are masked to the liquid_us dynamic eligibility matrix before correlation analysis.",
            "Cross-sectional correlations are computed by date and then averaged across dates.",
        ],
    }


def _print_interpretation(report: dict[str, Any]) -> None:
    most = report.get("most_correlated_pairs", []) or []
    least = report.get("least_correlated_pairs", []) or []
    redundant = next((row for row in most if abs(float(row["mean_corr"])) >= 0.80), None)

    print("INTERPRETATION")
    print("--------------")
    if most:
        row = most[0]
        print(
            "Highly correlated pair: "
            f"{row['signal_a']} / {row['signal_b']} ({float(row['mean_corr']):.4f})."
        )
    else:
        print("Highly correlated pairs: none identified.")
    if least:
        row = least[0]
        print(
            "Low-correlation pair: "
            f"{row['signal_a']} / {row['signal_b']} ({float(row['mean_corr']):.4f})."
        )
    else:
        print("Low-correlation pairs: none identified.")
    if redundant is not None:
        print(
            "Potential redundancy: "
            f"{redundant['signal_a']} and {redundant['signal_b']} show very high average correlation."
        )
    else:
        print("No factor appears clearly redundant based on the average correlation matrix alone.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--max_tickers", type=int, default=MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    parser.add_argument("--method", choices=["spearman", "pearson"], default=CORRELATION_METHOD)
    args = parser.parse_args()

    results_root = Path(str(args.results_root))
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / ts
    outdir.mkdir(parents=True, exist_ok=True)

    close, volume = _load_panels(start=str(args.start), end=str(args.end), max_tickers=int(args.max_tickers))
    fundamentals_aligned = _load_fundamentals_aligned(close=close)
    factor_scores = compute_factors(
        factor_names=TARGET_FACTORS,
        close=close,
        factor_params=_factor_params_map(fundamentals_aligned=fundamentals_aligned),
    )
    eligibility = build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=5.0,
        min_avg_dollar_volume=10_000_000.0,
        adv_window=20,
        min_history=252,
    )
    signal_panels = _masked_signal_panels(factor_scores=factor_scores, eligibility=eligibility)
    report = run_signal_correlation(signal_panels=signal_panels, method=str(args.method))

    matrix_df = pd.DataFrame(report["average_correlation_matrix"]).reindex(index=TARGET_FACTORS, columns=TARGET_FACTORS)
    coverage_df = pd.DataFrame(report["coverage_summary"]).copy()
    timeseries_df = _pairwise_timeseries_frame(report.get("pairwise_correlation_by_date", {}))

    matrix_path = outdir / "factor_score_correlation_matrix.csv"
    coverage_path = outdir / "factor_score_correlation_coverage.csv"
    timeseries_path = outdir / "factor_score_correlation_timeseries.csv"
    manifest_path = outdir / "manifest.json"

    matrix_df.to_csv(matrix_path, float_format="%.10g")
    coverage_df.to_csv(coverage_path, index=False, float_format="%.10g")
    timeseries_df.to_csv(timeseries_path, index=False, float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                outdir=outdir,
                method=str(args.method),
                max_tickers=int(args.max_tickers),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    _copy_latest(
        files={
            "factor_score_correlation_matrix.csv": matrix_path,
            "factor_score_correlation_coverage.csv": coverage_path,
            "factor_score_correlation_timeseries.csv": timeseries_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    display = matrix_df.copy()
    for col in display.columns:
        display[col] = display[col].map(_format_float)
    print("FACTOR SCORE CORRELATION MATRIX")
    print("-------------------------------")
    print(display.to_string())
    print("")
    _print_interpretation(report)
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
