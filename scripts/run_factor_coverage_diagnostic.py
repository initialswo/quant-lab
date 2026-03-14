"""Measure raw factor coverage through time for the six-factor liquid_us research set."""

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
from quant_lab.universe.liquid_us import build_liquid_us_universe


RESULTS_ROOT = Path("results") / "factor_coverage_diagnostic"
START = "2000-01-01"
END = "2024-12-31"
UNIVERSE = "liquid_us"
UNIVERSE_MODE = "dynamic"
MAX_TICKERS = 300
DATA_ROOT = "data/equities"
FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
FUNDAMENTALS_FALLBACK_LAG_DAYS = 60
TOP_N = 50
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


def _coverage_by_date(
    close: pd.DataFrame,
    eligibility: pd.DataFrame,
    factor_scores: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    base_universe_count = close.notna().sum(axis=1).astype(int)
    eligible_count = eligibility.fillna(False).astype(bool).sum(axis=1).astype(int)
    rows: list[pd.DataFrame] = []

    for factor_name in TARGET_FACTORS:
        masked_scores = (
            factor_scores[factor_name]
            .reindex(index=eligibility.index, columns=eligibility.columns)
            .where(eligibility.fillna(False).astype(bool))
            .astype(float)
        )
        nonnull_count = masked_scores.notna().sum(axis=1).astype(int)
        factor_rows = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(close.index),
                "factor_name": factor_name,
                "base_universe_count": base_universe_count.to_numpy(dtype=int),
                "liquid_us_eligible_count": eligible_count.to_numpy(dtype=int),
                "factor_nonnull_count": nonnull_count.to_numpy(dtype=int),
                # Standalone sleeves rank exactly the finite raw scores that survive the dynamic eligibility mask.
                "final_tradable_count": nonnull_count.to_numpy(dtype=int),
            }
        )
        rows.append(factor_rows)

    out = pd.concat(rows, axis=0, ignore_index=True)
    out["year"] = pd.DatetimeIndex(out["date"]).year.astype(int)
    out["top50_degenerate_flag"] = out["final_tradable_count"].le(TOP_N)
    return out


def _coverage_by_year(coverage_by_date: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        coverage_by_date.groupby(["year", "factor_name"], observed=False)
        .agg(
            median_eligible=("liquid_us_eligible_count", "median"),
            median_nonnull=("factor_nonnull_count", "median"),
            median_final_tradable=("final_tradable_count", "median"),
            min_final_tradable=("final_tradable_count", "min"),
            max_final_tradable=("final_tradable_count", "max"),
        )
        .reset_index()
    )
    for col in [
        "median_eligible",
        "median_nonnull",
        "median_final_tradable",
        "min_final_tradable",
        "max_final_tradable",
    ]:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").round().astype(int)
    grouped["top50_degenerate_flag"] = grouped["median_final_tradable"].le(TOP_N)
    return grouped.sort_values(["factor_name", "year"], kind="mergesort").reset_index(drop=True)


def _summary_text(yearly_df: pd.DataFrame) -> str:
    constrained = yearly_df.sort_values(
        ["median_final_tradable", "min_final_tradable", "factor_name", "year"],
        kind="mergesort",
    ).head(12)
    bad = yearly_df.loc[yearly_df["top50_degenerate_flag"]].copy()

    lines = [
        "FACTOR COVERAGE SUMMARY",
        "=======================",
        "",
        "Most coverage-constrained factor-years:",
    ]
    for _, row in constrained.iterrows():
        lines.append(
            f"- {int(row['year'])} {row['factor_name']}: "
            f"median_final_tradable={int(row['median_final_tradable'])}, "
            f"min_final_tradable={int(row['min_final_tradable'])}, "
            f"median_nonnull={int(row['median_nonnull'])}, "
            f"median_eligible={int(row['median_eligible'])}"
        )
    lines.extend(["", "Top-50 degenerate factor-years:"])
    if bad.empty:
        lines.append("- None.")
    else:
        for _, row in bad.iterrows():
            lines.append(
                f"- {int(row['year'])} {row['factor_name']}: "
                f"median_final_tradable={int(row['median_final_tradable'])}"
            )
    return "\n".join(lines) + "\n"


def _build_manifest(outdir: Path, start: str, end: str, max_tickers: int) -> dict[str, Any]:
    return {
        "timestamp_utc": outdir.name,
        "results_dir": str(outdir),
        "date_range": {"start": start, "end": end},
        "universe": UNIVERSE,
        "universe_mode": UNIVERSE_MODE,
        "max_tickers": max_tickers,
        "top_n_reference": TOP_N,
        "data_root": DATA_ROOT,
        "fundamentals_path": FUNDAMENTALS_PATH,
        "fundamentals_fallback_lag_days": FUNDAMENTALS_FALLBACK_LAG_DAYS,
        "factors": TARGET_FACTORS,
        "notes": [
            "Coverage diagnostic uses direct OHLCV loading plus PIT-aligned fundamentals where required.",
            "Signals are masked to the liquid_us dynamic eligibility matrix before coverage counts are measured.",
            "final_tradable_count is measured from finite raw scores that survive the liquid_us eligibility mask.",
        ],
    }


def _print_console(yearly_df: pd.DataFrame) -> None:
    display = yearly_df[
        [
            "year",
            "factor_name",
            "median_eligible",
            "median_nonnull",
            "median_final_tradable",
            "top50_degenerate_flag",
        ]
    ].copy()
    print("FACTOR COVERAGE DIAGNOSTIC")
    print("--------------------------")
    print(display.to_string(index=False))
    print("")

    constrained = yearly_df.sort_values(
        ["median_final_tradable", "min_final_tradable", "factor_name", "year"],
        kind="mergesort",
    )
    worst = constrained.head(5)
    degenerate = yearly_df.loc[yearly_df["top50_degenerate_flag"]]

    print("INTERPRETATION")
    print("--------------")
    if not worst.empty:
        row = worst.iloc[0]
        print(
            "Most coverage-constrained factor-year: "
            f"{row['factor_name']} in {int(row['year'])} "
            f"(median_final_tradable={int(row['median_final_tradable'])})."
        )
    if degenerate.empty:
        print("No factor-year has median final tradable count at or below top_n=50.")
    else:
        years = ", ".join(
            f"{row.factor_name}:{int(row.year)}"
            for row in degenerate.sort_values(["factor_name", "year"], kind="mergesort").itertuples()
        )
        print(f"Problematic factor-years for top_n=50: {years}.")
        factors = sorted(set(str(x) for x in degenerate["factor_name"]))
        print(f"top_n=50 appears too large for: {', '.join(factors)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--max_tickers", type=int, default=MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    start = str(args.start)
    end = str(args.end)
    max_tickers = int(args.max_tickers)

    results_root = Path(str(args.results_root))
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / ts
    outdir.mkdir(parents=True, exist_ok=True)

    close, volume = _load_panels(start=start, end=end, max_tickers=max_tickers)
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

    coverage_by_date = _coverage_by_date(close=close, eligibility=eligibility, factor_scores=factor_scores)
    coverage_by_year = _coverage_by_year(coverage_by_date=coverage_by_date)
    summary_text = _summary_text(yearly_df=coverage_by_year)

    by_date_path = outdir / "factor_coverage_by_date.csv"
    by_year_path = outdir / "factor_coverage_by_year.csv"
    summary_path = outdir / "factor_coverage_summary.txt"
    manifest_path = outdir / "manifest.json"

    coverage_by_date.to_csv(by_date_path, index=False, float_format="%.10g")
    coverage_by_year.to_csv(by_year_path, index=False, float_format="%.10g")
    summary_path.write_text(summary_text, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                outdir=outdir,
                start=start,
                end=end,
                max_tickers=max_tickers,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    _copy_latest(
        files={
            "factor_coverage_by_date.csv": by_date_path,
            "factor_coverage_by_year.csv": by_year_path,
            "factor_coverage_summary.txt": summary_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    _print_console(yearly_df=coverage_by_year)
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
