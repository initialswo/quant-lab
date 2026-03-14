"""CLI utility: point-in-time check for fundamentals-based factor coverage."""

from __future__ import annotations

import argparse

import pandas as pd

from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel, load_fundamentals_file
from quant_lab.data.loaders import load_ohlcv_for_research
from quant_lab.factors.registry import compute_factor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check PIT alignment and coverage for gross_profitability.")
    p.add_argument("--fundamentals_path", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--data_root", default="data/equities")
    p.add_argument("--max_tickers", type=int, default=2000)
    p.add_argument("--fallback_lag_days", type=int, default=60)
    return p


def main() -> None:
    args = build_parser().parse_args()
    loader = load_ohlcv_for_research(
        start=str(args.start),
        end=str(args.end),
        max_tickers=int(args.max_tickers),
        store_root=str(args.data_root),
    )
    close = loader.panels.get("close", pd.DataFrame()).astype(float)
    if close.empty:
        raise ValueError("No close panel loaded; check data_root/start/end.")

    fundamentals = load_fundamentals_file(
        path=str(args.fundamentals_path),
        fallback_lag_days=int(args.fallback_lag_days),
    )
    aligned = align_fundamentals_to_daily_panel(
        fundamentals=fundamentals,
        price_index=pd.DatetimeIndex(close.index),
        price_columns=close.columns,
    )
    scores = compute_factor(
        "gross_profitability",
        close,
        fundamentals_aligned=aligned,
    )

    non_null = scores.notna()
    coverage_all = float(non_null.mean().mean()) if not scores.empty else 0.0
    ticker_coverage = non_null.mean(axis=0).sort_values(ascending=False)
    date_coverage = non_null.mean(axis=1)
    print("GROSS_PROFITABILITY COVERAGE")
    print("----------------------------")
    print(f"Rows (dates): {scores.shape[0]}")
    print(f"Columns (tickers): {scores.shape[1]}")
    print(f"Overall coverage: {coverage_all:.4%}")
    print(f"Date coverage median: {float(date_coverage.median()):.4%}")
    print(f"Date coverage min: {float(date_coverage.min()):.4%}")
    print(f"Date coverage max: {float(date_coverage.max()):.4%}")

    top_tickers = ticker_coverage.head(10)
    print("\nTop ticker coverage (10):")
    for t, v in top_tickers.items():
        print(f"{t}: {float(v):.4%}")

    sample = scores.stack(dropna=True).reset_index()
    sample.columns = ["Date", "Ticker", "gross_profitability"]
    print("\nSample non-null rows:")
    if sample.empty:
        print("(none)")
    else:
        print(sample.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
