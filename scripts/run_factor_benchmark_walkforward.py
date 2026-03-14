"""Walk-forward validation for the six-factor benchmark portfolio."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import run_composite_vs_sleeves as composite
import run_factor_benchmark_portfolio as benchmark


RESULTS_ROOT = Path("results") / "factor_benchmark_walkforward"
TRAIN_YEARS = 10
TEST_YEARS = 3
STEP_YEARS = 3


def _generate_windows(
    start: str,
    end: str,
    train_years: int,
    test_years: int,
    step_years: int,
) -> list[dict[str, str | int]]:
    start_ts = pd.Timestamp(str(start))
    end_ts = pd.Timestamp(str(end))
    windows: list[dict[str, str | int]] = []
    train_start = start_ts
    window_id = 1

    while True:
        train_end = train_start + pd.DateOffset(years=int(train_years)) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=int(test_years)) - pd.Timedelta(days=1)
        if test_start > end_ts or test_end > end_ts:
            break
        windows.append(
            {
                "window_id": int(window_id),
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
            }
        )
        train_start = train_start + pd.DateOffset(years=int(step_years))
        window_id += 1

    if not windows:
        raise ValueError("No valid walk-forward windows generated for the requested period.")
    return windows


def _test_slice(series: pd.Series, start: str, end: str) -> pd.Series:
    idx = pd.DatetimeIndex(series.index)
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return pd.to_numeric(series.loc[mask], errors="coerce").fillna(0.0).astype(float)


def _window_metrics(
    window_id: int,
    test_start: str,
    test_end: str,
    daily_return: pd.Series,
    daily_turnover: pd.Series,
) -> dict[str, float | int | str]:
    test_return = _test_slice(daily_return, start=test_start, end=test_end)
    test_turnover = _test_slice(daily_turnover, start=test_start, end=test_end)
    row = composite._portfolio_metrics(
        name=f"wf_{window_id}",
        daily_return=test_return,
        turnover=test_turnover,
    )
    return {
        "window_id": int(window_id),
        "train_start": "",
        "train_end": "",
        "test_start": str(test_start),
        "test_end": str(test_end),
        "CAGR": float(row["CAGR"]),
        "Vol": float(row["Vol"]),
        "Sharpe": float(row["Sharpe"]),
        "MaxDD": float(row["MaxDD"]),
        "Turnover": float(row["Turnover"]),
    }


def _build_manifest(
    outdir: Path,
    windows: list[dict[str, str | int]],
    max_tickers: int,
) -> dict[str, object]:
    return {
        "timestamp_utc": outdir.name,
        "results_dir": str(outdir),
        "date_range": {"start": benchmark.START, "end": benchmark.END},
        "walkforward": {
            "train_years": TRAIN_YEARS,
            "test_years": TEST_YEARS,
            "step_years": STEP_YEARS,
            "windows": windows,
        },
        "universe": "liquid_us",
        "universe_mode": "dynamic",
        "rebalance": "weekly",
        "top_n": 50,
        "weighting": "equal",
        "costs_bps": 10.0,
        "max_tickers": max_tickers,
        "factors": benchmark.FACTOR_NAMES,
        "portfolio_definition": {
            "name": "factor_benchmark_equal_weight",
            "weights": {factor: float(1.0 / len(benchmark.FACTOR_NAMES)) for factor in benchmark.FACTOR_NAMES},
            "construction": "return-level equal-weight sleeve combination",
        },
        "notes": [
            "Sleeves are rerun on each train+test span, but metrics are computed only on the test window.",
            "book_to_market and asset_growth reuse the benchmark helper for PIT-aligned factor params.",
            "This script does not use engine walkforward mode; it performs a transparent return-level walk-forward study.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=benchmark.START)
    parser.add_argument("--end", default=benchmark.END)
    parser.add_argument("--max_tickers", type=int, default=benchmark.MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    benchmark.START = str(args.start)
    benchmark.END = str(args.end)
    benchmark.MAX_TICKERS = int(args.max_tickers)

    results_root = Path(str(args.results_root))
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / ts
    outdir.mkdir(parents=True, exist_ok=True)

    windows = _generate_windows(
        start=benchmark.START,
        end=benchmark.END,
        train_years=TRAIN_YEARS,
        test_years=TEST_YEARS,
        step_years=STEP_YEARS,
    )

    run_cache: dict[str, object] = {}
    rows: list[dict[str, float | int | str]] = []
    daily_return_records: list[pd.DataFrame] = []

    for win in windows:
        train_start = str(win["train_start"])
        train_end = str(win["train_end"])
        test_start = str(win["test_start"])
        test_end = str(win["test_end"])
        window_id = int(win["window_id"])
        combined_end = test_end

        benchmark.START = train_start
        benchmark.END = combined_end

        special_factor_params = benchmark._build_special_factor_params(
            start=train_start,
            end=combined_end,
            max_tickers=benchmark.MAX_TICKERS,
        )
        sleeve_payloads, _ = benchmark._run_sleeves(special_factor_params=special_factor_params)
        benchmark_payload = benchmark._combined_payload(sleeve_payloads=sleeve_payloads)

        row = _window_metrics(
            window_id=window_id,
            test_start=test_start,
            test_end=test_end,
            daily_return=benchmark_payload["daily_return"],
            daily_turnover=benchmark_payload["daily_turnover"],
        )
        row["train_start"] = train_start
        row["train_end"] = train_end
        rows.append(row)

        test_return = _test_slice(benchmark_payload["daily_return"], start=test_start, end=test_end)
        daily_return_records.append(
            pd.DataFrame(
                {
                    "window_id": window_id,
                    "date": pd.DatetimeIndex(test_return.index),
                    "daily_return": test_return.to_numpy(dtype=float),
                }
            )
        )

    results_df = pd.DataFrame(rows).sort_values("window_id", kind="mergesort").reset_index(drop=True)
    daily_returns_df = pd.concat(daily_return_records, axis=0, ignore_index=True)

    results_path = outdir / "walkforward_results.csv"
    daily_returns_path = outdir / "daily_returns.csv"
    manifest_path = outdir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index=False, float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                outdir=outdir,
                windows=windows,
                max_tickers=benchmark.MAX_TICKERS,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    composite._copy_latest(
        files={
            "walkforward_results.csv": results_path,
            "daily_returns.csv": daily_returns_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    table = results_df[["window_id", "test_start", "test_end", "CAGR", "Vol", "Sharpe", "MaxDD"]].copy()
    table["test_period"] = table["test_start"].str.slice(0, 4) + "-" + table["test_end"].str.slice(0, 4)
    table = table[["window_id", "test_period", "CAGR", "Vol", "Sharpe", "MaxDD"]]
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD"]:
        table[col] = table[col].map(composite._format_float)

    sharpe = pd.to_numeric(results_df["Sharpe"], errors="coerce")
    print("FACTOR BENCHMARK WALK-FORWARD TEST")
    print("----------------------------------")
    print("")
    print(table.to_string(index=False))
    print("")
    print(f"Average test Sharpe: {float(sharpe.mean()):.4f}")
    print(f"Median test Sharpe: {float(sharpe.median()):.4f}")
    print(f"Worst test Sharpe: {float(sharpe.min()):.4f}")
    print(f"Best test Sharpe: {float(sharpe.max()):.4f}")
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
