"""Analyze sleeve-level factor performance by walk-forward regime/window."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import run_composite_vs_sleeves as composite
import run_factor_benchmark_portfolio as benchmark
import run_factor_benchmark_walkforward as benchmark_walkforward


RESULTS_ROOT = Path("results") / "factor_regime_analysis"
DISPLAY_FACTORS = [
    "momentum_12_1",
    "reversal_1m",
    "low_vol_20",
    "gross_profitability",
    "book_to_market",
    "asset_growth",
]


def _test_slice(series: pd.Series, start: str, end: str) -> pd.Series:
    idx = pd.DatetimeIndex(series.index)
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return pd.to_numeric(series.loc[mask], errors="coerce").fillna(0.0).astype(float)


def _metric_row(
    window_id: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    factor_name: str,
    daily_return: pd.Series,
) -> dict[str, float | int | str]:
    row = composite._portfolio_metrics(
        name=factor_name,
        daily_return=_test_slice(daily_return, start=test_start, end=test_end),
        turnover=pd.Series(0.0, index=_test_slice(daily_return, start=test_start, end=test_end).index, dtype=float),
    )
    return {
        "window_id": int(window_id),
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "factor_name": factor_name,
        "CAGR": float(row["CAGR"]),
        "Vol": float(row["Vol"]),
        "Sharpe": float(row["Sharpe"]),
        "MaxDD": float(row["MaxDD"]),
        "TotalReturn": float(row["TotalReturn"]),
    }


def _build_manifest(
    outdir: Path,
    windows: list[dict[str, str | int]],
    max_tickers: int,
    overall_start: str,
    overall_end: str,
) -> dict[str, object]:
    return {
        "timestamp_utc": outdir.name,
        "results_dir": str(outdir),
        "date_range": {"start": overall_start, "end": overall_end},
        "walkforward": {
            "train_years": benchmark_walkforward.TRAIN_YEARS,
            "test_years": benchmark_walkforward.TEST_YEARS,
            "step_years": benchmark_walkforward.STEP_YEARS,
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
        "notes": [
            "This is a sleeve-level regime analysis using the benchmark walk-forward windows.",
            "Sleeve metrics are computed strictly on each test window only.",
            "book_to_market and asset_growth reuse benchmark helper logic for PIT-aligned factor params.",
        ],
    }


def _print_window_tables(results_df: pd.DataFrame) -> None:
    for window_id in sorted(results_df["window_id"].unique()):
        sub = results_df.loc[results_df["window_id"].eq(window_id)].copy()
        test_period = f"{sub['test_start'].iloc[0][:4]}-{sub['test_end'].iloc[0][:4]}"
        table = sub[["window_id", "factor_name", "CAGR", "Vol", "Sharpe", "MaxDD"]].copy()
        table.insert(1, "test_period", test_period)
        for col in ["CAGR", "Vol", "Sharpe", "MaxDD"]:
            table[col] = table[col].map(composite._format_float)
        print(table.to_string(index=False))
        print("")


def _print_summary(rankings_df: pd.DataFrame) -> None:
    best = rankings_df.loc[rankings_df["sharpe_rank"].eq(1)].sort_values("window_id", kind="mergesort")
    worst = rankings_df.loc[rankings_df["sharpe_rank"].eq(rankings_df.groupby("window_id")["sharpe_rank"].transform("max"))]
    worst = worst.sort_values("window_id", kind="mergesort")

    print("SUMMARY")
    print("-------")
    print("Best factor by Sharpe in each regime:")
    for _, row in best.iterrows():
        print(f"  Window {int(row['window_id'])}: {row['factor_name']}")
    print("Worst factor by Sharpe in each regime:")
    for _, row in worst.iterrows():
        print(f"  Window {int(row['window_id'])}: {row['factor_name']}")

    reversal_best = int(best["factor_name"].eq("reversal_1m").sum())
    if reversal_best == int(best["window_id"].nunique()):
        print("Reversal dominates consistently: yes.")
    else:
        print(f"Reversal dominates consistently: no ({reversal_best}/{int(best['window_id'].nunique())} windows).")

    for factor_name, label in [
        ("momentum_12_1", "Momentum"),
        ("book_to_market", "Value"),
        ("gross_profitability", "Profitability"),
    ]:
        wins = best.loc[best["factor_name"].eq(factor_name), "window_id"].tolist()
        if wins:
            joined = ", ".join(str(int(x)) for x in wins)
            print(f"{label} strengthens in windows: {joined}.")
        else:
            print(f"{label} does not lead any walk-forward test window.")


def _warn_identical_series(window_id: int, factor_returns: dict[str, pd.Series]) -> None:
    names = list(factor_returns)
    for i, left in enumerate(names):
        left_series = pd.to_numeric(factor_returns[left], errors="coerce").fillna(0.0).astype(float)
        for right in names[i + 1 :]:
            right_series = pd.to_numeric(factor_returns[right], errors="coerce").fillna(0.0).astype(float)
            aligned = pd.concat([left_series.rename("left"), right_series.rename("right")], axis=1).fillna(0.0)
            if aligned["left"].equals(aligned["right"]):
                print(
                    f"WARNING: window_id={window_id} has identical daily return series for "
                    f"{left} and {right}."
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=benchmark.START)
    parser.add_argument("--end", default=benchmark.END)
    parser.add_argument("--max_tickers", type=int, default=benchmark.MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    overall_start = str(args.start)
    overall_end = str(args.end)
    benchmark.START = overall_start
    benchmark.END = overall_end
    benchmark.MAX_TICKERS = int(args.max_tickers)

    results_root = Path(str(args.results_root))
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / ts
    outdir.mkdir(parents=True, exist_ok=True)

    windows = benchmark_walkforward._generate_windows(
        start=benchmark.START,
        end=benchmark.END,
        train_years=benchmark_walkforward.TRAIN_YEARS,
        test_years=benchmark_walkforward.TEST_YEARS,
        step_years=benchmark_walkforward.STEP_YEARS,
    )

    result_rows: list[dict[str, float | int | str]] = []
    daily_return_rows: list[pd.DataFrame] = []

    for win in windows:
        window_id = int(win["window_id"])
        train_start = str(win["train_start"])
        train_end = str(win["train_end"])
        test_start = str(win["test_start"])
        test_end = str(win["test_end"])

        benchmark.START = train_start
        benchmark.END = test_end
        special_factor_params = benchmark._build_special_factor_params(
            start=train_start,
            end=test_end,
            max_tickers=benchmark.MAX_TICKERS,
        )
        sleeve_payloads, _ = benchmark._run_sleeves(special_factor_params=special_factor_params)
        by_factor = {
            spec["factor_name"]: next(p for p in sleeve_payloads if p["name"] == spec["strategy_name"])
            for spec in benchmark.SLEEVE_SPECS
        }
        factor_test_returns = {
            factor_name: _test_slice(by_factor[factor_name]["daily_return"], start=test_start, end=test_end)
            for factor_name in DISPLAY_FACTORS
        }
        _warn_identical_series(window_id=window_id, factor_returns=factor_test_returns)

        for factor_name in DISPLAY_FACTORS:
            payload = by_factor[factor_name]
            result_rows.append(
                _metric_row(
                    window_id=window_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    factor_name=factor_name,
                    daily_return=payload["daily_return"],
                )
            )
            test_return = _test_slice(payload["daily_return"], start=test_start, end=test_end)
            daily_return_rows.append(
                pd.DataFrame(
                    {
                        "window_id": window_id,
                        "train_start": train_start,
                        "train_end": train_end,
                        "test_start": test_start,
                        "test_end": test_end,
                        "factor_name": factor_name,
                        "date": pd.DatetimeIndex(test_return.index),
                        "daily_return": test_return.to_numpy(dtype=float),
                    }
                )
            )

    results_df = pd.DataFrame(result_rows).sort_values(
        ["window_id", "Sharpe", "CAGR"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    rankings_df = results_df[["window_id", "test_start", "test_end", "factor_name", "Sharpe", "CAGR"]].copy()
    rankings_df["test_period"] = rankings_df["test_start"].str.slice(0, 4) + "-" + rankings_df["test_end"].str.slice(0, 4)
    rankings_df["sharpe_rank"] = rankings_df.groupby("window_id")["Sharpe"].rank(
        method="dense",
        ascending=False,
    ).astype(int)
    rankings_df["cagr_rank"] = rankings_df.groupby("window_id")["CAGR"].rank(
        method="dense",
        ascending=False,
    ).astype(int)
    rankings_df = rankings_df[["window_id", "test_period", "factor_name", "sharpe_rank", "cagr_rank"]]
    rankings_df = rankings_df.sort_values(["window_id", "sharpe_rank", "cagr_rank", "factor_name"], kind="mergesort")

    daily_returns_df = pd.concat(daily_return_rows, axis=0, ignore_index=True)

    results_path = outdir / "factor_regime_results.csv"
    rankings_path = outdir / "factor_regime_rankings.csv"
    daily_returns_path = outdir / "daily_returns_by_window.csv"
    manifest_path = outdir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    rankings_df.to_csv(rankings_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index=False, float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                outdir=outdir,
                windows=windows,
                max_tickers=benchmark.MAX_TICKERS,
                overall_start=overall_start,
                overall_end=overall_end,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    composite._copy_latest(
        files={
            "factor_regime_results.csv": results_path,
            "factor_regime_rankings.csv": rankings_path,
            "daily_returns_by_window.csv": daily_returns_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    print("FACTOR REGIME ANALYSIS")
    print("----------------------")
    print("")
    _print_window_tables(results_df)
    _print_summary(rankings_df)
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
