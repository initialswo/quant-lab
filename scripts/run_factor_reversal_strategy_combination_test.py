"""Diversification test: Benchmark v1.1 vs short-term reversal strategy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest


OUTDIR = Path("results/factor_reversal_strategy_combination_test")

BENCHMARK_V11: dict[str, Any] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
    "universe": "sp500",
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "factor_name": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
    "factor_names": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
    "factor_weights": None,
    "dynamic_factor_weights": True,
    "factor_aggregation_method": "geometric_rank",
    "top_n": 50,
    "rank_buffer": 20,
    "rebalance": "monthly",
    "execution_delay_days": 0,
    "target_vol": 0.14,
    "port_vol_lookback": 20,
    "max_leverage": 1.5,
    "regime_filter": False,
    "regime_benchmark": "SPY",
    "bear_exposure_scale": 1.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
    "costs_bps": 10.0,
    "save_artifacts": True,
}

REVERSAL_CANDIDATE: dict[str, Any] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
    "universe": "sp500",
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "factor_name": ["reversal_1m"],
    "factor_names": ["reversal_1m"],
    "factor_weights": [1.0],
    "dynamic_factor_weights": False,
    "factor_aggregation_method": "geometric_rank",
    "top_n": 50,
    "rank_buffer": 20,
    "rebalance": "weekly",
    "execution_delay_days": 0,
    "target_vol": 0.14,
    "port_vol_lookback": 20,
    "max_leverage": 1.5,
    "regime_filter": False,
    "regime_benchmark": "SPY",
    "bear_exposure_scale": 1.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
    "costs_bps": 10.0,
    "save_artifacts": True,
}


def _read_returns(path: Path, col_name: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])[["date", "returns"]].set_index("date").sort_index()
    s = pd.to_numeric(df["returns"], errors="coerce").rename(col_name)
    return s


def _drawdown(ret: pd.Series) -> pd.Series:
    eq = (1.0 + ret.astype(float).fillna(0.0)).cumprod()
    return eq / eq.cummax() - 1.0


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Running Benchmark v1.1 backtest...")
    _, bench_outdir = run_backtest(**BENCHMARK_V11)
    print("Running reversal candidate backtest...")
    _, rev_outdir = run_backtest(**REVERSAL_CANDIDATE)

    bench_path = Path(bench_outdir) / "equity_curve.csv"
    rev_path = Path(rev_outdir) / "equity_curve.csv"
    if not bench_path.exists():
        raise FileNotFoundError(f"Missing benchmark equity curve: {bench_path}")
    if not rev_path.exists():
        raise FileNotFoundError(f"Missing reversal equity curve: {rev_path}")

    aligned = pd.concat(
        [_read_returns(bench_path, "benchmark_return"), _read_returns(rev_path, "reversal_return")],
        axis=1,
        join="inner",
    ).dropna()

    full_corr = float(aligned["benchmark_return"].corr(aligned["reversal_return"]))
    rolling_corr = aligned["benchmark_return"].rolling(252).corr(aligned["reversal_return"])

    dd_bench = _drawdown(aligned["benchmark_return"])
    dd_rev = _drawdown(aligned["reversal_return"])
    in_dd_bench = dd_bench < 0.0
    in_dd_rev = dd_rev < 0.0
    in_dd_both = in_dd_bench & in_dd_rev
    in_dd_either = in_dd_bench | in_dd_rev
    dd_overlap_all_days = float(in_dd_both.mean())
    dd_overlap_when_either = (
        float(in_dd_both.sum() / in_dd_either.sum()) if bool(in_dd_either.any()) else float("nan")
    )

    blends: list[tuple[str, float, float]] = [
        ("100% Benchmark", 1.0, 0.0),
        ("100% Reversal", 0.0, 1.0),
        ("75/25", 0.75, 0.25),
        ("50/50", 0.50, 0.50),
        ("25/75", 0.25, 0.75),
    ]

    combined = aligned.copy()
    portfolio_rows: list[dict[str, float | str]] = []
    for name, w_bench, w_rev in blends:
        col = f"portfolio_{name.replace('%', 'pct').replace('/', '_').replace(' ', '').lower()}"
        combined[col] = w_bench * combined["benchmark_return"] + w_rev * combined["reversal_return"]
        m = compute_metrics(combined[col])
        portfolio_rows.append(
            {
                "Portfolio": name,
                "WeightBenchmark": w_bench,
                "WeightReversal": w_rev,
                "CAGR": float(m["CAGR"]),
                "Vol": float(m["Vol"]),
                "Sharpe": float(m["Sharpe"]),
                "MaxDD": float(m["MaxDD"]),
            }
        )

    corr_row = {
        "full_period_corr": full_corr,
        "rolling_corr_252_mean": float(rolling_corr.mean()),
        "rolling_corr_252_median": float(rolling_corr.median()),
        "rolling_corr_252_min": float(rolling_corr.min()),
        "rolling_corr_252_max": float(rolling_corr.max()),
        "drawdown_overlap_all_days": dd_overlap_all_days,
        "drawdown_overlap_conditional": dd_overlap_when_either,
        "benchmark_equity_curve": str(bench_path),
        "reversal_equity_curve": str(rev_path),
    }

    combined.assign(
        drawdown_benchmark=dd_bench,
        drawdown_reversal=dd_rev,
        rolling_corr_252=rolling_corr,
        in_drawdown_benchmark=in_dd_bench.astype(int),
        in_drawdown_reversal=in_dd_rev.astype(int),
        in_drawdown_both=in_dd_both.astype(int),
    ).reset_index().to_csv(OUTDIR / "combined_returns.csv", index=False, float_format="%.10g")

    pd.DataFrame([corr_row]).to_csv(OUTDIR / "strategy_correlation.csv", index=False, float_format="%.10g")
    portfolio_df = pd.DataFrame(portfolio_rows)
    portfolio_df.to_csv(OUTDIR / "portfolio_summary.csv", index=False, float_format="%.10g")

    payload = {
        "benchmark_config": BENCHMARK_V11,
        "reversal_config": REVERSAL_CANDIDATE,
        "correlation": corr_row,
        "portfolio_summary": portfolio_rows,
    }
    (OUTDIR / "combination_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("FACTOR REVERSAL STRATEGY COMBINATION TEST")
    print("-----------------------------------------")
    print(pd.DataFrame([corr_row]).to_string(index=False))
    print("")
    print(portfolio_df.to_string(index=False))
    print("")
    print(f"Saved: {OUTDIR / 'combined_returns.csv'}")
    print(f"Saved: {OUTDIR / 'strategy_correlation.csv'}")
    print(f"Saved: {OUTDIR / 'portfolio_summary.csv'}")
    print(f"Saved: {OUTDIR / 'combination_summary.json'}")


if __name__ == "__main__":
    main()
