"""Walk-forward factor weight optimization using train Sharpe selection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_lab.engine.runner import run_backtest


FACTOR_NAMES = ["momentum_12_1", "reversal_1m", "low_vol_20"]
WEIGHT_GRID = [
    (0.6, 0.2, 0.2),
    (0.5, 0.3, 0.2),
    (0.7, 0.2, 0.1),
    (0.4, 0.4, 0.2),
    (0.5, 0.2, 0.3),
]
WF_WINDOWS = [
    {
        "train_start": "2005-01-01",
        "train_end": "2014-12-31",
        "test_start": "2015-01-01",
        "test_end": "2019-12-31",
    },
    {
        "train_start": "2010-01-01",
        "train_end": "2019-12-31",
        "test_start": "2020-01-01",
        "test_end": "2024-12-31",
    },
]


def _run_one(start: str, end: str, weights: tuple[float, float, float]) -> dict:
    summary, _ = run_backtest(
        start=start,
        end=end,
        max_tickers=2000,
        top_n=50,
        rebalance="monthly",
        costs_bps=10.0,
        data_source="parquet",
        data_cache_dir="data/equities",
        factor_name=FACTOR_NAMES,
        factor_names=FACTOR_NAMES,
        factor_weights=list(weights),
        target_vol=0.15,
        port_vol_lookback=20,
        max_leverage=1.5,
    )
    return summary


def _weight_label(weights: tuple[float, float, float]) -> str:
    return f"({weights[0]:.1f},{weights[1]:.1f},{weights[2]:.1f})"


def main() -> None:
    rows: list[dict] = []
    for win in WF_WINDOWS:
        train_scores: list[tuple[tuple[float, float, float], float]] = []
        for w in WEIGHT_GRID:
            s_train = _run_one(start=win["train_start"], end=win["train_end"], weights=w)
            train_scores.append((w, float(s_train.get("Sharpe", float("-inf")))))
        best_weights, best_train_sharpe = max(
            train_scores,
            key=lambda x: float("-inf") if pd.isna(x[1]) else x[1],
        )

        s_test = _run_one(start=win["test_start"], end=win["test_end"], weights=best_weights)
        rows.append(
            {
                "TrainStart": win["train_start"],
                "TrainEnd": win["train_end"],
                "TestStart": win["test_start"],
                "TestEnd": win["test_end"],
                "BestWeights": _weight_label(best_weights),
                "BestWeightsRaw": ",".join(str(x) for x in best_weights),
                "TrainSharpe": best_train_sharpe,
                "TestSharpe": float(s_test.get("Sharpe", float("nan"))),
                "TestCAGR": float(s_test.get("CAGR", float("nan"))),
                "TestMaxDD": float(s_test.get("MaxDD", float("nan"))),
                "TestOutdir": str(s_test.get("Outdir", "")),
            }
        )

    df = pd.DataFrame(rows)
    print("## WALK-FORWARD WEIGHT OPTIMIZATION")
    print("")
    print("| Train Period | Test Period | Best Weights | Test Sharpe | Test CAGR | Test MaxDD |")
    print("|---|---|---|---:|---:|---:|")
    for _, r in df.iterrows():
        train_period = f"{r['TrainStart']} -> {r['TrainEnd']}"
        test_period = f"{r['TestStart']} -> {r['TestEnd']}"
        print(
            f"| {train_period} | {test_period} | {r['BestWeights']} | "
            f"{r['TestSharpe']:.4f} | {r['TestCAGR']:.4f} | {r['TestMaxDD']:.4f} |"
        )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / f"walkforward_weight_optimization_{ts}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.10g")
    print("")
    print(f"Saved CSV: {out}")


if __name__ == "__main__":
    main()

