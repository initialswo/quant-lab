"""Benchmark grid for Long/Short Equity v1 research sleeve."""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.loaders import load_ohlcv_for_research
from quant_lab.engine.runner import run_backtest
from quant_lab.research.long_short_equity import run_long_short_backtest


START = "2005-01-01"
END = "2024-12-31"
LONG_N_VALUES: list[int] = [25, 50]
SHORT_N_VALUES: list[int] = [5, 10, 15, 25, 50]
WEIGHTING_VALUES: list[str] = ["equal", "inv_vol"]
REBALANCE = "weekly"
COSTS_BPS = 10.0

BASE_FACTOR_RUN: dict[str, Any] = {
    "start": START,
    "end": END,
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
    "rebalance": REBALANCE,
    "execution_delay_days": 0,
    "target_vol": 0.0,
    "port_vol_lookback": 20,
    "max_leverage": 1.0,
    "regime_filter": False,
    "regime_benchmark": "SPY",
    "bear_exposure_scale": 1.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
    "costs_bps": COSTS_BPS,
    "save_artifacts": True,
}


def _variant_name(long_n: int, short_n: int, weighting: str) -> str:
    return f"long{long_n}_short{short_n}_{weighting}"


def _to_native_map(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if hasattr(v, "item"):
            try:
                out[str(k)] = v.item()
                continue
            except Exception:
                pass
        out[str(k)] = v
    return out


def _build_daily_score_panel(snapshot_csv: Path, close_index: pd.DatetimeIndex) -> pd.DataFrame:
    snap = pd.read_csv(snapshot_csv, parse_dates=["date"]).set_index("date").sort_index()
    cols = [str(c) for c in snap.columns]
    daily = pd.DataFrame(index=close_index, columns=cols, dtype=float)
    shared_idx = snap.index.intersection(close_index)
    daily.loc[shared_idx, cols] = snap.loc[shared_idx, cols].astype(float)
    return daily.astype(float)


def main() -> None:
    print("Running base factor backtest to generate composite score snapshot...")
    _, factor_outdir = run_backtest(**BASE_FACTOR_RUN)
    factor_dir = Path(factor_outdir)
    score_snapshot_path = factor_dir / "composite_scores_snapshot.csv"
    if not score_snapshot_path.exists():
        raise FileNotFoundError(f"Missing composite score snapshot: {score_snapshot_path}")

    data = load_ohlcv_for_research(
        start=START,
        end=END,
        universe="sp500",
        max_tickers=2000,
        store_root="data/equities",
    )
    close = data.panels["close"].astype(float).sort_index()
    scores_daily = _build_daily_score_panel(score_snapshot_path, pd.DatetimeIndex(close.index))
    common_cols = [c for c in scores_daily.columns if c in close.columns]
    close = close.reindex(columns=common_cols)
    scores_daily = scores_daily.reindex(columns=common_cols)

    rows: list[dict[str, Any]] = []
    best_key: tuple[int, int, str] | None = None
    best_sharpe = float("-inf")
    combos = list(itertools.product(LONG_N_VALUES, SHORT_N_VALUES, WEIGHTING_VALUES))
    for long_n, short_n, weighting in combos:
        sim, weights, summary = run_long_short_backtest(
            scores=scores_daily,
            close=close,
            long_n=int(long_n),
            short_n=int(short_n),
            rebalance=REBALANCE,
            weighting=str(weighting),
            vol_lookback=20,
            costs_bps=COSTS_BPS,
            slippage_bps=0.0,
            execution_delay_days=0,
            gross_exposure=1.0,
            net_exposure=0.0,
        )
        row = {
            "VariantName": _variant_name(int(long_n), int(short_n), str(weighting)),
            "long_n": int(long_n),
            "short_n": int(short_n),
            "LongShortRatio": float(long_n) / float(short_n),
            "IsAsymmetric": bool(int(short_n) < int(long_n)),
            "weighting": str(weighting),
            "CAGR": float(summary["CAGR"]),
            "Vol": float(summary["Vol"]),
            "Sharpe": float(summary["Sharpe"]),
            "MaxDD": float(summary["MaxDD"]),
            "AnnualTurnover": float(summary["AnnualTurnover"]),
            "AvgGrossExposure": float(summary["AvgGrossExposure"]),
            "AvgNetExposure": float(summary["AvgNetExposure"]),
            "_sim": sim,
            "_weights": weights,
        }
        rows.append(row)
        if float(row["Sharpe"]) > best_sharpe:
            best_sharpe = float(row["Sharpe"])
            best_key = (int(long_n), int(short_n), str(weighting))

    if not rows or best_key is None:
        raise RuntimeError("No long/short variants were evaluated.")

    grid = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    grid = grid.sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(drop=True)

    best_row = next(r for r in rows if (r["long_n"], r["short_n"], r["weighting"]) == best_key)
    best_sim = best_row["_sim"].copy()
    best_weights = best_row["_weights"].copy()

    best_returns = pd.DataFrame(
        {
            "date": best_sim.index,
            "equity": best_sim["Equity"].to_numpy(dtype=float),
            "returns": best_sim["DailyReturn"].to_numpy(dtype=float),
            "turnover": best_sim["Turnover"].to_numpy(dtype=float),
            "effective_cost_bps": best_sim["EffectiveCostBps"].to_numpy(dtype=float),
            "gross_exposure": best_weights.abs().sum(axis=1).to_numpy(dtype=float),
            "net_exposure": best_weights.sum(axis=1).to_numpy(dtype=float),
        }
    )
    best_weights_out = best_weights.reset_index().rename(columns={"index": "date"})

    best_by_sharpe = grid.sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").iloc[0]
    best_by_cagr = grid.sort_values(["CAGR", "Sharpe"], ascending=[False, False], kind="mergesort").iloc[0]
    best_by_maxdd = grid.sort_values(["MaxDD", "Sharpe"], ascending=[False, False], kind="mergesort").iloc[0]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path("results/long_short_equity_benchmark")
    outdir = root / ts
    outdir.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    grid_path = outdir / "long_short_grid.csv"
    returns_path = outdir / "best_variant_returns.csv"
    weights_path = outdir / "best_variant_weights.csv"
    summary_path = outdir / "best_variant_summary.json"

    grid.to_csv(grid_path, index=False, float_format="%.10g")
    best_returns.to_csv(returns_path, index=False, float_format="%.10g")
    best_weights_out.to_csv(weights_path, index=False, float_format="%.10g")

    summary = {
        "start": START,
        "end": END,
        "rebalance": REBALANCE,
        "costs_bps": COSTS_BPS,
        "factors": BASE_FACTOR_RUN["factor_names"],
        "factor_source_outdir": str(factor_dir),
        "score_snapshot_path": str(score_snapshot_path),
        "variant_count": int(len(grid)),
        "best_by_sharpe": _to_native_map(best_by_sharpe),
        "best_by_cagr": _to_native_map(best_by_cagr),
        "best_by_least_severe_drawdown": _to_native_map(best_by_maxdd),
        "artifacts": {
            "grid": str(grid_path),
            "best_returns": str(returns_path),
            "best_weights": str(weights_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    grid.to_csv(root / "long_short_grid.csv", index=False, float_format="%.10g")
    best_returns.to_csv(root / "best_variant_returns.csv", index=False, float_format="%.10g")
    best_weights_out.to_csv(root / "best_variant_weights.csv", index=False, float_format="%.10g")
    (root / "best_variant_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("LONG/SHORT EQUITY BENCHMARK")
    print("---------------------------")
    print("\nTop variants by Sharpe:")
    print(grid.head(10).to_string(index=False))
    print("\nBest by Sharpe:")
    print(pd.DataFrame([best_by_sharpe]).to_string(index=False))
    print("\nBest by CAGR:")
    print(pd.DataFrame([best_by_cagr]).to_string(index=False))
    print("\nBest by Least Severe Drawdown:")
    print(pd.DataFrame([best_by_maxdd]).to_string(index=False))
    print(f"\nSaved: {grid_path}")
    print(f"Saved: {returns_path}")
    print(f"Saved: {weights_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
