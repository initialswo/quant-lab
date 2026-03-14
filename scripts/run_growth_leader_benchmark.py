"""Benchmark grid for Growth Leader (IBD-style hypothesis) Equity Sleeve v1."""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel, load_fundamentals_file
from quant_lab.data.loaders import load_ohlcv_for_research
from quant_lab.research.growth_leader_equity import (
    apply_growth_screen,
    build_growth_scores,
    run_growth_leader_backtest,
)


START = "2005-01-01"
END = "2024-12-31"
TOP_N_VALUES: list[int] = [20, 30, 40]
REBALANCE_VALUES: list[str] = ["weekly"]
WEIGHTING_VALUES: list[str] = ["equal", "inv_vol"]
COSTS_BPS = 10.0
FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"


def _variant_name(top_n: int, rebalance: str, weighting: str) -> str:
    return f"top{top_n}_{rebalance}_{weighting}"


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


def main() -> None:
    data = load_ohlcv_for_research(
        start=START,
        end=END,
        universe="sp500",
        max_tickers=2000,
        store_root="data/equities",
    )
    close = data.panels["close"].astype(float).sort_index()
    volume = data.panels["volume"].astype(float).reindex(index=close.index, columns=close.columns)

    fundamentals = load_fundamentals_file(path=FUNDAMENTALS_PATH, fallback_lag_days=60)
    fundamentals_aligned = align_fundamentals_to_daily_panel(
        fundamentals=fundamentals,
        price_index=pd.DatetimeIndex(close.index),
        price_columns=close.columns,
    )

    screen = apply_growth_screen(
        fundamentals_aligned=fundamentals_aligned,
        prices=close,
        volume=volume,
        min_price=10.0,
        min_avg_dollar_volume=5_000_000.0,
        adv_lookback=20,
        require_positive_momentum=False,
    )
    scores = build_growth_scores(
        prices=close,
        fundamentals_aligned=fundamentals_aligned,
        screen_mask=screen,
    )

    rows: list[dict[str, Any]] = []
    best_key: tuple[int, str, str] | None = None
    best_sharpe = float("-inf")
    combos = list(itertools.product(TOP_N_VALUES, REBALANCE_VALUES, WEIGHTING_VALUES))
    for top_n, rebalance, weighting in combos:
        sim, weights, summary = run_growth_leader_backtest(
            score_panel=scores,
            close_prices=close,
            top_n=int(top_n),
            rebalance=str(rebalance),
            weighting=str(weighting),
            costs_bps=COSTS_BPS,
            vol_lookback=20,
        )
        row = {
            "VariantName": _variant_name(int(top_n), str(rebalance), str(weighting)),
            "top_n": int(top_n),
            "rebalance": str(rebalance),
            "weighting": str(weighting),
            "CAGR": float(summary["CAGR"]),
            "Vol": float(summary["Vol"]),
            "Sharpe": float(summary["Sharpe"]),
            "MaxDD": float(summary["MaxDD"]),
            "AnnualTurnover": float(summary["AnnualTurnover"]),
            "_sim": sim,
            "_weights": weights,
        }
        rows.append(row)
        if float(row["Sharpe"]) > best_sharpe:
            best_sharpe = float(row["Sharpe"])
            best_key = (int(top_n), str(rebalance), str(weighting))

    if not rows or best_key is None:
        raise RuntimeError("No growth leader variants were evaluated.")

    grid = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    grid = grid.sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(drop=True)

    best_row = next(r for r in rows if (r["top_n"], r["rebalance"], r["weighting"]) == best_key)
    best_sim = best_row["_sim"].copy()
    best_weights = best_row["_weights"].copy()

    best_returns = pd.DataFrame(
        {
            "date": best_sim.index,
            "equity": best_sim["Equity"].to_numpy(dtype=float),
            "returns": best_sim["DailyReturn"].to_numpy(dtype=float),
            "turnover": best_sim["Turnover"].to_numpy(dtype=float),
            "effective_cost_bps": best_sim["EffectiveCostBps"].to_numpy(dtype=float),
            "holdings_count": (best_weights > 0.0).sum(axis=1).to_numpy(dtype=int),
        }
    )
    best_weights_out = best_weights.reset_index().rename(columns={"index": "date"})

    best_by_sharpe = grid.sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").iloc[0]
    best_by_cagr = grid.sort_values(["CAGR", "Sharpe"], ascending=[False, False], kind="mergesort").iloc[0]
    best_by_maxdd = grid.sort_values(["MaxDD", "Sharpe"], ascending=[False, False], kind="mergesort").iloc[0]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path("results/growth_leader_benchmark")
    outdir = root / ts
    outdir.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    grid_path = outdir / "growth_leader_grid.csv"
    returns_path = outdir / "best_variant_returns.csv"
    weights_path = outdir / "best_variant_weights.csv"
    summary_path = outdir / "best_variant_summary.json"

    grid.to_csv(grid_path, index=False, float_format="%.10g")
    best_returns.to_csv(returns_path, index=False, float_format="%.10g")
    best_weights_out.to_csv(weights_path, index=False, float_format="%.10g")

    summary = {
        "start": START,
        "end": END,
        "costs_bps": COSTS_BPS,
        "factors": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
        "screen_rules": {
            "gross_profitability_positive": True,
            "earnings_yield_positive_if_available": True,
            "min_price": 10.0,
            "min_avg_dollar_volume": 5_000_000.0,
            "require_positive_momentum": False,
        },
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

    grid.to_csv(root / "growth_leader_grid.csv", index=False, float_format="%.10g")
    best_returns.to_csv(root / "best_variant_returns.csv", index=False, float_format="%.10g")
    best_weights_out.to_csv(root / "best_variant_weights.csv", index=False, float_format="%.10g")
    (root / "best_variant_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("GROWTH LEADER BENCHMARK")
    print("-----------------------")
    print("\nTop 10 variants by Sharpe:")
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
