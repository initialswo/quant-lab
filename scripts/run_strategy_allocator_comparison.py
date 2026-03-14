"""Research comparison of static and dynamic strategy allocators."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.portfolio.allocator import (
    inverse_vol_allocator,
    simulate_allocator,
    smoothed_inverse_vol_allocator,
    static_weights,
)
from quant_lab.portfolio.strategy_panel import StrategyPanel


BENCHMARK_RETURNS_CSV: Path | None = None
CROSS_ASSET_RETURNS_CSV = Path("results/cross_asset_trend_v2_grid_latest/combined_returns.csv")


def _find_benchmark_v11_equity_curve() -> Path:
    candidates: list[tuple[float, Path]] = []
    for s in Path("results").glob("*/summary.json"):
        try:
            d = json.loads(s.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(d.get("FactorNames", "")) != "momentum_12_1;reversal_1m;low_vol_20;gross_profitability":
            continue
        if str(d.get("FactorAggregationMethod", "")).lower() != "geometric_rank":
            continue
        if str(d.get("PortfolioMode", "composite")) != "composite":
            continue
        if bool(d.get("DynamicFactorWeights", False)) is not True:
            continue
        if int(d.get("RankBuffer", -1)) != 20:
            continue
        if str(d.get("Rebalance", "")).lower() != "monthly":
            continue
        if float(d.get("TargetVol", 0.0)) != 0.14:
            continue
        if float(d.get("BearExposureScale", 0.0)) != 1.0:
            continue
        if str(d.get("Start", "")) != "2005-01-01" or str(d.get("End", "")) != "2024-12-31":
            continue
        p = s.parent / "equity_curve.csv"
        if p.exists():
            candidates.append((s.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError("Could not locate Benchmark v1.1 equity_curve.csv in results/*")
    candidates.sort()
    return candidates[-1][1]


def _summary_row(name: str, returns: pd.Series) -> dict[str, float | str]:
    m = compute_metrics(returns)
    return {
        "Allocator": name,
        "CAGR": float(m["CAGR"]),
        "Vol": float(m["Vol"]),
        "Sharpe": float(m["Sharpe"]),
        "MaxDD": float(m["MaxDD"]),
    }


def main() -> None:
    benchmark_path = BENCHMARK_RETURNS_CSV or _find_benchmark_v11_equity_curve()
    cross_asset_path = Path(CROSS_ASSET_RETURNS_CSV)
    if not cross_asset_path.exists():
        raise FileNotFoundError(f"Cross-asset returns file missing: {cross_asset_path}")

    panel = StrategyPanel.from_csv_files(
        {
            "benchmark": Path(benchmark_path),
            "cross_asset_v2": cross_asset_path,
        }
    )
    rets = panel.returns

    alloc_weights: dict[str, pd.DataFrame] = {
        "100% Benchmark": static_weights(rets, {"benchmark": 1.0, "cross_asset_v2": 0.0}),
        "80/20 Blend": static_weights(rets, {"benchmark": 0.8, "cross_asset_v2": 0.2}),
        "70/30 Blend": static_weights(rets, {"benchmark": 0.7, "cross_asset_v2": 0.3}),
        "60/40 Blend": static_weights(rets, {"benchmark": 0.6, "cross_asset_v2": 0.4}),
        "Inverse Vol": inverse_vol_allocator(rets, lookback=63),
        "Smoothed Inv Vol": smoothed_inverse_vol_allocator(rets, lookback=63, smoothing=0.2),
    }

    sim_map: dict[str, pd.DataFrame] = {name: simulate_allocator(rets, w) for name, w in alloc_weights.items()}

    summary_rows = [_summary_row(name, sim["returns"]) for name, sim in sim_map.items()]
    summary_df = pd.DataFrame(summary_rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)

    returns_cols = {name: sim["returns"] for name, sim in sim_map.items()}
    equity_cols = {f"{name} Equity": sim["equity"] for name, sim in sim_map.items()}
    turnover_cols = {name: sim["turnover"] for name, sim in sim_map.items()}

    allocator_returns_df = pd.concat([pd.DataFrame(returns_cols), pd.DataFrame(equity_cols)], axis=1)
    allocator_returns_df.index.name = "date"

    weight_records: list[pd.DataFrame] = []
    for alloc_name, w in alloc_weights.items():
        ww = w.copy()
        ww["allocator"] = alloc_name
        ww = ww.reset_index().rename(columns={"index": "date"})
        weight_records.append(ww)
    allocator_weights_df = pd.concat(weight_records, axis=0, ignore_index=True)
    cols = ["date", "allocator", *panel.strategies]
    allocator_weights_df = allocator_weights_df[cols]

    allocator_turnover_df = pd.DataFrame(turnover_cols)
    allocator_turnover_df.index.name = "date"

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = Path("results/strategy_allocator_test") / ts
    outdir.mkdir(parents=True, exist_ok=True)

    allocator_returns_df.reset_index().to_csv(outdir / "allocator_returns.csv", index=False, float_format="%.10g")
    allocator_weights_df.to_csv(outdir / "allocator_weights.csv", index=False, float_format="%.10g")
    allocator_turnover_df.reset_index().to_csv(outdir / "allocator_turnover.csv", index=False, float_format="%.10g")
    summary_df.to_csv(outdir / "allocator_summary.csv", index=False, float_format="%.10g")

    latest = Path("results/strategy_allocator_test")
    latest.mkdir(parents=True, exist_ok=True)
    allocator_returns_df.reset_index().to_csv(latest / "allocator_returns.csv", index=False, float_format="%.10g")
    allocator_weights_df.to_csv(latest / "allocator_weights.csv", index=False, float_format="%.10g")
    allocator_turnover_df.reset_index().to_csv(latest / "allocator_turnover.csv", index=False, float_format="%.10g")
    summary_df.to_csv(latest / "allocator_summary.csv", index=False, float_format="%.10g")

    print("STRATEGY ALLOCATOR COMPARISON")
    print("-----------------------------")
    print(summary_df.to_string(index=False))
    print("")
    print(f"Saved: {outdir / 'allocator_returns.csv'}")
    print(f"Saved: {outdir / 'allocator_weights.csv'}")
    print(f"Saved: {outdir / 'allocator_turnover.csv'}")
    print(f"Saved: {outdir / 'allocator_summary.csv'}")


if __name__ == "__main__":
    main()

