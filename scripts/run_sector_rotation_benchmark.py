"""Grid benchmark for the Sector Rotation v1 research sleeve."""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.research.sector_rotation import (
    DEFAULT_SECTOR_UNIVERSE,
    build_sector_rotation_weights,
    compute_sector_momentum_signals,
    load_sector_prices,
    run_sector_rotation_backtest,
)


LOOKBACKS: list[int] = [63, 126, 252]
SIGNAL_TYPES: list[str] = ["relative", "absolute"]
TOP_N_VALUES: list[int] = [3, 4]
WEIGHTING_METHODS: list[str] = ["equal", "inv_vol"]
REBALANCE = "monthly"
COSTS_BPS = 5.0
START = "2005-01-01"
END = "2024-12-31"


def _variant_name(lookback: int, signal_type: str, top_n: int, weighting: str) -> str:
    return f"lb{lookback}_{signal_type}_top{top_n}_{weighting}"


def _best_variant_rows(grid: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    by_sharpe = grid.sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").iloc[0]
    by_cagr = grid.sort_values(["CAGR", "Sharpe"], ascending=[False, False], kind="mergesort").iloc[0]
    by_maxdd = grid.sort_values(["MaxDD", "Sharpe"], ascending=[False, False], kind="mergesort").iloc[0]
    return by_sharpe, by_cagr, by_maxdd


def _to_native(row: pd.Series) -> dict[str, Any]:
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
    close = load_sector_prices(
        universe=DEFAULT_SECTOR_UNIVERSE,
        data_roots=("data/sector_etfs", "data/cross_asset"),
        start=START,
        end=END,
    )
    if close.empty or close.shape[1] == 0:
        raise FileNotFoundError(
            "No sector ETF price files found. Run scripts/fetch_sector_etf_data.py first."
        )

    combos = list(itertools.product(LOOKBACKS, SIGNAL_TYPES, TOP_N_VALUES, WEIGHTING_METHODS))
    rows: list[dict[str, Any]] = []
    best_sharpe_key: tuple[int, str, int, str] | None = None
    best_sharpe = float("-inf")

    for lookback, signal_type, top_n, weighting in combos:
        sim, summary = run_sector_rotation_backtest(
            close=close,
            lookback=int(lookback),
            signal_type=str(signal_type),
            top_n=int(top_n),
            weighting=str(weighting),
            rebalance=REBALANCE,
            costs_bps=COSTS_BPS,
        )
        row = {
            "VariantName": _variant_name(int(lookback), str(signal_type), int(top_n), str(weighting)),
            "lookback": int(lookback),
            "signal_type": str(signal_type),
            "top_n": int(top_n),
            "weighting": str(weighting),
            "CAGR": float(summary["CAGR"]),
            "Vol": float(summary["Vol"]),
            "Sharpe": float(summary["Sharpe"]),
            "MaxDD": float(summary["MaxDD"]),
            "AnnualTurnover": float(summary["AnnualTurnover"]),
        }
        rows.append(row)
        if float(row["Sharpe"]) > best_sharpe:
            best_sharpe = float(row["Sharpe"])
            best_sharpe_key = (int(lookback), str(signal_type), int(top_n), str(weighting))

    if not rows or best_sharpe_key is None:
        raise RuntimeError("No sector rotation variants were evaluated.")

    grid = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(
        drop=True
    )
    best_by_sharpe, best_by_cagr, best_by_maxdd = _best_variant_rows(grid)

    lb, st, tn, wt = best_sharpe_key
    best_sim, _ = run_sector_rotation_backtest(
        close=close,
        lookback=lb,
        signal_type=st,
        top_n=tn,
        weighting=wt,
        rebalance=REBALANCE,
        costs_bps=COSTS_BPS,
    )
    best_scores = compute_sector_momentum_signals(close=close, lookback=lb, signal_type=st)
    best_weights = build_sector_rotation_weights(
        scores=best_scores,
        close=close,
        top_n=tn,
        rebalance=REBALANCE,
        weighting=wt,
        vol_lookback=20,
    )

    best_returns = pd.DataFrame(
        {
            "date": best_sim.index,
            "equity": best_sim["Equity"].to_numpy(dtype=float),
            "returns": best_sim["DailyReturn"].to_numpy(dtype=float),
            "turnover": best_sim["Turnover"].to_numpy(dtype=float),
            "assets_held": (best_weights > 0.0).sum(axis=1).to_numpy(dtype=int),
        }
    )
    best_weights_out = best_weights.reset_index().rename(columns={"index": "date"})

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path("results/sector_rotation_benchmark")
    outdir = root / ts
    outdir.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    grid_path = outdir / "sector_rotation_grid.csv"
    returns_path = outdir / "best_variant_returns.csv"
    weights_path = outdir / "best_variant_weights.csv"
    summary_path = outdir / "summary.json"
    legacy_returns_path = outdir / "sector_rotation_returns.csv"
    legacy_weights_path = outdir / "sector_rotation_weights.csv"
    legacy_summary_path = outdir / "sector_rotation_summary.json"

    grid.to_csv(grid_path, index=False, float_format="%.10g")
    best_returns.to_csv(returns_path, index=False, float_format="%.10g")
    best_weights_out.to_csv(weights_path, index=False, float_format="%.10g")
    best_returns.to_csv(legacy_returns_path, index=False, float_format="%.10g")
    best_weights_out.to_csv(legacy_weights_path, index=False, float_format="%.10g")

    summary = {
        "start": START,
        "end": END,
        "rebalance": REBALANCE,
        "costs_bps": COSTS_BPS,
        "default_universe": DEFAULT_SECTOR_UNIVERSE,
        "assets_loaded": list(close.columns),
        "assets_loaded_count": int(close.shape[1]),
        "variant_count": int(len(grid)),
        "best_by_sharpe": _to_native(best_by_sharpe),
        "best_by_cagr": _to_native(best_by_cagr),
        "best_by_least_severe_drawdown": _to_native(best_by_maxdd),
        "artifacts": {
            "grid": str(grid_path),
            "best_returns": str(returns_path),
            "best_weights": str(weights_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    legacy_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    grid.to_csv(root / "sector_rotation_grid.csv", index=False, float_format="%.10g")
    best_returns.to_csv(root / "best_variant_returns.csv", index=False, float_format="%.10g")
    best_weights_out.to_csv(root / "best_variant_weights.csv", index=False, float_format="%.10g")
    best_returns.to_csv(root / "sector_rotation_returns.csv", index=False, float_format="%.10g")
    best_weights_out.to_csv(root / "sector_rotation_weights.csv", index=False, float_format="%.10g")
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "sector_rotation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("SECTOR ROTATION BENCHMARK GRID")
    print("------------------------------")
    print(f"Assets loaded ({close.shape[1]}): {', '.join(close.columns)}")
    print("\nTop 10 by Sharpe:")
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
