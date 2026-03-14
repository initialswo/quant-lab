"""Lightweight parameter sweep runner for Quant Lab experiments."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_runtime import SweepState, run_sweep_variants


SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]

BASE_CONFIG: dict[str, object] = {
    "factor": "momentum_12_1,reversal_1m,low_vol_20",
    "factor_weights": "0.5,0.3,0.2",
    "top_n": 50,
    "rebalance": "monthly",
    "start": "2005-01-01",
    "end": "2024-12-31",
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_root": "data/equities",
    "target_vol": 0.15,
    "port_vol_lookback": 20,
    "max_leverage": 1.5,
    "regime_filter": 1,
    "regime_benchmark": "SPY",
    "bear_exposure_scale": 1.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
    "costs_bps": 10.0,
}

SWEEP_GRID: dict[str, list[object]] = {
    "bear_exposure_scale": [1.0, 0.75, 0.5],
    "min_avg_dollar_volume": [0.0, 5_000_000.0, 10_000_000.0],
}

ENABLE_SUBPERIOD_EVAL = True


def _parse_csv_list(raw: str) -> list[str]:
    out = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError("Expected non-empty CSV list.")
    return out


def _parse_csv_floats(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected non-empty CSV float list.")
    return vals


def _fmt_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def _to_run_kwargs(cfg: dict[str, object]) -> dict[str, object]:
    run_cfg = dict(cfg)
    factor_raw = str(run_cfg.pop("factor"))
    factor_names = _parse_csv_list(factor_raw)
    factor_weights_raw = run_cfg.pop("factor_weights")
    if isinstance(factor_weights_raw, str):
        factor_weights = _parse_csv_floats(factor_weights_raw)
    else:
        factor_weights = [float(x) for x in list(factor_weights_raw)]  # type: ignore[arg-type]
    if len(factor_names) != len(factor_weights):
        raise ValueError("factor and factor_weights lengths must match.")
    data_root = str(run_cfg.pop("data_root"))

    run_cfg["factor_name"] = factor_names
    run_cfg["factor_names"] = factor_names
    run_cfg["factor_weights"] = factor_weights
    run_cfg["data_cache_dir"] = data_root
    run_cfg["regime_filter"] = bool(run_cfg.get("regime_filter", 0))
    return run_cfg


def _variant_name(combo: dict[str, object]) -> str:
    parts: list[str] = []
    for k, v in combo.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return "|".join(parts)


def _run_one(cfg: dict[str, object], run_cache: dict[str, Any], cache_debug: bool) -> dict:
    summary, _ = run_backtest(**_to_run_kwargs(cfg), run_cache=run_cache, cache_debug=cache_debug)
    if str(summary.get("DataSource", "")).lower() != "parquet":
        raise ValueError(f"Unexpected DataSource={summary.get('DataSource')}; expected parquet.")
    if int(summary.get("DataFetchedOrRefreshed", 0)) != 0:
        raise ValueError(
            "Backtest unexpectedly refreshed/fetched external data: "
            f"DataFetchedOrRefreshed={summary.get('DataFetchedOrRefreshed')}"
        )
    return summary


def _sweep_combinations(base_cfg: dict[str, object], grid: dict[str, list[object]]) -> list[dict[str, object]]:
    keys = list(grid.keys())
    combos: list[dict[str, object]] = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        combo = {k: v for k, v in zip(keys, vals)}
        merged = dict(base_cfg)
        merged.update(combo)
        merged["_sweep_values"] = combo
        combos.append(merged)
    return combos


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results") / "parameter_sweep")
    p.add_argument("--resume", action="store_true", help="Explicitly resume from existing registry/artifacts.")
    p.add_argument("--overwrite", action="store_true", help="Rerun all variants and overwrite prior sweep state.")
    p.add_argument("--fail-fast", action="store_true", help="Stop sweep on first variant failure.")
    p.add_argument("--cache-debug", action="store_true", help="Print runner cache hit/miss diagnostics.")
    p.add_argument("--no-subperiods", action="store_true", help="Disable subperiod stability evaluation.")
    p.add_argument("--max-variants", type=int, default=0, help="Run only first N variants (0 means all).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    state = SweepState(results_dir=args.results_dir, overwrite=bool(args.overwrite))
    run_cache: dict[str, Any] = {}

    if args.resume and not args.overwrite:
        print(f"Resuming sweep in {args.results_dir} (completed variants will be skipped).")
    elif not args.overwrite and state.registry_path.exists():
        print(f"Found existing registry in {args.results_dir}; auto-resume is active.")

    combos = _sweep_combinations(BASE_CONFIG, SWEEP_GRID)
    if int(args.max_variants) > 0:
        combos = combos[: int(args.max_variants)]
    grid_keys = list(SWEEP_GRID.keys())
    enable_subperiods = ENABLE_SUBPERIOD_EVAL and (not bool(args.no_subperiods))

    variants: list[dict[str, Any]] = []
    for cfg in combos:
        sweep_vals = dict(cfg.get("_sweep_values", {}))
        variants.append(
            {
                "name": _variant_name(sweep_vals),
                "params": {k: sweep_vals[k] for k in sweep_vals},
                "cfg": dict(cfg),
            }
        )

    def _run_variant(variant: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(variant["cfg"])
        sweep_vals = dict(cfg.pop("_sweep_values"))
        variant_name = str(variant["name"])

        t0 = time.perf_counter()
        summary = _run_one(cfg, run_cache=run_cache, cache_debug=bool(args.cache_debug))
        row = {
            "VariantName": variant_name,
            **sweep_vals,
            "CAGR": float(summary.get("CAGR", float("nan"))),
            "Vol": float(summary.get("Vol", float("nan"))),
            "Sharpe": float(summary.get("Sharpe", float("nan"))),
            "MaxDD": float(summary.get("MaxDD", float("nan"))),
            "LeverageAvg": float(summary.get("LeverageAvg", float("nan"))),
            "LeverageMax": float(summary.get("LeverageMax", float("nan"))),
            "regime_pct_bull": float(summary.get("regime_pct_bull", float("nan"))),
            "regime_pct_bear_or_volatile": float(
                summary.get("regime_pct_bear_or_volatile", float("nan"))
            ),
            "LiquidityEligibleMedian": float(summary.get("LiquidityEligibleMedian", float("nan"))),
            "SanityWarningsCount": int(summary.get("SanityWarningsCount", 0)),
            "Outdir": str(summary.get("Outdir", "")),
        }
        sub_rows: list[dict[str, Any]] = []
        sub_timing: list[dict[str, Any]] = []

        if enable_subperiods:
            for start, end in SUBPERIODS:
                t_sub = time.perf_counter()
                cfg_sub = dict(cfg)
                cfg_sub["start"] = start
                cfg_sub["end"] = end
                s_sub = _run_one(cfg_sub, run_cache=run_cache, cache_debug=bool(args.cache_debug))
                sub_timing.append({"Start": start, "End": end, "duration_seconds": time.perf_counter() - t_sub})
                sub_rows.append(
                    {
                        "VariantName": variant_name,
                        **sweep_vals,
                        "Start": start,
                        "End": end,
                        "Sharpe": float(s_sub.get("Sharpe", float("nan"))),
                    }
                )

        return {
            "variant": variant_name,
            "params": sweep_vals,
            "full_row": row,
            "sub_rows": sub_rows,
            "timing": {
                "variant_seconds": time.perf_counter() - t0,
                "run_backtest_timing": summary.get("Timing", {}),
                "subperiod_timing": sub_timing,
            },
            "output_rows": 1 + len(sub_rows),
        }

    run_sweep_variants(
        variants=variants,
        state=state,
        run_variant=_run_variant,
        fail_fast=bool(args.fail_fast),
    )

    payloads = state.load_all_payloads()
    rows = [dict(p["full_row"]) for p in payloads if "full_row" in p]
    sub_rows = [r for p in payloads for r in list(p.get("sub_rows", []))]

    if not rows:
        print("No completed variant artifacts found; nothing to aggregate.")
        return

    df = pd.DataFrame(rows)
    sub_df = pd.DataFrame(sub_rows)

    if enable_subperiods and not sub_df.empty:
        stats = (
            sub_df.groupby("VariantName", dropna=False)["Sharpe"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    "mean": "MeanSubSharpe",
                    "std": "StdSubSharpe",
                    "min": "MinSubSharpe",
                    "max": "MaxSubSharpe",
                }
            )
        )
        df = df.merge(stats, on="VariantName", how="left")
        df["StabilityScore"] = df["MeanSubSharpe"] - 0.5 * df["StdSubSharpe"]
        df = df.sort_values(["StabilityScore", "Sharpe"], ascending=[False, False]).reset_index(drop=True)
    else:
        df["MeanSubSharpe"] = np.nan
        df["StdSubSharpe"] = np.nan
        df["MinSubSharpe"] = np.nan
        df["MaxSubSharpe"] = np.nan
        df["StabilityScore"] = np.nan
        df = df.sort_values(["Sharpe"], ascending=[False]).reset_index(drop=True)

    display_cols = ["VariantName", *grid_keys, "CAGR", "Vol", "Sharpe", "MaxDD", "StabilityScore"]
    disp = df[display_cols].copy()
    for c in ["CAGR", "Vol", "Sharpe", "MaxDD", "StabilityScore"]:
        disp[c] = disp[c].map(_fmt_float)
    for k in grid_keys:
        if pd.api.types.is_numeric_dtype(df[k]):
            disp[k] = df[k].map(_fmt_float)

    print("PARAMETER SWEEP RESULTS")
    print("-----------------------")
    print(disp.to_string(index=False))

    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "parameter_sweep.csv"
    out_json = outdir / "parameter_sweep.json"

    df.to_csv(out_csv, index=False, float_format="%.10g")
    out_json.write_text(
        json.dumps(
            {
                "base_config": BASE_CONFIG,
                "sweep_grid": SWEEP_GRID,
                "subperiod_eval_enabled": enable_subperiods,
                "subperiods": [{"start": s, "end": e} for s, e in SUBPERIODS],
                "results": df.to_dict(orient="records"),
                "subperiod_results": sub_df.to_dict(orient="records"),
                "results_dir": str(outdir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
