"""Focused local refinement around best 4-factor quality variant."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
    "top_n": 50,
    "rebalance": "monthly",
    "start": "2005-01-01",
    "end": "2024-12-31",
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "target_vol": 0.15,
    "port_vol_lookback": 20,
    "max_leverage": 1.5,
    "regime_filter": True,
    "regime_benchmark": "SPY",
    "bear_exposure_scale": 1.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
    "costs_bps": 10.0,
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
}

FACTOR_3F = ["momentum_12_1", "reversal_1m", "low_vol_20"]
FACTOR_4F = ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"]

VARIANTS: list[dict[str, object]] = [
    {"Variant": "Baseline_3F", "factor_names": FACTOR_3F, "factor_weights": [0.5, 0.3, 0.2]},
    {"Variant": "Q4F_Control_0.45_0.25_0.15_0.15", "factor_names": FACTOR_4F, "factor_weights": [0.45, 0.25, 0.15, 0.15]},
    {"Variant": "Q4F_0.47_0.23_0.15_0.15", "factor_names": FACTOR_4F, "factor_weights": [0.47, 0.23, 0.15, 0.15]},
    {"Variant": "Q4F_0.45_0.23_0.17_0.15", "factor_names": FACTOR_4F, "factor_weights": [0.45, 0.23, 0.17, 0.15]},
    {"Variant": "Q4F_0.45_0.27_0.13_0.15", "factor_names": FACTOR_4F, "factor_weights": [0.45, 0.27, 0.13, 0.15]},
    {"Variant": "Q4F_0.43_0.25_0.17_0.15", "factor_names": FACTOR_4F, "factor_weights": [0.43, 0.25, 0.17, 0.15]},
    {"Variant": "Q4F_0.45_0.25_0.17_0.13", "factor_names": FACTOR_4F, "factor_weights": [0.45, 0.25, 0.17, 0.13]},
    {"Variant": "Q4F_0.43_0.27_0.15_0.15", "factor_names": FACTOR_4F, "factor_weights": [0.43, 0.27, 0.15, 0.15]},
    {"Variant": "Q4F_0.47_0.25_0.13_0.15", "factor_names": FACTOR_4F, "factor_weights": [0.47, 0.25, 0.13, 0.15]},
]


def _fmt(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def _run_one(cfg: dict[str, object], run_cache: dict[str, Any], cache_debug: bool) -> dict:
    run_cfg = dict(cfg)
    run_cfg["factor_name"] = run_cfg["factor_names"]
    summary, _ = run_backtest(**run_cfg, run_cache=run_cache, cache_debug=cache_debug)
    return summary


def _disp(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Variant",
        "CAGR",
        "Vol",
        "Sharpe",
        "MaxDD",
        "MeanSubSharpe",
        "StdSubSharpe",
        "MinSubSharpe",
        "MaxSubSharpe",
        "StabilityScore",
    ]
    out = df[cols].copy()
    for c in cols[1:]:
        out[c] = out[c].map(_fmt)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results") / "quality_factor_refinement")
    p.add_argument("--resume", action="store_true", help="Explicitly resume from existing registry/artifacts.")
    p.add_argument("--overwrite", action="store_true", help="Rerun all variants and overwrite prior sweep state.")
    p.add_argument("--fail-fast", action="store_true", help="Stop sweep on first variant failure.")
    p.add_argument("--cache-debug", action="store_true", help="Print runner cache hit/miss diagnostics.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    state = SweepState(results_dir=args.results_dir, overwrite=bool(args.overwrite))
    run_cache: dict[str, Any] = {}

    if args.resume and not args.overwrite:
        print(f"Resuming sweep in {args.results_dir} (completed variants will be skipped).")
    elif not args.overwrite and state.registry_path.exists():
        print(f"Found existing registry in {args.results_dir}; auto-resume is active.")

    variants = [
        {
            "name": str(v["Variant"]),
            "params": {
                "factor_names": list(v["factor_names"]),
                "factor_weights": list(v["factor_weights"]),
            },
        }
        for v in VARIANTS
    ]

    def _run_variant(variant: dict[str, Any]) -> dict[str, Any]:
        variant_name = str(variant["name"])
        params = dict(variant["params"])
        cfg = dict(BASE_CONFIG)
        cfg.update(params)

        t0 = time.perf_counter()
        full = _run_one(cfg, run_cache=run_cache, cache_debug=bool(args.cache_debug))
        sub_rows: list[dict[str, Any]] = []
        sub_timing: list[dict[str, Any]] = []

        for s, e in SUBPERIODS:
            t_sub = time.perf_counter()
            cfg_sub = dict(cfg)
            cfg_sub["start"] = s
            cfg_sub["end"] = e
            sub = _run_one(cfg_sub, run_cache=run_cache, cache_debug=bool(args.cache_debug))
            sub_timing.append({"Start": s, "End": e, "duration_seconds": time.perf_counter() - t_sub})
            sub_rows.append(
                {
                    "Variant": variant_name,
                    "Start": s,
                    "End": e,
                    "Sharpe": float(sub.get("Sharpe", float("nan"))),
                }
            )

        row = {
            "Variant": variant_name,
            "Factors": ",".join(cfg["factor_names"]),  # type: ignore[index]
            "Weights": ",".join(f"{float(x):.2f}" for x in cfg["factor_weights"]),  # type: ignore[index]
            "CAGR": float(full.get("CAGR", float("nan"))),
            "Vol": float(full.get("Vol", float("nan"))),
            "Sharpe": float(full.get("Sharpe", float("nan"))),
            "MaxDD": float(full.get("MaxDD", float("nan"))),
            "Outdir": str(full.get("Outdir", "")),
        }
        return {
            "variant": variant_name,
            "params": params,
            "full_row": row,
            "sub_rows": sub_rows,
            "timing": {
                "variant_seconds": time.perf_counter() - t0,
                "run_backtest_timing": full.get("Timing", {}),
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
    stats = (
        sub_df.groupby("Variant", dropna=False)["Sharpe"]
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
    df = df.merge(stats, on="Variant", how="left")
    df["StabilityScore"] = df["MeanSubSharpe"] - 0.5 * df["StdSubSharpe"]

    by_stability = df.sort_values(["StabilityScore", "Sharpe"], ascending=[False, False]).reset_index(drop=True)
    by_sharpe = df.sort_values(["Sharpe", "StabilityScore"], ascending=[False, False]).reset_index(drop=True)
    by_maxdd = df.sort_values(["MaxDD", "StabilityScore"], ascending=[False, False]).reset_index(drop=True)

    print("QUALITY REFINEMENT — SORTED BY STABILITY")
    print("----------------------------------------")
    print(_disp(by_stability).to_string(index=False))
    print("")
    print("QUALITY REFINEMENT — SORTED BY SHARPE")
    print("-------------------------------------")
    print(_disp(by_sharpe).to_string(index=False))
    print("")
    print("QUALITY REFINEMENT — SORTED BY MAXDD")
    print("------------------------------------")
    print(_disp(by_maxdd).to_string(index=False))

    baseline = df.loc[df["Variant"] == "Baseline_3F"].head(1)
    original = df.loc[df["Variant"] == "Q4F_Control_0.45_0.25_0.15_0.15"].head(1)
    refined_4f = by_stability.loc[by_stability["Variant"] != "Baseline_3F"].head(1)
    ablation = pd.concat([baseline, original, refined_4f], axis=0, ignore_index=True)
    print("")
    print("ABLATION SNAPSHOT")
    print("-----------------")
    print(_disp(ablation).to_string(index=False))

    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    full_csv = outdir / "quality_factor_refinement_full.csv"
    sub_csv = outdir / "quality_factor_refinement_subperiods.csv"
    summary_json = outdir / "quality_factor_refinement_summary.json"

    by_stability.to_csv(full_csv, index=False, float_format="%.10g")
    sub_df.to_csv(sub_csv, index=False, float_format="%.10g")
    summary_json.write_text(
        json.dumps(
            {
                "base_config": BASE_CONFIG,
                "variants": VARIANTS,
                "subperiods": [{"start": s, "end": e} for s, e in SUBPERIODS],
                "results_by_stability": by_stability.to_dict(orient="records"),
                "results_by_sharpe": by_sharpe.to_dict(orient="records"),
                "results_by_maxdd": by_maxdd.to_dict(orient="records"),
                "ablation": ablation.to_dict(orient="records"),
                "subperiod_results": sub_df.to_dict(orient="records"),
                "results_dir": str(outdir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print(f"Saved full results: {full_csv}")
    print(f"Saved subperiod results: {sub_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
