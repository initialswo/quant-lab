"""Generic experiment grid orchestrator for Quant Lab research sweeps."""

from __future__ import annotations

import itertools
import json
import hashlib
import os
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.runner import run_backtest


# Edit this dictionary to define the sweep grid.
experiment: dict[str, list[Any]] = {
    "factors": [
        ["reversal_1m"],
    ],
    "top_n": [20, 30, 50],
    "rank_buffer": [0, 10, 20],
    "rebalance": ["daily", "weekly"],
    "factor_aggregation_method": ["geometric_rank"],
    "execution_delay_days": [0, 1],
}

# Shared baseline config for each run_backtest call.
BASE_CONFIG: dict[str, Any] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
    "universe": "sp500",
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "factor_weights": None,  # Filled per variant as equal-weight over selected factors.
    "dynamic_factor_weights": True,
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
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
    "save_artifacts": False,
}

# Used to compute StabilityScore consistently with existing research scripts.
SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]
CACHE_DIR = Path("results") / "experiment_cache"


def _build_combinations(cfg: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(cfg.keys())
    combinations: list[dict[str, Any]] = []
    for values in itertools.product(*(cfg[k] for k in keys)):
        combinations.append(dict(zip(keys, values)))
    return combinations


def _format_variant_name(index: int, combo: dict[str, Any]) -> str:
    _ = index
    return (
        f"rev1m_top{combo['top_n']}_buf{combo['rank_buffer']}_"
        f"reb{combo['rebalance']}_delay{combo['execution_delay_days']}"
    )


def _run_summary(cfg: dict[str, Any], run_cache: dict[str, Any]) -> dict[str, Any]:
    summary, _ = run_backtest(**cfg, run_cache=run_cache)
    return summary


def _compute_stability(cfg: dict[str, Any], run_cache: dict[str, Any]) -> float:
    sharpes: list[float] = []
    for start, end in SUBPERIODS:
        sub_cfg = dict(cfg)
        sub_cfg["start"] = start
        sub_cfg["end"] = end
        sub_summary = _run_summary(sub_cfg, run_cache=run_cache)
        sharpes.append(float(sub_summary.get("Sharpe", float("nan"))))

    sharpe_series = pd.Series(sharpes, dtype=float)
    return float(sharpe_series.mean() - 0.5 * sharpe_series.std(ddof=0))


def params_to_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def run_variant(params: dict[str, Any]) -> dict[str, Any]:
    variant_params = dict(params)
    variant_name = str(variant_params.pop("VariantName"))
    factors = list(variant_params.pop("factors"))
    top_n = int(variant_params.pop("top_n"))
    rank_buffer = int(variant_params.pop("rank_buffer"))
    rebalance = str(variant_params.pop("rebalance"))
    execution_delay_days = int(variant_params.pop("execution_delay_days"))
    idx = int(variant_params.pop("_idx"))
    total = int(variant_params.pop("_total"))

    print(f"[{idx}/{total}] {variant_name}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    params_hash = params_to_hash(variant_params)
    cache_path = CACHE_DIR / f"{params_hash}.json"

    if cache_path.exists():
        print(f"cache hit: {variant_name}")
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        metrics = dict(cached.get("metrics", {}))
    else:
        run_cache: dict[str, Any] = {}
        summary, _ = run_backtest(**variant_params, run_cache=run_cache)
        stability = _compute_stability(variant_params, run_cache=run_cache)
        metrics = {
            "Sharpe": float(summary.get("Sharpe", float("nan"))),
            "CAGR": float(summary.get("CAGR", float("nan"))),
            "Vol": float(summary.get("Vol", float("nan"))),
            "MaxDD": float(summary.get("MaxDD", float("nan"))),
            "StabilityScore": stability,
        }
        cache_payload = {"variant_name": variant_name, "params": variant_params, "metrics": metrics}
        tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(cache_payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, cache_path)

    return {
        "VariantName": variant_name,
        "factors": ",".join(factors),
        "top_n": top_n,
        "rank_buffer": rank_buffer,
        "rebalance": rebalance,
        "execution_delay_days": execution_delay_days,
        "Sharpe": float(metrics.get("Sharpe", float("nan"))),
        "CAGR": float(metrics.get("CAGR", float("nan"))),
        "Vol": float(metrics.get("Vol", float("nan"))),
        "MaxDD": float(metrics.get("MaxDD", float("nan"))),
        "StabilityScore": float(metrics.get("StabilityScore", float("nan"))),
    }


def main() -> None:
    combos = _build_combinations(experiment)
    variants: list[dict[str, Any]] = []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    total = len(combos)
    for idx, combo in enumerate(combos, start=1):
        factors = list(combo["factors"])
        n_factors = len(factors)
        weights = [1.0 / n_factors] * n_factors

        cfg = dict(BASE_CONFIG)
        cfg.update(
            {
                "factor_name": factors,
                "factor_names": factors,
                "factor_weights": weights,
                "top_n": int(combo["top_n"]),
                "rank_buffer": int(combo["rank_buffer"]),
                "rebalance": str(combo["rebalance"]),
                "factor_aggregation_method": str(combo["factor_aggregation_method"]),
                "execution_delay_days": int(combo["execution_delay_days"]),
            }
        )

        variants.append(
            {
                **cfg,
                "VariantName": _format_variant_name(idx, combo),
                "factors": factors,
                "top_n": int(combo["top_n"]),
                "rank_buffer": int(combo["rank_buffer"]),
                "rebalance": str(combo["rebalance"]),
                "execution_delay_days": int(combo["execution_delay_days"]),
                "_idx": idx,
                "_total": total,
            }
        )

    workers = max(cpu_count() - 1, 1)
    print(f"Running {len(variants)} variants across {workers} workers")
    if workers == 1:
        results = [run_variant(params) for params in variants]
    else:
        with Pool(workers) as pool:
            results = pool.map(run_variant, variants)

    rows = results
    df = pd.DataFrame(rows)
    df = df.sort_values(["StabilityScore", "Sharpe"], ascending=[False, False]).reset_index(drop=True)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"experiment_grid_reversal_{ts}.csv"
    df.to_csv(out_path, index=False, float_format="%.10g")

    print("\nTop 10 variants by StabilityScore:")
    print(df.head(10).to_string(index=False))
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
