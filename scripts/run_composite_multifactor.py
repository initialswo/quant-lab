"""Run the first-generation composite multi-factor portfolio experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel, load_fundamentals_file
from quant_lab.engine import runner
from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover
from quant_lab.strategies.topn import rebalance_mask


RESULTS_ROOT = Path("results") / "composite_multifactor"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE = "liquid_us"
DEFAULT_FACTORS = [
    "momentum_12_1",
    "reversal_1m",
    "gross_profitability",
    "book_to_market",
]
DEFAULT_TOP_N = 10
DEFAULT_REBALANCE = "weekly"
DEFAULT_COSTS_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 0.0
DEFAULT_MAX_TICKERS = 2000
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
FUNDAMENTALS_FALLBACK_LAG_DAYS = 60
COMPOSITE_NOTES = "Mean percentile-rank composite with equal factor weights."
RESULT_COLUMNS = [
    "strategy_name",
    "factors",
    "start",
    "end",
    "universe",
    "top_n",
    "rebalance",
    "CAGR",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "turnover",
    "hit_rate",
    "n_rebalance_dates",
    "median_selected_names",
]
SUMMARY_COLUMNS = [
    "strategy_name",
    "factor_count",
    "sharpe",
    "cagr",
    "max_drawdown",
    "turnover",
    "excess_vs_baseline",
    "notes",
]
SPECIAL_FUNDAMENTAL_FACTORS = {"book_to_market", "asset_growth"}


def _load_close_panel(start: str, end: str, max_tickers: int) -> pd.DataFrame:
    tickers = runner._load_universe_seed_tickers(
        universe="liquid_us",
        max_tickers=int(max_tickers),
        data_cache_dir=DATA_CACHE_DIR,
    )
    ohlcv_map, _ = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(start),
        end=str(end),
        cache_dir=DATA_CACHE_DIR,
        data_source=DATA_SOURCE,
        refresh=False,
        bulk_prepare=False,
    )
    close_cols, _, missing_tickers, rejected_tickers, _ = runner._collect_close_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=tickers,
    )
    if not close_cols:
        raise ValueError(
            "No valid OHLCV data found. "
            f"missing={len(missing_tickers)} rejected={len(rejected_tickers)}"
        )
    close = pd.concat(close_cols, axis=1, join="outer")
    return runner._prepare_close_panel(close_raw=close, price_fill_mode="ffill").astype(float)


def build_special_factor_params(start: str, end: str, factor_names: list[str]) -> dict[str, dict[str, Any]]:
    needed = set(str(x) for x in factor_names).intersection(SPECIAL_FUNDAMENTAL_FACTORS)
    if not needed:
        return {}
    close = _load_close_panel(start=start, end=end, max_tickers=DEFAULT_MAX_TICKERS)
    fundamentals = load_fundamentals_file(
        path=FUNDAMENTALS_PATH,
        fallback_lag_days=int(FUNDAMENTALS_FALLBACK_LAG_DAYS),
    )
    aligned = align_fundamentals_to_daily_panel(
        fundamentals=fundamentals,
        price_index=pd.DatetimeIndex(close.index),
        price_columns=close.columns,
    )
    out: dict[str, dict[str, Any]] = {}
    if "book_to_market" in needed:
        out["book_to_market"] = {"fundamentals_aligned": aligned}
    if "asset_growth" in needed:
        out["asset_growth"] = {"fundamentals_aligned": aligned, "lag_days": 252}
    return out


ALT_STRATEGIES: list[dict[str, Any]] = [
    {
        "strategy_name": "composite_price_only",
        "factors": ["momentum_12_1", "reversal_1m"],
        "notes": "Price-only comparison composite.",
    },
    {
        "strategy_name": "composite_fundamental_only",
        "factors": ["gross_profitability", "book_to_market"],
        "notes": "Fundamental-only comparison composite.",
    },
    {
        "strategy_name": "composite_plus_lowvol",
        "factors": [
            "momentum_12_1",
            "reversal_1m",
            "gross_profitability",
            "book_to_market",
            "low_vol_20",
        ],
        "notes": "Core composite plus low-vol comparison case.",
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--factors", default=",".join(DEFAULT_FACTORS))
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--rebalance", choices=["daily", "weekly", "biweekly", "monthly"], default=DEFAULT_REBALANCE)
    parser.add_argument("--costs_bps", type=float, default=DEFAULT_COSTS_BPS)
    parser.add_argument("--slippage_bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--compare_single_factors", action="store_true")
    parser.add_argument("--compare_alt_sets", action="store_true")
    return parser.parse_args()


def parse_factor_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def format_factor_set(factors: list[str]) -> str:
    return ",".join(str(x) for x in factors)


def equal_weights(factors: list[str]) -> list[float]:
    if not factors:
        raise ValueError("factor set must be non-empty")
    return [1.0 / float(len(factors))] * len(factors)


def strategy_name_for_main(factors: list[str], explicit_name: str = "") -> str:
    if str(explicit_name).strip():
        return str(explicit_name).strip()
    return "composite_core" if list(factors) == DEFAULT_FACTORS else "composite_custom"


def build_strategy_specs(
    factors: list[str],
    compare_single_factors: bool,
    compare_alt_sets: bool,
    name: str = "",
) -> list[dict[str, Any]]:
    main_name = strategy_name_for_main(factors=factors, explicit_name=name)
    specs: list[dict[str, Any]] = [
        {
            "strategy_name": main_name,
            "factors": list(factors),
            "portfolio_mode": "composite",
            "factor_aggregation_method": "mean_rank",
            "notes": COMPOSITE_NOTES,
        }
    ]
    seen = {main_name}

    if compare_alt_sets:
        for spec in ALT_STRATEGIES:
            if str(spec["strategy_name"]) in seen:
                continue
            specs.append(
                {
                    "strategy_name": str(spec["strategy_name"]),
                    "factors": list(spec["factors"]),
                    "portfolio_mode": "composite",
                    "factor_aggregation_method": "mean_rank",
                    "notes": str(spec["notes"]),
                }
            )
            seen.add(str(spec["strategy_name"]))

    if compare_single_factors:
        for factor_name in factors:
            strategy_name = str(factor_name)
            if strategy_name in seen:
                continue
            specs.append(
                {
                    "strategy_name": strategy_name,
                    "factors": [str(factor_name)],
                    "portfolio_mode": "composite",
                    "factor_aggregation_method": "mean_rank",
                    "notes": "Single-factor Top-N comparison using the same backtest settings.",
                }
            )
            seen.add(strategy_name)
    return specs


def build_run_config(
    args: argparse.Namespace,
    factors: list[str],
    factor_params_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "start": str(args.start),
        "end": str(args.end),
        "universe": str(args.universe),
        "top_n": int(args.top_n),
        "rebalance": str(args.rebalance),
        "weighting": "equal",
        "costs_bps": float(args.costs_bps),
        "slippage_bps": float(args.slippage_bps),
        "max_tickers": DEFAULT_MAX_TICKERS,
        "data_source": DATA_SOURCE,
        "data_cache_dir": DATA_CACHE_DIR,
        "factor_name": list(factors),
        "factor_names": list(factors),
        "factor_weights": equal_weights(factors),
        "portfolio_mode": "composite",
        "factor_aggregation_method": "mean_rank",
        "use_factor_normalization": True,
        "use_sector_neutralization": False,
        "use_size_neutralization": False,
        "orthogonalize_factors": False,
        "save_artifacts": True,
        "fundamentals_path": FUNDAMENTALS_PATH,
        "fundamentals_fallback_lag_days": FUNDAMENTALS_FALLBACK_LAG_DAYS,
    }
    if factor_params_map:
        cfg["factor_params"] = {k: dict(v) for k, v in factor_params_map.items() if k in set(factors)}
    if str(args.universe).strip().lower() == "liquid_us":
        cfg["universe_mode"] = "dynamic"
    return cfg


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def artifact_stats(outdir: str | Path, rebalance: str, skipped_count: int) -> dict[str, float]:
    outdir_path = Path(outdir)
    equity = _read_csv(outdir_path / "equity.csv")
    holdings = _read_csv(outdir_path / "holdings.csv")
    daily_return = pd.to_numeric(equity.get("DailyReturn"), errors="coerce").dropna()
    rb_mask = rebalance_mask(pd.DatetimeIndex(holdings.index), rebalance).reindex(holdings.index).fillna(False)
    selected_counts = (holdings.loc[rb_mask].abs() > 0.0).sum(axis=1).astype(float)
    scheduled_rebalances = int(rb_mask.sum())
    n_rebalance_dates = max(0, scheduled_rebalances - int(skipped_count))
    return {
        "hit_rate": float((daily_return > 0.0).mean()) if not daily_return.empty else float("nan"),
        "median_selected_names": float(selected_counts.median()) if not selected_counts.empty else float("nan"),
        "n_rebalance_dates": int(n_rebalance_dates),
    }


def result_row(
    summary: dict[str, Any],
    outdir: str,
    strategy_name: str,
    factors: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stats = artifact_stats(
        outdir=outdir,
        rebalance=str(args.rebalance),
        skipped_count=int(summary.get("RebalanceSkippedCount", 0)),
    )
    return {
        "strategy_name": str(strategy_name),
        "factors": format_factor_set(factors),
        "start": str(args.start),
        "end": str(args.end),
        "universe": str(args.universe),
        "top_n": int(args.top_n),
        "rebalance": str(args.rebalance),
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "ann_vol": float(summary.get("Vol", float("nan"))),
        "sharpe": float(summary.get("Sharpe", float("nan"))),
        "max_drawdown": float(summary.get("MaxDD", float("nan"))),
        "turnover": extract_annual_turnover(summary=summary, outdir=outdir),
        "hit_rate": float(stats["hit_rate"]),
        "n_rebalance_dates": int(stats["n_rebalance_dates"]),
        "median_selected_names": float(stats["median_selected_names"]),
    }


def build_summary(results_df: pd.DataFrame, baseline_name: str, notes_map: dict[str, str]) -> pd.DataFrame:
    baseline_cagr = float(
        results_df.loc[results_df["strategy_name"].eq(baseline_name), "CAGR"].iloc[0]
    ) if baseline_name in set(results_df["strategy_name"]) else float("nan")
    rows: list[dict[str, Any]] = []
    for row in results_df.to_dict(orient="records"):
        rows.append(
            {
                "strategy_name": str(row["strategy_name"]),
                "factor_count": int(len(parse_factor_list(str(row["factors"])))),
                "sharpe": float(row["sharpe"]),
                "cagr": float(row["CAGR"]),
                "max_drawdown": float(row["max_drawdown"]),
                "turnover": float(row["turnover"]),
                "excess_vs_baseline": float(row["CAGR"] - baseline_cagr) if pd.notna(baseline_cagr) else float("nan"),
                "notes": str(notes_map.get(str(row["strategy_name"]), "")),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)


def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"



def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": [int(obj.shape[0]), int(obj.shape[1])],
            "columns_sample": [str(c) for c in list(obj.columns[:5])],
        }
    if isinstance(obj, pd.Series):
        return {
            "type": "Series",
            "length": int(obj.shape[0]),
            "name": str(obj.name),
        }
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def main() -> None:
    args = _parse_args()
    factors = parse_factor_list(args.factors)
    if not factors:
        raise ValueError("At least one factor must be provided via --factors.")
    if int(args.top_n) <= 0:
        raise ValueError("--top_n must be > 0")

    strategy_specs = build_strategy_specs(
        factors=factors,
        compare_single_factors=bool(args.compare_single_factors),
        compare_alt_sets=bool(args.compare_alt_sets),
        name=str(args.name),
    )
    factor_params_map = build_special_factor_params(
        start=str(args.start),
        end=str(args.end),
        factor_names=sorted({factor for spec in strategy_specs for factor in list(spec["factors"])}),
    )
    baseline_name = strategy_specs[0]["strategy_name"]
    notes_map = {str(spec["strategy_name"]): str(spec["notes"]) for spec in strategy_specs}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{str(args.name).strip()}" if str(args.name).strip() and not str(args.output_dir).strip() else ""
    output_dir = Path(str(args.output_dir)) if str(args.output_dir).strip() else RESULTS_ROOT / f"{timestamp}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []
    daily_returns_map: dict[str, str] = {}
    holdings_map: dict[str, str] = {}

    print("")
    print("COMPOSITE MULTIFACTOR SUMMARY")
    print("-----------------------------")
    for spec in strategy_specs:
        cfg = build_run_config(args=args, factors=list(spec["factors"]), factor_params_map=factor_params_map)
        cfg["portfolio_mode"] = str(spec["portfolio_mode"])
        cfg["factor_aggregation_method"] = str(spec["factor_aggregation_method"])
        strategy_name = str(spec["strategy_name"])
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        row = result_row(
            summary=summary,
            outdir=run_outdir,
            strategy_name=strategy_name,
            factors=list(spec["factors"]),
            args=args,
        )
        rows.append(row)
        run_manifest.append(
            {
                "strategy_name": strategy_name,
                "factors": list(spec["factors"]),
                "run_config": _to_serializable(cfg),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
                "notes": str(spec["notes"]),
            }
        )

        equity_src = Path(run_outdir) / "equity.csv"
        holdings_src = Path(run_outdir) / "holdings.csv"
        equity_dst = output_dir / f"{strategy_name}_daily_returns.csv"
        holdings_dst = output_dir / f"{strategy_name}_holdings.csv"
        if equity_src.exists():
            shutil.copy2(equity_src, equity_dst)
            daily_returns_map[strategy_name] = str(equity_dst)
        if holdings_src.exists():
            shutil.copy2(holdings_src, holdings_dst)
            holdings_map[strategy_name] = str(holdings_dst)

        print(
            f"{strategy_name:26s} "
            f"Sharpe={_format_float(float(row['sharpe']))} "
            f"CAGR={_format_float(float(row['CAGR']))} "
            f"MaxDD={_format_float(float(row['max_drawdown']))} "
            f"Turnover={_format_float(float(row['turnover']))}"
        )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_df = results_df.sort_values(["sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    summary_df = build_summary(results_df=results_df, baseline_name=str(baseline_name), notes_map=notes_map)
    summary_df = summary_df.sort_values(["sharpe", "cagr"], ascending=[False, False], kind="mergesort").reset_index(drop=True)

    runtime_seconds = time.perf_counter() - t0
    results_path = output_dir / "composite_multifactor_results.csv"
    summary_path = output_dir / "composite_multifactor_summary.csv"
    manifest_path = output_dir / "manifest.json"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")
    manifest = {
        "timestamp": timestamp,
        "script_name": "scripts/run_composite_multifactor.py",
        "start": str(args.start),
        "end": str(args.end),
        "universe": str(args.universe),
        "factor_sets_run": [
            {"strategy_name": str(spec["strategy_name"]), "factors": list(spec["factors"]), "notes": str(spec["notes"])}
            for spec in strategy_specs
        ],
        "top_n": int(args.top_n),
        "rebalance": str(args.rebalance),
        "costs_bps": float(args.costs_bps),
        "slippage_bps": float(args.slippage_bps),
        "output_dir": str(output_dir),
        "output_paths": {
            "results": str(results_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "daily_returns": daily_returns_map,
            "holdings": holdings_map,
        },
        "runtime_seconds": float(runtime_seconds),
        "baseline_strategy": str(baseline_name),
        "compare_single_factors": bool(args.compare_single_factors),
        "compare_alt_sets": bool(args.compare_alt_sets),
        "composite_ranking_method": (
            "The runner computes each factor using existing direction conventions, applies project-standard robust preprocessing, "
            "converts each factor to a cross-sectional percentile rank, and averages those ranks equally using factor_aggregation_method='mean_rank'."
        ),
        "missing_data_policy": (
            "For liquid_us dynamic-universe runs, the runner requires all selected factors to be present on a date before a security is eligible for the composite score."
        ),
        "runs": run_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if output_dir.parent == RESULTS_ROOT:
        _copy_latest(
            files={
                "composite_multifactor_results.csv": results_path,
                "composite_multifactor_summary.csv": summary_path,
                "manifest.json": manifest_path,
            },
            latest_root=RESULTS_ROOT / "latest",
        )

    print("")
    print(results_df[["strategy_name", "CAGR", "sharpe", "max_drawdown", "turnover"]].to_string(index=False, float_format=_format_float))
    print("")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
