"""Single-factor sleeve decomposition study for the benchmark factor set."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover


FACTORS: list[str] = [
    "momentum_12_1",
    "reversal_1m",
    "low_vol_20",
    "gross_profitability",
]
UNIVERSE_CONFIGS: dict[str, dict[str, Any]] = {
    "sp500": {},
    "liquid_us": {"universe_mode": "dynamic"},
}
RESULT_COLUMNS: list[str] = [
    "universe",
    "factor",
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "AnnualTurnover",
    "EligMedian",
    "TradableMedian",
    "RebalanceSkipped",
]

BASE_CONFIG: dict[str, Any] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
    "top_n": 50,
    "weighting": "equal",
    "rebalance": "weekly",
    "costs_bps": 10.0,
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "save_artifacts": True,
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
}


def build_run_config(universe: str, factor: str) -> dict[str, Any]:
    if universe not in UNIVERSE_CONFIGS:
        raise ValueError(f"Unsupported universe: {universe}")
    cfg = dict(BASE_CONFIG)
    cfg.update(UNIVERSE_CONFIGS[universe])
    cfg["universe"] = str(universe)
    cfg["factor_name"] = [str(factor)]
    cfg["factor_names"] = [str(factor)]
    cfg["factor_weights"] = [1.0]
    return cfg


def summary_to_result_row(summary: dict[str, Any], outdir: str, universe: str, factor: str) -> dict[str, Any]:
    return {
        "universe": str(universe),
        "factor": str(factor),
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "Vol": float(summary.get("Vol", float("nan"))),
        "Sharpe": float(summary.get("Sharpe", float("nan"))),
        "MaxDD": float(summary.get("MaxDD", float("nan"))),
        "AnnualTurnover": extract_annual_turnover(summary=summary, outdir=outdir),
        "EligMedian": float(summary.get("EligibleOnRebalanceMedian", float("nan"))),
        "TradableMedian": float(summary.get("FinalTradableOnRebalanceMedian", float("nan"))),
        "RebalanceSkipped": int(summary.get("RebalanceSkippedCount", 0)),
    }


def load_equity_curve(outdir: str | Path) -> pd.DataFrame:
    eq = pd.read_csv(Path(outdir) / "equity_curve.csv", parse_dates=["date"])
    eq = eq.sort_values("date", kind="mergesort").reset_index(drop=True)
    return eq


def copy_sleeve_artifacts(run_outdir: str | Path, dest_dir: Path) -> dict[str, str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in ["equity_curve.csv", "summary.json", "run_config.json", "holdings.csv"]:
        src = Path(run_outdir) / name
        if not src.exists():
            continue
        dst = dest_dir / name
        shutil.copy2(src, dst)
        copied[name] = str(dst)
    return copied


def build_return_panel(run_manifest: list[dict[str, Any]], universe: str) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    for run in run_manifest:
        if run["universe"] != universe:
            continue
        eq = load_equity_curve(run["backtest_outdir"])
        series = eq.set_index("date")["returns"].astype(float).rename(str(run["factor"]))
        series_map[str(run["factor"])] = series
    if not series_map:
        return pd.DataFrame(index=pd.Index([], name="date"))
    panel = pd.concat(series_map.values(), axis=1).sort_index()
    panel.index.name = "date"
    panel = panel.reindex(columns=FACTORS)
    return panel


def compute_rolling_average_correlation(return_panel: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    out = pd.DataFrame(index=FACTORS, columns=FACTORS, dtype=float)
    for left in FACTORS:
        for right in FACTORS:
            pair = return_panel[[left, right]].dropna()
            if pair.empty:
                out.loc[left, right] = float("nan")
                continue
            if left == right:
                out.loc[left, right] = 1.0
                continue
            rolling = pair[left].rolling(window=window, min_periods=window).corr(pair[right])
            out.loc[left, right] = float(rolling.mean()) if rolling.notna().any() else float("nan")
    out.index.name = "factor"
    return out


def average_pairwise_correlation(corr: pd.DataFrame) -> float:
    values: list[float] = []
    for i, left in enumerate(FACTORS):
        for right in FACTORS[i + 1 :]:
            val = corr.loc[left, right]
            if pd.notna(val):
                values.append(float(val))
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def build_contribution_summary(results_df: pd.DataFrame, corr_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for universe in UNIVERSE_CONFIGS:
        sub = results_df.loc[results_df["universe"].eq(universe)].copy()
        if sub.empty:
            continue
        best_sharpe_row = sub.sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").iloc[0]
        best_cagr_row = sub.sort_values(["CAGR", "Sharpe"], ascending=[False, False], kind="mergesort").iloc[0]
        lowest_dd_row = sub.sort_values(["MaxDD", "Sharpe"], ascending=[False, False], kind="mergesort").iloc[0]
        rows.append(
            {
                "universe": universe,
                "BestStandaloneSharpeFactor": str(best_sharpe_row["factor"]),
                "BestStandaloneSharpe": float(best_sharpe_row["Sharpe"]),
                "BestStandaloneCAGRFactor": str(best_cagr_row["factor"]),
                "BestStandaloneCAGR": float(best_cagr_row["CAGR"]),
                "LowestStandaloneDrawdownFactor": str(lowest_dd_row["factor"]),
                "LowestStandaloneDrawdown": float(lowest_dd_row["MaxDD"]),
                "AveragePairwiseCorrelation": average_pairwise_correlation(corr_map[universe]),
            }
        )
    return pd.DataFrame(rows)


def run_study() -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, pd.DataFrame]]:
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for universe in UNIVERSE_CONFIGS:
        for factor in FACTORS:
            cfg = build_run_config(universe=universe, factor=factor)
            print(f"Running universe={universe} factor={factor}")
            summary, outdir = run_backtest(**cfg, run_cache=run_cache)
            rows.append(summary_to_result_row(summary=summary, outdir=outdir, universe=universe, factor=factor))
            run_manifest.append(
                {
                    "universe": str(universe),
                    "factor": str(factor),
                    "backtest_outdir": str(outdir),
                    "summary_path": str(Path(outdir) / "summary.json"),
                    "run_config": cfg,
                }
            )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_df = results_df.sort_values(["universe", "factor"], kind="mergesort").reset_index(drop=True)
    return_panels = {universe: build_return_panel(run_manifest=run_manifest, universe=universe) for universe in UNIVERSE_CONFIGS}
    return results_df, run_manifest, return_panels


def write_outputs(
    results_df: pd.DataFrame,
    run_manifest: list[dict[str, Any]],
    return_panels: dict[str, pd.DataFrame],
    runtime_seconds: float,
    results_root: Path = Path("results") / "factor_decomposition",
) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    sleeves_root = outdir / "sleeves"
    corr_map: dict[str, pd.DataFrame] = {}
    rolling_map: dict[str, pd.DataFrame] = {}
    sleeve_artifacts: list[dict[str, Any]] = []

    for run in run_manifest:
        sleeve_dir = sleeves_root / str(run["universe"]) / str(run["factor"])
        copied = copy_sleeve_artifacts(run_outdir=run["backtest_outdir"], dest_dir=sleeve_dir)
        sleeve_artifacts.append(
            {
                "universe": str(run["universe"]),
                "factor": str(run["factor"]),
                "source_outdir": str(run["backtest_outdir"]),
                "copied_artifacts": copied,
            }
        )

    for universe, panel in return_panels.items():
        panel_path = outdir / f"factor_sleeve_returns_{universe}.csv"
        panel.reset_index().to_csv(panel_path, index=False, float_format="%.10g")

        corr = panel.corr().reindex(index=FACTORS, columns=FACTORS)
        corr.index.name = "factor"
        corr_map[universe] = corr
        corr_path = outdir / f"factor_correlation_{universe}.csv"
        corr.to_csv(corr_path, float_format="%.10g")
        corr.to_csv(results_root / f"factor_correlation_{universe}_latest.csv", float_format="%.10g")

        rolling_corr = compute_rolling_average_correlation(panel, window=252)
        rolling_map[universe] = rolling_corr
        rolling_corr.to_csv(outdir / f"factor_correlation_rolling252_avg_{universe}.csv", float_format="%.10g")

    contribution_df = build_contribution_summary(results_df=results_df, corr_map=corr_map)

    results_path = outdir / "factor_decomposition_results.csv"
    contribution_path = outdir / "factor_decomposition_contribution_summary.csv"
    manifest_path = outdir / "factor_decomposition_manifest.json"
    latest_results_path = results_root / "factor_decomposition_results_latest.csv"

    results_df.to_csv(results_path, index=False, float_format="%.10g")
    results_df.to_csv(latest_results_path, index=False, float_format="%.10g")
    contribution_df.to_csv(contribution_path, index=False, float_format="%.10g")

    manifest_path.write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp,
                "results_dir": str(outdir),
                "runtime_seconds": float(runtime_seconds),
                "base_config": BASE_CONFIG,
                "factors": FACTORS,
                "universes": list(UNIVERSE_CONFIGS.keys()),
                "runs": run_manifest,
                "sleeve_artifacts": sleeve_artifacts,
                "rolling_correlation_files": {
                    universe: str(outdir / f"factor_correlation_rolling252_avg_{universe}.csv")
                    for universe in UNIVERSE_CONFIGS
                },
                "contribution_summary_path": str(contribution_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_payload = {
        "timestamp_utc": timestamp,
        "runtime_seconds": float(runtime_seconds),
        "results_path": str(results_path),
        "contribution_summary_path": str(contribution_path),
        "correlation_files": {
            universe: str(outdir / f"factor_correlation_{universe}.csv") for universe in UNIVERSE_CONFIGS
        },
        "rolling_correlation_files": {
            universe: str(outdir / f"factor_correlation_rolling252_avg_{universe}.csv") for universe in UNIVERSE_CONFIGS
        },
    }
    (outdir / "factor_decomposition_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    print("")
    print("FACTOR DECOMPOSITION RESULTS")
    print("----------------------------")
    print(results_df.to_string(index=False))
    print("")
    for universe in UNIVERSE_CONFIGS:
        print(f"{universe} correlation matrix")
        print(corr_map[universe].to_string(float_format=lambda x: f"{x:.4f}"))
        print("")
    print(f"Saved: {results_path}")
    print(f"Saved: {latest_results_path}")
    print(f"Saved: {contribution_path}")
    for universe in UNIVERSE_CONFIGS:
        print(f"Saved: {outdir / f'factor_correlation_{universe}.csv'}")
        print(f"Saved: {results_root / f'factor_correlation_{universe}_latest.csv'}")
    print(f"Saved: {manifest_path}")
    return outdir


def main() -> None:
    t0 = time.perf_counter()
    results_df, run_manifest, return_panels = run_study()
    runtime_seconds = time.perf_counter() - t0
    write_outputs(
        results_df=results_df,
        run_manifest=run_manifest,
        return_panels=return_panels,
        runtime_seconds=runtime_seconds,
    )


if __name__ == "__main__":
    main()
