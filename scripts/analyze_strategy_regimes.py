"""Analyze regime stability for selected core-factor composite strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_metrics


RESULTS_ROOT = Path("results") / "core_factor_composites"
TARGET_STRATEGIES = [
    "profitability_only",
    "reversal_only",
    "reversal_profitability",
]
REGIMES: list[tuple[str, str, str]] = [
    ("2010-2013", "2010-01-01", "2013-12-31"),
    ("2014-2017", "2014-01-01", "2017-12-31"),
    ("2018-2020", "2018-01-01", "2020-12-31"),
    ("2021-2024", "2021-01-01", "2024-12-31"),
]
OUTPUT_NAME = "regime_analysis.csv"



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    parser.add_argument("--run_dir", default="")
    return parser.parse_args()



def _latest_run_dir(results_root: Path) -> Path:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    candidates = [p for p in results_root.iterdir() if p.is_dir() and p.name != "latest"]
    if not candidates:
        raise FileNotFoundError(f"No timestamped run directories found under: {results_root}")
    return sorted(candidates, key=lambda p: p.name)[-1]



def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))



def _strategy_backtest_dir(manifest: dict[str, Any], strategy_name: str) -> Path | None:
    runs = manifest.get("runs", []) or []
    for run in runs:
        if str(run.get("strategy_name", "")) == str(strategy_name):
            outdir = str(run.get("backtest_outdir", "")).strip()
            return Path(outdir) if outdir else None
    return None



def _read_daily_return_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    for col in ["DailyReturn", "daily_return", "returns"]:
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce").dropna().astype(float)
            series.index = pd.DatetimeIndex(series.index)
            series.name = "DailyReturn"
            return series
    raise ValueError(f"No daily-return column found in: {path}")



def _load_strategy_returns(run_dir: Path, manifest: dict[str, Any], strategy_name: str) -> pd.Series:
    candidate_paths = [
        run_dir / f"{strategy_name}_daily_returns.csv",
        run_dir / strategy_name / "daily_returns.csv",
        run_dir / strategy_name / "equity.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return _read_daily_return_series(path)

    backtest_dir = _strategy_backtest_dir(manifest=manifest, strategy_name=strategy_name)
    if backtest_dir is not None:
        for name in ["daily_returns.csv", "equity.csv"]:
            path = backtest_dir / name
            if path.exists():
                return _read_daily_return_series(path)

    raise FileNotFoundError(
        f"No daily return series found for strategy '{strategy_name}' in {run_dir} or manifest-linked backtest outputs."
    )



def _regime_metrics(strategy_name: str, returns: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = pd.DatetimeIndex(returns.index)
    for label, start, end in REGIMES:
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        part = returns.loc[mask]
        metrics = compute_metrics(part)
        rows.append(
            {
                "strategy": str(strategy_name),
                "period": str(label),
                "CAGR": float(metrics.get("CAGR", float("nan"))),
                "Volatility": float(metrics.get("Vol", float("nan"))),
                "Sharpe": float(metrics.get("Sharpe", float("nan"))),
                "MaxDD": float(metrics.get("MaxDD", float("nan"))),
            }
        )
    return rows



def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"



def main() -> None:
    args = _parse_args()
    results_root = Path(str(args.results_root)).expanduser()
    run_dir = Path(str(args.run_dir)).expanduser() if str(args.run_dir).strip() else _latest_run_dir(results_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    manifest = _load_manifest(run_dir)
    rows: list[dict[str, Any]] = []
    for strategy_name in TARGET_STRATEGIES:
        returns = _load_strategy_returns(run_dir=run_dir, manifest=manifest, strategy_name=strategy_name)
        rows.extend(_regime_metrics(strategy_name=strategy_name, returns=returns))

    regime_df = pd.DataFrame(rows, columns=["strategy", "period", "CAGR", "Volatility", "Sharpe", "MaxDD"])
    out_path = run_dir / OUTPUT_NAME
    regime_df.to_csv(out_path, index=False, float_format="%.10g")

    print("REGIME ANALYSIS")
    print("---------------")
    for strategy_name in TARGET_STRATEGIES:
        subset = regime_df.loc[regime_df["strategy"].eq(strategy_name)].copy()
        print(f"Strategy: {strategy_name}")
        for row in subset.to_dict(orient="records"):
            print(
                f"{row['period']:10s} "
                f"Sharpe={_format_float(float(row['Sharpe']))} "
                f"CAGR={_format_float(float(row['CAGR']))} "
                f"MaxDD={_format_float(float(row['MaxDD']))}"
            )
        print("")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
