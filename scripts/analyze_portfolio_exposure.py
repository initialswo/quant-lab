"""Analyze invested-versus-cash exposure for the latest Top-N sweep run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


RESULTS_ROOT = Path("results") / "topn_sweep"
OUTPUT_NAME = "exposure_analysis.csv"



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
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))



def _load_results(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "topn_sweep_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results CSV: {path}")
    return pd.read_csv(path)



def _run_backtest_dir(manifest: dict[str, Any], top_n: int) -> Path:
    for run in manifest.get("runs", []) or []:
        if int(run.get("top_n", -1)) == int(top_n):
            outdir = str(run.get("backtest_outdir", "")).strip()
            if outdir:
                return Path(outdir)
    raise FileNotFoundError(f"No backtest_outdir found in manifest for top_n={top_n}")



def _read_holdings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing holdings artifact: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index().astype(float)



def _exposure_row(strategy: str, holdings: pd.DataFrame) -> dict[str, Any]:
    position_counts = (holdings.abs() > 0.0).sum(axis=1).astype(int)
    total_days = int(len(position_counts))
    days_with_positions = int((position_counts > 0).sum())
    days_with_zero_positions = int((position_counts == 0).sum())
    percentage_cash = (
        float(days_with_zero_positions) / float(total_days)
        if total_days > 0
        else float("nan")
    )

    first_date_with_positions = (
        position_counts.index[position_counts > 0][0]
        if days_with_positions > 0
        else pd.NaT
    )
    last_date_with_zero_positions = (
        position_counts.index[position_counts == 0][-1]
        if days_with_zero_positions > 0
        else pd.NaT
    )

    return {
        "strategy": str(strategy),
        "total_days": int(total_days),
        "days_with_positions": int(days_with_positions),
        "days_with_zero_positions": int(days_with_zero_positions),
        "percentage_cash": float(percentage_cash),
        "first_date_with_positions": (
            pd.Timestamp(first_date_with_positions).date().isoformat()
            if pd.notna(first_date_with_positions)
            else ""
        ),
        "last_date_with_zero_positions": (
            pd.Timestamp(last_date_with_zero_positions).date().isoformat()
            if pd.notna(last_date_with_zero_positions)
            else ""
        ),
    }



def _format_fraction(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.2%}"



def main() -> None:
    args = _parse_args()
    results_root = Path(str(args.results_root)).expanduser()
    run_dir = Path(str(args.run_dir)).expanduser() if str(args.run_dir).strip() else _latest_run_dir(results_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    manifest = _load_manifest(run_dir)
    results_df = _load_results(run_dir)
    rows: list[dict[str, Any]] = []

    for result in results_df.to_dict(orient="records"):
        top_n = int(result["top_n"])
        strategy = f"Top{top_n}"
        backtest_dir = _run_backtest_dir(manifest=manifest, top_n=top_n)
        holdings = _read_holdings(backtest_dir / "holdings.csv")
        rows.append(_exposure_row(strategy=strategy, holdings=holdings))

    exposure_df = pd.DataFrame(
        rows,
        columns=[
            "strategy",
            "total_days",
            "days_with_positions",
            "days_with_zero_positions",
            "percentage_cash",
            "first_date_with_positions",
            "last_date_with_zero_positions",
        ],
    )
    exposure_df = exposure_df.sort_values("strategy", key=lambda s: s.str.extract(r"(\d+)").astype(int)[0], kind="mergesort")

    out_path = run_dir / OUTPUT_NAME
    exposure_df.to_csv(out_path, index=False, float_format="%.10g")

    print("PORTFOLIO EXPOSURE")
    print("------------------")
    print(f"{'Strategy':10s} {'CashDays':>10s} {'TotalDays':>10s} {'CashFraction':>14s}")
    for row in exposure_df.to_dict(orient="records"):
        print(
            f"{str(row['strategy']):10s} "
            f"{int(row['days_with_zero_positions']):10d} "
            f"{int(row['total_days']):10d} "
            f"{_format_fraction(float(row['percentage_cash'])):>14s}"
        )
    print("")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
