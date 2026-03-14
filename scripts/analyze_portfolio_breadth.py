"""Analyze effective portfolio breadth for the latest Top-N sweep run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


RESULTS_ROOT = Path("results") / "topn_sweep"
OUTPUT_NAME = "breadth_analysis.csv"



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



def _read_holdings_counts(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing holdings artifact: {path}")
    holdings = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    counts = (holdings.abs() > 0.0).sum(axis=1).astype(float)
    counts.index = pd.DatetimeIndex(counts.index)
    counts.name = "positions"
    return counts



def _read_universe_counts(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing universe eligibility artifact: {path}")
    frame = pd.read_csv(path)
    if "date" not in frame.columns or "eligible_count" not in frame.columns:
        raise ValueError(f"Universe eligibility artifact missing required columns: {path}")
    series = pd.Series(
        pd.to_numeric(frame["eligible_count"], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(frame["date"], errors="coerce"),
        name="eligible_universe_size",
    ).dropna()
    series.index = pd.DatetimeIndex(series.index)
    return series.sort_index()



def _breadth_row(strategy: str, positions: pd.Series, universe: pd.Series) -> dict[str, Any]:
    aligned = pd.concat([positions.rename("positions"), universe.rename("eligible_universe_size")], axis=1, join="inner")
    aligned = aligned.dropna()
    if aligned.empty:
        return {
            "strategy": str(strategy),
            "avg_positions": float("nan"),
            "min_positions": float("nan"),
            "max_positions": float("nan"),
            "avg_universe_size": float("nan"),
            "min_universe_size": float("nan"),
            "max_universe_size": float("nan"),
            "avg_fraction_of_universe": float("nan"),
        }
    avg_pos = float(aligned["positions"].mean())
    avg_universe = float(aligned["eligible_universe_size"].mean())
    fraction = avg_pos / avg_universe if pd.notna(avg_universe) and avg_universe > 0.0 else float("nan")
    return {
        "strategy": str(strategy),
        "avg_positions": avg_pos,
        "min_positions": float(aligned["positions"].min()),
        "max_positions": float(aligned["positions"].max()),
        "avg_universe_size": avg_universe,
        "min_universe_size": float(aligned["eligible_universe_size"].min()),
        "max_universe_size": float(aligned["eligible_universe_size"].max()),
        "avg_fraction_of_universe": float(fraction),
    }



def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.2f}"



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
        positions = _read_holdings_counts(backtest_dir / "holdings.csv")
        universe = _read_universe_counts(backtest_dir / "universe_eligibility_summary.csv")
        rows.append(_breadth_row(strategy=strategy, positions=positions, universe=universe))

    breadth_df = pd.DataFrame(
        rows,
        columns=[
            "strategy",
            "avg_positions",
            "min_positions",
            "max_positions",
            "avg_universe_size",
            "min_universe_size",
            "max_universe_size",
            "avg_fraction_of_universe",
        ],
    )
    breadth_df = breadth_df.sort_values("strategy", key=lambda s: s.str.extract(r"(\d+)").astype(int)[0], kind="mergesort")

    out_path = run_dir / OUTPUT_NAME
    breadth_df.to_csv(out_path, index=False, float_format="%.10g")

    print("PORTFOLIO BREADTH")
    print("-----------------")
    print("")
    print(f"{'Strategy':10s} {'AvgPos':>8s} {'MinPos':>8s} {'AvgUniverse':>12s} {'Fraction':>10s}")
    for row in breadth_df.to_dict(orient="records"):
        print(
            f"{str(row['strategy']):10s} "
            f"{_format_float(float(row['avg_positions'])):>8s} "
            f"{_format_float(float(row['min_positions'])):>8s} "
            f"{_format_float(float(row['avg_universe_size'])):>12s} "
            f"{_format_float(float(row['avg_fraction_of_universe'])):>10s}"
        )
    print("")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
