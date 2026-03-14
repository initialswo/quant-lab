#!/usr/bin/env python3
"""Generate a synthetic historical membership CSV from an actual run's loaded tickers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_run_metadata(run_dir: Path) -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json: {run_config_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    selected = run_config.get("selected_tickers", [])
    if not isinstance(selected, list) or not selected:
        raise ValueError(
            "run_config.json does not contain a non-empty 'selected_tickers' list."
        )

    start_raw = summary.get("Start")
    end_raw = summary.get("End")
    if not start_raw or not end_raw:
        raise ValueError("summary.json is missing Start/End.")
    start = pd.Timestamp(start_raw)
    end = pd.Timestamp(end_raw)
    if start >= end:
        raise ValueError(f"Invalid Start/End in summary.json: {start_raw}, {end_raw}")
    return [str(t).strip() for t in selected if str(t).strip()], start, end


def _phase_dates(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    total_days = max((end - start).days, 1)
    early = start
    middle = start + pd.Timedelta(days=int(total_days * 0.50))
    later = start + pd.Timedelta(days=int(total_days * 0.75))
    return early.normalize(), middle.normalize(), later.normalize()


def generate_smoke_membership(run_dir: Path, output_csv: Path) -> Path:
    loaded_tickers, start, end = _load_run_metadata(run_dir)
    early_dt, middle_dt, later_dt = _phase_dates(start, end)

    # Pattern is based on loaded tickers only: first 2 -> first 4 -> first 3.
    k_early = min(2, len(loaded_tickers))
    k_middle = min(4, len(loaded_tickers))
    k_later = min(3, len(loaded_tickers))

    rows: list[dict[str, object]] = []
    for dt, k in ((early_dt, k_early), (middle_dt, k_middle), (later_dt, k_later)):
        members = set(loaded_tickers[:k])
        for ticker in loaded_tickers:
            rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "is_member": 1 if ticker in members else 0,
                }
            )

    out_df = pd.DataFrame(rows, columns=["date", "ticker", "is_member"])
    out_df = out_df.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate data/universe/historical_membership_smoke.csv from a real run's loaded tickers."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to results/<run_tag> containing run_config.json and summary.json",
    )
    parser.add_argument(
        "--output",
        default="data/universe/historical_membership_smoke.csv",
        help="Output CSV path (default: data/universe/historical_membership_smoke.csv)",
    )
    args = parser.parse_args()

    out = generate_smoke_membership(Path(args.run_dir), Path(args.output))
    print(str(out))


if __name__ == "__main__":
    main()

