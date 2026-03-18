"""Stage FMP fundamentals snapshot for a limited dry-run cohort."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_lab.utils.env import get_required_env, load_project_env
from quant_lab.data.fmp_fundamentals import fetch_fmp_fundamentals_frame


def _load_env_key() -> str:
    load_project_env()
    return get_required_env("FMP_API_KEY")

def _read_cohort(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"cohort file not found: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        col = "ticker" if "ticker" in frame.columns else frame.columns[0]
        vals = frame[col].astype(str).tolist()
    else:
        vals = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    seen: set[str] = set()
    for raw in vals:
        t = str(raw).strip().upper().replace(".US", "").replace(".", "-")
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    if not out:
        raise ValueError("cohort file produced 0 tickers")
    return out


def run_staged_fmp_fundamentals_dry_run(
    cohort: list[str],
    out_dir: Path,
    available_lag_days: int = 60,
    limit: int = 120,
    source_label: str = "fmp_phase2a_dryrun",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_payloads"
    raw_dir.mkdir(parents=True, exist_ok=True)

    api_key = _load_env_key()
    session = requests.Session()
    frames: list[pd.DataFrame] = []
    report_rows: list[dict[str, Any]] = []

    for i, ticker in enumerate(cohort, start=1):
        frame, raw = fetch_fmp_fundamentals_frame(
            session=session,
            ticker=ticker,
            api_key=api_key,
            available_lag_days=int(available_lag_days),
            limit=int(limit),
            statement_period="quarter",
            timeout_seconds=20.0,
            max_retries=2,
        )
        error = str((raw or {}).get("error") or "")
        report = {
            "ticker": ticker,
            "rows": int(len(frame)),
            "error": error,
            "income_status": int((((raw or {}).get("income") or {}).get("status_code") or 0)),
            "balance_status": int((((raw or {}).get("balance") or {}).get("status_code") or 0)),
        }
        report_rows.append(report)
        (raw_dir / f"{ticker}.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
        if not frame.empty:
            staged = frame.copy()
            staged["source"] = str(source_label)
            staged["fetch_ts"] = datetime.now(UTC).isoformat()
            frames.append(staged)
        if i % 20 == 0 or i == len(cohort):
            ok_count = sum(1 for x in report_rows if not x.get("error"))
            print(f"[fmp] {i}/{len(cohort)} complete ok={ok_count} fail={len(report_rows)-ok_count}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.sort_values(["ticker", "available_date", "period_end"]).reset_index(drop=True)
    out_parquet = out_dir / "fmp_fundamentals_snapshot.parquet"
    combined.to_parquet(out_parquet, index=False)

    report_df = pd.DataFrame(report_rows).sort_values(["error", "ticker"]).reset_index(drop=True)
    report_path = out_dir / "fmp_fetch_report.csv"
    report_df.to_csv(report_path, index=False)

    summary = {
        "rows": int(len(combined)),
        "tickers_requested": int(len(cohort)),
        "tickers_succeeded": int((report_df["error"].astype(str) == "").sum()) if not report_df.empty else 0,
        "tickers_failed": int((report_df["error"].astype(str) != "").sum()) if not report_df.empty else 0,
        "parquet_path": str(out_parquet),
        "report_path": str(report_path),
    }
    (out_dir / "fmp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Stage FMP fundamentals snapshot for a dry-run ticker cohort.")
    p.add_argument("--cohort-file", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--available-lag-days", type=int, default=60)
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--source-label", default="fmp_phase2a_dryrun")
    args = p.parse_args()

    cohort = _read_cohort(Path(args.cohort_file))
    summary = run_staged_fmp_fundamentals_dry_run(
        cohort=cohort,
        out_dir=Path(args.out_dir),
        available_lag_days=int(args.available_lag_days),
        limit=int(args.limit),
        source_label=str(args.source_label),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
