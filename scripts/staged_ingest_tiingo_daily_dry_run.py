"""Stage Tiingo daily equity snapshot for a limited dry-run cohort."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def _load_env_key() -> str:
    key = str(os.getenv("TIINGO_API_KEY", "")).strip()
    if key:
        return key
    env_path = Path(".env")
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "TIINGO_API_KEY":
                key = v.strip().strip("\"'").strip()
                if key:
                    return key
    raise ValueError("TIINGO_API_KEY not found in environment or .env")


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


def _fetch_one(session: requests.Session, ticker: str, api_key: str, start_date: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    url = f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices"
    params = {"startDate": str(start_date), "resampleFreq": "daily"}
    headers = {"Authorization": f"Token {api_key}"}
    try:
        response = session.get(url, params=params, headers=headers, timeout=30)
        payload: Any
        try:
            payload = response.json()
        except Exception:
            payload = {"raw_text": response.text[:4000]}
    except Exception as exc:
        return pd.DataFrame(), {
            "ticker": ticker,
            "status_code": 0,
            "rows": 0,
            "error": f"tiingo_request_error:{type(exc).__name__}",
        }
    ok = response.status_code == 200 and isinstance(payload, list) and len(payload) > 0
    if not ok:
        return pd.DataFrame(), {
            "ticker": ticker,
            "status_code": int(response.status_code),
            "rows": 0,
            "error": f"tiingo_http_{response.status_code}",
        }

    frame = pd.DataFrame(payload)
    for col in ["date", "open", "high", "low", "close", "volume", "adjClose"]:
        if col not in frame.columns:
            frame[col] = pd.NA
    out = frame[["date", "open", "high", "low", "close", "volume", "adjClose"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "volume", "adjClose"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.loc[out["date"].notna()].copy()
    out["ticker"] = f"{ticker}.US"
    out["source"] = "tiingo_phase2a_dryrun"
    out["fetch_ts"] = datetime.now(UTC).isoformat()
    return out, {
        "ticker": ticker,
        "status_code": int(response.status_code),
        "rows": int(len(out)),
        "error": "",
    }


def run_staged_tiingo_dry_run(
    cohort: list[str],
    out_dir: Path,
    start_date: str = "2000-01-01",
    source_label: str = "tiingo_phase2a_dryrun",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_payloads"
    raw_dir.mkdir(parents=True, exist_ok=True)

    api_key = _load_env_key()
    session = requests.Session()
    records: list[pd.DataFrame] = []
    report_rows: list[dict[str, Any]] = []

    for i, ticker in enumerate(cohort, start=1):
        frame, rep = _fetch_one(session=session, ticker=ticker, api_key=api_key, start_date=start_date)
        report_rows.append(rep)
        (raw_dir / f"{ticker}.json").write_text(
            json.dumps({"ticker": ticker, "report": rep}, indent=2),
            encoding="utf-8",
        )
        if not frame.empty:
            frame["source"] = str(source_label)
            records.append(frame)
        if i % 20 == 0 or i == len(cohort):
            ok_count = sum(1 for x in report_rows if not x.get("error"))
            print(f"[tiingo] {i}/{len(cohort)} complete ok={ok_count} fail={len(report_rows)-ok_count}")

    combined = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=["date", "open", "high", "low", "close", "volume", "adjClose", "ticker", "source", "fetch_ts"]
    )
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    parquet_path = out_dir / "tiingo_daily_snapshot.parquet"
    combined.to_parquet(parquet_path, index=False)
    report = pd.DataFrame(report_rows).sort_values(["error", "ticker"]).reset_index(drop=True)
    report_path = out_dir / "tiingo_fetch_report.csv"
    report.to_csv(report_path, index=False)

    summary = {
        "rows": int(len(combined)),
        "tickers_requested": int(len(cohort)),
        "tickers_succeeded": int((report["error"].astype(str) == "").sum()) if not report.empty else 0,
        "tickers_failed": int((report["error"].astype(str) != "").sum()) if not report.empty else 0,
        "parquet_path": str(parquet_path),
        "report_path": str(report_path),
    }
    (out_dir / "tiingo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Stage Tiingo daily snapshot for a dry-run ticker cohort.")
    p.add_argument("--cohort-file", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start-date", default="2000-01-01")
    p.add_argument("--source-label", default="tiingo_phase2a_dryrun")
    args = p.parse_args()

    cohort = _read_cohort(Path(args.cohort_file))
    summary = run_staged_tiingo_dry_run(
        cohort=cohort,
        out_dir=Path(args.out_dir),
        start_date=str(args.start_date),
        source_label=str(args.source_label),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
