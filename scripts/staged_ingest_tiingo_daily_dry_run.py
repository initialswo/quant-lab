"""Stage Tiingo daily equity snapshot for a limited dry-run cohort."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from quant_lab.utils.env import get_required_env, load_project_env


REPORT_COLUMNS = [
    "ticker",
    "status_code",
    "rows",
    "error",
    "attempts",
    "retries_used",
    "chunk_path",
    "fetch_ts",
]
SNAPSHOT_COLUMNS = ["date", "open", "high", "low", "close", "volume", "adjClose", "ticker", "source", "fetch_ts"]


def _load_env_key() -> str:
    load_project_env()
    return get_required_env("TIINGO_API_KEY")


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


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SNAPSHOT_COLUMNS)


def _coerce_snapshot_frame(frame: pd.DataFrame, ticker: str, source_label: str) -> pd.DataFrame:
    for col in ["date", "open", "high", "low", "close", "volume", "adjClose"]:
        if col not in frame.columns:
            frame[col] = pd.NA
    out = frame[["date", "open", "high", "low", "close", "volume", "adjClose"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "volume", "adjClose"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.loc[out["date"].notna()].copy()
    out["ticker"] = f"{ticker}.US"
    out["source"] = str(source_label)
    out["fetch_ts"] = datetime.now(UTC).isoformat()
    return out[SNAPSHOT_COLUMNS]


def _response_payload(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return {"raw_text": response.text[:4000]}


def _retry_delay_seconds(
    response: requests.Response | None,
    attempt: int,
    initial_backoff_seconds: float,
    backoff_multiplier: float,
    max_backoff_seconds: float,
) -> float:
    if response is not None:
        retry_after = str(response.headers.get("Retry-After", "")).strip()
        if retry_after:
            try:
                return max(0.0, min(float(retry_after), float(max_backoff_seconds)))
            except ValueError:
                pass
    delay = float(initial_backoff_seconds) * (float(backoff_multiplier) ** max(0, attempt - 1))
    return max(0.0, min(delay, float(max_backoff_seconds)))


def _fetch_one(
    session: requests.Session,
    ticker: str,
    api_key: str,
    start_date: str,
    source_label: str,
    timeout_seconds: float,
    max_retries: int,
    initial_backoff_seconds: float,
    backoff_multiplier: float,
    max_backoff_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    url = f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices"
    params = {"startDate": str(start_date), "resampleFreq": "daily"}
    headers = {"Authorization": f"Token {api_key}"}

    for attempt in range(1, int(max_retries) + 2):
        response: requests.Response | None = None
        try:
            response = session.get(url, params=params, headers=headers, timeout=float(timeout_seconds))
            payload = _response_payload(response)
        except Exception as exc:
            retryable = attempt <= int(max_retries)
            if retryable:
                time.sleep(
                    _retry_delay_seconds(
                        response=None,
                        attempt=attempt,
                        initial_backoff_seconds=initial_backoff_seconds,
                        backoff_multiplier=backoff_multiplier,
                        max_backoff_seconds=max_backoff_seconds,
                    )
                )
                continue
            return _empty_frame(), {
                "ticker": ticker,
                "status_code": 0,
                "rows": 0,
                "error": f"tiingo_request_error:{type(exc).__name__}",
                "attempts": attempt,
                "retries_used": max(0, attempt - 1),
                "chunk_path": "",
                "fetch_ts": datetime.now(UTC).isoformat(),
            }

        ok = response.status_code == 200 and isinstance(payload, list) and len(payload) > 0
        if ok:
            out = _coerce_snapshot_frame(pd.DataFrame(payload), ticker=ticker, source_label=source_label)
            return out, {
                "ticker": ticker,
                "status_code": int(response.status_code),
                "rows": int(len(out)),
                "error": "",
                "attempts": attempt,
                "retries_used": max(0, attempt - 1),
                "chunk_path": "",
                "fetch_ts": datetime.now(UTC).isoformat(),
            }

        retryable_http = int(response.status_code) == 429 or int(response.status_code) >= 500
        if retryable_http and attempt <= int(max_retries):
            time.sleep(
                _retry_delay_seconds(
                    response=response,
                    attempt=attempt,
                    initial_backoff_seconds=initial_backoff_seconds,
                    backoff_multiplier=backoff_multiplier,
                    max_backoff_seconds=max_backoff_seconds,
                )
            )
            continue

        error = f"tiingo_http_{int(response.status_code)}"
        if int(response.status_code) == 200 and isinstance(payload, list) and len(payload) == 0:
            error = "tiingo_http_200_empty"
        return _empty_frame(), {
            "ticker": ticker,
            "status_code": int(response.status_code),
            "rows": 0,
            "error": error,
            "attempts": attempt,
            "retries_used": max(0, attempt - 1),
            "chunk_path": "",
            "fetch_ts": datetime.now(UTC).isoformat(),
        }

    raise RuntimeError("unreachable")


def _load_report_state(raw_dir: Path, report_path: Path) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    if report_path.exists():
        report = pd.read_csv(report_path)
        for row in report.to_dict(orient="records"):
            ticker = str(row.get("ticker", "")).strip().upper()
            if ticker:
                state[ticker] = row
    if raw_dir.exists():
        for payload_path in sorted(raw_dir.glob("*.json")):
            try:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            report = payload.get("report", {})
            ticker = str(report.get("ticker") or payload.get("ticker") or payload_path.stem).strip().upper()
            if ticker:
                state[ticker] = report
    return state


def _write_report(report_rows: dict[str, dict[str, Any]], report_path: Path) -> pd.DataFrame:
    if report_rows:
        report = pd.DataFrame(report_rows.values())
    else:
        report = pd.DataFrame(columns=REPORT_COLUMNS)
    for col in REPORT_COLUMNS:
        if col not in report.columns:
            report[col] = pd.NA
    report = report[REPORT_COLUMNS].sort_values(["error", "ticker"]).reset_index(drop=True)
    report.to_csv(report_path, index=False)
    return report


def _combine_ticker_chunks(chunk_dir: Path, parquet_path: Path) -> int:
    chunk_paths = sorted(chunk_dir.glob("*.parquet"))
    if not chunk_paths:
        _empty_frame().to_parquet(parquet_path, index=False)
        return 0

    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for chunk_path in chunk_paths:
            table = pq.read_table(chunk_path)
            row_count += int(table.num_rows)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return row_count


def _write_summary(
    out_dir: Path,
    report: pd.DataFrame,
    parquet_path: Path,
    tickers_requested: int,
    resumed_successes: int,
) -> dict[str, Any]:
    rows = int(pq.ParquetFile(parquet_path).metadata.num_rows) if parquet_path.exists() else 0
    summary = {
        "rows": rows,
        "tickers_requested": int(tickers_requested),
        "tickers_succeeded": int((report["error"].fillna("") == "").sum()) if not report.empty else 0,
        "tickers_failed": int((report["error"].fillna("") != "").sum()) if not report.empty else 0,
        "resumed_success_tickers": int(resumed_successes),
        "parquet_path": str(parquet_path),
        "report_path": str(out_dir / "tiingo_fetch_report.csv"),
    }
    (out_dir / "tiingo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_staged_tiingo_dry_run(
    cohort: list[str],
    out_dir: Path,
    start_date: str = "2000-01-01",
    source_label: str = "tiingo_phase2a_dryrun",
    batch_size: int = 100,
    batch_pause_seconds: float = 0.5,
    timeout_seconds: float = 30.0,
    max_retries: int = 6,
    initial_backoff_seconds: float = 2.0,
    backoff_multiplier: float = 2.0,
    max_backoff_seconds: float = 120.0,
    resume: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_payloads"
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = out_dir / "ticker_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "tiingo_fetch_report.csv"
    parquet_path = out_dir / "tiingo_daily_snapshot.parquet"

    if not resume and (report_path.exists() or any(chunk_dir.glob("*.parquet"))):
        raise ValueError(f"out_dir already contains prior Tiingo artifacts: {out_dir}; use resume or a new out_dir")

    api_key = _load_env_key()
    session = requests.Session()

    report_rows = _load_report_state(raw_dir=raw_dir, report_path=report_path) if resume else {}
    completed_successes = {path.stem.upper() for path in chunk_dir.glob("*.parquet")}
    for ticker in sorted(completed_successes):
        if ticker in report_rows and str(report_rows[ticker].get("error", "")).strip() == "":
            continue
        chunk_rows = int(pq.ParquetFile(chunk_dir / f"{ticker}.parquet").metadata.num_rows)
        report_rows[ticker] = {
            "ticker": ticker,
            "status_code": 200,
            "rows": chunk_rows,
            "error": "",
            "attempts": 0,
            "retries_used": 0,
            "chunk_path": str(chunk_dir / f"{ticker}.parquet"),
            "fetch_ts": datetime.now(UTC).isoformat(),
        }

    pending = [ticker for ticker in cohort if ticker not in completed_successes]
    resumed_successes = len(completed_successes)

    for batch_start in range(0, len(pending), max(1, int(batch_size))):
        batch = pending[batch_start : batch_start + max(1, int(batch_size))]
        for ticker in batch:
            frame, rep = _fetch_one(
                session=session,
                ticker=ticker,
                api_key=api_key,
                start_date=start_date,
                source_label=source_label,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                initial_backoff_seconds=initial_backoff_seconds,
                backoff_multiplier=backoff_multiplier,
                max_backoff_seconds=max_backoff_seconds,
            )
            chunk_path = chunk_dir / f"{ticker}.parquet"
            if not frame.empty:
                frame.to_parquet(chunk_path, index=False)
                rep["chunk_path"] = str(chunk_path)
                completed_successes.add(ticker)
            else:
                rep["chunk_path"] = ""
            report_rows[ticker] = rep
            (raw_dir / f"{ticker}.json").write_text(
                json.dumps({"ticker": ticker, "report": rep}, indent=2),
                encoding="utf-8",
            )

        report = _write_report(report_rows=report_rows, report_path=report_path)
        ok_count = int((report["error"].fillna("") == "").sum()) if not report.empty else 0
        fail_count = int((report["error"].fillna("") != "").sum()) if not report.empty else 0
        processed = min(batch_start + len(batch), len(pending))
        total_done = resumed_successes + processed
        print(f"[tiingo] {total_done}/{len(cohort)} complete ok={ok_count} fail={fail_count}")
        if batch_pause_seconds > 0 and (batch_start + len(batch)) < len(pending):
            time.sleep(float(batch_pause_seconds))

    report = _write_report(report_rows=report_rows, report_path=report_path)
    _combine_ticker_chunks(chunk_dir=chunk_dir, parquet_path=parquet_path)
    summary = _write_summary(
        out_dir=out_dir,
        report=report,
        parquet_path=parquet_path,
        tickers_requested=len(cohort),
        resumed_successes=resumed_successes,
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Stage Tiingo daily snapshot for a dry-run ticker cohort.")
    p.add_argument("--cohort-file", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start-date", default="2000-01-01")
    p.add_argument("--source-label", default="tiingo_phase2a_dryrun")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--batch-pause-seconds", type=float, default=0.5)
    p.add_argument("--timeout-seconds", type=float, default=30.0)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--initial-backoff-seconds", type=float, default=2.0)
    p.add_argument("--backoff-multiplier", type=float, default=2.0)
    p.add_argument("--max-backoff-seconds", type=float, default=120.0)
    p.add_argument("--resume", type=int, choices=[0, 1], default=1)
    args = p.parse_args()

    cohort = _read_cohort(Path(args.cohort_file))
    summary = run_staged_tiingo_dry_run(
        cohort=cohort,
        out_dir=Path(args.out_dir),
        start_date=str(args.start_date),
        source_label=str(args.source_label),
        batch_size=int(args.batch_size),
        batch_pause_seconds=float(args.batch_pause_seconds),
        timeout_seconds=float(args.timeout_seconds),
        max_retries=int(args.max_retries),
        initial_backoff_seconds=float(args.initial_backoff_seconds),
        backoff_multiplier=float(args.backoff_multiplier),
        max_backoff_seconds=float(args.max_backoff_seconds),
        resume=bool(int(args.resume)),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
