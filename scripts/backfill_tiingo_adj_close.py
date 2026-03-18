"""Backfill true Tiingo adjusted closes into the active research parquet store."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_lab.utils.env import get_required_env, load_project_env

TIINGO_SOURCES = {"tiingo", "tiingo_cache"}
BATCH_SIZE = 50
SLEEP_SECONDS = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-path", default="data/equities/daily_ohlcv.parquet")
    parser.add_argument("--results-root", default="results/adj_close_backfill")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sleep-seconds", type=float, default=SLEEP_SECONDS)
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional cap for a smaller controlled run.")
    return parser.parse_args()


def _load_env_key() -> str:
    load_project_env()
    return get_required_env("TIINGO_API_KEY")

def _to_vendor_symbol(store_ticker: str) -> str:
    symbol = str(store_ticker).strip().upper()
    if symbol.endswith(".US"):
        symbol = symbol[:-3]
    return symbol.replace(".", "-")


def _to_store_ticker(vendor_symbol: str) -> str:
    symbol = str(vendor_symbol).strip().upper().replace(".", "-")
    return f"{symbol}.US"


def _fetch_tiingo_adj_close(
    session: requests.Session,
    api_key: str,
    store_ticker: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    vendor_symbol = _to_vendor_symbol(store_ticker)
    url = f"https://api.tiingo.com/tiingo/daily/{vendor_symbol.lower()}/prices"
    params = {"startDate": "1900-01-01", "resampleFreq": "daily"}
    headers = {"Authorization": f"Token {api_key}"}
    try:
        response = session.get(url, params=params, headers=headers, timeout=30)
        try:
            payload: Any = response.json()
        except Exception:
            payload = {"raw_text": response.text[:4000]}
    except Exception as exc:
        return pd.DataFrame(columns=["ticker", "date", "adj_close_vendor"]), {
            "ticker": str(store_ticker),
            "status_code": 0,
            "rows": 0,
            "error": f"request_error:{type(exc).__name__}",
        }
    if response.status_code != 200 or not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=["ticker", "date", "adj_close_vendor"]), {
            "ticker": str(store_ticker),
            "status_code": int(response.status_code),
            "rows": 0,
            "error": f"tiingo_http_{response.status_code}",
        }
    frame = pd.DataFrame(payload)
    for col in ["date", "close", "adjClose"]:
        if col not in frame.columns:
            frame[col] = pd.NA
    out = frame[["date", "close", "adjClose"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["adjClose"] = pd.to_numeric(out["adjClose"], errors="coerce")
    out = out.loc[out["date"].notna()].copy()
    out["ticker"] = _to_store_ticker(vendor_symbol)
    out = out.rename(columns={"adjClose": "adj_close_vendor"})
    out = out[["ticker", "date", "close", "adj_close_vendor"]]
    return out, {
        "ticker": str(store_ticker),
        "status_code": int(response.status_code),
        "rows": int(len(out)),
        "error": "",
    }


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), max(1, int(size)))]


def main() -> None:
    args = _parse_args()
    daily_path = Path(str(args.daily_path)).expanduser()
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing parquet file: {daily_path}")

    results_root = Path(str(args.results_root)).expanduser()
    run_dir = results_root / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(daily_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["date", "ticker", "close", "adj_close", "source"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"daily parquet missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["source"] = df["source"].astype(str).str.strip().str.lower()

    rows_total = int(len(df))
    tiingo_mask = df["source"].isin(TIINGO_SOURCES)
    rows_tiingo = int(tiingo_mask.sum())
    tickers = sorted(df.loc[tiingo_mask, "ticker"].astype(str).dropna().unique().tolist())
    if int(args.max_tickers) > 0:
        tickers = tickers[: int(args.max_tickers)]

    backup_path = run_dir / "daily_ohlcv_before.parquet"
    df.to_parquet(backup_path, index=False)

    report_rows: list[dict[str, Any]] = []
    session = requests.Session()
    api_key = _load_env_key()
    rows_updated = 0
    tickers_failed: list[str] = []
    processed = 0

    for batch_idx, batch in enumerate(_chunked(tickers, int(args.batch_size)), start=1):
        batch_vendor_frames: list[pd.DataFrame] = []
        for ticker in batch:
            vendor_df, rep = _fetch_tiingo_adj_close(session=session, api_key=api_key, store_ticker=ticker)
            report_rows.append(rep)
            processed += 1
            if rep.get("error"):
                tickers_failed.append(str(ticker))
            else:
                batch_vendor_frames.append(vendor_df)
            time.sleep(float(args.sleep_seconds))

        if batch_vendor_frames:
            vendor_batch = pd.concat(batch_vendor_frames, ignore_index=True)
            vendor_batch = vendor_batch.drop_duplicates(subset=["ticker", "date"], keep="last")
            batch_mask = tiingo_mask & df["ticker"].isin(batch)
            batch_slice = df.loc[batch_mask, ["ticker", "date", "adj_close"]].copy()
            batch_slice["row_id"] = batch_slice.index
            merged = batch_slice.merge(
                vendor_batch[["ticker", "date", "adj_close_vendor"]],
                on=["ticker", "date"],
                how="left",
                sort=False,
            )
            update_mask = merged["adj_close_vendor"].notna()
            if bool(update_mask.any()):
                row_ids = merged.loc[update_mask, "row_id"].to_numpy()
                new_vals = merged.loc[update_mask, "adj_close_vendor"].to_numpy()
                old_vals = pd.to_numeric(df.loc[row_ids, "adj_close"], errors="coerce").to_numpy()
                changed_mask = pd.Series(old_vals).ne(pd.Series(new_vals)).fillna(True).to_numpy()
                if changed_mask.any():
                    df.loc[row_ids[changed_mask], "adj_close"] = new_vals[changed_mask]
                    rows_updated += int(changed_mask.sum())

        ok_count = sum(1 for row in report_rows if not row.get("error"))
        print(
            f"[batch {batch_idx}] processed={processed}/{len(tickers)} "
            f"ok={ok_count} fail={len(report_rows) - ok_count} rows_updated={rows_updated}"
        )

    if int(len(df)) != rows_total:
        raise RuntimeError("Row count changed unexpectedly before write; aborting")

    df.to_parquet(daily_path, index=False)

    reloaded = pd.read_parquet(daily_path, columns=["ticker", "source", "close", "adj_close"])
    if int(len(reloaded)) != rows_total:
        raise RuntimeError("Row count changed unexpectedly after write; backfill is not safe")

    diff_mask = pd.to_numeric(reloaded["adj_close"], errors="coerce").ne(
        pd.to_numeric(reloaded["close"], errors="coerce")
    )
    diff_rows = int(diff_mask.sum())
    diff_sample = reloaded.loc[diff_mask, ["ticker", "source", "close", "adj_close"]].head(10)

    report_df = pd.DataFrame(report_rows)
    report_path = run_dir / "tiingo_adj_close_report.csv"
    report_df.to_csv(report_path, index=False)

    summary = {
        "daily_path": str(daily_path),
        "backup_path": str(backup_path),
        "report_path": str(report_path),
        "rows_total": rows_total,
        "rows_tiingo": rows_tiingo,
        "rows_updated": int(rows_updated),
        "tickers_processed": int(len(tickers)),
        "tickers_failed": int(len(set(tickers_failed))),
        "rows_where_adj_close_differs": diff_rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not diff_sample.empty:
        print("sample_rows_where_adj_close_differs")
        print(diff_sample.to_string(index=False))


if __name__ == "__main__":
    main()
