"""Run Phase 3 full bulk ingest with strict validation and optional promotion."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from quant_lab.data.tiingo_universe import (
    DEFAULT_TIINGO_US_COMMON_EQUITY_MANIFEST,
    load_symbol_manifest,
)
from quant_lab.utils.env import load_project_env


def _norm(raw: object) -> str:
    s = str(raw or "").strip().upper()
    if s.endswith(".US"):
        s = s[:-3]
    return s.replace(".", "-")


def _normalize_date_series(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_convert(None).dt.normalize()


def _run(cmd: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _load_ticker_universe(
    equities_root: Path,
    symbol_file: Path,
    max_tickers: int = 0,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if symbol_file.exists():
        for raw in load_symbol_manifest(symbol_file):
            t = _norm(raw)
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    if not out:
        md = pd.read_parquet(equities_root / "metadata.parquet", columns=["ticker"])
        for raw in md["ticker"].astype(str).tolist():
            t = _norm(raw)
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    if int(max_tickers) > 0:
        out = out[: int(max_tickers)]
    return out


def _enrich_metadata(base_metadata: pd.DataFrame, tiingo_snapshot: pd.DataFrame) -> pd.DataFrame:
    md = base_metadata.copy()
    md["ticker"] = md["ticker"].astype(str).str.strip().str.upper()
    existing_raw = set(md["ticker"].astype(str))
    if tiingo_snapshot.empty:
        return md

    snap = tiingo_snapshot.copy()
    snap["ticker"] = snap["ticker"].astype(str).str.strip().str.upper()
    snap["date"] = _normalize_date_series(snap["date"])
    g = snap.groupby("ticker")["date"].agg(first_date="min", last_date="max").reset_index()
    global_last = g["last_date"].max()
    g["active_flag"] = g["last_date"].eq(global_last)

    missing = g.loc[~g["ticker"].isin(existing_raw)].copy()
    if missing.empty:
        return md

    extra = pd.DataFrame(
        {
            "ticker": missing["ticker"],
            "name": pd.NA,
            "exchange": pd.NA,
            "sector": pd.NA,
            "industry": pd.NA,
            "first_date": missing["first_date"],
            "last_date": missing["last_date"],
            "active_flag": missing["active_flag"].astype(bool),
            "source": "tiingo_phase3_inferred",
        }
    )
    out = pd.concat([md, extra], ignore_index=True)
    out = out.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run Phase 3 full bulk ingest and strict validation.")
    p.add_argument("--equities-root", default="data/equities")
    p.add_argument("--fundamentals-path", default="data/fundamentals/fundamentals_fmp.parquet")
    p.add_argument("--warehouse-root", default="data/warehouse")
    p.add_argument("--symbol-file", default=str(DEFAULT_TIINGO_US_COMMON_EQUITY_MANIFEST))
    p.add_argument("--staging-root", default="data/staging/phase3")
    p.add_argument("--validation-root", default="results/data_validation")
    p.add_argument("--results-root", default="results/ingest/phase3")
    p.add_argument("--run-id", default="", help="Optional existing run id/timestamp to resume in-place.")
    p.add_argument("--max-tickers", type=int, default=0, help="Optional cap for emergency partial run.")
    p.add_argument("--tiingo-batch-size", type=int, default=100)
    p.add_argument("--tiingo-batch-pause-seconds", type=float, default=0.5)
    p.add_argument("--tiingo-timeout-seconds", type=float, default=30.0)
    p.add_argument("--tiingo-max-retries", type=int, default=6)
    p.add_argument("--tiingo-initial-backoff-seconds", type=float, default=2.0)
    p.add_argument("--tiingo-backoff-multiplier", type=float, default=2.0)
    p.add_argument("--tiingo-max-backoff-seconds", type=float, default=120.0)
    p.add_argument("--tiingo-resume", type=int, choices=[0, 1], default=1)
    p.add_argument("--duckdb-memory-limit", default="2GB")
    p.add_argument("--duckdb-threads", type=int, default=4)
    p.add_argument("--promote", type=int, choices=[0, 1], default=1, help="Promote staged warehouse into canonical data/warehouse after validation succeeds.")
    args = p.parse_args()
    load_project_env()

    ts = str(args.run_id).strip() or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.staging_root) / ts
    raw_root = run_root / "raw"
    source_root = run_root / "source"
    staged_equities = source_root / "equities"
    staged_fundamentals = source_root / "fundamentals"
    staged_warehouse = run_root / "warehouse"
    result_dir = Path(args.results_root) / ts
    for d in [raw_root, staged_equities, staged_fundamentals, staged_warehouse, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    equities_root = Path(args.equities_root)
    fundamentals_path = Path(args.fundamentals_path)
    baseline_warehouse_root = Path(args.warehouse_root)
    validation_phase_root = Path(args.validation_root) / "phase3"

    tickers = _load_ticker_universe(
        equities_root=equities_root,
        symbol_file=Path(args.symbol_file),
        max_tickers=int(args.max_tickers),
    )
    cohort_path = run_root / "bulk_tickers.csv"
    pd.DataFrame({"ticker": tickers}).to_csv(cohort_path, index=False)

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    # Priority 1: FMP fundamentals expansion.
    _run(
        [
            sys.executable,
            "scripts/build_fundamentals_database.py",
            "--source",
            "fmp",
            "--tickers-file",
            str(cohort_path),
            "--output",
            str(raw_root / "fmp" / "fundamentals_fmp_bulk.parquet"),
            "--available-lag-days",
            "60",
            "--limit",
            "120",
            "--statement-period",
            "quarter",
            "--batch-size",
            "50",
            "--pause-seconds",
            "0.1",
            "--batch-pause-seconds",
            "0.5",
            "--timeout-seconds",
            "20",
            "--max-retries",
            "2",
        ],
        env=env,
    )

    # Priority 2: Tiingo price refresh/completion.
    _run(
        [
            sys.executable,
            "scripts/staged_ingest_tiingo_daily_dry_run.py",
            "--cohort-file",
            str(cohort_path),
            "--out-dir",
            str(raw_root / "tiingo"),
            "--start-date",
            "1900-01-01",
            "--source-label",
            "tiingo_phase3_bulk",
            "--batch-size",
            str(args.tiingo_batch_size),
            "--batch-pause-seconds",
            str(args.tiingo_batch_pause_seconds),
            "--timeout-seconds",
            str(args.tiingo_timeout_seconds),
            "--max-retries",
            str(args.tiingo_max_retries),
            "--initial-backoff-seconds",
            str(args.tiingo_initial_backoff_seconds),
            "--backoff-multiplier",
            str(args.tiingo_backoff_multiplier),
            "--max-backoff-seconds",
            str(args.tiingo_max_backoff_seconds),
            "--resume",
            str(args.tiingo_resume),
        ],
        env=env,
    )

    _run(
        [
            sys.executable,
            "scripts/continue_phase3_bulk_ingest_from_raw.py",
            "--run-root",
            str(run_root),
            "--equities-root",
            str(equities_root),
            "--fundamentals-path",
            str(fundamentals_path),
            "--warehouse-root",
            str(baseline_warehouse_root),
            "--validation-root",
            str(validation_phase_root),
            "--duckdb-memory-limit",
            str(args.duckdb_memory_limit),
            "--duckdb-threads",
            str(args.duckdb_threads),
        ],
        env=env,
    )

    # Collect summary metrics.
    tiingo_report = pd.read_csv(raw_root / "tiingo" / "tiingo_fetch_report.csv")
    stability = pd.read_parquet(staged_warehouse / "ticker_id_stability_report.parquet")
    staged_selected_path = staged_warehouse / "equity_prices_daily.parquet"
    staged_fund_path = staged_warehouse / "equity_fundamentals_pit.parquet"
    base_selected_path = baseline_warehouse_root / "equity_prices_daily.parquet"
    base_daily_path = equities_root / "daily_ohlcv.parquet"
    base_fund_path = fundamentals_path
    raw_fmp_path = raw_root / "fmp" / "fundamentals_fmp_bulk.parquet"

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={max(int(args.duckdb_threads), 1)}")
        norm_expr = "REPLACE(CASE WHEN UPPER(CAST(ticker AS VARCHAR)) LIKE '%.US' THEN LEFT(UPPER(CAST(ticker AS VARCHAR)), LENGTH(UPPER(CAST(ticker AS VARCHAR))) - 3) ELSE UPPER(CAST(ticker AS VARCHAR)) END, '.', '-')"
        new_price_rows_added = int(con.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT date, ticker_id FROM read_parquet('{staged_selected_path}')
                EXCEPT
                SELECT CAST(date AS DATE) AS date, ticker_id FROM read_parquet('{base_selected_path}')
            )
            """
        ).fetchone()[0])
        replaced_by_precedence = int(con.execute(
            f"""
            WITH base_src AS (
                SELECT CAST(date AS DATE) AS date, ticker_id, CAST(source AS VARCHAR) AS base_source
                FROM read_parquet('{base_selected_path}')
            )
            SELECT COUNT(*)
            FROM read_parquet('{staged_selected_path}') s
            JOIN base_src b USING (date, ticker_id)
            WHERE COALESCE(CAST(s.source AS VARCHAR), '') != COALESCE(b.base_source, '')
            """
        ).fetchone()[0])
        base_fund_rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{base_fund_path}')").fetchone()[0])
        staged_fund_rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{staged_fund_path}')").fetchone()[0])
        new_fund_rows_added = int(staged_fund_rows - base_fund_rows)
        new_symbols_seen = int(con.execute(
            f"""
            WITH base_symbols AS (
                SELECT DISTINCT {norm_expr} AS sym
                FROM read_parquet('{base_daily_path}')
            ), final_symbols AS (
                SELECT DISTINCT CAST(canonical_symbol AS VARCHAR) AS sym
                FROM read_parquet('{staged_selected_path}')
            )
            SELECT COUNT(*)
            FROM final_symbols
            WHERE sym NOT IN (SELECT sym FROM base_symbols)
            """
        ).fetchone()[0])
        fmp_succeeded_tickers = int(
            con.execute(f"SELECT COUNT(DISTINCT UPPER(CAST(ticker AS VARCHAR))) FROM read_parquet('{raw_fmp_path}')").fetchone()[0]
        ) if raw_fmp_path.exists() else 0
    finally:
        con.close()

    ticker_id_reused = int(stability["status"].eq("reused").sum())
    ticker_id_new = int(stability["status"].eq("new").sum())
    ticker_id_changed = int(stability["status"].eq("changed").sum())

    val_summary_path = validation_phase_root / "latest" / "validation_summary.json"
    val = json.loads(val_summary_path.read_text(encoding="utf-8"))
    cov = val.get("coverage", {})

    ingest_summary = {
        "run_root": str(run_root),
        "requested_tickers": int(len(tickers)),
        "tiingo_succeeded_tickers": int((tiingo_report["rows"] > 0).sum()) if not tiingo_report.empty else 0,
        "fmp_succeeded_tickers": int(fmp_succeeded_tickers),
        "new_price_rows_added": new_price_rows_added,
        "price_rows_replaced_by_precedence": replaced_by_precedence,
        "new_fundamentals_rows_added": new_fund_rows_added,
        "new_symbols_seen": new_symbols_seen,
        "ticker_id_reuse": ticker_id_reused,
        "ticker_id_new": ticker_id_new,
        "ticker_id_changed": ticker_id_changed,
        "final_price_ticker_count": int(cov.get("unique_price_tickers", 0)),
        "final_fundamentals_covered_price_tickers": int(cov.get("fundamentals_covered_tickers", 0)),
        "final_coverage_ratio": float(cov.get("fundamentals_coverage_ratio", 0.0)),
        "validation_failed": bool(val.get("thresholds", {}).get("failed", True)),
        "validation_summary_path": str(val_summary_path),
        "warehouse_staged_path": str(staged_warehouse),
        "symbol_file": str(args.symbol_file),
        "promoted": False,
    }

    # Promote staged validated warehouse into canonical path only when requested.
    if ingest_summary["validation_failed"]:
        (run_root / "phase3_ingest_summary.json").write_text(json.dumps(ingest_summary, indent=2), encoding="utf-8")
        (result_dir / "phase3_ingest_summary.json").write_text(json.dumps(ingest_summary, indent=2), encoding="utf-8")
        raise RuntimeError("Phase 3 validation failed; refusing to promote staged warehouse.")

    if bool(args.promote):
        backup_root = Path(args.results_root) / "backups" / ts
        backup_root.mkdir(parents=True, exist_ok=True)
        for name in [
            "security_master.parquet",
            "symbol_history.parquet",
            "ticker_id_stability_report.parquet",
            "symbol_collision_report.parquet",
            "equity_prices_daily.parquet",
            "equity_prices_daily_versions.parquet",
            "equity_fundamentals_pit.parquet",
            "universe_membership_daily.parquet",
            "ingestion_audit.parquet",
        ]:
            src = baseline_warehouse_root / name
            if src.exists():
                shutil.copy2(src, backup_root / name)

        for file_path in staged_warehouse.glob("*.parquet"):
            shutil.copy2(file_path, baseline_warehouse_root / file_path.name)
        ingest_summary["promoted"] = True
    else:
        ingest_summary["promotion_skipped"] = True

    (run_root / "phase3_ingest_summary.json").write_text(json.dumps(ingest_summary, indent=2), encoding="utf-8")
    (result_dir / "phase3_ingest_summary.json").write_text(json.dumps(ingest_summary, indent=2), encoding="utf-8")
    print(json.dumps(ingest_summary, indent=2))


if __name__ == "__main__":
    main()
