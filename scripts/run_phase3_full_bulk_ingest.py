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

import pandas as pd

from quant_lab.data.tiingo_universe import (
    DEFAULT_TIINGO_US_COMMON_EQUITY_MANIFEST,
    load_symbol_manifest,
)


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
    out = sorted(out)
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
    p.add_argument("--max-tickers", type=int, default=0, help="Optional cap for emergency partial run.")
    p.add_argument("--promote", type=int, choices=[0, 1], default=1, help="Promote staged warehouse into canonical data/warehouse after validation succeeds.")
    args = p.parse_args()

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
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
        ],
        env=env,
    )

    # Build staged source tables.
    base_daily = pd.read_parquet(equities_root / "daily_ohlcv.parquet")
    base_meta = pd.read_parquet(equities_root / "metadata.parquet")
    base_membership = pd.read_parquet(equities_root / "universe_membership.parquet")
    base_fund = pd.read_parquet(fundamentals_path)
    base_selected_prices = pd.read_parquet(baseline_warehouse_root / "equity_prices_daily.parquet")

    tiingo_new = pd.read_parquet(raw_root / "tiingo" / "tiingo_daily_snapshot.parquet")
    tiingo_append = (
        tiingo_new.rename(columns={"adjClose": "adj_close"})[
            ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
        ].copy()
        if not tiingo_new.empty
        else pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"])
    )
    merged_daily = pd.concat([base_daily, tiingo_append], ignore_index=True)
    merged_daily["date"] = _normalize_date_series(merged_daily["date"])
    merged_daily["ticker"] = merged_daily["ticker"].astype(str).str.upper()
    merged_daily = merged_daily.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    fmp_new = pd.read_parquet(raw_root / "fmp" / "fundamentals_fmp_bulk.parquet")
    fmp_append = (
        fmp_new[
            [
                "ticker",
                "period_end",
                "available_date",
                "revenue",
                "cogs",
                "gross_profit",
                "total_assets",
                "shareholders_equity",
                "net_income",
                "shares_outstanding",
            ]
        ].copy()
        if not fmp_new.empty
        else pd.DataFrame(columns=base_fund.columns)
    )
    merged_fund = pd.concat([base_fund, fmp_append], ignore_index=True)
    merged_fund["ticker"] = merged_fund["ticker"].astype(str).str.upper().map(_norm)
    merged_fund["period_end"] = pd.to_datetime(merged_fund["period_end"], errors="coerce").dt.normalize()
    merged_fund["available_date"] = pd.to_datetime(merged_fund["available_date"], errors="coerce").dt.normalize()
    merged_fund = (
        merged_fund.dropna(subset=["ticker", "period_end", "available_date"])
        .drop_duplicates(subset=["ticker", "period_end", "available_date"], keep="last")
        .sort_values(["ticker", "available_date", "period_end"])
        .reset_index(drop=True)
    )

    merged_meta = _enrich_metadata(base_metadata=base_meta, tiingo_snapshot=tiingo_new)

    base_daily.to_parquet(run_root / "baseline_daily.parquet", index=False)
    base_fund.to_parquet(run_root / "baseline_fundamentals.parquet", index=False)
    merged_daily.to_parquet(staged_equities / "daily_ohlcv.parquet", index=False)
    merged_meta.to_parquet(staged_equities / "metadata.parquet", index=False)
    base_membership.to_parquet(staged_equities / "universe_membership.parquet", index=False)
    merged_fund.to_parquet(staged_fundamentals / "fundamentals_fmp.parquet", index=False)

    # Build warehouse and validate strict.
    _run(
        [
            sys.executable,
            "scripts/build_equity_warehouse.py",
            "--equities-root",
            str(staged_equities),
            "--fundamentals-path",
            str(staged_fundamentals / "fundamentals_fmp.parquet"),
            "--warehouse-root",
            str(staged_warehouse),
            "--existing-security-master-path",
            str(baseline_warehouse_root / "security_master.parquet"),
        ],
        env=env,
    )
    _run(
        [
            sys.executable,
            "scripts/validate_equity_warehouse.py",
            "--warehouse-root",
            str(staged_warehouse),
            "--out-root",
            str(Path(args.validation_root) / "phase3"),
            "--max-duplicate-rows",
            "0",
            "--max-unmatched-symbols",
            "0",
            "--max-critical-null-frac",
            "0",
            "--max-ticker-id-instability",
            "0",
        ],
        env=env,
    )

    # Collect summary metrics.
    tiingo_report = pd.read_csv(raw_root / "tiingo" / "tiingo_fetch_report.csv")
    fmp_report = pd.DataFrame()
    if (raw_root / "fmp" / "fundamentals_fmp_bulk.parquet").exists():
        # Build per-ticker success proxy from output rows (no separate report in build_fundamentals_database.py).
        fmp_out = pd.read_parquet(raw_root / "fmp" / "fundamentals_fmp_bulk.parquet")
        fmp_report = fmp_out.groupby("ticker").size().rename("rows").reset_index().rename(columns={"ticker": "ticker"})
    staged_selected = pd.read_parquet(staged_warehouse / "equity_prices_daily.parquet")
    staged_versions = pd.read_parquet(staged_warehouse / "equity_prices_daily_versions.parquet")
    staged_fund_out = pd.read_parquet(staged_warehouse / "equity_fundamentals_pit.parquet")
    stability = pd.read_parquet(staged_warehouse / "ticker_id_stability_report.parquet")

    base_keys = base_selected_prices[["date", "ticker_id"]].copy()
    base_keys["date"] = pd.to_datetime(base_keys["date"], errors="coerce").dt.normalize()
    base_keys["key"] = base_keys["date"].astype(str) + "|" + base_keys["ticker_id"].astype(str)
    staged_keys = staged_selected[["date", "ticker_id"]].copy()
    staged_keys["date"] = pd.to_datetime(staged_keys["date"], errors="coerce").dt.normalize()
    staged_keys["key"] = staged_keys["date"].astype(str) + "|" + staged_keys["ticker_id"].astype(str)
    base_key_set = set(base_keys["key"].astype(str))
    staged_key_set = set(staged_keys["key"].astype(str))
    new_price_rows_added = int(len(staged_key_set - base_key_set))

    base_src = base_selected_prices[["date", "ticker_id", "source"]].copy()
    base_src["date"] = pd.to_datetime(base_src["date"], errors="coerce").dt.normalize()
    cmp = staged_selected[["date", "ticker_id", "source"]].merge(
        base_src.rename(columns={"source": "base_source"}),
        on=["date", "ticker_id"],
        how="left",
    )
    replaced_by_precedence = int(
        (cmp["base_source"].notna() & (cmp["base_source"].astype(str) != cmp["source"].astype(str))).sum()
    )

    base_fund_rows = int(len(base_fund))
    new_fund_rows_added = int(len(staged_fund_out) - base_fund_rows)

    base_symbols = set(base_daily["ticker"].map(_norm).dropna().astype(str))
    final_symbols = set(staged_selected["canonical_symbol"].astype(str))
    new_symbols_seen = int(len(final_symbols - base_symbols))

    ticker_id_reused = int(stability["status"].eq("reused").sum())
    ticker_id_new = int(stability["status"].eq("new").sum())
    ticker_id_changed = int(stability["status"].eq("changed").sum())

    val_summary_path = Path(args.validation_root) / "phase3" / "latest" / "validation_summary.json"
    val = json.loads(val_summary_path.read_text(encoding="utf-8"))
    cov = val.get("coverage", {})

    ingest_summary = {
        "run_root": str(run_root),
        "requested_tickers": int(len(tickers)),
        "tiingo_succeeded_tickers": int((tiingo_report["rows"] > 0).sum()) if not tiingo_report.empty else 0,
        "fmp_succeeded_tickers": int(fmp_report["ticker"].nunique()) if not fmp_report.empty else 0,
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
