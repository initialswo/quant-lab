"""Run Phase 2A staged ingest dry-run for Tiingo + FMP on a controlled cohort."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quant_lab.utils.env import load_project_env


def _norm(raw: object) -> str:
    s = str(raw or "").strip().upper()
    if s.endswith(".US"):
        s = s[:-3]
    return s.replace(".", "-")


def _normalize_date_series(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_convert(None).dt.normalize()


def _pick_cohort(equities_root: Path, fundamentals_path: Path, size: int) -> pd.DataFrame:
    md = pd.read_parquet(equities_root / "metadata.parquet", columns=["ticker", "active_flag"])
    daily = pd.read_parquet(equities_root / "daily_ohlcv.parquet", columns=["date", "ticker", "close", "volume"])
    fund = pd.read_parquet(fundamentals_path, columns=["ticker"])

    md["canonical"] = md["ticker"].map(_norm)
    daily["canonical"] = daily["ticker"].map(_norm)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["dollar_vol"] = pd.to_numeric(daily["close"], errors="coerce") * pd.to_numeric(
        daily["volume"], errors="coerce"
    )
    last = daily["date"].max()
    recent = daily.loc[daily["date"] >= (last - pd.Timedelta(days=90))].copy()
    adv = recent.groupby("canonical")["dollar_vol"].mean().rename("adv_90d").reset_index()

    fund_set = set(fund["ticker"].map(_norm).dropna().astype(str))
    active_set = set(md.loc[md["active_flag"].fillna(False), "canonical"].astype(str))
    inactive_set = set(md.loc[~md["active_flag"].fillna(False), "canonical"].astype(str))
    price_set = set(daily["canonical"].dropna().astype(str))
    edge_set = {x for x in price_set if "-" in x}

    stats = adv.copy()
    stats["has_fundamentals"] = stats["canonical"].isin(fund_set)
    stats["is_active"] = stats["canonical"].isin(active_set)
    stats["is_inactive"] = stats["canonical"].isin(inactive_set)
    stats["is_edge"] = stats["canonical"].isin(edge_set)
    stats = stats.sort_values(["adv_90d", "canonical"], ascending=[False, True]).reset_index(drop=True)

    sel: list[dict] = []
    used: set[str] = set()

    def add_from(mask: pd.Series, target: int, label: str) -> None:
        chosen = stats.loc[mask].copy()
        for _, row in chosen.iterrows():
            t = str(row["canonical"])
            if t in used:
                continue
            used.add(t)
            sel.append(
                {
                    "ticker": t,
                    "category": label,
                    "adv_90d": float(row["adv_90d"]) if pd.notna(row["adv_90d"]) else 0.0,
                    "is_active": bool(row["is_active"]),
                    "is_inactive": bool(row["is_inactive"]),
                    "is_edge": bool(row["is_edge"]),
                    "has_fundamentals": bool(row["has_fundamentals"]),
                }
            )
            if len([x for x in sel if x["category"] == label]) >= target:
                break

    add_from((stats["is_active"]) & (~stats["has_fundamentals"]), target=35, label="active_liquid_missing_fund")
    add_from((stats["is_active"]) & (stats["has_fundamentals"]), target=20, label="active_liquid_with_fund")
    add_from((stats["is_inactive"]) & (stats["has_fundamentals"]), target=15, label="inactive_with_fund")
    add_from((stats["is_edge"]), target=10, label="symbol_edge_case")

    if len(sel) < size:
        add_from(stats["canonical"].isin(price_set), target=size - len(sel), label="backfill")

    out = pd.DataFrame(sel).drop_duplicates(subset=["ticker"], keep="first").head(size).reset_index(drop=True)
    return out


def _run(cmd: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run Phase 2A staged ingest dry-run.")
    p.add_argument("--equities-root", default="data/equities")
    p.add_argument("--fundamentals-path", default="data/fundamentals/fundamentals_fmp.parquet")
    p.add_argument("--cohort-size", type=int, default=80)
    p.add_argument("--staging-root", default="data/staging/phase2a")
    p.add_argument("--validation-root", default="results/data_validation")
    p.add_argument("--baseline-warehouse-root", default="data/warehouse")
    args = p.parse_args()
    load_project_env()

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.staging_root) / ts
    raw_root = run_root / "raw"
    source_root = run_root / "source"
    staged_equities = source_root / "equities"
    staged_fundamentals = source_root / "fundamentals"
    staged_warehouse = run_root / "warehouse"

    for d in [raw_root, staged_equities, staged_fundamentals, staged_warehouse]:
        d.mkdir(parents=True, exist_ok=True)

    cohort = _pick_cohort(
        equities_root=Path(args.equities_root),
        fundamentals_path=Path(args.fundamentals_path),
        size=int(args.cohort_size),
    )
    cohort_path = run_root / "cohort.csv"
    cohort.to_csv(cohort_path, index=False)

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    _run(
        [
            sys.executable,
            "scripts/staged_ingest_tiingo_daily_dry_run.py",
            "--cohort-file",
            str(cohort_path),
            "--out-dir",
            str(raw_root / "tiingo"),
            "--start-date",
            "2000-01-01",
        ],
        env=env,
    )
    _run(
        [
            sys.executable,
            "scripts/staged_ingest_fmp_fundamentals_dry_run.py",
            "--cohort-file",
            str(cohort_path),
            "--out-dir",
            str(raw_root / "fmp"),
            "--available-lag-days",
            "60",
            "--limit",
            "120",
        ],
        env=env,
    )

    # Build staged source tables (copy baseline, then append dry-run snapshots).
    base_daily = pd.read_parquet(Path(args.equities_root) / "daily_ohlcv.parquet")
    base_meta = pd.read_parquet(Path(args.equities_root) / "metadata.parquet")
    base_membership = pd.read_parquet(Path(args.equities_root) / "universe_membership.parquet")
    base_fund = pd.read_parquet(Path(args.fundamentals_path))

    tiingo_new = pd.read_parquet(raw_root / "tiingo" / "tiingo_daily_snapshot.parquet")
    if not tiingo_new.empty:
        tiingo_append = tiingo_new.rename(columns={"adjClose": "adj_close"}).copy()
        tiingo_append = tiingo_append[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]]
    else:
        tiingo_append = pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"])

    merged_daily = pd.concat([base_daily, tiingo_append], ignore_index=True)
    merged_daily["date"] = _normalize_date_series(merged_daily["date"])
    merged_daily["ticker"] = merged_daily["ticker"].astype(str).str.upper()
    merged_daily = (
        merged_daily.dropna(subset=["date"])
        .drop_duplicates(subset=["date", "ticker"], keep="last")
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    fmp_new = pd.read_parquet(raw_root / "fmp" / "fmp_fundamentals_snapshot.parquet")
    fmp_append = fmp_new[
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
    ].copy() if not fmp_new.empty else pd.DataFrame(columns=base_fund.columns)
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

    base_daily.to_parquet(run_root / "baseline_daily.parquet", index=False)
    base_fund.to_parquet(run_root / "baseline_fundamentals.parquet", index=False)
    merged_daily.to_parquet(staged_equities / "daily_ohlcv.parquet", index=False)
    base_meta.to_parquet(staged_equities / "metadata.parquet", index=False)
    base_membership.to_parquet(staged_equities / "universe_membership.parquet", index=False)
    merged_fund.to_parquet(staged_fundamentals / "fundamentals_fmp.parquet", index=False)

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
            str(Path(args.baseline_warehouse_root) / "security_master.parquet"),
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
            str(Path(args.validation_root) / "phase2a"),
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

    # Build summary stats for checkpoint.
    base_norm = base_daily["ticker"].map(_norm).nunique()
    merged_norm = merged_daily["ticker"].map(_norm).nunique()
    price_rows_added = int(len(merged_daily) - len(base_daily))
    fund_rows_added = int(len(merged_fund) - len(base_fund))
    new_symbols_seen = int(max(0, merged_norm - base_norm))

    stability = pd.read_parquet(staged_warehouse / "ticker_id_stability_report.parquet")
    reuse = int(stability["status"].eq("reused").sum())
    new_ids = int(stability["status"].eq("new").sum())
    changed_ids = int(stability["status"].eq("changed").sum())

    baseline_sm_path = Path(args.baseline_warehouse_root) / "security_master.parquet"
    if baseline_sm_path.exists():
        baseline_sm = pd.read_parquet(baseline_sm_path)[["ticker_id", "canonical_symbol"]].copy()
        staged_sm = pd.read_parquet(staged_warehouse / "security_master.parquet")[
            ["ticker_id", "canonical_symbol"]
        ].copy()
        base_map = baseline_sm.drop_duplicates(subset=["canonical_symbol"], keep="first")
        staged_map = staged_sm.drop_duplicates(subset=["canonical_symbol"], keep="first")
        cmp_df = staged_map.merge(
            base_map.rename(columns={"ticker_id": "baseline_ticker_id"}),
            on="canonical_symbol",
            how="left",
        )
        reuse_vs_baseline = int(
            (cmp_df["baseline_ticker_id"].notna() & cmp_df["baseline_ticker_id"].eq(cmp_df["ticker_id"])).sum()
        )
        new_vs_baseline = int(cmp_df["baseline_ticker_id"].isna().sum())
        changed_vs_baseline = int(
            (cmp_df["baseline_ticker_id"].notna() & ~cmp_df["baseline_ticker_id"].eq(cmp_df["ticker_id"])).sum()
        )
    else:
        reuse_vs_baseline = reuse
        new_vs_baseline = new_ids
        changed_vs_baseline = changed_ids

    latest_val = Path(args.validation_root) / "phase2a" / "latest" / "validation_summary.json"
    val = json.loads(latest_val.read_text(encoding="utf-8"))

    summary = {
        "run_root": str(run_root),
        "cohort_size": int(len(cohort)),
        "cohort_path": str(cohort_path),
        "tiingo_summary_path": str(raw_root / "tiingo" / "tiingo_summary.json"),
        "fmp_summary_path": str(raw_root / "fmp" / "fmp_summary.json"),
        "price_rows_added": price_rows_added,
        "fundamentals_rows_added": fund_rows_added,
        "new_symbols_seen": new_symbols_seen,
        "ticker_id_reused": reuse,
        "ticker_id_new": new_ids,
        "ticker_id_changed": changed_ids,
        "ticker_id_reused_vs_baseline": reuse_vs_baseline,
        "ticker_id_new_vs_baseline": new_vs_baseline,
        "ticker_id_changed_vs_baseline": changed_vs_baseline,
        "validation_summary_path": str(latest_val),
        "validation_failed": bool(val.get("thresholds", {}).get("failed", False)),
    }
    (run_root / "phase2a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
