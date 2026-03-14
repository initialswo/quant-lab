"""Run Phase 2B staged ingest dry-run with expanded edge-case cohort."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


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


def _build_metadata_symbol_stats(metadata: pd.DataFrame) -> pd.DataFrame:
    md = metadata.copy()
    md["raw_symbol"] = md["ticker"].astype(str).str.strip().str.upper()
    md["canonical"] = md["raw_symbol"].map(_norm)
    md["active_flag"] = md["active_flag"].fillna(False).astype(bool)
    stats = (
        md.groupby("canonical")
        .agg(
            raw_symbol_count=("raw_symbol", "nunique"),
            raw_symbol_examples=("raw_symbol", lambda x: ";".join(sorted(set(x.astype(str)))[:10])),
            active_any=("active_flag", "max"),
            inactive_any=("active_flag", lambda x: bool((~pd.Series(x).astype(bool)).any())),
            active_true_count=("active_flag", lambda x: int(pd.Series(x).astype(bool).sum())),
            active_false_count=("active_flag", lambda x: int((~pd.Series(x).astype(bool)).sum())),
        )
        .reset_index()
    )
    stats["active_flag_conflict"] = (stats["active_true_count"] > 0) & (stats["active_false_count"] > 0)
    stats["is_edge_symbol"] = stats["canonical"].str.contains("-", regex=False)
    return stats


def _pick_expanded_cohort(equities_root: Path, fundamentals_path: Path, size: int) -> pd.DataFrame:
    md = pd.read_parquet(equities_root / "metadata.parquet", columns=["ticker", "active_flag"])
    daily = pd.read_parquet(equities_root / "daily_ohlcv.parquet", columns=["date", "ticker", "close", "volume"])
    fund = pd.read_parquet(fundamentals_path, columns=["ticker"])

    daily["canonical"] = daily["ticker"].map(_norm)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["dollar_vol"] = pd.to_numeric(daily["close"], errors="coerce") * pd.to_numeric(
        daily["volume"], errors="coerce"
    )
    last = daily["date"].max()
    recent = daily.loc[daily["date"] >= (last - pd.Timedelta(days=120))].copy()
    adv = recent.groupby("canonical")["dollar_vol"].mean().rename("adv_120d").reset_index()

    fund_set = set(fund["ticker"].map(_norm).dropna().astype(str))
    md_stats = _build_metadata_symbol_stats(md)
    stats = adv.merge(md_stats, on="canonical", how="left")
    stats["has_fundamentals"] = stats["canonical"].isin(fund_set)
    stats["active_any"] = stats["active_any"].fillna(False).astype(bool)
    stats["inactive_any"] = stats["inactive_any"].fillna(False).astype(bool)
    stats["active_flag_conflict"] = stats["active_flag_conflict"].fillna(False).astype(bool)
    stats["is_edge_symbol"] = stats["is_edge_symbol"].fillna(False).astype(bool)
    stats["raw_symbol_count"] = stats["raw_symbol_count"].fillna(1).astype(int)
    stats = stats.sort_values(["adv_120d", "canonical"], ascending=[False, True]).reset_index(drop=True)

    selections: list[dict] = []
    used: set[str] = set()

    def _add(mask: pd.Series, target: int, category: str) -> None:
        subset = stats.loc[mask].copy()
        for _, row in subset.iterrows():
            t = str(row["canonical"])
            if not t or t in used:
                continue
            used.add(t)
            selections.append(
                {
                    "ticker": t,
                    "category": category,
                    "adv_120d": float(row["adv_120d"]) if pd.notna(row["adv_120d"]) else 0.0,
                    "has_fundamentals": bool(row["has_fundamentals"]),
                    "active_any": bool(row["active_any"]),
                    "inactive_any": bool(row["inactive_any"]),
                    "active_flag_conflict": bool(row["active_flag_conflict"]),
                    "raw_symbol_count": int(row["raw_symbol_count"]),
                    "is_edge_symbol": bool(row["is_edge_symbol"]),
                    "raw_symbol_examples": str(row.get("raw_symbol_examples", "")),
                }
            )
            if len([x for x in selections if x["category"] == category]) >= target:
                break

    _add((stats["active_any"]) & (~stats["has_fundamentals"]), target=40, category="active_liquid_missing_fund")
    _add((stats["active_any"]) & (stats["has_fundamentals"]), target=20, category="active_liquid_with_fund")
    _add((stats["inactive_any"]) & (stats["has_fundamentals"]), target=20, category="inactive_or_delisted_with_fund")
    _add((stats["active_flag_conflict"]), target=20, category="metadata_active_conflict")
    _add((stats["is_edge_symbol"]), target=10, category="symbol_edge_share_class")
    _add((stats["raw_symbol_count"] > 1), target=10, category="symbol_ambiguity_multiraw")

    if len(selections) < size:
        _add(stats["canonical"].notna(), target=size - len(selections), category="backfill")

    cohort = (
        pd.DataFrame(selections)
        .drop_duplicates(subset=["ticker"], keep="first")
        .head(size)
        .sort_values(["category", "adv_120d"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return cohort


def main() -> None:
    p = argparse.ArgumentParser(description="Run Phase 2B expanded staged ingest dry-run.")
    p.add_argument("--equities-root", default="data/equities")
    p.add_argument("--fundamentals-path", default="data/fundamentals/fundamentals_fmp.parquet")
    p.add_argument("--baseline-warehouse-root", default="data/warehouse")
    p.add_argument("--cohort-size", type=int, default=120)
    p.add_argument("--staging-root", default="data/staging/phase2b")
    p.add_argument("--validation-root", default="results/data_validation")
    args = p.parse_args()

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.staging_root) / ts
    raw_root = run_root / "raw"
    source_root = run_root / "source"
    staged_equities = source_root / "equities"
    staged_fundamentals = source_root / "fundamentals"
    staged_warehouse = run_root / "warehouse"
    for d in [raw_root, staged_equities, staged_fundamentals, staged_warehouse]:
        d.mkdir(parents=True, exist_ok=True)

    cohort = _pick_expanded_cohort(
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
            "--source-label",
            "tiingo_phase2b_dryrun",
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
            "--source-label",
            "fmp_phase2b_dryrun",
        ],
        env=env,
    )

    base_daily = pd.read_parquet(Path(args.equities_root) / "daily_ohlcv.parquet")
    base_meta = pd.read_parquet(Path(args.equities_root) / "metadata.parquet")
    base_membership = pd.read_parquet(Path(args.equities_root) / "universe_membership.parquet")
    base_fund = pd.read_parquet(Path(args.fundamentals_path))

    tiingo_new = pd.read_parquet(raw_root / "tiingo" / "tiingo_daily_snapshot.parquet")
    tiingo_append = (
        tiingo_new.rename(columns={"adjClose": "adj_close"})[
            ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
        ].copy()
        if not tiingo_new.empty
        else pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"])
    )

    # Keep overlapping rows to exercise source precedence in warehouse build.
    merged_daily = pd.concat([base_daily, tiingo_append], ignore_index=True)
    merged_daily["date"] = _normalize_date_series(merged_daily["date"])
    merged_daily["ticker"] = merged_daily["ticker"].astype(str).str.upper()
    merged_daily = merged_daily.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    fmp_new = pd.read_parquet(raw_root / "fmp" / "fmp_fundamentals_snapshot.parquet")
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
            str(Path(args.validation_root) / "phase2b"),
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

    base_norm = base_daily["ticker"].map(_norm).nunique()
    merged_norm = merged_daily["ticker"].map(_norm).nunique()
    price_rows_added = int(len(merged_daily) - len(base_daily))
    fund_rows_added = int(len(merged_fund) - len(base_fund))
    new_symbols_seen = int(max(0, merged_norm - base_norm))

    staged_versions = pd.read_parquet(staged_warehouse / "equity_prices_daily_versions.parquet")
    staged_selected = pd.read_parquet(staged_warehouse / "equity_prices_daily.parquet")
    precedence_replaced = int(len(staged_versions) - len(staged_selected))

    stability = pd.read_parquet(staged_warehouse / "ticker_id_stability_report.parquet")
    reuse = int(stability["status"].eq("reused").sum())
    new_ids = int(stability["status"].eq("new").sum())
    changed_ids = int(stability["status"].eq("changed").sum())

    latest_val = Path(args.validation_root) / "phase2b" / "latest" / "validation_summary.json"
    val = json.loads(latest_val.read_text(encoding="utf-8"))

    summary = {
        "run_root": str(run_root),
        "cohort_size": int(len(cohort)),
        "cohort_path": str(cohort_path),
        "tiingo_summary_path": str(raw_root / "tiingo" / "tiingo_summary.json"),
        "fmp_summary_path": str(raw_root / "fmp" / "fmp_summary.json"),
        "price_rows_staged": price_rows_added,
        "fundamentals_rows_added": fund_rows_added,
        "new_symbols_seen": new_symbols_seen,
        "price_version_rows": int(len(staged_versions)),
        "price_selected_rows": int(len(staged_selected)),
        "price_rows_replaced_by_precedence": precedence_replaced,
        "ticker_id_reused": reuse,
        "ticker_id_new": new_ids,
        "ticker_id_changed": changed_ids,
        "validation_summary_path": str(latest_val),
        "validation_failed": bool(val.get("thresholds", {}).get("failed", False)),
    }
    (run_root / "phase2b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
