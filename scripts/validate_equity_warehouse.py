"""Validate Phase 1 local equity warehouse and emit audit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def _duplicate_count(df: pd.DataFrame, keys: list[str]) -> int:
    if df.empty:
        return 0
    return int(df.duplicated(subset=keys, keep=False).sum())


def _null_profile(df: pd.DataFrame) -> dict:
    rows = int(len(df))
    out = {"row_count": rows, "column_nulls": {}, "column_null_frac": {}}
    for c in df.columns:
        n = int(df[c].isna().sum())
        out["column_nulls"][c] = n
        out["column_null_frac"][c] = float(n / rows) if rows > 0 else 0.0
    return out


def _safe_examples(series: pd.Series, k: int = 10) -> str:
    vals = [str(x) for x in series.dropna().astype(str).unique().tolist()[:k]]
    return ";".join(vals)


def _security_master_metadata_completeness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ['name', 'exchange', 'sector', 'industry']:
        null_count = int(df[column].isna().sum()) if column in df.columns else int(len(df))
        rows.append(
            {
                'column': column,
                'row_count': int(len(df)),
                'null_count': null_count,
                'non_null_count': int(len(df) - null_count),
                'null_frac': float(null_count / max(1, len(df))),
                'sample_values': _safe_examples(df[column], k=10) if column in df.columns else '',
            }
        )
    return pd.DataFrame(rows)


def _reporting_delay_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return pd.DataFrame([{
            "row_count": 0,
            "negative_reporting_delay_rows": 0,
            "min_reporting_delay_days": 0.0,
            "sample_negative_rows": "",
        }]), 0

    period_end = pd.to_datetime(df.get("period_end"), errors="coerce")
    available_date = pd.to_datetime(df.get("available_date"), errors="coerce")
    delay_days = (available_date - period_end).dt.days
    bad_mask = period_end.notna() & available_date.notna() & (delay_days < 0)
    sample = ""
    if bool(bad_mask.any()):
        sample_rows = df.loc[bad_mask, [c for c in ["ticker_id", "canonical_symbol", "raw_source_symbol", "period_end", "available_date"] if c in df.columns]].head(10)
        sample = sample_rows.astype(str).agg("|".join, axis=1).str.cat(sep=';')
    audit = pd.DataFrame([{
        "row_count": int(len(df)),
        "negative_reporting_delay_rows": int(bad_mask.sum()),
        "min_reporting_delay_days": float(delay_days.min()) if delay_days.notna().any() else 0.0,
        "sample_negative_rows": sample,
    }])
    return audit, int(bad_mask.sum())


def _parse_null_thresholds(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    txt = str(raw or "").strip()
    if not txt:
        return out
    for part in txt.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid null threshold '{item}'. Use col:frac format.")
        col, val = item.split(":", 1)
        out[col.strip()] = float(val.strip())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Validate data/warehouse tables and emit data quality artifacts.")
    p.add_argument("--warehouse-root", default="data/warehouse")
    p.add_argument("--out-root", default="results/data_validation")
    p.add_argument("--max-duplicate-rows", type=int, default=0)
    p.add_argument("--max-unmatched-symbols", type=int, default=0)
    p.add_argument("--max-critical-null-frac", type=float, default=0.0)
    p.add_argument("--max-ticker-id-instability", type=int, default=0)
    p.add_argument(
        "--critical-null-columns",
        default="security_master:ticker_id,security_master:canonical_symbol,equity_prices_daily:ticker_id,equity_prices_daily:date,equity_fundamentals_pit:ticker_id,equity_fundamentals_pit:available_date",
        help="Comma-separated table:column list checked against --max-critical-null-frac",
    )
    p.add_argument(
        "--critical-null-frac-overrides",
        default="",
        help="Optional overrides in table:column:frac format, comma-separated.",
    )
    args = p.parse_args()

    warehouse_root = Path(args.warehouse_root)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    security = pd.read_parquet(warehouse_root / "security_master.parquet")
    prices = pd.read_parquet(warehouse_root / "equity_prices_daily.parquet")
    membership = pd.read_parquet(warehouse_root / "universe_membership_daily.parquet")
    fundamentals = pd.read_parquet(warehouse_root / "equity_fundamentals_pit.parquet")
    symbol_history = pd.read_parquet(warehouse_root / "symbol_history.parquet")
    ticker_id_stability = pd.read_parquet(warehouse_root / "ticker_id_stability_report.parquet")
    symbol_collision = pd.read_parquet(warehouse_root / "symbol_collision_report.parquet")
    audit = pd.read_parquet(warehouse_root / "ingestion_audit.parquet")

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    fundamentals["available_date"] = pd.to_datetime(fundamentals["available_date"], errors="coerce")
    symbol_history["effective_from"] = pd.to_datetime(symbol_history["effective_from"], errors="coerce")
    symbol_history["effective_to"] = pd.to_datetime(symbol_history["effective_to"], errors="coerce")

    coverage = (
        prices.assign(year=prices["date"].dt.year)
        .groupby("year", dropna=False)
        .agg(
            rows=("ticker_id", "size"),
            unique_tickers=("ticker_id", "nunique"),
            avg_daily_rows=("ticker_id", lambda x: float(len(x) / max(1, prices.loc[x.index, "date"].nunique()))),
        )
        .reset_index()
        .sort_values("year")
    )
    coverage.to_csv(out_dir / "coverage_by_year.csv", index=False)

    fund_cov = security[["ticker_id", "canonical_symbol"]].copy()
    fstats = (
        fundamentals.groupby("ticker_id")
        .agg(
            fundamentals_rows=("ticker_id", "size"),
            first_available_date=("available_date", "min"),
            last_available_date=("available_date", "max"),
        )
        .reset_index()
    )
    pstats = prices.groupby("ticker_id").size().rename("price_rows").reset_index()
    fund_cov = fund_cov.merge(fstats, on="ticker_id", how="left").merge(pstats, on="ticker_id", how="left")
    fund_cov["fundamentals_rows"] = fund_cov["fundamentals_rows"].fillna(0).astype(int)
    fund_cov["price_rows"] = fund_cov["price_rows"].fillna(0).astype(int)
    fund_cov["has_fundamentals"] = fund_cov["fundamentals_rows"] > 0
    fund_cov["has_prices"] = fund_cov["price_rows"] > 0
    fund_cov.to_csv(out_dir / "fundamentals_coverage.csv", index=False)

    source_frames = {
        "equity_prices_daily": prices,
        "universe_membership_daily": membership,
        "equity_fundamentals_pit": fundamentals,
    }
    sm_symbols = set(security["canonical_symbol"].astype(str))
    match_rows = []
    for name, df in source_frames.items():
        c = df["canonical_symbol"].astype(str)
        unique_syms = sorted(set(c.tolist()))
        unmatched = [s for s in unique_syms if s not in sm_symbols]
        match_rows.append(
            {
                "table_name": name,
                "unique_raw_source_symbols": int(df["raw_source_symbol"].astype(str).nunique()),
                "unique_canonical_symbols": int(len(unique_syms)),
                "matched_symbols": int(len(unique_syms) - len(unmatched)),
                "unmatched_symbols": int(len(unmatched)),
                "unmatched_examples": ";".join(unmatched[:20]),
            }
        )
    symbol_match = pd.DataFrame(match_rows)
    symbol_match.to_csv(out_dir / "symbol_match_report.csv", index=False)

    dup_rows = [
        {
            "table_name": "security_master",
            "key_columns": "ticker_id|canonical_symbol",
            "duplicate_rows": _duplicate_count(security, ["ticker_id"]) + _duplicate_count(security, ["canonical_symbol"]),
        },
        {
            "table_name": "equity_prices_daily",
            "key_columns": "date|ticker_id",
            "duplicate_rows": _duplicate_count(prices, ["date", "ticker_id"]),
        },
        {
            "table_name": "universe_membership_daily",
            "key_columns": "date|universe|ticker_id",
            "duplicate_rows": _duplicate_count(membership, ["date", "universe", "ticker_id"]),
        },
        {
            "table_name": "equity_fundamentals_pit",
            "key_columns": "ticker_id|period_end|available_date",
            "duplicate_rows": _duplicate_count(fundamentals, ["ticker_id", "period_end", "available_date"]),
        },
    ]
    dup_df = pd.DataFrame(dup_rows)
    dup_df.to_csv(out_dir / "duplicate_key_report.csv", index=False)

    # Copy-through hardening diagnostics from warehouse build stage.
    ticker_id_stability.to_csv(out_dir / "ticker_id_stability_report.csv", index=False)
    symbol_collision.to_csv(out_dir / "symbol_collision_report.csv", index=False)

    symbol_history_cov = (
        symbol_history.groupby("ticker_id", as_index=False)
        .agg(
            raw_symbol_count=("raw_source_symbol", "nunique"),
            first_effective_from=("effective_from", "min"),
            last_effective_to=("effective_to", "max"),
            change_types=("change_type", lambda x: ";".join(sorted(set(x.astype(str))))),
        )
        .merge(security[["ticker_id", "canonical_symbol"]], on="ticker_id", how="left")
    )
    symbol_history_cov.to_csv(out_dir / "symbol_history_coverage.csv", index=False)

    security_meta_completeness = _security_master_metadata_completeness(security)
    security_meta_completeness.to_csv(out_dir / "security_master_metadata_completeness.csv", index=False)

    reporting_delay_audit, negative_reporting_delay_rows = _reporting_delay_audit(fundamentals)
    reporting_delay_audit.to_csv(out_dir / "fundamentals_reporting_delay_audit.csv", index=False)

    null_profile = {
        "security_master": _null_profile(security),
        "equity_prices_daily": _null_profile(prices),
        "universe_membership_daily": _null_profile(membership),
        "equity_fundamentals_pit": _null_profile(fundamentals),
        "ingestion_audit": _null_profile(audit),
    }
    (out_dir / "null_profile.json").write_text(json.dumps(null_profile, indent=2), encoding="utf-8")

    fundamentals_covered = int((fund_cov["has_fundamentals"] & fund_cov["has_prices"]).sum())
    total_price_symbols = int(fund_cov["has_prices"].sum())

    roll = prices.groupby("ticker_id")["date"].agg(["min", "max", "count"]).reset_index()
    irregular = int((roll["count"] < 252).sum())

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "warehouse_root": str(warehouse_root),
        "validation_output_dir": str(out_dir),
        "table_rows": {
            "security_master": int(len(security)),
            "equity_prices_daily": int(len(prices)),
            "universe_membership_daily": int(len(membership)),
            "equity_fundamentals_pit": int(len(fundamentals)),
            "ingestion_audit": int(len(audit)),
        },
        "coverage": {
            "price_date_min": str(prices["date"].min().date()) if not prices.empty else "",
            "price_date_max": str(prices["date"].max().date()) if not prices.empty else "",
            "unique_price_tickers": int(prices["ticker_id"].nunique()),
            "fundamentals_coverage_ratio": float(fundamentals_covered / max(1, total_price_symbols)),
            "fundamentals_covered_tickers": fundamentals_covered,
            "total_price_tickers": total_price_symbols,
            "short_history_tickers_lt_252": irregular,
        },
        "duplicates": {
            row["table_name"]: int(row["duplicate_rows"]) for row in dup_rows
        },
        "security_master_metadata_completeness": {
            row['column']: {
                'null_count': int(row['null_count']),
                'non_null_count': int(row['non_null_count']),
                'null_frac': float(row['null_frac']),
                'sample_values': row['sample_values'],
            }
            for row in security_meta_completeness.to_dict(orient='records')
        },
        "fundamentals_reporting_delay": reporting_delay_audit.to_dict(orient='records')[0],
        "symbol_match": {
            row["table_name"]: {
                "matched": int(row["matched_symbols"]),
                "unmatched": int(row["unmatched_symbols"]),
                "unmatched_examples": row["unmatched_examples"],
            }
            for row in match_rows
        },
        "artifacts": {
            "coverage_by_year": str(out_dir / "coverage_by_year.csv"),
            "fundamentals_coverage": str(out_dir / "fundamentals_coverage.csv"),
            "symbol_match_report": str(out_dir / "symbol_match_report.csv"),
            "duplicate_key_report": str(out_dir / "duplicate_key_report.csv"),
            "ticker_id_stability_report": str(out_dir / "ticker_id_stability_report.csv"),
            "symbol_collision_report": str(out_dir / "symbol_collision_report.csv"),
            "symbol_history_coverage": str(out_dir / "symbol_history_coverage.csv"),
            "security_master_metadata_completeness": str(out_dir / "security_master_metadata_completeness.csv"),
            "fundamentals_reporting_delay_audit": str(out_dir / "fundamentals_reporting_delay_audit.csv"),
            "null_profile": str(out_dir / "null_profile.json"),
            "validation_summary": str(out_dir / "validation_summary.json"),
        },
    }

    # Threshold evaluation (fail-fast support).
    total_duplicate_rows = int(sum(int(r["duplicate_rows"]) for r in dup_rows))
    total_unmatched_symbols = int(sum(int(r["unmatched_symbols"]) for r in match_rows))
    ticker_id_instability = int(ticker_id_stability.get("stability_issue", pd.Series(dtype=bool)).astype(bool).sum())

    null_table_map = {
        "security_master": security,
        "equity_prices_daily": prices,
        "universe_membership_daily": membership,
        "equity_fundamentals_pit": fundamentals,
        "symbol_history": symbol_history,
    }
    critical_pairs = []
    for raw_pair in str(args.critical_null_columns).split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Invalid --critical-null-columns entry '{pair}'. Use table:column.")
        t, c = pair.split(":", 1)
        critical_pairs.append((t.strip(), c.strip()))

    override_map: dict[tuple[str, str], float] = {}
    for raw_item in str(args.critical_null_frac_overrides).split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.split(":")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --critical-null-frac-overrides entry '{item}'. Use table:column:frac."
            )
        override_map[(parts[0], parts[1])] = float(parts[2])

    critical_null_checks: list[dict] = []
    critical_null_failures = 0
    for table, col in critical_pairs:
        frame = null_table_map.get(table)
        if frame is None:
            critical_null_checks.append(
                {
                    "table": table,
                    "column": col,
                    "null_frac": 1.0,
                    "threshold": 0.0,
                    "breach": True,
                    "reason": "missing_table",
                }
            )
            critical_null_failures += 1
            continue
        if col not in frame.columns:
            critical_null_checks.append(
                {
                    "table": table,
                    "column": col,
                    "null_frac": 1.0,
                    "threshold": 0.0,
                    "breach": True,
                    "reason": "missing_column",
                }
            )
            critical_null_failures += 1
            continue
        null_frac = float(frame[col].isna().mean()) if len(frame) > 0 else 0.0
        threshold = float(override_map.get((table, col), float(args.max_critical_null_frac)))
        breach = bool(null_frac > threshold)
        critical_null_checks.append(
            {
                "table": table,
                "column": col,
                "null_frac": null_frac,
                "threshold": threshold,
                "breach": breach,
                "reason": "",
            }
        )
        if breach:
            critical_null_failures += 1

    thresholds = {
        "max_duplicate_rows": int(args.max_duplicate_rows),
        "max_unmatched_symbols": int(args.max_unmatched_symbols),
        "max_critical_null_frac": float(args.max_critical_null_frac),
        "max_ticker_id_instability": int(args.max_ticker_id_instability),
        "critical_null_columns": [f"{t}:{c}" for t, c in critical_pairs],
        "critical_null_frac_overrides": {
            f"{t}:{c}": float(v) for (t, c), v in override_map.items()
        },
    }
    (out_dir / "validation_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    breaches = {
        "duplicate_rows_breach": total_duplicate_rows > int(args.max_duplicate_rows),
        "unmatched_symbols_breach": total_unmatched_symbols > int(args.max_unmatched_symbols),
        "critical_nulls_breach": critical_null_failures > 0,
        "ticker_id_instability_breach": ticker_id_instability > int(args.max_ticker_id_instability),
        "negative_reporting_delay_breach": negative_reporting_delay_rows > 0,
    }
    threshold_status = {
        "total_duplicate_rows": total_duplicate_rows,
        "total_unmatched_symbols": total_unmatched_symbols,
        "ticker_id_instability_count": ticker_id_instability,
        "critical_null_failures": critical_null_failures,
        "negative_reporting_delay_rows": negative_reporting_delay_rows,
        "critical_null_checks": critical_null_checks,
        "breaches": breaches,
        "failed": bool(any(breaches.values())),
    }
    summary["thresholds"] = threshold_status

    (out_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest = Path(args.out_root) / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for name in [
        "coverage_by_year.csv",
        "fundamentals_coverage.csv",
        "symbol_match_report.csv",
        "duplicate_key_report.csv",
        "ticker_id_stability_report.csv",
        "symbol_collision_report.csv",
        "symbol_history_coverage.csv",
        "security_master_metadata_completeness.csv",
        "fundamentals_reporting_delay_audit.csv",
        "validation_thresholds.json",
        "null_profile.json",
        "validation_summary.json",
    ]:
        src = out_dir / name
        dst = latest / name
        dst.write_bytes(src.read_bytes())

    print(f"saved validation bundle: {out_dir}")
    print(json.dumps(summary["coverage"], indent=2))
    print(json.dumps(summary["thresholds"], indent=2))
    if summary["thresholds"]["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
