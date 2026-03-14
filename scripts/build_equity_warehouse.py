"""Build Phase 1 equity warehouse parquet tables from existing local datasets."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from build_security_master import build_security_master, normalize_symbol


SOURCE_PRECEDENCE = {
    # vendor direct pulls
    "tiingo_phase2a_dryrun": 400,
    "tiingo_phase2b_dryrun": 400,
    "tiingo": 350,
    # cached/legacy sources
    "tiingo_cache": 300,
    "stooq_cache": 200,
    "sector_stooq": 150,
    "unknown": 10,
}


def _append_ingestion_audit(path: Path, rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    if path.exists():
        prev = pd.read_parquet(path)
        out = pd.concat([prev, new], ignore_index=True)
    else:
        out = new
    out.to_parquet(path, index=False)
    return out


def _table_audit(
    run_id: str,
    table_name: str,
    source_path: str,
    rows_in: int,
    rows_out: int,
    unmatched_symbols: int,
    status: str,
) -> dict:
    return {
        "run_id": run_id,
        "table_name": table_name,
        "source_path": source_path,
        "rows_in": int(rows_in),
        "rows_out": int(rows_out),
        "unmatched_symbols": int(unmatched_symbols),
        "status": status,
        "load_ts": datetime.now(UTC).isoformat(),
    }


def _source_rank(source: object) -> int:
    key = str(source or "").strip().lower()
    return int(SOURCE_PRECEDENCE.get(key, 100))


def _build_symbol_history(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    p = prices[["ticker_id", "canonical_symbol", "raw_source_symbol", "date"]].copy()
    p = p.rename(columns={"date": "effective_date"})
    p["source_table"] = "equity_prices_daily"
    frames.append(p)

    m = membership[["ticker_id", "canonical_symbol", "raw_source_symbol", "date"]].copy()
    m = m.rename(columns={"date": "effective_date"})
    m["source_table"] = "universe_membership_daily"
    frames.append(m)

    f = fundamentals[["ticker_id", "canonical_symbol", "raw_source_symbol", "available_date"]].copy()
    f = f.rename(columns={"available_date": "effective_date"})
    f["source_table"] = "equity_fundamentals_pit"
    frames.append(f)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["effective_date"] = pd.to_datetime(all_rows["effective_date"], errors="coerce").dt.normalize()
    all_rows = all_rows.loc[
        all_rows["ticker_id"].notna()
        & all_rows["canonical_symbol"].notna()
        & all_rows["raw_source_symbol"].notna()
        & all_rows["effective_date"].notna()
    ].copy()

    hist = (
        all_rows.groupby(["ticker_id", "canonical_symbol", "raw_source_symbol"], as_index=False)["effective_date"]
        .agg(effective_from="min", effective_to="max")
        .sort_values(["ticker_id", "effective_from", "raw_source_symbol"])
    )
    raw_counts = hist.groupby("ticker_id")["raw_source_symbol"].nunique().rename("raw_count")
    hist = hist.merge(raw_counts, on="ticker_id", how="left")
    hist["change_type"] = "stable"
    hist.loc[hist["raw_count"] > 1, "change_type"] = "alias_or_change"
    first_rows = hist.groupby("ticker_id")["effective_from"].transform("min").eq(hist["effective_from"])
    hist.loc[(hist["raw_count"] > 1) & first_rows, "change_type"] = "initial"
    hist = hist.drop(columns=["raw_count"])
    hist["source"] = "phase1_local"
    return hist[
        [
            "ticker_id",
            "canonical_symbol",
            "raw_source_symbol",
            "effective_from",
            "effective_to",
            "change_type",
            "source",
        ]
    ].sort_values(["ticker_id", "effective_from", "raw_source_symbol"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build Phase 1 data warehouse tables from local parquet inputs.")
    p.add_argument("--equities-root", default="data/equities")
    p.add_argument("--fundamentals-path", default="data/fundamentals/fundamentals_fmp.parquet")
    p.add_argument("--warehouse-root", default="data/warehouse")
    p.add_argument("--existing-security-master-path", default="")
    args = p.parse_args()

    equities_root = Path(args.equities_root)
    fundamentals_path = Path(args.fundamentals_path)
    warehouse_root = Path(args.warehouse_root)
    warehouse_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    security_master, stability_report, collision_report, _ = build_security_master(
        equities_root=equities_root,
        fundamentals_path=fundamentals_path,
        warehouse_root=warehouse_root,
        existing_security_master_path=(
            Path(args.existing_security_master_path)
            if str(args.existing_security_master_path).strip()
            else None
        ),
    )
    security_master.to_parquet(warehouse_root / "security_master.parquet", index=False)
    stability_report.to_parquet(warehouse_root / "ticker_id_stability_report.parquet", index=False)
    collision_report.to_parquet(warehouse_root / "symbol_collision_report.parquet", index=False)

    map_df = security_master[["ticker_id", "canonical_symbol"]].copy()

    daily_path = equities_root / "daily_ohlcv.parquet"
    membership_path = equities_root / "universe_membership.parquet"

    daily = pd.read_parquet(daily_path)
    daily["raw_source_symbol"] = daily["ticker"].astype(str).str.strip().str.upper()
    daily["canonical_symbol"] = daily["raw_source_symbol"].map(normalize_symbol)
    daily = daily.merge(map_df, on="canonical_symbol", how="left")
    unmatched_prices = int(daily["ticker_id"].isna().sum())
    price_versions = daily.loc[daily["ticker_id"].notna()].copy()
    price_versions["date"] = pd.to_datetime(price_versions["date"], errors="coerce").dt.normalize()
    price_versions["adj_close"] = pd.to_numeric(price_versions["close"], errors="coerce")
    price_versions["source_rank"] = price_versions["source"].map(_source_rank).astype(int)
    price_versions["_has_us_suffix"] = price_versions["raw_source_symbol"].astype(str).str.endswith(".US")
    price_versions["load_batch_id"] = run_id
    price_versions["load_ts"] = datetime.now(UTC).isoformat()
    price_versions = price_versions[
        [
            "date",
            "ticker_id",
            "raw_source_symbol",
            "canonical_symbol",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "source",
            "source_rank",
            "load_batch_id",
            "load_ts",
            "_has_us_suffix",
        ]
    ].sort_values(
        ["date", "ticker_id", "source_rank", "_has_us_suffix", "raw_source_symbol", "load_ts"],
        ascending=[True, True, False, True, True, False],
    )

    # Versioning policy:
    # keep all source rows in equity_prices_daily_versions for auditability,
    # then select one canonical row per (date,ticker_id) using precedence order above.
    versions_path = warehouse_root / "equity_prices_daily_versions.parquet"
    price_versions_out = price_versions.drop(columns=["_has_us_suffix"]).reset_index(drop=True)
    price_versions_out.to_parquet(versions_path, index=False)

    prices = price_versions.drop_duplicates(subset=["date", "ticker_id"], keep="first").copy()
    prices = prices.drop(columns=["_has_us_suffix"]).sort_values(["date", "ticker_id"]).reset_index(drop=True)
    prices.to_parquet(warehouse_root / "equity_prices_daily.parquet", index=False)

    membership = pd.read_parquet(membership_path)
    membership["raw_source_symbol"] = membership["ticker"].astype(str).str.strip().str.upper()
    membership["canonical_symbol"] = membership["raw_source_symbol"].map(normalize_symbol)
    membership = membership.merge(map_df, on="canonical_symbol", how="left")
    unmatched_membership = int(membership["ticker_id"].isna().sum())
    membership_out = membership.loc[membership["ticker_id"].notna()].copy()
    membership_out["date"] = pd.to_datetime(membership_out["date"], errors="coerce").dt.normalize()
    membership_out["universe"] = membership_out["universe"].astype(str).str.strip().str.lower()
    if membership_out["in_universe"].dtype != bool:
        membership_out["in_universe"] = membership_out["in_universe"].astype(bool)
    membership_out["load_ts"] = datetime.now(UTC).isoformat()
    membership_out = membership_out[
        [
            "date",
            "universe",
            "ticker_id",
            "raw_source_symbol",
            "canonical_symbol",
            "in_universe",
            "load_ts",
        ]
    ].sort_values(["date", "universe", "ticker_id"])
    membership_out.to_parquet(warehouse_root / "universe_membership_daily.parquet", index=False)

    fundamentals = pd.read_parquet(fundamentals_path)
    fundamentals["raw_source_symbol"] = fundamentals["ticker"].astype(str).str.strip().str.upper()
    fundamentals["canonical_symbol"] = fundamentals["raw_source_symbol"].map(normalize_symbol)
    fundamentals = fundamentals.merge(map_df, on="canonical_symbol", how="left")
    unmatched_fundamentals = int(fundamentals["ticker_id"].isna().sum())
    fundamentals_out = fundamentals.loc[fundamentals["ticker_id"].notna()].copy()
    fundamentals_out["period_end"] = pd.to_datetime(fundamentals_out["period_end"], errors="coerce").dt.normalize()
    fundamentals_out["available_date"] = pd.to_datetime(
        fundamentals_out["available_date"], errors="coerce"
    ).dt.normalize()
    fundamentals_out["source"] = "fmp"
    fundamentals_out["load_ts"] = datetime.now(UTC).isoformat()

    keep_cols = [
        "ticker_id",
        "raw_source_symbol",
        "canonical_symbol",
        "period_end",
        "available_date",
        "revenue",
        "cogs",
        "gross_profit",
        "total_assets",
        "shareholders_equity",
        "net_income",
        "shares_outstanding",
        "source",
        "load_ts",
    ]
    fundamentals_out = fundamentals_out[keep_cols].sort_values(["ticker_id", "available_date", "period_end"])
    fundamentals_out.to_parquet(warehouse_root / "equity_fundamentals_pit.parquet", index=False)

    symbol_history = _build_symbol_history(
        prices=prices,
        membership=membership_out,
        fundamentals=fundamentals_out,
    )
    symbol_history.to_parquet(warehouse_root / "symbol_history.parquet", index=False)

    audit_rows = [
        _table_audit(
            run_id=run_id,
            table_name="security_master",
            source_path=str(equities_root),
            rows_in=int(len(security_master)),
            rows_out=int(len(security_master)),
            unmatched_symbols=0,
            status="ok",
        ),
        _table_audit(
            run_id=run_id,
            table_name="equity_prices_daily_versions",
            source_path=str(daily_path),
            rows_in=int(len(daily)),
            rows_out=int(len(price_versions_out)),
            unmatched_symbols=unmatched_prices,
            status="ok" if unmatched_prices == 0 else "warning",
        ),
        _table_audit(
            run_id=run_id,
            table_name="equity_prices_daily",
            source_path=str(daily_path),
            rows_in=int(len(daily)),
            rows_out=int(len(prices)),
            unmatched_symbols=unmatched_prices,
            status="ok" if unmatched_prices == 0 else "warning",
        ),
        _table_audit(
            run_id=run_id,
            table_name="universe_membership_daily",
            source_path=str(membership_path),
            rows_in=int(len(membership)),
            rows_out=int(len(membership_out)),
            unmatched_symbols=unmatched_membership,
            status="ok" if unmatched_membership == 0 else "warning",
        ),
        _table_audit(
            run_id=run_id,
            table_name="equity_fundamentals_pit",
            source_path=str(fundamentals_path),
            rows_in=int(len(fundamentals)),
            rows_out=int(len(fundamentals_out)),
            unmatched_symbols=unmatched_fundamentals,
            status="ok" if unmatched_fundamentals == 0 else "warning",
        ),
        _table_audit(
            run_id=run_id,
            table_name="symbol_history",
            source_path="warehouse_internal",
            rows_in=int(len(prices) + len(membership_out) + len(fundamentals_out)),
            rows_out=int(len(symbol_history)),
            unmatched_symbols=0,
            status="ok",
        ),
        _table_audit(
            run_id=run_id,
            table_name="ticker_id_stability_report",
            source_path="warehouse_internal",
            rows_in=int(len(stability_report)),
            rows_out=int(len(stability_report)),
            unmatched_symbols=int(stability_report["stability_issue"].sum()),
            status="ok" if int(stability_report["stability_issue"].sum()) == 0 else "warning",
        ),
        _table_audit(
            run_id=run_id,
            table_name="symbol_collision_report",
            source_path="warehouse_internal",
            rows_in=int(len(collision_report)),
            rows_out=int(len(collision_report)),
            unmatched_symbols=int(collision_report["ambiguous_mapping_flag"].sum()),
            status="ok" if int(collision_report["ambiguous_mapping_flag"].sum()) == 0 else "warning",
        ),
    ]

    audit_path = warehouse_root / "ingestion_audit.parquet"
    all_audit = _append_ingestion_audit(path=audit_path, rows=audit_rows)

    print(f"saved: {warehouse_root / 'security_master.parquet'} rows={len(security_master)}")
    print(f"saved: {warehouse_root / 'equity_prices_daily_versions.parquet'} rows={len(price_versions_out)}")
    print(f"saved: {warehouse_root / 'equity_prices_daily.parquet'} rows={len(prices)}")
    print(f"saved: {warehouse_root / 'universe_membership_daily.parquet'} rows={len(membership_out)}")
    print(f"saved: {warehouse_root / 'equity_fundamentals_pit.parquet'} rows={len(fundamentals_out)}")
    print(f"saved: {warehouse_root / 'symbol_history.parquet'} rows={len(symbol_history)}")
    print(f"saved: {warehouse_root / 'ticker_id_stability_report.parquet'} rows={len(stability_report)}")
    print(f"saved: {warehouse_root / 'symbol_collision_report.parquet'} rows={len(collision_report)}")
    print(f"saved: {audit_path} rows={len(all_audit)}")


if __name__ == "__main__":
    main()
