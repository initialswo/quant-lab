#!/usr/bin/env python3
"""Materialize legacy research-store compatibility outputs from warehouse tables."""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PRICE_REQUIRED_COLUMNS = [
    "date",
    "ticker_id",
    "canonical_symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
]
FUNDAMENTALS_REQUIRED_COLUMNS = [
    "ticker_id",
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
]
MEMBERSHIP_REQUIRED_COLUMNS = [
    "date",
    "universe",
    "ticker_id",
    "canonical_symbol",
    "in_universe",
]
SECURITY_REQUIRED_COLUMNS = [
    "ticker_id",
    "canonical_symbol",
    "name",
    "exchange",
    "sector",
    "industry",
    "first_date",
    "last_date",
    "consolidated_active_flag",
    "source",
]
LEGACY_PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
LEGACY_FUNDAMENTALS_COLUMNS = [
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
LEGACY_MEMBERSHIP_COLUMNS = ["date", "universe", "ticker", "in_universe"]
LEGACY_METADATA_COLUMNS = [
    "ticker",
    "name",
    "exchange",
    "sector",
    "industry",
    "first_date",
    "last_date",
    "active_flag",
    "source",
]


def _log(message: str) -> None:
    print(f"[stage-a] {message}")



def _require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")



def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0.0)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})



def _normalize_dates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out



def _validate_security_master(security: pd.DataFrame) -> pd.DataFrame:
    _require_columns(security, SECURITY_REQUIRED_COLUMNS, "security_master")
    sec = security[SECURITY_REQUIRED_COLUMNS].copy()
    sec["ticker_id"] = sec["ticker_id"].astype(str).str.strip()
    sec["canonical_symbol"] = sec["canonical_symbol"].astype(str).str.strip().str.upper()
    if sec["ticker_id"].eq("").any() or sec["canonical_symbol"].eq("").any():
        raise ValueError("security_master contains blank ticker_id or canonical_symbol values")
    dup_ticker_id = int(sec["ticker_id"].duplicated().sum())
    dup_symbol = int(sec["canonical_symbol"].duplicated().sum())
    if dup_ticker_id or dup_symbol:
        raise ValueError(
            "security_master mapping must be one-to-one; "
            f"duplicate ticker_id={dup_ticker_id} duplicate canonical_symbol={dup_symbol}"
        )
    sec = _normalize_dates(sec, ["first_date", "last_date"])
    sec["active_flag"] = _coerce_bool(sec["consolidated_active_flag"])
    return sec



def _map_legacy_ticker(frame: pd.DataFrame, label: str, id_to_symbol: pd.Series) -> tuple[pd.Series, int]:
    mapped = frame["ticker_id"].astype(str).str.strip().map(id_to_symbol)
    unmapped = int(mapped.isna().sum())
    if unmapped:
        sample = frame.loc[mapped.isna(), ["ticker_id"]].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"{label} contains unmapped ticker_id values; count={unmapped} sample={sample}")
    if "canonical_symbol" in frame.columns:
        lhs = frame["canonical_symbol"].astype(str).str.strip().str.upper()
        rhs = mapped.astype(str)
        mismatch_mask = lhs.ne(rhs)
        mismatch_count = int(mismatch_mask.sum())
        if mismatch_count:
            sample = frame.loc[mismatch_mask, ["ticker_id", "canonical_symbol"]].head(10).to_dict("records")
            raise ValueError(
                f"{label} canonical_symbol disagrees with security_master mapping; "
                f"count={mismatch_count} sample={sample}"
            )
    return mapped.astype(str), unmapped



def _validate_no_duplicates(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    dup_count = int(frame.duplicated(subset=keys).sum())
    if dup_count:
        sample = frame.loc[frame.duplicated(subset=keys, keep=False), keys].head(10).to_dict("records")
        raise ValueError(f"{label} contains duplicate keys on {keys}; count={dup_count} sample={sample}")



def _summarize_dates(frame: pd.DataFrame, column: str) -> tuple[str | None, str | None]:
    if frame.empty:
        return None, None
    vals = pd.to_datetime(frame[column], errors="coerce")
    vals = vals.loc[vals.notna()]
    if vals.empty:
        return None, None
    return str(vals.min().date()), str(vals.max().date())



def _write_parquet_atomic(frame: pd.DataFrame, dest: Path, run_id: str) -> dict[str, str | bool]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.parent / f".{dest.name}.{run_id}.tmp.parquet"
    backup_path = dest.parent / f"{dest.name}.{run_id}.bak"
    frame.to_parquet(tmp_path, index=False)
    backup_created = False
    if dest.exists():
        shutil.copy2(dest, backup_path)
        backup_created = True
    os.replace(tmp_path, dest)
    return {
        "destination": str(dest),
        "backup_path": str(backup_path) if backup_created else "",
        "backup_created": backup_created,
    }



def _materialize_prices(prices_path: Path, id_to_symbol: pd.Series) -> tuple[pd.DataFrame, dict[str, object]]:
    _log(f"loading warehouse prices: {prices_path}")
    prices = pd.read_parquet(prices_path, columns=PRICE_REQUIRED_COLUMNS)
    _require_columns(prices, PRICE_REQUIRED_COLUMNS, "equity_prices_daily")
    prices = _normalize_dates(prices, ["date"])
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    prices["ticker"] = _map_legacy_ticker(prices, "equity_prices_daily", id_to_symbol)[0]
    out = prices[LEGACY_PRICE_COLUMNS].copy()
    out = out.sort_values(["date", "ticker", "source"]).reset_index(drop=True)
    _validate_no_duplicates(out, ["date", "ticker"], "legacy daily_ohlcv")
    date_min, date_max = _summarize_dates(out, "date")
    stats = {
        "input_path": str(prices_path),
        "input_rows": int(len(prices)),
        "output_rows": int(len(out)),
        "distinct_tickers": int(out["ticker"].nunique()),
        "date_min": date_min,
        "date_max": date_max,
        "unmapped_symbols": 0,
        "dropped_rows": int(len(prices) - len(out)),
    }
    return out, stats



def _materialize_fundamentals(fund_path: Path, id_to_symbol: pd.Series) -> tuple[pd.DataFrame, dict[str, object]]:
    _log(f"loading warehouse fundamentals: {fund_path}")
    fundamentals = pd.read_parquet(fund_path, columns=FUNDAMENTALS_REQUIRED_COLUMNS)
    _require_columns(fundamentals, FUNDAMENTALS_REQUIRED_COLUMNS, "equity_fundamentals_pit")
    fundamentals = _normalize_dates(fundamentals, ["period_end", "available_date"])
    for col in LEGACY_FUNDAMENTALS_COLUMNS[3:]:
        fundamentals[col] = pd.to_numeric(fundamentals[col], errors="coerce")
    fundamentals["ticker"] = _map_legacy_ticker(fundamentals, "equity_fundamentals_pit", id_to_symbol)[0]
    out = fundamentals[LEGACY_FUNDAMENTALS_COLUMNS].copy()
    out = out.sort_values(["ticker", "available_date", "period_end"]).reset_index(drop=True)
    _validate_no_duplicates(out, ["ticker", "period_end", "available_date"], "legacy fundamentals_fmp")
    avail_min, avail_max = _summarize_dates(out, "available_date")
    stats = {
        "input_path": str(fund_path),
        "input_rows": int(len(fundamentals)),
        "output_rows": int(len(out)),
        "distinct_tickers": int(out["ticker"].nunique()),
        "available_date_min": avail_min,
        "available_date_max": avail_max,
        "unmapped_symbols": 0,
        "dropped_rows": int(len(fundamentals) - len(out)),
    }
    return out, stats



def _materialize_membership(membership_path: Path, id_to_symbol: pd.Series) -> tuple[pd.DataFrame, dict[str, object]]:
    _log(f"loading warehouse membership: {membership_path}")
    membership = pd.read_parquet(membership_path, columns=MEMBERSHIP_REQUIRED_COLUMNS)
    _require_columns(membership, MEMBERSHIP_REQUIRED_COLUMNS, "universe_membership_daily")
    membership = _normalize_dates(membership, ["date"])
    membership["universe"] = membership["universe"].astype(str).str.strip().str.lower()
    membership["in_universe"] = _coerce_bool(membership["in_universe"])
    membership["ticker"] = _map_legacy_ticker(membership, "universe_membership_daily", id_to_symbol)[0]
    out = membership[LEGACY_MEMBERSHIP_COLUMNS].copy()
    out = out.sort_values(["date", "universe", "ticker"]).reset_index(drop=True)
    _validate_no_duplicates(out, ["date", "universe", "ticker"], "legacy universe_membership")
    date_min, date_max = _summarize_dates(out, "date")
    stats = {
        "input_path": str(membership_path),
        "input_rows": int(len(membership)),
        "output_rows": int(len(out)),
        "distinct_tickers": int(out["ticker"].nunique()),
        "date_min": date_min,
        "date_max": date_max,
        "unmapped_symbols": 0,
        "dropped_rows": int(len(membership) - len(out)),
    }
    return out, stats



def _materialize_metadata(security: pd.DataFrame, price_tickers: set[str], source_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    _log(f"projecting warehouse security master: {source_path}")
    out = security.copy()
    out["ticker"] = out["canonical_symbol"].astype(str)
    out = out.loc[out["ticker"].isin(price_tickers)].copy()
    out = out.rename(columns={"consolidated_active_flag": "_unused_consolidated_active_flag"})
    out["source"] = out["source"].astype(str)
    out = out[LEGACY_METADATA_COLUMNS].copy()
    out = out.sort_values(["ticker"]).reset_index(drop=True)
    _validate_no_duplicates(out, ["ticker"], "legacy metadata")
    date_min, date_max = _summarize_dates(out, "first_date")
    last_min, last_max = _summarize_dates(out, "last_date")
    stats = {
        "input_path": str(source_path),
        "input_rows": int(len(security)),
        "output_rows": int(len(out)),
        "distinct_tickers": int(out["ticker"].nunique()),
        "first_date_min": date_min,
        "first_date_max": date_max,
        "last_date_min": last_min,
        "last_date_max": last_max,
        "unmapped_symbols": 0,
        "dropped_rows": int(len(security) - len(out)),
        "dropped_non_price_backed_tickers": int(len(security) - len(out)),
    }
    return out, stats



def _print_report(report: dict[str, dict[str, object]], writes: dict[str, dict[str, object]]) -> None:
    _log("validation report")
    for name, stats in report.items():
        print(f"[{name}]")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        write_info = writes.get(name, {})
        print(f"  output_path: {write_info.get('destination', '')}")
        print(f"  backup_created: {write_info.get('backup_created', False)}")
        if write_info.get("backup_path"):
            print(f"  backup_path: {write_info['backup_path']}")
        print(f"  derived_from_warehouse: true")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse-root", default="data/warehouse")
    parser.add_argument("--equities-root", default="data/equities")
    parser.add_argument("--fundamentals-root", default="data/fundamentals")
    args = parser.parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    warehouse_root = Path(args.warehouse_root)
    equities_root = Path(args.equities_root)
    fundamentals_root = Path(args.fundamentals_root)

    security_path = warehouse_root / "security_master.parquet"
    prices_path = warehouse_root / "equity_prices_daily.parquet"
    membership_path = warehouse_root / "universe_membership_daily.parquet"
    fundamentals_path = warehouse_root / "equity_fundamentals_pit.parquet"

    for path in [security_path, prices_path, membership_path, fundamentals_path]:
        if not path.exists():
            raise FileNotFoundError(f"missing required warehouse input: {path}")

    _log(f"loading warehouse security master: {security_path}")
    security_raw = pd.read_parquet(security_path, columns=SECURITY_REQUIRED_COLUMNS)
    security = _validate_security_master(security_raw)
    id_to_symbol = security.set_index("ticker_id")["canonical_symbol"]

    legacy_prices, prices_stats = _materialize_prices(prices_path, id_to_symbol)
    legacy_fundamentals, fundamentals_stats = _materialize_fundamentals(fundamentals_path, id_to_symbol)
    legacy_membership, membership_stats = _materialize_membership(membership_path, id_to_symbol)
    legacy_metadata, metadata_stats = _materialize_metadata(
        security=security,
        price_tickers=set(legacy_prices["ticker"].astype(str).unique().tolist()),
        source_path=security_path,
    )

    outputs = {
        "prices": equities_root / "daily_ohlcv.parquet",
        "fundamentals": fundamentals_root / "fundamentals_fmp.parquet",
        "membership": equities_root / "universe_membership.parquet",
        "metadata": equities_root / "metadata.parquet",
    }
    frames = {
        "prices": legacy_prices,
        "fundamentals": legacy_fundamentals,
        "membership": legacy_membership,
        "metadata": legacy_metadata,
    }

    writes: dict[str, dict[str, object]] = {}
    for name in ["prices", "fundamentals", "membership", "metadata"]:
        _log(f"writing {name} compatibility output: {outputs[name]}")
        writes[name] = _write_parquet_atomic(frames[name], outputs[name], run_id)

    report = {
        "prices": prices_stats,
        "fundamentals": fundamentals_stats,
        "membership": membership_stats,
        "metadata": metadata_stats,
    }
    _print_report(report, writes)


if __name__ == "__main__":
    main()
