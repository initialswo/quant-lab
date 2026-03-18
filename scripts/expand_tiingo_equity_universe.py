"""Safely append missing Tiingo U.S. common-equity tickers to the active research parquet store."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_lab.utils.env import get_required_env, load_project_env
from quant_lab.data.loaders import load_ohlcv_for_research
from quant_lab.data.parquet_store import DAILY_COLUMNS, METADATA_COLUMNS

RESULTS_ROOT = Path("results") / "ingest" / "tiingo_expand"
DEFAULT_EQUITIES_ROOT = Path("data/equities")
DEFAULT_FUNDAMENTALS_PATH = Path("data/fundamentals/fundamentals_fmp.parquet")
TIINGO_META_URL = "https://api.tiingo.com/tiingo/fundamentals/meta"
NON_COMMON_NAME_RE = re.compile(
    r"(?:ETF|ETN|EXCHANGE TRADED|TRUST|FUND|ISHARES|SPDR|PROSHARES|DIREXION|VANGUARD|INVESCO|INDEX|UNITS?\\b|WARRANTS?\\b|RIGHTS?\\b|PREFERRED|PREF\\b|DEPOSITARY|ADR\\b|ADS\\b)",
    re.IGNORECASE,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equities-root", default=str(DEFAULT_EQUITIES_ROOT))
    parser.add_argument("--fundamentals-path", default=str(DEFAULT_FUNDAMENTALS_PATH))
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--start-date", default="1900-01-01")
    parser.add_argument("--source-label", default="tiingo")
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional cap for a smaller controlled run.")
    parser.add_argument("--skip-download", action="store_true", help="Reuse an existing staged snapshot in --results-root/<ts>/raw/tiingo.")
    parser.add_argument("--run-dir", default="", help="Optional existing run directory to resume from.")
    return parser.parse_args()


def _load_env_key() -> str:
    load_project_env()
    return get_required_env("TIINGO_API_KEY")

def _norm(raw: object) -> str:
    text = str(raw or "").strip().upper()
    if text.endswith(".US"):
        text = text[:-3]
    return text.replace(".", "-")


def _safe_name_present(series: pd.Series) -> pd.Series:
    txt = series.astype(str).fillna("").str.strip()
    return txt.ne("") & ~txt.str.lower().eq("nan")


def _load_existing_daily(equities_root: Path) -> pd.DataFrame:
    path = equities_root / "daily_ohlcv.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing active daily parquet: {path}")
    frame = pd.read_parquet(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = [c for c in DAILY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Active daily parquet is missing required columns: {missing}")
    frame = frame[DAILY_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    return frame.loc[frame["date"].notna() & frame["ticker"].ne("")].reset_index(drop=True)


def _load_existing_metadata(equities_root: Path) -> pd.DataFrame:
    path = equities_root / "metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing active metadata parquet: {path}")
    frame = pd.read_parquet(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    for col in METADATA_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame = frame[METADATA_COLUMNS].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["first_date"] = pd.to_datetime(frame["first_date"], errors="coerce").dt.normalize()
    frame["last_date"] = pd.to_datetime(frame["last_date"], errors="coerce").dt.normalize()
    return frame.loc[frame["ticker"].ne("")].drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)


def _fetch_tiingo_meta(api_key: str) -> pd.DataFrame:
    response = requests.get(TIINGO_META_URL, params={"token": api_key, "format": "json"}, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected Tiingo fundamentals/meta payload shape")
    frame = pd.DataFrame(payload)
    if frame.empty:
        raise ValueError("Tiingo fundamentals/meta returned 0 rows")
    frame["ticker"] = frame.get("ticker", pd.Series(index=frame.index, dtype=object)).astype(str).str.strip().str.upper()
    frame["name"] = frame.get("name", pd.Series(index=frame.index, dtype=object))
    frame["permaTicker"] = frame.get("permaTicker", pd.Series(index=frame.index, dtype=object)).astype(str)
    frame["isADR"] = frame.get("isADR", False).fillna(False).astype(bool)
    frame["isActive"] = frame.get("isActive", False).fillna(False).astype(bool)
    frame["norm"] = frame["ticker"].map(_norm)
    return frame


def _filter_common_equities(meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if meta.empty:
        return meta.copy(), {"candidate_rows": 0, "candidate_unique_norm": 0}
    unique_norm = meta.groupby("norm")["norm"].transform("size").eq(1)
    name_ok = _safe_name_present(meta["name"])
    perma_us = meta["permaTicker"].astype(str).str.startswith("US")
    commonish = ~meta["name"].astype(str).str.contains(NON_COMMON_NAME_RE, na=False)
    mask = unique_norm & name_ok & perma_us & ~meta["isADR"] & commonish
    filtered = meta.loc[mask].copy()
    filtered = filtered.drop_duplicates(subset=["norm"], keep="first").sort_values(["ticker"]).reset_index(drop=True)
    stats = {
        "meta_rows": int(len(meta)),
        "meta_unique_norm": int(meta["norm"].nunique()),
        "excluded_duplicate_norm_rows": int((~unique_norm).sum()),
        "excluded_missing_name_rows": int((~name_ok).sum()),
        "excluded_non_us_permaticker_rows": int((~perma_us).sum()),
        "excluded_adr_rows": int(meta["isADR"].sum()),
        "excluded_non_commonish_rows": int((~commonish).sum()),
        "candidate_rows": int(len(filtered)),
        "candidate_unique_norm": int(filtered["norm"].nunique()),
    }
    return filtered, stats


def _build_missing_candidates(existing_daily: pd.DataFrame, candidate_meta: pd.DataFrame, max_tickers: int) -> pd.DataFrame:
    existing_norm = set(existing_daily["ticker"].map(_norm))
    missing = candidate_meta.loc[~candidate_meta["norm"].isin(existing_norm)].copy()
    missing = missing.sort_values(["ticker"]).reset_index(drop=True)
    if int(max_tickers) > 0:
        missing = missing.head(int(max_tickers)).copy()
    return missing


def _stage_tiingo_download(run_dir: Path, tickers: list[str], start_date: str, source_label: str, skip_download: bool) -> tuple[Path, Path]:
    raw_root = run_dir / "raw" / "tiingo"
    raw_root.mkdir(parents=True, exist_ok=True)
    cohort_path = run_dir / "missing_tickers.csv"
    pd.DataFrame({"ticker": tickers}).to_csv(cohort_path, index=False)
    snapshot_path = raw_root / "tiingo_daily_snapshot.parquet"
    report_path = raw_root / "tiingo_fetch_report.csv"
    if skip_download and snapshot_path.exists() and report_path.exists():
        return snapshot_path, report_path
    cmd = [
        sys.executable,
        "scripts/staged_ingest_tiingo_daily_dry_run.py",
        "--cohort-file",
        str(cohort_path),
        "--out-dir",
        str(raw_root),
        "--start-date",
        str(start_date),
        "--source-label",
        str(source_label),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Tiingo staging download failed with exit code {proc.returncode}")
    if not snapshot_path.exists() or not report_path.exists():
        raise FileNotFoundError("Expected staged Tiingo snapshot/report were not created")
    return snapshot_path, report_path


def _normalize_staged_daily(snapshot_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(snapshot_path)
    if frame.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    frame = frame.rename(columns={"adjClose": "adj_close"}).copy()
    for col in DAILY_COLUMNS:
        if col not in frame.columns:
            if col == "source":
                frame[col] = "tiingo"
            else:
                frame[col] = pd.NA
    out = frame[DAILY_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["adj_close"] = out["adj_close"].where(out["adj_close"].notna(), out["close"])
    out["source"] = out["source"].astype(str).str.strip().replace({"": "tiingo"})
    out = out.loc[out["date"].notna() & out["ticker"].ne("")].copy()
    out = out.drop_duplicates(subset=["date", "ticker"], keep="last").sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def _build_new_metadata_rows(staged_daily: pd.DataFrame, missing_meta: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if staged_daily.empty:
        return pd.DataFrame(columns=METADATA_COLUMNS)
    daily = staged_daily.copy()
    daily["norm"] = daily["ticker"].map(_norm)
    grouped = daily.groupby("norm", as_index=False)["date"].agg(first_date="min", last_date="max")
    meta = missing_meta[["norm", "name"]].drop_duplicates(subset=["norm"], keep="first").copy()
    out = grouped.merge(meta, on="norm", how="left")
    out["ticker"] = out["norm"].astype(str) + ".US"
    global_last = pd.to_datetime(staged_daily["date"], errors="coerce").max()
    out["active_flag"] = out["last_date"].eq(global_last)
    out["exchange"] = pd.NA
    out["sector"] = pd.NA
    out["industry"] = pd.NA
    out["source"] = str(source_label)
    out = out[["ticker", "name", "exchange", "sector", "industry", "first_date", "last_date", "active_flag", "source"]]
    for col in METADATA_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[METADATA_COLUMNS].sort_values(["ticker"]).reset_index(drop=True)


def _append_with_backups(
    equities_root: Path,
    existing_daily: pd.DataFrame,
    existing_meta: pd.DataFrame,
    incoming_daily: pd.DataFrame,
    incoming_meta: pd.DataFrame,
    run_dir: Path,
) -> dict[str, Any]:
    backups_dir = run_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    daily_path = equities_root / "daily_ohlcv.parquet"
    meta_path = equities_root / "metadata.parquet"
    shutil.copy2(daily_path, backups_dir / "daily_ohlcv_before.parquet")
    shutil.copy2(meta_path, backups_dir / "metadata_before.parquet")

    existing_raw = set(existing_daily["ticker"].astype(str))
    existing_norm = set(existing_daily["ticker"].map(_norm))
    incoming_raw = set(incoming_daily["ticker"].astype(str))
    incoming_norm = set(incoming_daily["ticker"].map(_norm))
    raw_overlap = sorted(existing_raw & incoming_raw)
    norm_overlap = sorted(existing_norm & incoming_norm)
    if raw_overlap:
        raise ValueError(f"Refusing to append overlapping raw tickers: sample={raw_overlap[:20]}")
    if norm_overlap:
        raise ValueError(f"Refusing to append overlapping normalized tickers: sample={norm_overlap[:20]}")

    existing_meta_raw = set(existing_meta["ticker"].astype(str))
    incoming_meta_raw = set(incoming_meta["ticker"].astype(str))
    meta_overlap = sorted(existing_meta_raw & incoming_meta_raw)
    if meta_overlap:
        raise ValueError(f"Refusing to append overlapping metadata tickers: sample={meta_overlap[:20]}")

    merged_daily = pd.concat([existing_daily, incoming_daily], ignore_index=True)
    merged_daily = merged_daily.sort_values(["date", "ticker"]).reset_index(drop=True)
    merged_meta = pd.concat([existing_meta, incoming_meta], ignore_index=True)
    merged_meta = merged_meta.drop_duplicates(subset=["ticker"], keep="last").sort_values(["ticker"]).reset_index(drop=True)

    merged_daily.to_parquet(daily_path, index=False)
    merged_meta.to_parquet(meta_path, index=False)

    return {
        "backup_daily_path": str(backups_dir / "daily_ohlcv_before.parquet"),
        "backup_metadata_path": str(backups_dir / "metadata_before.parquet"),
        "daily_rows_written": int(len(merged_daily)),
        "metadata_rows_written": int(len(merged_meta)),
    }


def _summarize_counts(daily: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, int]:
    return {
        "daily_rows": int(len(daily)),
        "daily_tickers": int(daily["ticker"].astype(str).nunique()),
        "daily_norm_tickers": int(daily["ticker"].map(_norm).nunique()),
        "metadata_rows": int(len(metadata)),
        "metadata_tickers": int(metadata["ticker"].astype(str).nunique()),
    }


def _validate_loader(equities_root: Path) -> dict[str, Any]:
    loader = load_ohlcv_for_research(start="2018-01-01", end="2020-12-31", universe=None, store_root=str(equities_root))
    close = loader.panels["close"]
    adj_close = loader.panels["adj_close"]
    volume = loader.panels["volume"]
    return {
        "close_shape": [int(close.shape[0]), int(close.shape[1])],
        "adj_close_shape": [int(adj_close.shape[0]), int(adj_close.shape[1])],
        "volume_shape": [int(volume.shape[0]), int(volume.shape[1])],
        "close_nonnull": int(close.notna().sum().sum()),
        "adj_close_nonnull": int(adj_close.notna().sum().sum()),
    }


def _load_failed_tickers(report_path: Path) -> tuple[list[str], list[str]]:
    report = pd.read_csv(report_path)
    if report.empty:
        return [], []
    ok = sorted(report.loc[pd.to_numeric(report.get("rows", 0), errors="coerce").fillna(0).gt(0), "ticker"].astype(str).str.upper().tolist())
    failed = sorted(report.loc[pd.to_numeric(report.get("rows", 0), errors="coerce").fillna(0).le(0), "ticker"].astype(str).str.upper().tolist())
    return ok, failed


def main() -> None:
    args = _parse_args()
    equities_root = Path(str(args.equities_root)).expanduser()
    fundamentals_path = Path(str(args.fundamentals_path)).expanduser()
    results_root = Path(str(args.results_root)).expanduser()
    results_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(str(args.run_dir)).expanduser() if str(args.run_dir).strip() else results_root / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    existing_daily = _load_existing_daily(equities_root)
    existing_meta = _load_existing_metadata(equities_root)
    before_counts = _summarize_counts(existing_daily, existing_meta)

    tiingo_meta = _fetch_tiingo_meta(api_key=_load_env_key())
    candidate_meta, filter_stats = _filter_common_equities(tiingo_meta)
    candidate_meta.to_csv(run_dir / "candidate_tiingo_common_equities.csv", index=False)

    missing_meta = _build_missing_candidates(existing_daily=existing_daily, candidate_meta=candidate_meta, max_tickers=int(args.max_tickers))
    missing_meta.to_csv(run_dir / "missing_tiingo_tickers.csv", index=False)
    missing_tickers = missing_meta["ticker"].astype(str).str.upper().tolist()

    if not missing_tickers:
        summary = {
            "status": "no_missing_tickers",
            "before": before_counts,
            "filter_stats": filter_stats,
            "fundamentals_unique_tickers": int(pd.read_parquet(fundamentals_path, columns=["ticker"])["ticker"].astype(str).str.upper().nunique()) if fundamentals_path.exists() else None,
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    snapshot_path, report_path = _stage_tiingo_download(
        run_dir=run_dir,
        tickers=missing_tickers,
        start_date=str(args.start_date),
        source_label=str(args.source_label),
        skip_download=bool(args.skip_download),
    )
    ok_tickers, failed_tickers = _load_failed_tickers(report_path)
    staged_daily = _normalize_staged_daily(snapshot_path)
    staged_daily.to_parquet(run_dir / "incoming_daily_normalized.parquet", index=False)
    staged_meta = missing_meta.loc[missing_meta["ticker"].astype(str).str.upper().isin(ok_tickers)].copy()
    incoming_meta = _build_new_metadata_rows(staged_daily=staged_daily, missing_meta=staged_meta, source_label=str(args.source_label))
    incoming_meta.to_parquet(run_dir / "incoming_metadata_normalized.parquet", index=False)

    if staged_daily.empty:
        raise RuntimeError("Staged Tiingo snapshot is empty; refusing to touch the active warehouse.")
    append_stats = _append_with_backups(
        equities_root=equities_root,
        existing_daily=existing_daily,
        existing_meta=existing_meta,
        incoming_daily=staged_daily,
        incoming_meta=incoming_meta,
        run_dir=run_dir,
    )

    after_daily = _load_existing_daily(equities_root)
    after_meta = _load_existing_metadata(equities_root)
    after_counts = _summarize_counts(after_daily, after_meta)
    loader_stats = _validate_loader(equities_root)

    expected_new_rows = int(len(staged_daily))
    expected_new_raw_tickers = int(staged_daily["ticker"].astype(str).nunique())
    if after_counts["daily_rows"] != before_counts["daily_rows"] + expected_new_rows:
        raise RuntimeError("Daily row count validation failed after append")
    if after_counts["daily_tickers"] != before_counts["daily_tickers"] + expected_new_raw_tickers:
        raise RuntimeError("Daily ticker count validation failed after append")

    summary = {
        "status": "appended",
        "equities_root": str(equities_root),
        "fundamentals_path": str(fundamentals_path),
        "run_dir": str(run_dir),
        "source_label": str(args.source_label),
        "filter_policy": {
            "source_endpoint": TIINGO_META_URL,
            "rules": [
                "keep permaTicker values starting with 'US'",
                "exclude ADRs via isADR",
                "exclude ambiguous reused symbols where normalized ticker appears multiple times in Tiingo metadata",
                "exclude obvious non-common-share instruments by name regex: ETF/ETN/trust/fund/units/warrants/rights/preferred/depositary/ADR/ADS",
                "treat an existing normalized ticker as already present even if only a non-.US alias exists in the active store",
            ],
            "warning": "Tiingo fundamentals/meta does not expose an explicit assetType or exchangeCode field in the list endpoint, so common-equity filtering remains heuristic.",
        },
        "before": before_counts,
        "after": after_counts,
        "filter_stats": filter_stats,
        "candidate_tickers_found": int(candidate_meta['norm'].nunique()),
        "already_present_tickers": int(candidate_meta['norm'].nunique() - missing_meta['norm'].nunique()),
        "missing_tickers_requested": int(len(missing_tickers)),
        "new_tickers_added": int(staged_daily['ticker'].astype(str).nunique()),
        "new_rows_added": int(len(staged_daily)),
        "failed_or_skipped_tickers": int(len(failed_tickers)),
        "failed_tickers_sample": failed_tickers[:50],
        "fundamentals_unique_tickers": int(pd.read_parquet(fundamentals_path, columns=['ticker'])['ticker'].astype(str).str.upper().nunique()) if fundamentals_path.exists() else None,
        "fundamentals_not_in_prices_after": int(len(set(pd.read_parquet(fundamentals_path, columns=['ticker'])['ticker'].astype(str).str.upper()) - set(after_daily['ticker'].astype(str).str.upper()))) if fundamentals_path.exists() else None,
        "append_stats": append_stats,
        "loader_validation": loader_stats,
        "artifacts": {
            "candidate_csv": str(run_dir / 'candidate_tiingo_common_equities.csv'),
            "missing_csv": str(run_dir / 'missing_tiingo_tickers.csv'),
            "fetch_report_csv": str(report_path),
            "incoming_daily_parquet": str(run_dir / 'incoming_daily_normalized.parquet'),
            "incoming_metadata_parquet": str(run_dir / 'incoming_metadata_normalized.parquet'),
        },
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
