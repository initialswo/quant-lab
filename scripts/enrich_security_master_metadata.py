"""Enrich warehouse security_master metadata using Tiingo metadata caches and API lookups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.security_master_metadata import (
    DEFAULT_TIINGO_BULK_META_CACHE,
    DEFAULT_TIINGO_DAILY_META_CACHE,
    enrich_security_master_metadata,
    load_or_fetch_tiingo_bulk_meta,
    load_or_fetch_tiingo_daily_meta,
)
from quant_lab.utils.env import load_project_env


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--security-master-path', required=True)
    p.add_argument('--out-path', default='')
    p.add_argument('--report-path', default='')
    p.add_argument('--bulk-meta-cache-path', default=str(DEFAULT_TIINGO_BULK_META_CACHE))
    p.add_argument('--daily-meta-cache-path', default=str(DEFAULT_TIINGO_DAILY_META_CACHE))
    p.add_argument('--refresh-bulk-meta', type=int, choices=[0, 1], default=0)
    p.add_argument('--refresh-daily-meta', type=int, choices=[0, 1], default=0)
    p.add_argument('--max-workers', type=int, default=8)
    p.add_argument('--timeout-seconds', type=float, default=20.0)
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--initial-backoff-seconds', type=float, default=1.0)
    p.add_argument('--backoff-multiplier', type=float, default=2.0)
    p.add_argument('--max-backoff-seconds', type=float, default=16.0)
    args = p.parse_args()

    load_project_env()

    security_master_path = Path(args.security_master_path)
    out_path = Path(args.out_path) if str(args.out_path).strip() else security_master_path
    report_path = Path(args.report_path) if str(args.report_path).strip() else out_path.with_name(out_path.stem + '_metadata_enrichment_report.json')

    security_master = pd.read_parquet(security_master_path)
    request_tickers = security_master['canonical_symbol'].astype(str).str.strip().tolist()

    bulk_meta = load_or_fetch_tiingo_bulk_meta(
        cache_path=Path(args.bulk_meta_cache_path),
        refresh=bool(args.refresh_bulk_meta),
    )
    daily_meta = load_or_fetch_tiingo_daily_meta(
        tickers=request_tickers,
        cache_path=Path(args.daily_meta_cache_path),
        refresh_missing=bool(args.refresh_daily_meta),
        max_workers=args.max_workers,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        initial_backoff_seconds=args.initial_backoff_seconds,
        backoff_multiplier=args.backoff_multiplier,
        max_backoff_seconds=args.max_backoff_seconds,
    )

    enriched, summary = enrich_security_master_metadata(
        security_master=security_master,
        bulk_meta=bulk_meta,
        daily_meta=daily_meta,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    enriched.to_parquet(tmp, index=False)
    tmp.replace(out_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'security_master_path': str(out_path),
        'bulk_meta_cache_path': str(Path(args.bulk_meta_cache_path)),
        'daily_meta_cache_path': str(Path(args.daily_meta_cache_path)),
        **summary,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
