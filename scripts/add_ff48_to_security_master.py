#!/usr/bin/env python3
"""Add Fama-French 48 industry classifications to warehouse security_master using SEC SIC codes."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

FF48_SOURCE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes48.zip"
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
DEFAULT_SECURITY_MASTER_PATH = Path("data/warehouse/security_master.parquet")
DEFAULT_SEC_CACHE_PATH = Path("data/reference/sec/sec_company_sic.parquet")
DEFAULT_FF48_CACHE_PATH = Path("data/reference/fama_french/ff48_sic_mapping.parquet")
DEFAULT_RESULTS_ROOT = Path("results") / "data_integration" / "ff48_security_master"
SEC_USER_AGENT = "QuantLab Research ops@quantlab.local"


@dataclass(frozen=True)
class FF48Rule:
    ff48_num: int
    ff48_code: str
    ff48_name: str
    sic_start: int
    sic_end: int
    sic_desc: str


class RateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = max(float(min_interval_seconds), 0.0)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0.0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                wait_for = self._next_allowed - now
                if wait_for <= 0.0:
                    self._next_allowed = now + self.min_interval_seconds
                    return
            time.sleep(min(wait_for, self.min_interval_seconds))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--security-master-path', default=str(DEFAULT_SECURITY_MASTER_PATH))
    p.add_argument('--sec-cache-path', default=str(DEFAULT_SEC_CACHE_PATH))
    p.add_argument('--ff48-cache-path', default=str(DEFAULT_FF48_CACHE_PATH))
    p.add_argument('--results-root', default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument('--refresh-sec-cache', action='store_true')
    p.add_argument('--refresh-ff48-cache', action='store_true')
    p.add_argument('--max-workers', type=int, default=6)
    p.add_argument('--requests-per-second', type=float, default=8.0)
    p.add_argument('--timeout-seconds', type=float, default=20.0)
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--active-recent-days', type=int, default=30)
    return p.parse_args()


def _load_security_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing security master: {path}')
    df = pd.read_parquet(path)
    required = {'ticker_id', 'canonical_symbol', 'consolidated_active_flag', 'price_last_date'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'security_master missing required columns: {sorted(missing)}')
    out = df.copy()
    out['canonical_symbol'] = out['canonical_symbol'].astype(str).str.strip().str.upper()
    out['price_last_date'] = pd.to_datetime(out['price_last_date'], errors='coerce').dt.normalize()
    out['consolidated_active_flag'] = out['consolidated_active_flag'].fillna(False).astype(bool)
    return out


def _download_ff48_text(timeout_seconds: float) -> str:
    r = requests.get(FF48_SOURCE_URL, timeout=timeout_seconds)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    if not names:
        raise ValueError('FF48 zip had no members')
    return zf.read(names[0]).decode('latin1')


def _parse_ff48_rules(raw_text: str) -> list[FF48Rule]:
    import re

    header_re = re.compile(r'^\s*(\d+)\s+([A-Za-z0-9]+)\s{2,}(.+?)\s*$')
    range_re = re.compile(r'^\s*(\d{4})-(\d{4})\s+(.+?)\s*$')
    rules: list[FF48Rule] = []
    current_num: int | None = None
    current_code: str | None = None
    current_name: str | None = None
    for raw_line in raw_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        m_header = header_re.match(line)
        if m_header:
            current_num = int(m_header.group(1))
            current_code = str(m_header.group(2)).strip()
            current_name = str(m_header.group(3)).strip()
            continue
        m_range = range_re.match(line)
        if m_range and current_num is not None and current_code is not None and current_name is not None:
            rules.append(
                FF48Rule(
                    ff48_num=current_num,
                    ff48_code=current_code,
                    ff48_name=current_name,
                    sic_start=int(m_range.group(1)),
                    sic_end=int(m_range.group(2)),
                    sic_desc=str(m_range.group(3)).strip(),
                )
            )
    if not rules:
        raise ValueError('Failed to parse any FF48 SIC rules from official source')
    return rules


def _load_or_refresh_ff48(cache_path: Path, refresh: bool, timeout_seconds: float) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        df = pd.read_parquet(cache_path)
        if not df.empty:
            return df
    text = _download_ff48_text(timeout_seconds=timeout_seconds)
    rules = _parse_ff48_rules(text)
    df = pd.DataFrame([r.__dict__ for r in rules]).sort_values(['sic_start', 'sic_end', 'ff48_num']).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + '.tmp')
    df.to_parquet(tmp, index=False)
    tmp.replace(cache_path)
    return df


def _load_sec_ticker_map(timeout_seconds: float) -> pd.DataFrame:
    headers = {'User-Agent': SEC_USER_AGENT, 'Accept-Encoding': 'gzip, deflate', 'Host': 'www.sec.gov'}
    r = requests.get(SEC_TICKER_URL, headers=headers, timeout=timeout_seconds)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict) or 'fields' not in payload or 'data' not in payload:
        raise ValueError('Unexpected SEC ticker payload shape')
    df = pd.DataFrame(payload['data'], columns=payload['fields'])
    df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
    df['cik'] = pd.to_numeric(df['cik'], errors='coerce').astype('Int64')
    return df[['ticker', 'cik', 'name', 'exchange']].dropna(subset=['ticker', 'cik']).drop_duplicates(subset=['ticker'], keep='first').reset_index(drop=True)


def _fetch_sec_submission(cik: int, timeout_seconds: float, max_retries: int, limiter: RateLimiter) -> dict[str, Any]:
    headers = {'User-Agent': SEC_USER_AGENT, 'Accept-Encoding': 'gzip, deflate', 'Host': 'data.sec.gov'}
    url = SEC_SUBMISSIONS_URL.format(cik=int(cik))
    last_error: str | None = None
    for attempt in range(1, max(int(max_retries), 1) + 1):
        try:
            limiter.wait()
            r = requests.get(url, headers=headers, timeout=timeout_seconds)
            if r.status_code == 200:
                payload = r.json()
                return {
                    'cik': int(cik),
                    'sic': payload.get('sic'),
                    'sic_description': payload.get('sicDescription'),
                    'sec_name': payload.get('name'),
                    'entity_type': payload.get('entityType'),
                    'fetch_status': int(r.status_code),
                    'fetch_error': pd.NA,
                    'fetched_at': datetime.now(UTC).isoformat(),
                }
            last_error = f'HTTP {r.status_code}: {r.text[:200]}'
        except Exception as exc:
            last_error = repr(exc)
        if attempt < max(int(max_retries), 1):
            time.sleep(min(2.0 ** (attempt - 1), 8.0))
    return {
        'cik': int(cik),
        'sic': pd.NA,
        'sic_description': pd.NA,
        'sec_name': pd.NA,
        'entity_type': pd.NA,
        'fetch_status': pd.NA,
        'fetch_error': last_error or pd.NA,
        'fetched_at': datetime.now(UTC).isoformat(),
    }


def _load_or_refresh_sec_sic(
    tickers: pd.Series,
    cache_path: Path,
    refresh: bool,
    max_workers: int,
    requests_per_second: float,
    timeout_seconds: float,
    max_retries: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sec_map = _load_sec_ticker_map(timeout_seconds=timeout_seconds)
    needed = pd.DataFrame({'ticker': tickers.astype(str).str.strip().str.upper()}).drop_duplicates()
    needed = needed.merge(sec_map, on='ticker', how='left')

    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
    else:
        cached = pd.DataFrame(columns=['cik', 'sic', 'sic_description', 'sec_name', 'entity_type', 'fetch_status', 'fetch_error', 'fetched_at'])
    if not cached.empty:
        cached['cik'] = pd.to_numeric(cached['cik'], errors='coerce').astype('Int64')
        cached = cached.dropna(subset=['cik']).drop_duplicates(subset=['cik'], keep='last').reset_index(drop=True)

    missing_ciks = sorted({int(c) for c in needed['cik'].dropna().astype(int).tolist() if int(c) not in set(cached.get('cik', pd.Series(dtype='Int64')).dropna().astype(int).tolist())})
    fetched_rows: list[dict[str, Any]] = []
    if missing_ciks:
        limiter = RateLimiter(1.0 / max(float(requests_per_second), 0.001))
        with ThreadPoolExecutor(max_workers=max(int(max_workers), 1)) as pool:
            futures = {
                pool.submit(_fetch_sec_submission, cik, timeout_seconds, max_retries, limiter): cik
                for cik in missing_ciks
            }
            for future in as_completed(futures):
                fetched_rows.append(future.result())
    combined = pd.concat([cached, pd.DataFrame(fetched_rows)], ignore_index=True) if fetched_rows else cached.copy()
    if not combined.empty:
        combined['cik'] = pd.to_numeric(combined['cik'], errors='coerce').astype('Int64')
        combined = combined.dropna(subset=['cik']).drop_duplicates(subset=['cik'], keep='last').sort_values('cik').reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + '.tmp')
        combined.to_parquet(tmp, index=False)
        tmp.replace(cache_path)
    out = needed.merge(combined, on='cik', how='left')
    return out, sec_map


def _map_ff48_from_sic(sic_series: pd.Series, ff48_rules: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    sic_num = pd.to_numeric(sic_series, errors='coerce').astype('Int64')
    ff48_code = pd.Series(pd.NA, index=sic_series.index, dtype='object')
    ff48_name = pd.Series(pd.NA, index=sic_series.index, dtype='object')
    ff48_num = pd.Series(pd.NA, index=sic_series.index, dtype='Int64')
    for row in ff48_rules.itertuples(index=False):
        mask = sic_num.notna() & sic_num.ge(int(row.sic_start)) & sic_num.le(int(row.sic_end)) & ff48_code.isna()
        if bool(mask.any()):
            ff48_code.loc[mask] = str(row.ff48_code)
            ff48_name.loc[mask] = str(row.ff48_name)
            ff48_num.loc[mask] = int(row.ff48_num)
    return ff48_code, ff48_name, ff48_num


def _safe_replace_parquet(df: pd.DataFrame, out_path: Path) -> Path:
    out_path = Path(out_path)
    timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    backup_path = out_path.with_suffix(out_path.suffix + f'.{timestamp}.bak')
    if out_path.exists():
        shutil.copy2(out_path, backup_path)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return backup_path


def _is_active_recent(df: pd.DataFrame, recent_days: int) -> pd.Series:
    max_date = pd.to_datetime(df['price_last_date'], errors='coerce').max()
    cutoff = max_date - pd.Timedelta(days=max(int(recent_days), 0)) if pd.notna(max_date) else pd.NaT
    return df['consolidated_active_flag'].fillna(False).astype(bool) & pd.to_datetime(df['price_last_date'], errors='coerce').ge(cutoff)


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    security_master_path = Path(args.security_master_path)
    sec_cache_path = Path(args.sec_cache_path)
    ff48_cache_path = Path(args.ff48_cache_path)
    results_root = Path(args.results_root)
    run_dir = results_root / datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)

    print('Loading security master...')
    security = _load_security_master(security_master_path)
    print(f'security_master rows={len(security)} tickers={security["canonical_symbol"].nunique()}')

    print('Loading official FF48 SIC mapping...')
    ff48_rules = _load_or_refresh_ff48(ff48_cache_path, refresh=bool(args.refresh_ff48_cache), timeout_seconds=float(args.timeout_seconds))
    print(f'ff48 rules={len(ff48_rules)} source={FF48_SOURCE_URL}')

    print('Loading SEC ticker map and SIC cache...')
    sec_sic, sec_ticker_map = _load_or_refresh_sec_sic(
        tickers=security['canonical_symbol'],
        cache_path=sec_cache_path,
        refresh=bool(args.refresh_sec_cache),
        max_workers=int(args.max_workers),
        requests_per_second=float(args.requests_per_second),
        timeout_seconds=float(args.timeout_seconds),
        max_retries=int(args.max_retries),
    )

    enriched = security.merge(sec_sic[['ticker', 'cik', 'sic', 'sic_description', 'sec_name', 'entity_type', 'fetch_status', 'fetch_error']], left_on='canonical_symbol', right_on='ticker', how='left')
    ff48_code, ff48_name, ff48_num = _map_ff48_from_sic(enriched['sic'], ff48_rules)
    enriched['industry_ff48'] = ff48_code
    enriched['industry_ff48_name'] = ff48_name
    enriched['industry_ff48_num'] = ff48_num

    active_mask = _is_active_recent(enriched, recent_days=int(args.active_recent_days))
    mapped_mask = enriched['industry_ff48'].notna()
    sec_match_mask = enriched['cik'].notna()

    active_total = int(active_mask.sum())
    active_mapped = int((active_mask & mapped_mask).sum())
    total_mapped = int(mapped_mask.sum())
    total_unmapped = int((~mapped_mask).sum())
    active_unmapped = int((active_mask & ~mapped_mask).sum())

    distribution = (
        enriched.loc[mapped_mask]
        .groupby(['industry_ff48', 'industry_ff48_name'], dropna=False)
        .agg(
            ticker_count=('canonical_symbol', 'nunique'),
            active_ticker_count=('canonical_symbol', lambda s: int(active_mask.loc[s.index].sum())),
        )
        .reset_index()
        .sort_values(['active_ticker_count', 'ticker_count', 'industry_ff48'], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    unmapped = enriched.loc[~mapped_mask, ['ticker_id', 'canonical_symbol', 'consolidated_active_flag', 'price_last_date', 'cik', 'sic', 'fetch_status', 'fetch_error']].copy()
    unmatched_sec = enriched.loc[~sec_match_mask, ['ticker_id', 'canonical_symbol', 'consolidated_active_flag', 'price_last_date']].copy()

    output_security = enriched.drop(columns=['ticker', 'cik', 'sic', 'sic_description', 'sec_name', 'entity_type', 'fetch_status', 'fetch_error', 'industry_ff48_name', 'industry_ff48_num']).copy()
    backup_path = _safe_replace_parquet(output_security, security_master_path)

    distribution_path = run_dir / 'ff48_distribution.csv'
    unmapped_path = run_dir / 'ff48_unmapped_tickers.csv'
    unmatched_sec_path = run_dir / 'sec_unmatched_tickers.csv'
    ff48_rules_path = run_dir / 'ff48_rules.csv'
    report_path = run_dir / 'ff48_integration_report.json'

    distribution.to_csv(distribution_path, index=False)
    unmapped.to_csv(unmapped_path, index=False)
    unmatched_sec.to_csv(unmatched_sec_path, index=False)
    ff48_rules.to_csv(ff48_rules_path, index=False)

    report = {
        'timestamp': datetime.now(UTC).isoformat(),
        'security_master_path': str(security_master_path),
        'security_master_backup_path': str(backup_path),
        'ff48_source_url': FF48_SOURCE_URL,
        'ff48_cache_path': str(ff48_cache_path),
        'sec_cache_path': str(sec_cache_path),
        'rows_total': int(len(enriched)),
        'tickers_total': int(enriched['canonical_symbol'].nunique()),
        'tickers_sec_matched': int(sec_match_mask.sum()),
        'tickers_sec_unmatched': int((~sec_match_mask).sum()),
        'tickers_mapped_ff48': int(total_mapped),
        'tickers_unmapped_ff48': int(total_unmapped),
        'active_recent_days': int(args.active_recent_days),
        'active_tickers_total': int(active_total),
        'active_tickers_mapped_ff48': int(active_mapped),
        'active_tickers_unmapped_ff48': int(active_unmapped),
        'active_mapping_rate': float(active_mapped / active_total) if active_total else None,
        'runtime_seconds': float(time.perf_counter() - started),
        'outputs': {
            'distribution_csv': str(distribution_path),
            'unmapped_csv': str(unmapped_path),
            'sec_unmatched_csv': str(unmatched_sec_path),
            'ff48_rules_csv': str(ff48_rules_path),
            'report_json': str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('')
    print('FF48 INTEGRATION REPORT')
    print('-----------------------')
    print(f'mapped_tickers={total_mapped}')
    print(f'unmapped_tickers={total_unmapped}')
    print(f'active_tickers={active_total}')
    print(f'active_tickers_mapped={active_mapped}')
    print(f'active_tickers_unmapped={active_unmapped}')
    if active_total:
        print(f'active_mapping_rate={active_mapped / active_total:.4%}')
    print('')
    print('Top industries by active ticker count:')
    preview = distribution.head(20)
    for row in preview.itertuples(index=False):
        print(f'{row.industry_ff48:>6s}  {row.industry_ff48_name:<30s} active={int(row.active_ticker_count):4d} total={int(row.ticker_count):4d}')
    print('')
    print(f'Updated security master: {security_master_path}')
    print(f'Backup created: {backup_path}')
    print(f'Report saved: {report_path}')


if __name__ == '__main__':
    main()
