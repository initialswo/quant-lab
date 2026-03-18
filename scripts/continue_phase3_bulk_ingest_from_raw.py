"""Continue a Phase 3 bulk ingest from existing raw outputs without refetching."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd

from quant_lab.data.fundamentals import enforce_available_date_floor
from quant_lab.utils.env import load_project_env


def _quote(path: Path) -> str:
    return str(path).replace("'", "''")


def _run(cmd: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def _tiingo_metadata_stats(
    raw_snapshot_path: Path,
    duckdb_memory_limit: str = '1GB',
    duckdb_temp_dir: Path | None = None,
    duckdb_threads: int = 4,
) -> pd.DataFrame:
    if duckdb_temp_dir is None:
        duckdb_temp_dir = raw_snapshot_path.parent / '.duckdb_tmp_metadata'
    duckdb_temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA memory_limit='{duckdb_memory_limit}'")
        con.execute(f"PRAGMA temp_directory='{_quote(duckdb_temp_dir)}'")
        con.execute(f"PRAGMA threads={max(int(duckdb_threads), 1)}")
        sql = f"""
            WITH stats AS (
                SELECT
                    UPPER(CAST(ticker AS VARCHAR)) AS ticker,
                    MIN(CAST(date AS DATE)) AS first_date,
                    MAX(CAST(date AS DATE)) AS last_date
                FROM read_parquet('{_quote(raw_snapshot_path)}')
                GROUP BY 1
            ), global_last AS (
                SELECT MAX(last_date) AS max_last_date FROM stats
            )
            SELECT
                s.ticker,
                s.first_date,
                s.last_date,
                s.last_date = g.max_last_date AS active_flag
            FROM stats s
            CROSS JOIN global_last g
            ORDER BY s.ticker
        """
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def _enrich_metadata(base_metadata: pd.DataFrame, tiingo_stats: pd.DataFrame) -> pd.DataFrame:
    md = base_metadata.copy()
    md['ticker'] = md['ticker'].astype(str).str.strip().str.upper()
    existing_raw = set(md['ticker'].astype(str))
    if tiingo_stats.empty:
        return md

    missing = tiingo_stats.loc[~tiingo_stats['ticker'].isin(existing_raw)].copy()
    if missing.empty:
        return md

    extra = pd.DataFrame(
        {
            'ticker': missing['ticker'],
            'name': pd.NA,
            'exchange': pd.NA,
            'sector': pd.NA,
            'industry': pd.NA,
            'first_date': pd.to_datetime(missing['first_date'], errors='coerce').dt.normalize(),
            'last_date': pd.to_datetime(missing['last_date'], errors='coerce').dt.normalize(),
            'active_flag': missing['active_flag'].astype(bool),
            'source': 'tiingo_phase3_inferred',
        }
    )
    out = pd.concat([md, extra], ignore_index=True)
    return out.drop_duplicates(subset=['ticker'], keep='last').reset_index(drop=True)


def _build_source_daily_duckdb(
    base_daily_path: Path,
    raw_snapshot_path: Path,
    out_path: Path,
    duckdb_memory_limit: str = '1GB',
    duckdb_temp_dir: Path | None = None,
    duckdb_threads: int = 4,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix('.parquet.tmp')
    if tmp_path.exists():
        tmp_path.unlink()

    if duckdb_temp_dir is None:
        duckdb_temp_dir = out_path.parent / '.duckdb_tmp_source_daily'
    duckdb_temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA memory_limit='{duckdb_memory_limit}'")
        con.execute(f"PRAGMA temp_directory='{_quote(duckdb_temp_dir)}'")
        con.execute(f"PRAGMA threads={max(int(duckdb_threads), 1)}")
        sql = f"""
            COPY (
                SELECT * FROM (
                    SELECT
                        CAST(date AS DATE) AS date,
                        UPPER(CAST(ticker AS VARCHAR)) AS ticker,
                        TRY_CAST(open AS DOUBLE) AS open,
                        TRY_CAST(high AS DOUBLE) AS high,
                        TRY_CAST(low AS DOUBLE) AS low,
                        TRY_CAST(close AS DOUBLE) AS close,
                        TRY_CAST(adj_close AS DOUBLE) AS adj_close,
                        TRY_CAST(volume AS DOUBLE) AS volume,
                        CAST(source AS VARCHAR) AS source
                    FROM read_parquet('{_quote(base_daily_path)}')
                    WHERE date IS NOT NULL

                    UNION ALL

                    SELECT
                        CAST(date AS DATE) AS date,
                        UPPER(CAST(ticker AS VARCHAR)) AS ticker,
                        TRY_CAST(open AS DOUBLE) AS open,
                        TRY_CAST(high AS DOUBLE) AS high,
                        TRY_CAST(low AS DOUBLE) AS low,
                        TRY_CAST(close AS DOUBLE) AS close,
                        TRY_CAST(adjClose AS DOUBLE) AS adj_close,
                        TRY_CAST(volume AS DOUBLE) AS volume,
                        CAST(source AS VARCHAR) AS source
                    FROM read_parquet('{_quote(raw_snapshot_path)}')
                    WHERE date IS NOT NULL
                ) merged
                ORDER BY date, ticker
            ) TO '{_quote(tmp_path)}' (FORMAT PARQUET, CODEC 'ZSTD')
        """
        con.execute(sql)
    finally:
        con.close()
    tmp_path.replace(out_path)


def _build_source_fundamentals(base_fund_path: Path, raw_fmp_path: Path, out_path: Path) -> None:
    base_fund = pd.read_parquet(base_fund_path)
    fmp_new = pd.read_parquet(raw_fmp_path)
    fmp_append = (
        fmp_new[
            [
                'ticker',
                'period_end',
                'available_date',
                'revenue',
                'cogs',
                'gross_profit',
                'total_assets',
                'shareholders_equity',
                'net_income',
                'shares_outstanding',
            ]
        ].copy()
        if not fmp_new.empty
        else pd.DataFrame(columns=base_fund.columns)
    )
    merged = pd.concat([base_fund, fmp_append], ignore_index=True)
    merged['ticker'] = merged['ticker'].astype(str).str.upper().str.replace('.US', '', regex=False).str.replace('.', '-', regex=False)
    merged['period_end'] = pd.to_datetime(merged['period_end'], errors='coerce').dt.normalize()
    merged['available_date'] = pd.to_datetime(merged['available_date'], errors='coerce').dt.normalize()
    merged, corrected_count = enforce_available_date_floor(merged)
    if corrected_count > 0:
        print(
            '[fundamentals] floored available_date to period_end for '
            f'{corrected_count} rows before writing staged source fundamentals'
        )
    merged = (
        merged.dropna(subset=['ticker', 'period_end', 'available_date'])
        .drop_duplicates(subset=['ticker', 'period_end', 'available_date'], keep='last')
        .sort_values(['ticker', 'available_date', 'period_end'])
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix('.parquet.tmp')
    merged.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)


def build_source_tables(
    run_root: Path,
    equities_root: Path,
    fundamentals_path: Path,
    duckdb_memory_limit: str,
    duckdb_threads: int,
) -> None:
    raw_root = run_root / 'raw'
    source_root = run_root / 'source'
    staged_equities = source_root / 'equities'
    staged_fundamentals = source_root / 'fundamentals'

    raw_snapshot_path = raw_root / 'tiingo' / 'tiingo_daily_snapshot.parquet'
    raw_fmp_path = raw_root / 'fmp' / 'fundamentals_fmp_bulk.parquet'
    if not raw_snapshot_path.exists():
        raise FileNotFoundError(f'missing raw Tiingo snapshot: {raw_snapshot_path}')
    if not raw_fmp_path.exists():
        raise FileNotFoundError(f'missing raw FMP parquet: {raw_fmp_path}')

    _copy_if_missing(equities_root / 'daily_ohlcv.parquet', run_root / 'baseline_daily.parquet')
    _copy_if_missing(fundamentals_path, run_root / 'baseline_fundamentals.parquet')

    source_daily_path = staged_equities / 'daily_ohlcv.parquet'
    if not source_daily_path.exists():
        _build_source_daily_duckdb(
            base_daily_path=equities_root / 'daily_ohlcv.parquet',
            raw_snapshot_path=raw_snapshot_path,
            out_path=source_daily_path,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_temp_dir=run_root / '.duckdb_tmp_source_daily',
            duckdb_threads=duckdb_threads,
        )

    metadata_path = staged_equities / 'metadata.parquet'
    if not metadata_path.exists():
        base_meta = pd.read_parquet(equities_root / 'metadata.parquet')
        tiingo_stats = _tiingo_metadata_stats(
            raw_snapshot_path=raw_snapshot_path,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_temp_dir=run_root / '.duckdb_tmp_metadata',
            duckdb_threads=duckdb_threads,
        )
        merged_meta = _enrich_metadata(base_metadata=base_meta, tiingo_stats=tiingo_stats)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = metadata_path.with_suffix('.parquet.tmp')
        merged_meta.to_parquet(tmp_path, index=False)
        tmp_path.replace(metadata_path)

    _copy_if_missing(equities_root / 'universe_membership.parquet', staged_equities / 'universe_membership.parquet')

    source_fund_path = staged_fundamentals / 'fundamentals_fmp.parquet'
    if not source_fund_path.exists():
        _build_source_fundamentals(
            base_fund_path=fundamentals_path,
            raw_fmp_path=raw_fmp_path,
            out_path=source_fund_path,
        )


def main() -> None:
    p = argparse.ArgumentParser(description='Continue a Phase 3 bulk ingest from existing raw outputs.')
    p.add_argument('--run-root', required=True)
    p.add_argument('--equities-root', default='data/equities')
    p.add_argument('--fundamentals-path', default='data/fundamentals/fundamentals_fmp.parquet')
    p.add_argument('--warehouse-root', default='data/warehouse')
    p.add_argument('--validation-root', default='results/data_validation/phase3_raw_continuation')
    p.add_argument('--duckdb-memory-limit', default='1GB')
    p.add_argument('--duckdb-threads', type=int, default=4)
    args = p.parse_args()
    load_project_env()

    run_root = Path(args.run_root)
    equities_root = Path(args.equities_root)
    fundamentals_path = Path(args.fundamentals_path)
    baseline_warehouse_root = Path(args.warehouse_root)
    staged_equities = run_root / 'source' / 'equities'
    staged_fundamentals = run_root / 'source' / 'fundamentals'
    staged_warehouse = run_root / 'warehouse'
    staged_warehouse.mkdir(parents=True, exist_ok=True)

    build_source_tables(
        run_root=run_root,
        equities_root=equities_root,
        fundamentals_path=fundamentals_path,
        duckdb_memory_limit=str(args.duckdb_memory_limit).strip() or '1GB',
        duckdb_threads=args.duckdb_threads,
    )

    env = dict(os.environ)
    env['PYTHONPATH'] = 'src'

    versions_path = staged_warehouse / 'equity_prices_daily_versions.parquet'
    if not versions_path.exists():
        _run(
            [
                sys.executable,
                'scripts/build_equity_warehouse.py',
                '--equities-root', str(staged_equities),
                '--fundamentals-path', str(staged_fundamentals / 'fundamentals_fmp.parquet'),
                '--warehouse-root', str(staged_warehouse),
                '--existing-security-master-path', str(baseline_warehouse_root / 'security_master.parquet'),
                '--stop-after', 'prices_versions',
                '--duckdb-memory-limit', str(args.duckdb_memory_limit).strip() or '1GB',
                '--duckdb-temp-dir', str(run_root / '.duckdb_tmp_prices_versions'),
                '--duckdb-threads', str(args.duckdb_threads),
            ],
            env=env,
        )

    _run(
        [
            sys.executable,
            'scripts/finalize_equity_warehouse_from_versions.py',
            '--equities-root', str(staged_equities),
            '--fundamentals-path', str(staged_fundamentals / 'fundamentals_fmp.parquet'),
            '--warehouse-root', str(staged_warehouse),
        ],
        env=env,
    )
    _run(
        [
            sys.executable,
            'scripts/enrich_security_master_metadata.py',
            '--security-master-path', str(staged_warehouse / 'security_master.parquet'),
            '--report-path', str(staged_warehouse / 'security_master_metadata_enrichment_report.json'),
        ],
        env=env,
    )
    _run(
        [
            sys.executable,
            'scripts/validate_equity_warehouse.py',
            '--warehouse-root', str(staged_warehouse),
            '--out-root', str(Path(args.validation_root)),
            '--max-duplicate-rows', '0',
            '--max-unmatched-symbols', '0',
            '--max-critical-null-frac', '0',
            '--max-ticker-id-instability', '0',
        ],
        env=env,
    )


if __name__ == '__main__':
    main()
