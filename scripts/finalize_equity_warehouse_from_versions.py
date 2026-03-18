"""Finalize warehouse tables from a prebuilt equity_prices_daily_versions parquet."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_security_master import normalize_symbol
from quant_lab.data.fundamentals import enforce_available_date_floor


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


def _stream_selected_prices(versions_path: Path, out_path: Path) -> tuple[int, pd.DataFrame]:
    parquet = pq.ParquetFile(versions_path)
    tmp_path = out_path.with_suffix('.parquet.tmp')
    if tmp_path.exists():
        tmp_path.unlink()

    writer: pq.ParquetWriter | None = None
    carry: pd.DataFrame | None = None
    rows_out = 0
    price_extents: dict[tuple[str, str, str], list[pd.Timestamp]] = {}

    def _update_extents(df: pd.DataFrame) -> None:
        if df.empty:
            return
        grouped = (
            df.groupby(['ticker_id', 'canonical_symbol', 'raw_source_symbol'], as_index=False)['date']
            .agg(effective_from='min', effective_to='max')
        )
        for row in grouped.itertuples(index=False):
            key = (str(row.ticker_id), str(row.canonical_symbol), str(row.raw_source_symbol))
            start = pd.to_datetime(row.effective_from, errors='coerce')
            end = pd.to_datetime(row.effective_to, errors='coerce')
            prev = price_extents.get(key)
            if prev is None:
                price_extents[key] = [start, end]
                continue
            if pd.notna(start) and (pd.isna(prev[0]) or start < prev[0]):
                prev[0] = start
            if pd.notna(end) and (pd.isna(prev[1]) or end > prev[1]):
                prev[1] = end

    for batch in parquet.iter_batches(batch_size=250_000):
        df = batch.to_pandas()
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
        combined = pd.concat([carry, df], ignore_index=True) if carry is not None else df
        if combined.empty:
            carry = combined
            continue
        last_key = combined.iloc[-1][['date', 'ticker_id']]
        hold_mask = combined['date'].eq(last_key['date']) & combined['ticker_id'].eq(last_key['ticker_id'])
        ready = combined.loc[~hold_mask].copy()
        carry = combined.loc[hold_mask].copy()
        if ready.empty:
            continue
        selected = ready.drop_duplicates(subset=['date', 'ticker_id'], keep='first').reset_index(drop=True)
        rows_out += int(len(selected))
        _update_extents(selected)
        table = pa.Table.from_pandas(selected, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, table.schema)
        writer.write_table(table)

    if carry is not None and not carry.empty:
        selected = carry.drop_duplicates(subset=['date', 'ticker_id'], keep='first').reset_index(drop=True)
        rows_out += int(len(selected))
        _update_extents(selected)
        table = pa.Table.from_pandas(selected, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, table.schema)
        writer.write_table(table)

    if writer is None:
        pd.DataFrame(columns=pd.read_parquet(versions_path, engine='pyarrow').columns).to_parquet(tmp_path, index=False)
    else:
        writer.close()
    tmp_path.replace(out_path)

    price_hist = pd.DataFrame(
        [
            {
                'ticker_id': key[0],
                'canonical_symbol': key[1],
                'raw_source_symbol': key[2],
                'effective_from': vals[0],
                'effective_to': vals[1],
                'source_table': 'equity_prices_daily',
            }
            for key, vals in price_extents.items()
        ]
    )
    return rows_out, price_hist


def _build_symbol_history(price_hist: pd.DataFrame, membership: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    membership_hist = (
        membership.groupby(['ticker_id', 'canonical_symbol', 'raw_source_symbol'], as_index=False)['date']
        .agg(effective_from='min', effective_to='max')
    )
    membership_hist['source_table'] = 'universe_membership_daily'

    fund_hist = (
        fundamentals.groupby(['ticker_id', 'canonical_symbol', 'raw_source_symbol'], as_index=False)['available_date']
        .agg(effective_from='min', effective_to='max')
    )
    fund_hist['source_table'] = 'equity_fundamentals_pit'

    all_rows = pd.concat([price_hist, membership_hist, fund_hist], ignore_index=True)
    all_rows['effective_from'] = pd.to_datetime(all_rows['effective_from'], errors='coerce').dt.normalize()
    all_rows['effective_to'] = pd.to_datetime(all_rows['effective_to'], errors='coerce').dt.normalize()
    all_rows = all_rows.loc[
        all_rows['ticker_id'].notna()
        & all_rows['canonical_symbol'].notna()
        & all_rows['raw_source_symbol'].notna()
        & all_rows['effective_from'].notna()
        & all_rows['effective_to'].notna()
    ].copy()

    hist = (
        all_rows.groupby(['ticker_id', 'canonical_symbol', 'raw_source_symbol'], as_index=False)
        .agg(effective_from=('effective_from', 'min'), effective_to=('effective_to', 'max'))
        .sort_values(['ticker_id', 'effective_from', 'raw_source_symbol'])
    )
    raw_counts = hist.groupby('ticker_id')['raw_source_symbol'].nunique().rename('raw_count')
    hist = hist.merge(raw_counts, on='ticker_id', how='left')
    hist['change_type'] = 'stable'
    hist.loc[hist['raw_count'] > 1, 'change_type'] = 'alias_or_change'
    first_rows = hist.groupby('ticker_id')['effective_from'].transform('min').eq(hist['effective_from'])
    hist.loc[(hist['raw_count'] > 1) & first_rows, 'change_type'] = 'initial'
    hist = hist.drop(columns=['raw_count'])
    hist['source'] = 'phase1_local'
    return hist[
        [
            'ticker_id',
            'canonical_symbol',
            'raw_source_symbol',
            'effective_from',
            'effective_to',
            'change_type',
            'source',
        ]
    ].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description='Finalize warehouse tables from prices versions parquet.')
    p.add_argument('--equities-root', default='data/equities')
    p.add_argument('--fundamentals-path', default='data/fundamentals/fundamentals_fmp.parquet')
    p.add_argument('--warehouse-root', default='data/warehouse')
    args = p.parse_args()

    equities_root = Path(args.equities_root)
    fundamentals_path = Path(args.fundamentals_path)
    warehouse_root = Path(args.warehouse_root)

    versions_path = warehouse_root / 'equity_prices_daily_versions.parquet'
    selected_path = warehouse_root / 'equity_prices_daily.parquet'
    membership_out_path = warehouse_root / 'universe_membership_daily.parquet'
    fundamentals_out_path = warehouse_root / 'equity_fundamentals_pit.parquet'
    symbol_history_path = warehouse_root / 'symbol_history.parquet'
    audit_path = warehouse_root / 'ingestion_audit.parquet'

    security = pd.read_parquet(warehouse_root / 'security_master.parquet', columns=['ticker_id', 'canonical_symbol'])
    map_df = security[['ticker_id', 'canonical_symbol']].copy()

    selected_rows, price_hist = _stream_selected_prices(versions_path=versions_path, out_path=selected_path)

    membership = pd.read_parquet(equities_root / 'universe_membership.parquet')
    membership['raw_source_symbol'] = membership['ticker'].astype(str).str.strip().str.upper()
    membership['canonical_symbol'] = membership['raw_source_symbol'].map(normalize_symbol)
    membership = membership.merge(map_df, on='canonical_symbol', how='left')
    unmatched_membership = int(membership['ticker_id'].isna().sum())
    membership_out = membership.loc[membership['ticker_id'].notna()].copy()
    membership_out['date'] = pd.to_datetime(membership_out['date'], errors='coerce').dt.normalize()
    membership_out['universe'] = membership_out['universe'].astype(str).str.strip().str.lower()
    if membership_out['in_universe'].dtype != bool:
        membership_out['in_universe'] = membership_out['in_universe'].astype(bool)
    membership_out['load_ts'] = datetime.now(UTC).isoformat()
    membership_out = membership_out[
        ['date', 'universe', 'ticker_id', 'raw_source_symbol', 'canonical_symbol', 'in_universe', 'load_ts']
    ].sort_values(['date', 'universe', 'ticker_id']).reset_index(drop=True)
    membership_out.to_parquet(membership_out_path, index=False)

    fundamentals = pd.read_parquet(fundamentals_path)
    fundamentals['raw_source_symbol'] = fundamentals['ticker'].astype(str).str.strip().str.upper()
    fundamentals['canonical_symbol'] = fundamentals['raw_source_symbol'].map(normalize_symbol)
    fundamentals = fundamentals.merge(map_df, on='canonical_symbol', how='left')
    unmatched_fundamentals = int(fundamentals['ticker_id'].isna().sum())
    fundamentals_out = fundamentals.loc[fundamentals['ticker_id'].notna()].copy()
    fundamentals_out['period_end'] = pd.to_datetime(fundamentals_out['period_end'], errors='coerce').dt.normalize()
    fundamentals_out['available_date'] = pd.to_datetime(fundamentals_out['available_date'], errors='coerce').dt.normalize()
    fundamentals_out, corrected_count = enforce_available_date_floor(fundamentals_out)
    if corrected_count > 0:
        print(
            '[fundamentals] floored available_date to period_end for '
            f'{corrected_count} rows before writing equity_fundamentals_pit'
        )
    fundamentals_out['source'] = 'fmp'
    fundamentals_out['load_ts'] = datetime.now(UTC).isoformat()
    fundamentals_out = fundamentals_out[
        [
            'ticker_id', 'raw_source_symbol', 'canonical_symbol', 'period_end', 'available_date',
            'revenue', 'cogs', 'gross_profit', 'total_assets', 'shareholders_equity',
            'net_income', 'shares_outstanding', 'source', 'load_ts',
        ]
    ].sort_values(['ticker_id', 'available_date', 'period_end']).reset_index(drop=True)
    fundamentals_out.to_parquet(fundamentals_out_path, index=False)

    symbol_history = _build_symbol_history(price_hist=price_hist, membership=membership_out, fundamentals=fundamentals_out)
    symbol_history.to_parquet(symbol_history_path, index=False)

    source_daily_rows = int(pq.ParquetFile(equities_root / 'daily_ohlcv.parquet').metadata.num_rows)
    versions_rows = int(pq.ParquetFile(versions_path).metadata.num_rows)
    run_id = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    stability_report = pd.read_parquet(warehouse_root / 'ticker_id_stability_report.parquet')
    collision_report = pd.read_parquet(warehouse_root / 'symbol_collision_report.parquet')

    audit_rows = [
        _table_audit(run_id, 'security_master', str(equities_root), len(security), len(security), 0, 'ok'),
        _table_audit(
            run_id,
            'equity_prices_daily_versions',
            str(equities_root / 'daily_ohlcv.parquet'),
            source_daily_rows,
            versions_rows,
            source_daily_rows - versions_rows,
            'ok' if source_daily_rows == versions_rows else 'warning',
        ),
        _table_audit(
            run_id,
            'equity_prices_daily',
            str(equities_root / 'daily_ohlcv.parquet'),
            source_daily_rows,
            selected_rows,
            source_daily_rows - versions_rows,
            'ok' if source_daily_rows == versions_rows else 'warning',
        ),
        _table_audit(
            run_id,
            'universe_membership_daily',
            str(equities_root / 'universe_membership.parquet'),
            len(membership),
            len(membership_out),
            unmatched_membership,
            'ok' if unmatched_membership == 0 else 'warning',
        ),
        _table_audit(
            run_id,
            'equity_fundamentals_pit',
            str(fundamentals_path),
            len(fundamentals),
            len(fundamentals_out),
            unmatched_fundamentals,
            'ok' if unmatched_fundamentals == 0 else 'warning',
        ),
        _table_audit(
            run_id,
            'symbol_history',
            'warehouse_internal',
            int(selected_rows + len(membership_out) + len(fundamentals_out)),
            len(symbol_history),
            0,
            'ok',
        ),
        _table_audit(
            run_id,
            'ticker_id_stability_report',
            'warehouse_internal',
            len(stability_report),
            len(stability_report),
            int(stability_report.get('stability_issue', pd.Series(dtype=bool)).astype(bool).sum()),
            'ok' if int(stability_report.get('stability_issue', pd.Series(dtype=bool)).astype(bool).sum()) == 0 else 'warning',
        ),
        _table_audit(
            run_id,
            'symbol_collision_report',
            'warehouse_internal',
            len(collision_report),
            len(collision_report),
            int(collision_report.get('ambiguous_mapping_flag', pd.Series(dtype=bool)).astype(bool).sum()),
            'ok' if int(collision_report.get('ambiguous_mapping_flag', pd.Series(dtype=bool)).astype(bool).sum()) == 0 else 'warning',
        ),
    ]
    pd.DataFrame(audit_rows).to_parquet(audit_path, index=False)

    print({
        'selected_price_rows': selected_rows,
        'membership_rows': int(len(membership_out)),
        'fundamentals_rows': int(len(fundamentals_out)),
        'symbol_history_rows': int(len(symbol_history)),
        'audit_rows': len(audit_rows),
    })


if __name__ == '__main__':
    main()
