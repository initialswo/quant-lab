"""Build a Phase 3 Tiingo ingest manifest with a priority current-universe overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.tiingo_universe import (
    DEFAULT_TIINGO_US_COMMON_EQUITY_MANIFEST,
    normalize_symbol,
)

DEFAULT_OUTPUT = Path('data/universe/tiingo_us_common_equities_current_sp500_overlay.csv')


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors='coerce', utc=True).dt.tz_convert(None).dt.normalize()


def _load_base_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f'Base manifest is empty: {path}')
    frame['ticker'] = frame['ticker'].astype(str).str.strip().str.upper().map(normalize_symbol)
    if 'norm' in frame.columns:
        frame['norm'] = frame['norm'].astype(str).str.strip().str.upper().map(normalize_symbol)
    else:
        frame['norm'] = frame['ticker'].map(normalize_symbol)
    if 'isActive' in frame.columns:
        frame['isActive'] = frame['isActive'].fillna(False).astype(bool)
    else:
        frame['isActive'] = True
    if 'isADR' in frame.columns:
        frame['isADR'] = frame['isADR'].fillna(False).astype(bool)
    else:
        frame['isADR'] = False
    return frame.drop_duplicates(subset=['ticker'], keep='first').reset_index(drop=True)


def _load_baseline_norms(metadata_path: Path) -> set[str]:
    md = pd.read_parquet(metadata_path, columns=['ticker'])
    return set(md['ticker'].astype(str).map(normalize_symbol))


def _load_current_overlay(membership_path: Path, universe: str) -> tuple[pd.DataFrame, str]:
    membership = pd.read_parquet(membership_path, columns=['date', 'universe', 'ticker', 'in_universe'])
    membership['date'] = _normalize_date(membership['date'])
    membership['universe'] = membership['universe'].astype(str).str.strip().str.lower()
    membership['ticker'] = membership['ticker'].astype(str).str.strip().str.upper().map(normalize_symbol)
    membership['in_universe'] = membership['in_universe'].fillna(False).astype(bool)
    filtered = membership.loc[
        membership['universe'].eq(str(universe).strip().lower())
        & membership['in_universe']
        & membership['ticker'].ne('')
    ].copy()
    if filtered.empty:
        raise ValueError(f'No active membership rows found for universe={universe!r} in {membership_path}')
    latest = filtered['date'].max()
    overlay = filtered.loc[filtered['date'].eq(latest), ['ticker']].drop_duplicates().sort_values('ticker').reset_index(drop=True)
    overlay['norm'] = overlay['ticker'].map(normalize_symbol)
    return overlay, str(latest.date())


def build_priority_manifest(
    base_manifest_path: Path,
    membership_path: Path,
    metadata_path: Path,
    overlay_universe: str = 'sp500',
) -> tuple[pd.DataFrame, dict]:
    base = _load_base_manifest(base_manifest_path)
    baseline_norms = _load_baseline_norms(metadata_path)
    overlay, overlay_date = _load_current_overlay(membership_path, universe=overlay_universe)

    overlay_tickers = set(overlay['ticker'])
    overlay_norms = set(overlay['norm'])
    keep_base = base['isActive'] | base['norm'].isin(baseline_norms) | base['ticker'].isin(overlay_tickers) | base['norm'].isin(overlay_norms)
    base_kept = base.loc[keep_base].copy()

    base_by_ticker = base.drop_duplicates(subset=['ticker'], keep='first').set_index('ticker', drop=False)
    rows: list[dict] = []
    seen_tickers: set[str] = set()

    for ticker in overlay['ticker'].tolist():
        if ticker in base_by_ticker.index:
            row = base_by_ticker.loc[ticker].to_dict()
            from_base = True
        else:
            row = {
                'ticker': ticker,
                'norm': normalize_symbol(ticker),
                'name': pd.NA,
                'permaTicker': pd.NA,
                'isActive': True,
                'isADR': False,
            }
            from_base = False
        row['manifest_role'] = 'priority_current_overlay'
        row['priority_rank'] = 0
        row['overlay_universe'] = str(overlay_universe).strip().lower()
        row['overlay_membership_date'] = overlay_date
        row['from_base_manifest'] = bool(from_base)
        rows.append(row)
        seen_tickers.add(ticker)

    active_remaining = base_kept.loc[base_kept['isActive'] & ~base_kept['ticker'].isin(seen_tickers)].sort_values('ticker')
    for row in active_remaining.to_dict(orient='records'):
        row['manifest_role'] = 'broad_manifest_active'
        row['priority_rank'] = 1
        row['overlay_universe'] = pd.NA
        row['overlay_membership_date'] = pd.NA
        row['from_base_manifest'] = True
        rows.append(row)
        seen_tickers.add(str(row['ticker']))

    inactive_remaining = base_kept.loc[(~base_kept['isActive']) & ~base_kept['ticker'].isin(seen_tickers)].sort_values('ticker')
    for row in inactive_remaining.to_dict(orient='records'):
        row['manifest_role'] = 'baseline_existing_inactive'
        row['priority_rank'] = 2
        row['overlay_universe'] = pd.NA
        row['overlay_membership_date'] = pd.NA
        row['from_base_manifest'] = True
        rows.append(row)
        seen_tickers.add(str(row['ticker']))

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError('Priority manifest produced 0 rows')

    preferred_cols = [
        'ticker', 'norm', 'name', 'permaTicker', 'isActive', 'isADR',
        'manifest_role', 'priority_rank', 'overlay_universe', 'overlay_membership_date', 'from_base_manifest',
    ]
    extra_cols = [c for c in base.columns if c not in preferred_cols]
    ordered = [c for c in preferred_cols if c in out.columns] + extra_cols
    out = out[ordered].drop_duplicates(subset=['ticker'], keep='first').reset_index(drop=True)

    summary = {
        'base_manifest_path': str(base_manifest_path),
        'membership_path': str(membership_path),
        'metadata_path': str(metadata_path),
        'overlay_universe': str(overlay_universe).strip().lower(),
        'overlay_membership_date': overlay_date,
        'base_manifest_rows': int(len(base)),
        'base_manifest_active_rows': int(base['isActive'].sum()),
        'base_manifest_inactive_rows': int((~base['isActive']).sum()),
        'baseline_metadata_norm_count': int(len(baseline_norms)),
        'overlay_ticker_count': int(len(overlay)),
        'overlay_missing_from_broad_manifest': sorted(t for t in overlay_tickers if t not in set(base['ticker'])),
        'output_rows': int(len(out)),
        'output_priority_rows': int((out['priority_rank'] == 0).sum()),
        'output_active_broad_rows': int((out['manifest_role'] == 'broad_manifest_active').sum()),
        'output_inactive_baseline_rows': int((out['manifest_role'] == 'baseline_existing_inactive').sum()),
        'filtered_out_inactive_nonbaseline_rows': int((~keep_base).sum()),
    }
    return out, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-manifest', default=str(DEFAULT_TIINGO_US_COMMON_EQUITY_MANIFEST))
    p.add_argument('--membership-path', default='data/equities/universe_membership.parquet')
    p.add_argument('--metadata-path', default='data/equities/metadata.parquet')
    p.add_argument('--overlay-universe', default='sp500')
    p.add_argument('--output', default=str(DEFAULT_OUTPUT))
    p.add_argument('--summary-output', default='')
    args = p.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output) if str(args.summary_output).strip() else output_path.with_suffix('.summary.json')

    manifest, summary = build_priority_manifest(
        base_manifest_path=Path(args.base_manifest),
        membership_path=Path(args.membership_path),
        metadata_path=Path(args.metadata_path),
        overlay_universe=args.overlay_universe,
    )
    manifest.to_csv(output_path, index=False)
    summary.update({'output_path': str(output_path), 'summary_path': str(summary_path)})
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
