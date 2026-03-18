from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / 'scripts' / 'build_phase3_priority_tiingo_manifest.py'
    spec = importlib.util.spec_from_file_location('build_phase3_priority_tiingo_manifest', path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_priority_manifest_adds_overlay_and_filters_inactive_noise(tmp_path) -> None:
    mod = _load_module()
    base_manifest = tmp_path / 'base_manifest.csv'
    membership = tmp_path / 'membership.parquet'
    metadata = tmp_path / 'metadata.parquet'

    pd.DataFrame(
        [
            {'ticker': 'AAA', 'norm': 'AAA', 'name': 'AAA Corp', 'permaTicker': 'US1', 'isActive': True, 'isADR': False},
            {'ticker': 'BBB', 'norm': 'BBB', 'name': 'BBB Corp', 'permaTicker': 'US2', 'isActive': False, 'isADR': False},
            {'ticker': 'GOOGL', 'norm': 'GOOGL', 'name': 'Alphabet Inc - Class A', 'permaTicker': 'US3', 'isActive': True, 'isADR': False},
            {'ticker': 'ZZZ', 'norm': 'ZZZ', 'name': 'ZZZ Corp', 'permaTicker': 'US4', 'isActive': False, 'isADR': False},
        ]
    ).to_csv(base_manifest, index=False)

    pd.DataFrame(
        [
            {'date': '2026-03-05', 'universe': 'sp500', 'ticker': 'OLD', 'in_universe': True},
            {'date': '2026-03-06', 'universe': 'sp500', 'ticker': 'AAA', 'in_universe': True},
            {'date': '2026-03-06', 'universe': 'sp500', 'ticker': 'GOOG', 'in_universe': True},
        ]
    ).to_parquet(membership, index=False)

    pd.DataFrame([
        {'ticker': 'BBB', 'name': 'BBB Corp'},
    ]).to_parquet(metadata, index=False)

    manifest, summary = mod.build_priority_manifest(
        base_manifest_path=base_manifest,
        membership_path=membership,
        metadata_path=metadata,
        overlay_universe='sp500',
    )

    assert manifest['ticker'].tolist() == ['AAA', 'GOOG', 'GOOGL', 'BBB']
    assert manifest['manifest_role'].tolist() == [
        'priority_current_overlay',
        'priority_current_overlay',
        'broad_manifest_active',
        'baseline_existing_inactive',
    ]
    assert bool(manifest.loc[manifest['ticker'] == 'GOOG', 'from_base_manifest'].iloc[0]) is False
    assert 'GOOG' in summary['overlay_missing_from_broad_manifest']
    assert summary['filtered_out_inactive_nonbaseline_rows'] == 1
