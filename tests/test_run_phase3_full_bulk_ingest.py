from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / 'scripts' / 'run_phase3_full_bulk_ingest.py'
    spec = importlib.util.spec_from_file_location('run_phase3_full_bulk_ingest', path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_ticker_universe_preserves_manifest_order_for_priority_overlay(tmp_path) -> None:
    mod = _load_module()
    equities_root = tmp_path / 'equities'
    equities_root.mkdir(parents=True)
    pd.DataFrame([{'ticker': 'ZZZ'}]).to_parquet(equities_root / 'metadata.parquet', index=False)

    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame([
        {'ticker': 'GOOG'},
        {'ticker': 'AAA'},
        {'ticker': 'BBB'},
    ]).to_csv(manifest, index=False)

    out = mod._load_ticker_universe(equities_root=equities_root, symbol_file=manifest, max_tickers=2)
    assert out == ['GOOG', 'AAA']
