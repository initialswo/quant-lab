from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / 'scripts' / 'continue_phase3_bulk_ingest_from_raw.py'
    spec = importlib.util.spec_from_file_location('continue_phase3_bulk_ingest_from_raw', path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_source_daily_duckdb_merges_baseline_and_raw_snapshot(tmp_path) -> None:
    mod = _load_module()
    base_daily = tmp_path / 'base_daily.parquet'
    raw_snapshot = tmp_path / 'raw_snapshot.parquet'
    out_path = tmp_path / 'merged.parquet'

    pd.DataFrame([
        {'date': '2024-01-02', 'ticker': 'AAA.US', 'open': 1.0, 'high': 1.2, 'low': 0.9, 'close': 1.1, 'adj_close': 1.05, 'volume': 10, 'source': 'tiingo'},
    ]).to_parquet(base_daily, index=False)
    pd.DataFrame([
        {'date': '2024-01-03', 'ticker': 'BBB.US', 'open': 2.0, 'high': 2.2, 'low': 1.9, 'close': 2.1, 'adjClose': 2.05, 'volume': 20, 'source': 'tiingo_phase3_bulk'},
    ]).to_parquet(raw_snapshot, index=False)

    mod._build_source_daily_duckdb(base_daily_path=base_daily, raw_snapshot_path=raw_snapshot, out_path=out_path)
    merged = pd.read_parquet(out_path)

    assert merged[['ticker', 'source']].to_dict(orient='records') == [
        {'ticker': 'AAA.US', 'source': 'tiingo'},
        {'ticker': 'BBB.US', 'source': 'tiingo_phase3_bulk'},
    ]
    assert float(merged.loc[merged['ticker'] == 'BBB.US', 'adj_close'].iloc[0]) == 2.05
