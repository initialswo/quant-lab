"""One-time backfill to add adj_close to the active equity parquet without altering raw close."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_lab.data.parquet_store import DAILY_COLUMNS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--daily-path', default='data/equities/daily_ohlcv.parquet')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = Path(str(args.daily_path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'Missing parquet file: {path}')
    frame = pd.read_parquet(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if 'close' not in frame.columns:
        raise ValueError('daily_ohlcv.parquet must contain close before backfill')
    if 'adj_close' not in frame.columns:
        frame['adj_close'] = pd.to_numeric(frame['close'], errors='coerce')
    else:
        frame['adj_close'] = pd.to_numeric(frame['adj_close'], errors='coerce')
        frame['adj_close'] = frame['adj_close'].where(frame['adj_close'].notna(), pd.to_numeric(frame['close'], errors='coerce'))
    for col in DAILY_COLUMNS:
        if col not in frame.columns:
            raise ValueError(f'daily_ohlcv.parquet missing required column after backfill: {col}')
    frame = frame[DAILY_COLUMNS].copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce').dt.normalize()
    frame = frame.loc[frame['date'].notna()].sort_values(['date', 'ticker']).reset_index(drop=True)
    frame.to_parquet(path, index=False)
    print(f'Backfilled adj_close into {path}')
    print(frame.columns.tolist())
    print(f'rows={len(frame)} tickers={frame["ticker"].astype(str).nunique()}')


if __name__ == '__main__':
    main()
