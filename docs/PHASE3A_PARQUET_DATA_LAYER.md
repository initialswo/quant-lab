# Phase 3A: Large-Universe Parquet Data Layer

## Why Centralized Parquet

One-file-per-ticker research workflows are easy to start with, but become hard to scale for:

- broad universes (SP1500 / Russell 3000),
- cross-universe comparison,
- repeated parameter sweeps over long horizons.

Centralized long-format Parquet tables reduce repeated file I/O and provide one canonical source
for universe-aware loaders and diagnostics.

## Storage Layout

Under `data/equities/`:

- `daily_ohlcv.parquet`
  - long format: `date,ticker,open,high,low,close,volume,source`
- `metadata.parquet`
  - one row per ticker with:
    - `ticker,name,exchange,sector,industry,first_date,last_date,active_flag,source`
  - non-available fields stay nullable (no invented values)
- `universe_membership.parquet`
  - long format: `date,universe,ticker,in_universe`

## Universe Representation

Universe membership is point-in-time and date-indexed in long format.

- `sp500`:
  - fully supported when historical membership is ingested.
  - fallback to current-membership expansion if historical rows are missing.
- `sp1500`, `russell3000`:
  - interfaces are scaffolded.
  - real historical/current membership sources still need to be ingested.

## Preferred Research Loader

Use `quant_lab.data.loaders.load_ohlcv_for_research(...)`.

It returns:

- long-format OHLCV (`ohlcv_long`)
- pivoted wide panels (`open/high/low/close/volume`) for compatibility
- diagnostics summary (`rows_loaded`, `tickers_loaded`, date bounds, etc.)

## Ingestion Paths

CLI helpers:

- `python run.py ingest-equity-cache --cache_dir data/cache/stooq --source stooq_cache`
- `python run.py ingest-universe-membership --csv_path data/universe/sp500_historical_membership.csv --universe sp500`

These ingestion commands feed centralized Parquet storage while preserving existing raw cache files.
