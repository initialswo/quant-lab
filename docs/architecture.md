# quant_lab Architecture

## System Overview
`quant_lab` is organized as a modular research engine:
- **CLI orchestration**: [`run.py`](/workspaces/quant_lab/run.py)
- **Execution engine**: `src/quant_lab/engine/runner.py`
- **Data layer**: `src/quant_lab/data/*`
- **Signal/factor layer**: `src/quant_lab/factors/*`
- **Portfolio logic**: `src/quant_lab/strategies/topn.py`
- **Risk overlays**: `src/quant_lab/risk/*`
- **Research sleeves/utilities**: `src/quant_lab/research/*`
- **Artifact/registry outputs**: `src/quant_lab/results/*`

## Directory Map
- `src/quant_lab/data`: loaders, parquet store, fundamentals PIT alignment, universe utilities, ingest/downloader integrations.
- `src/quant_lab/factors`: factor plugins discovered by registry; normalization/combination/neutralization helpers.
- `src/quant_lab/strategies`: Top-N weight builders and portfolio simulation.
- `src/quant_lab/engine`: run orchestration and metrics.
- `src/quant_lab/risk`: trend/regime and factor-weight regime logic.
- `src/quant_lab/research`: sleeve-specific reusable research modules.
- `scripts/`: reproducible experiment runners (benchmark, sweeps, combination tests).
- `tests/`: unit/integration coverage for data, factors, runner, sleeves, and diagnostics.

## End-to-End Backtest Flow
Implemented primarily in `run_backtest(...)` within [`runner.py`](/workspaces/quant_lab/src/quant_lab/engine/runner.py).

1. **Parse and validate config**
- Universe mode, factor aggregation, risk/overlay settings, and output flags are validated.

2. **Universe seed selection**
- `sp500`: primary seed from parquet historical membership (`data/equities/universe_membership.parquet`), with CSV fallback if parquet membership is unavailable.
- `liquid_us` / `all*`: ticker set from parquet metadata/daily ticker universe.

3. **Load OHLCV**
- Uses `fetch_ohlcv_with_summary(...)` (parquet/default source abstraction).
- Builds aligned `close`/`volume` panels.
- Applies panel sanity checks and optional bad-series dropping.

4. **Build factor inputs**
- Computes selected factors via registry plugin calls.
- Injects PIT-aligned fundamentals for fundamentals-dependent factors.
- Applies preprocessing: winsor/zscore/rank and optional neutralization/orthogonalization.

5. **Combine factors**
- Linear or rank-based aggregation (`linear`, `mean_rank`, `geometric_rank`).
- Supports static or regime-dependent factor weights.

6. **Apply universe/liquidity/membership masking**
- Dynamic eligibility can combine:
  - price/history/validity constraints
  - liquidity thresholds
  - historical membership (SP500 path auto-loads parquet membership when no explicit membership file is provided)
  - optional universe-dataset build/use mode

7. **Construct portfolio weights**
- Top-N selection on rebalance schedule.
- Weighting modes: `equal`, `inv_vol`, `score`, `score_inv_vol`.
- Optional rank buffer, sector constraints, and volatility-scaled variant.

8. **Simulate returns (causal)**
- Daily return uses lagged holdings (`w[t-1] * r[t]`).
- Turnover-based costs on rebalance dates.
- Optional execution delay and slippage components.

9. **Apply overlays**
- Optional trend filter.
- Optional portfolio-level vol targeting (lagged leverage scale).
- Optional bear exposure scaling.

10. **Write outputs**
- Metrics summary, equity curve, holdings, diagnostics CSVs.
- Optional run registry append.

## Data Layer Design
Centralized parquet model:
- `daily_ohlcv.parquet`: long-format OHLCV
- `metadata.parquet`: ticker metadata (`first_date`, `last_date`, `active_flag`, etc.)
- `universe_membership.parquet`: date/universe/ticker membership rows

This structure supports large-universe experimentation without per-ticker file sprawl.

## Data Buildout Plan (Pre-Subscription Expiry)
To maximize long-term research value before external data subscriptions expire, the target local warehouse should be expanded to include:
- full daily US equity OHLCV history and adjustment metadata
- PIT fundamentals with explicit availability dates
- security master + symbol lifecycle history
- universe/reference membership snapshots
- corporate actions as first-class tables

Recommended target parquet tables:
- `data/warehouse/equity_prices_daily.parquet`: unadjusted + adjusted OHLCV, source IDs, and load audit fields.
- `data/warehouse/equity_fundamentals_quarterly_pit.parquet`: statement fundamentals keyed by `(ticker_id, period_end, available_date)`.
- `data/warehouse/security_master.parquet`: canonical security IDs, normalized symbols, exchange/asset metadata, first/last trading dates.
- `data/warehouse/symbol_history.parquet`: historical symbol mappings and effective date ranges.
- `data/warehouse/universe_membership_daily.parquet`: PIT membership rows by universe and date.
- `data/warehouse/corporate_actions.parquet`: splits, dividends, and optional adjustment factors.
- `data/warehouse/ingestion_audit.parquet`: per-run status, row counts, checksum hashes, and quality flags.

Core integrity rules:
- canonical symbol normalization before joins (`.`/`-`, suffixes, casing) with deterministic mapping tables
- PIT-safe joins based on `available_date` for fundamentals
- deterministic dedup precedence by source/run timestamp
- required post-load coverage + gap diagnostics emitted as CSV/JSON artifacts

Phase 1 status:
- implemented base warehouse tables:
  - `security_master.parquet`
  - `equity_prices_daily.parquet`
  - `universe_membership_daily.parquet`
  - `equity_fundamentals_pit.parquet`
  - `ingestion_audit.parquet`
- implemented validation bundle generation in `results/data_validation/<timestamp>/`

Phase 1.5 hardening status:
- added `symbol_history.parquet` with effective date ranges per `(ticker_id, raw_source_symbol)`
- added warehouse diagnostics:
  - `ticker_id_stability_report.parquet`
  - `symbol_collision_report.parquet`
- validation now supports fail-fast thresholds (nonzero exit) for duplicates, unmatched symbols, critical nulls, and ticker ID instability.

Phase 2A staging status:
- vendor dry-runs ingest into `data/staging/phase2a/<timestamp>/raw/` first
- staged source snapshots are materialized under `data/staging/phase2a/<timestamp>/source/`
- warehouse build/validation runs against staged source roots before any full bulk ingest.

Phase 2B hardening policy:
- metadata consolidation by canonical symbol is deterministic:
  - pick representative row by latest `last_date`
  - tie-break by non-`.US` symbol preference, then earliest `first_date`, then lexical ticker
  - `consolidated_active_flag` is taken from the selected representative row
  - conflict diagnostics retained via `active_flag_conflict` + metadata reports
- price source precedence is deterministic and versioned:
  - all competing rows are stored in `equity_prices_daily_versions.parquet`
  - one selected row per `(date, ticker_id)` is chosen by `source_rank` (descending), then symbol tie-breakers
  - selected panel is written to `equity_prices_daily.parquet` with `source_rank`, `load_batch_id`, and `load_ts`

Phase 3 operational flow:
- bulk vendor pulls are staged under `data/staging/phase3/<timestamp>/raw/`
- staged source snapshots are assembled under `data/staging/phase3/<timestamp>/source/`
- warehouse build + strict validation run on staged inputs
- security master metadata completeness is audited separately from price-layer promotion readiness
- validated price + identity outputs are promoted into canonical `data/warehouse/` even if `sector` / `industry` coverage remains incomplete
- classification-dependent research remains guarded until classification coverage improves
- pre-promotion warehouse snapshots are backed up under `results/ingest/phase3/backups/<timestamp>/`

## Causality / Lookahead Controls
- Portfolio returns are computed from lagged weights.
- Inverse-vol and related risk estimates use trailing windows with lag where applicable.
- Fundamentals are aligned by `available_date` and then forward-filled.
- Vol targeting uses lagged realized volatility to set leverage.

## Result Artifacts
Common outputs include:
- `summary.json`, `summary.txt`
- `equity.csv` or `equity_curve.csv`
- holdings and diagnostics tables
- optional universe and regime diagnostics

Experiment metadata is optionally appended to `results/registry.csv`.

## Research Script Layer
`scripts/` contains focused, reproducible experiment entry points:
- benchmark robustness sweeps
- sleeve benchmark grids
- strategy combination tests
- data ingestion/build utilities

This keeps core engine generic while allowing fast iteration without broad framework refactors.
