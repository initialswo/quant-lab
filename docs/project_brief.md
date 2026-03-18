# quant_lab Project Brief

## Purpose
`quant_lab` is a research-first systematic investing framework for testing equity and cross-asset strategy sleeves with consistent backtesting, diagnostics, and artifact outputs.

The current focus is:
- robust factor-based equity benchmarking
- diversification sleeve research (cross-asset trend, sector rotation, long/short, growth leader variants)
- disciplined experiment tracking and reproducibility

## Core Capabilities
- CLI runner for `backtest`, `walkforward`, and `sweep` via [`run.py`](/workspaces/quant_lab/run.py)
- Modular factor library in `src/quant_lab/factors`
- Top-N portfolio construction and causal mark-to-market simulation in `src/quant_lab/strategies/topn.py`
- Dynamic universe controls (price/history/liquidity eligibility, optional dataset mode)
- Parquet-backed SP500 historical membership alignment in the default benchmark path
- Point-in-time fundamentals alignment for quality/value factors
- Standardized result artifacts under `results/` + run registry logging

## Data Model (Current)
Primary local equity store is centralized parquet:
- `data/equities/daily_ohlcv.parquet`
- `data/equities/metadata.parquet`
- `data/equities/universe_membership.parquet`

Cross-asset and sector ETF research scripts also write/read local data under project conventions.

## Strategy Research Surface
Implemented research modules include:
- Equity factor composite benchmark infrastructure (runner + factor stack scripts)
- Cross-asset trend research (`v2` and `v3` grids/combination scripts)
- Sector rotation research sleeve
- Long/short equity research sleeve
- Growth leader equity sleeve

See `scripts/` for experiment runners and `src/quant_lab/research/` for reusable research logic.

## Typical Workflow
1. Build/refresh local data (`ingest-*`, downloader scripts).
2. Run strategy backtest or walkforward with controlled parameters.
3. Review `summary.json`, `equity.csv`, diagnostics CSVs, and registry rows.
4. Run benchmark/combo scripts for sleeve interaction analysis.
5. Promote or reject variants based on robustness and diversification behavior.

## Environment
API-using scripts now auto-load the repo-root `.env` via `python-dotenv`, so `TIINGO_API_KEY` and `FMP_API_KEY` are available even when commands are launched from a different working directory.

## Design Intent
- Keep strategy logic explicit and testable.
- Preserve causal/lag-safe mechanics.
- Avoid hidden optimizers in baseline sleeves.
- Iterate through small, interpretable research steps rather than broad refactors.
