# Quant Lab Data Status (Authoritative Summary)

## Overview

## Dataset Snapshot

Current promoted warehouse:

Equity universe size: ~2,528 tickers  
Price rows: ~9.48M  
Price history: 1962-01-02 → 2026-03-06  
Fundamentals coverage: 2395 / 2528 (~94.7%)

Data sources:

Tiingo — equity prices  
FMP — fundamentals  

All research experiments use the warehouse snapshot located in:

data/warehouse/

As of Phase 3 bulk ingest, Quant Lab uses a validated **local warehouse-backed equity dataset** for all research experiments.

The warehouse integrates data from:

- Tiingo — daily equity prices
- FMP (Financial Modeling Prep) — fundamentals
- historical index membership datasets
- internal identity and symbol-history mappings

All research runs use the promoted warehouse under:

data/warehouse/

This replaced earlier mixed pipelines that relied partially on raw caches.

---

# Warehouse Architecture

Canonical warehouse tables:

data/warehouse/
    security_master.parquet
    symbol_history.parquet
    equity_prices_daily.parquet
    equity_prices_daily_versions.parquet
    equity_fundamentals_pit.parquet
    universe_membership_daily.parquet
    ingestion_audit.parquet

---

## security_master

Identity table assigning stable ticker_id values.

Fields include:

ticker_id  
canonical_symbol  
raw_source_symbol  
exchange  
active_flag  
first_trade_date  
last_trade_date  

Ticker IDs are deterministic and reused across builds.

Validation confirmed:

ticker_id reused: 3122  
ticker_id changed: 0

---

## symbol_history

Tracks symbol lifecycle events.

ticker_id  
symbol  
effective_from  
effective_to  
change_type  

Used to resolve symbol changes and collisions.

---

## equity_prices_daily

Primary research price table.

Columns include:

date  
ticker_id  
source_symbol  
open  
high  
low  
close  
volume  
adj_close  
source  
load_ts  

Current coverage:

rows: ~9.48M  
tickers: 2528  
date range: 1962-01-02 → 2026-03-06

---

## equity_prices_daily_versions

Full version history of price rows before precedence selection.

Selection precedence example:

tiingo_phase3  
> tiingo  
> tiingo_cache  
> stooq_cache

This allows conflicts between vendors to be audited.

---

## equity_fundamentals_pit

Point-in-time fundamentals table.

Key columns:

ticker_id  
period_end  
filing_date  
accepted_date  
available_date  
revenue  
cogs  
gross_profit  
net_income  
total_assets  
shareholders_equity  
shares_outstanding  

Important property:

available_date determines when data becomes visible.

Coverage after bulk ingest:

2395 / 2528 tickers (~94.7%)

---

## universe_membership_daily

Tracks point-in-time universe membership.

date  
universe  
ticker_id  
in_universe  

Used to prevent index survivorship bias.

---

## ingestion_audit

Tracks ingestion runs:

run_id  
table_name  
rows_in  
rows_out  
duplicate_count  
status  
started_at  
finished_at  

---

# Validation Framework

Warehouse builds automatically run validation.

Artifacts stored under:

results/data_validation/

Checks include:

duplicate_key_report  
symbol_match_report  
coverage_by_year  
fundamentals_coverage  
symbol_collision_report  
ticker_id_stability_report  
null_profile  

Strict thresholds enforced:

duplicate rows: 0 tolerance  
unmatched symbols: 0 tolerance  
ticker identity instability: 0 tolerance  
critical null failures: 0 tolerance  

Latest validation status:

PASSED

---

# Warmup Period Behavior

Some strategies require historical lookback windows.

Example:

liquid_us dynamic universe  
minimum history ≈ 300 trading days

This produces early warmup behavior.

Example:

62 skipped rebalances  
2005-01-03 → 2006-03-06

This is expected initialization behavior.

---

# Research Reproducibility

After Phase 3 ingest, benchmark strategies were rerun.

Results were unchanged within floating-point precision.

This confirms:

data warehouse integrity  
engine determinism  
stable research environment

---

# Current Data Integrity Assessment

Quant Lab now has:

stable security master  
deterministic symbol normalization  
symbol history tracking  
versioned price data  
PIT fundamentals  
point-in-time universe membership  
strict validation gates  
reproducible backtests  

Additional protections:

- point-in-time fundamentals prevent lookahead bias
- point-in-time universe membership prevents index survivorship bias
- symbol history prevents ticker reuse contamination
- versioned price data enables vendor conflict auditing

This dataset is suitable for serious factor research.

# Known Data Limitations

Some vendor limitations still exist:

- Certain security types (units, warrants, SPAC instruments) may lack fundamentals from FMP.
- A small number of fundamental rows contain non-positive `total_assets` values and are filtered during factor construction.
- Vendor metadata occasionally contains conflicting `active_flag` values; deterministic consolidation rules resolve these conflicts.

These limitations do not materially affect benchmark factor experiments but should be considered when expanding the universe.