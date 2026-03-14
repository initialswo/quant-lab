# Quant Lab Research Principles

This document defines the research integrity rules used throughout Quant Lab.

All experiments should respect these principles.

---

# 1. No Lookahead Bias

Signals must never use information unavailable at the time of trading.

Correct timeline:

close(t) prices → compute factors  
rank stocks at t  
generate weights  
execute trades at t+1  
returns begin t+1

This prevents unrealistic foresight.

---

# 2. Point-in-Time Fundamentals

Fundamentals are aligned using availability date, not fiscal period end.

Example:

period_end: 2020-03-31  
filing_date: 2020-05-05  
available_date: 2020-05-05  

Data becomes visible only after available_date.

This prevents future knowledge of financial statements.

---

# 3. Point-in-Time Universe Membership

Universe membership must be applied as-of each historical date.

Example dataset:

universe_membership_daily.parquet

This avoids survivorship bias from using current index constituents.

---

# 4. Stable Security Identity

All securities use internal identifiers:

ticker_id

Symbols are not used as primary identifiers because symbols change.

Symbol changes are tracked in:

symbol_history.parquet

---

# 5. Deterministic Data Ingestion

Vendor conflicts must resolve deterministically.

Example precedence:

tiingo_phase3  
> tiingo  
> tiingo_cache  
> stooq_cache  

All candidate rows remain available in:

equity_prices_daily_versions.parquet

---

# 6. Strict Warehouse Validation

Each warehouse build must pass validation before promotion.

Validation checks include:

duplicate rows  
symbol mismatches  
ticker identity stability  
critical null coverage  

Validation artifacts stored in:

results/data_validation/

---

# 7. Explicit Strategy Warmup Periods

Many strategies require history windows.

Examples:

momentum lookback  
volatility estimation  
liquidity estimation  

Universe construction may require minimum history.

Example:

liquid_us universe requires ~300 trading days.

Early skipped rebalances should be treated as initialization behavior.

---

# 8. Reproducibility Requirement

Benchmark results must be reproducible.

If data changes, results should remain stable within floating-point precision.

Large deviations indicate potential data issues.

---

# 9. Research vs Production

Quant Lab is currently a research platform.

Models include realistic assumptions such as:

transaction costs  
slippage  
liquidity filters  

Production considerations like market impact are outside current scope.

---

# 10. Documentation of Experiments

All experiments should be logged in:

docs/experiment_log.md

Entries should include:

experiment description  
parameters  
artifact locations  
results  
interpretation

---

# Current Status

Quant Lab satisfies:

no lookahead bias  
point-in-time fundamentals  
point-in-time universe membership  
stable security identity  
validated warehouse  
reproducible benchmarks  

The platform is suitable for systematic factor research.