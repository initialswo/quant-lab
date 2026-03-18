# Quant Lab Data Fill Policy

## Purpose

This document defines the **official rules for handling missing data** in Quant Lab.

It exists to:

- prevent lookahead bias
- ensure consistent factor computation
- standardize data preprocessing across experiments

This document is **authoritative for all fill behavior**.

---

# 1. Core Principle

All data handling must be **causal**.

> No value from time t+k may influence any computation at time t.

Any operation that violates this rule introduces **lookahead bias**.

---

# 2. Price Data (OHLCV)

## Allowed

- Forward fill (`ffill`) is allowed for:
  - short gaps (e.g., non-trading days)
  - continuity in return calculations

## NOT Allowed

- Backward fill (`bfill`) is strictly prohibited

Reason:
- `bfill` uses future prices to fill past values
- this directly introduces forward leakage

## Missing Data Handling

If price data is missing:

- do NOT fill using future values
- either:
  - leave as NaN
  - or drop the asset for that date via universe masking

---

# 3. Volume / Liquidity Data

## Allowed

- Forward fill for short gaps only if necessary for rolling metrics

## Preferred

- Recompute rolling metrics using available data
- Require sufficient history (e.g., ADV20 requires 20 valid observations)

## NOT Allowed

- Backward fill
- filling long gaps

---

# 4. Fundamental Data (PIT)

Fundamentals must be:

- point-in-time aligned
- only available after their publication date

## Allowed

- Carry-forward last known value after it becomes available

## NOT Allowed

- Using future filings
- Backfilling values prior to release date

---

# 5. Factor Computation

Factors must respect:

- causal inputs only
- sufficient lookback windows

## Rules

- If required lookback is incomplete → result is NaN
- Do not fill missing factor values using future information

---

# 6. Universe Eligibility

Missing data should be handled through **eligibility filtering**, not filling.

Examples:

- insufficient price history → exclude asset
- insufficient volume history → exclude asset
- missing fundamentals → exclude asset

This ensures:

> clean inputs rather than artificially filled data

---

# 7. Panel Construction

When constructing wide panels:

## Allowed

- forward alignment of known values
- NaNs where data is unavailable

## NOT Allowed

- global filling across time
- filling that alters historical availability

---

# 8. Exceptions

Any deviation from this policy must:

- be explicitly documented in the experiment
- include justification
- be treated as **non-promotable** unless validated

---

# 9. Validation Requirements

All pipelines must ensure:

- no backward fill is applied
- no future data leaks into past rows
- PIT alignment is preserved

Tests should verify:

- monotonic availability of fundamentals
- no negative lags in data joins
- correct handling of missing values

---

# 10. Relationship to Research Protocol

See:

- `research_protocol.md` → governance rules
- `research_principles.md` → bias and PIT principles

This document defines the **implementation rules** for missing data.

---

# 11. Guiding Principle

> It is better to lose data than to corrupt it.

Missing data reduces sample size.  
Incorrectly filled data invalidates results.