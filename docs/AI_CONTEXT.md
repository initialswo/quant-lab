# Quant Lab — AI Context

This document provides minimal onboarding context for AI agents assisting with this repository.

For full documentation see:

`docs/project_brief.md`  
`docs/research_state.md`  
`docs/research_principles.md`  
`docs/research_roadmap.md`  
`docs/experiment_log.md`

---

# Project Purpose

Quant Lab is a Python-based quantitative research platform for systematic trading strategy research.

Primary research areas:

* equity factor strategies
* portfolio construction
* multi-strategy portfolio design

---

# Current Data Readiness

Canonical warehouse location:

`data/warehouse/`

Current promotion-ready staged candidate:

`data/staging/phase3_tiingo_manifest/20260315_phase3_tiingo_priority_overlay1/warehouse/`

Readiness split:

* canonical price layer: ready
* security identity layer: ready
* security classification layer: incomplete

Interpretation:

* Price and identity are sufficient for canonical warehouse promotion.
* Incomplete `sector` / `industry` coverage in `security_master` is a metadata-quality issue, not a price-promotion blocker.
* Classification-dependent research should remain guarded until coverage improves.

Primary vendors:

* Tiingo — equity prices and security metadata
* FMP — fundamentals

Key integrity protections:

* point-in-time fundamentals
* point-in-time universe membership
* stable ticker identity mapping
* strict validation gates

See:

`docs/research_state.md`

---

# Research Caveat

Sector-neutral and industry-neutral research should remain disabled or explicitly flagged as experimental until `security_master` classification coverage is materially improved.

---

# Current Benchmark Strategy

Factor stack:

`momentum_12_1`  
`reversal_1m`  
`low_vol_20`  
`gross_profitability`

---

# Research Integrity Rules

These rules must not be violated:

* no lookahead bias
* use point-in-time fundamentals
* use point-in-time universe membership
* maintain deterministic backtests
* avoid hidden optimizations

See:

`docs/research_principles.md`

---

# Current Research Phase

Infrastructure and canonical price-data promotion are ready.

Open infrastructure task:

* improve security classification metadata coverage for `sector` and `industry`

Next experiments should avoid treating classification-aware constraints as fully production-grade until that work is complete.

---

# AI Collaboration Rules

When assisting with this project:

1. Prioritize correctness over speed.
2. Preserve research integrity.
3. Avoid modifying strategy logic unless explicitly requested.
4. Prefer small, interpretable changes.
5. Return code changes as Codex prompts when possible.

All experiments should update `docs/experiment_log.md`.
Major research findings should also update `docs/research_state.md`.
