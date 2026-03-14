# Quant Lab — AI Context

This document provides minimal onboarding context for AI agents assisting with this repository.

For full documentation see:

docs/project_brief.md
docs/research_state.md
docs/research_principles.md
docs/research_roadmap.md
docs/experiment_log.md

---

# Project Purpose

Quant Lab is a Python-based quantitative research platform for systematic trading strategy research.

Primary research areas:

* equity factor strategies
* portfolio construction
* multi-strategy portfolio design

The system supports:

* bias-safe backtesting
* factor experimentation
* strategy combination analysis
* reproducible research workflows

---

# Current Dataset

All experiments run on a validated warehouse dataset located in:

data/warehouse/

Dataset properties:

* ~2,528 US equities
* ~9.48M daily price rows
* price history: 1962–2026
* ~94.7% fundamental coverage

Primary vendors:

* Tiingo — equity prices
* FMP — fundamentals

Key integrity protections:

* point-in-time fundamentals
* point-in-time universe membership
* stable ticker identity mapping
* strict validation gates

See:

docs/research_state.md

---

# Current Benchmark Strategy

Factor stack:

momentum_12_1
reversal_1m
low_vol_20
gross_profitability

Baseline configuration:

top_n = 50
equal weight
weekly rebalance
transaction cost = 10 bps

Universes tested:

sp500 (historical membership)
liquid_us (dynamic liquidity universe)

---

# Research Integrity Rules

These rules must not be violated:

* no lookahead bias
* use point-in-time fundamentals
* use point-in-time universe membership
* maintain deterministic backtests
* avoid hidden optimizations

See:

docs/research_principles.md

---

# Current Research Phase

Infrastructure and data ingestion are complete.

The project is now focused on:

* factor robustness analysis
* portfolio construction research
* multi-strategy portfolio design

Next experiments include:

* portfolio breadth sensitivity (top_n sweep)
* factor weight robustness
* rebalance frequency sensitivity
* regime stability testing

See:

docs/research_roadmap.md

---

# AI Collaboration Rules

When assisting with this project:

1. Prioritize correctness over speed.
2. Preserve research integrity.
3. Avoid modifying strategy logic unless explicitly requested.
4. Prefer small, interpretable changes.
5. Return code changes as Codex prompts when possible.

All experiments should update docs/experiment_log.md.
Major research findings should also update docs/research_state.md.
