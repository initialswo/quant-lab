# Quant Lab Research Roadmap

## Purpose

This document defines the structured research roadmap for Quant Lab.

The project progresses through stages:

data infrastructure → factor validation → portfolio construction → multi-strategy portfolios

---

# Phase 1 — Data Infrastructure (Completed)

Implemented warehouse system with:

security_master  
symbol_history  
equity_prices_daily  
equity_prices_daily_versions  
equity_fundamentals_pit  
universe_membership_daily  
ingestion_audit  

Data sources:

Tiingo (prices)  
FMP (fundamentals)

Coverage:

~2528 equities  
~9.48M price rows  
~94.7% fundamental coverage  
1962-2026 history

Benchmark reproducibility confirmed.

---

# Phase 2 — Factor Validation (Current)

Current benchmark factors:

momentum_12_1  
reversal_1m  
low_vol_20  
gross_profitability  

Initial findings:

gross_profitability: strongest standalone signal  
reversal_1m: strong but volatile  
momentum_12_1: moderate contributor  
low_vol_20: weakest standalone factor

Next step:

evaluate factor performance across subperiods.

---

# Phase 3 — Portfolio Construction Research

Current configuration:

top_n = 50  
equal weight  
weekly rebalance

Planned experiments:

Portfolio breadth sweep:

top_n = 25  
top_n = 50  
top_n = 100  
top_n = 150  
top_n = 200  

Run across:

sp500 universe  
liquid_us dynamic universe

Metrics:

CAGR  
Sharpe  
max drawdown  
turnover

---

# Rebalance Frequency

Test sensitivity to rebalance schedule:

weekly  
biweekly  
monthly

Goal:

reduce turnover while preserving alpha.

---

# Factor Weight Optimization

Current weights:

25% momentum  
25% reversal  
25% low_vol  
25% profitability

Future experiments:

profitability-heavy allocations  
momentum + profitability core  
reversal as satellite factor

---

# Phase 4 — Universe Research

Current universes:

sp500  
liquid_us dynamic

Future tests:

large-cap subset  
mid-cap subset  
expanded liquid universe

Goal:

identify where factors work best.

---

# Phase 5 — Regime Analysis

Evaluate strategy performance across regimes:

2008 financial crisis  
2010-2019 expansion  
2020 COVID shock  
2022-2024 inflation regime

Goal:

detect structural weaknesses.

---

# Phase 6 — Strategy Sleeve Construction

Possible sleeves:

Quality sleeve  
Profitability  
ROA  
Earnings yield  

Momentum sleeve  
12-month momentum  
relative strength  

Mean reversion sleeve  
short-term reversal  

Volatility sleeve  
low volatility signals

Each sleeve evaluated independently.

---

# Phase 7 — Multi-Strategy Portfolio

Combine sleeves into diversified portfolio.

Research tasks:

correlation analysis  
risk budgeting  
portfolio allocation

Goal:

improve Sharpe  
reduce drawdowns

---

# Phase 8 — Robustness Testing

Before final conclusions:

parameter sweeps  
walk-forward tests  
out-of-sample validation  
stress testing

Goal:

avoid overfitting.

---

# Current Research Focus

Immediate experiments:

portfolio breadth sweep  
factor regime robustness  
factor weight optimization

---

# Research Philosophy

Quant Lab emphasizes:

data integrity  
reproducibility  
systematic experimentation  
clear documentation

Strategy discovery must be evidence-driven.