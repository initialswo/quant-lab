# Experiment Log

## 2026-03-12 — Composite vs Independent Factor Sleeves (Smoke Validation)

### Objective
Validate a new experiment script that compares the benchmark composite rank against independent factor sleeves and a fixed equal-weight sleeve combination.

### Command
```bash
PYTHONPATH=src python scripts/run_composite_vs_sleeves.py --start 2018-01-01 --end 2020-12-31 --max_tickers 300 --results_root results/composite_vs_sleeves_smoke --skip_log
```

### Setup
- Universe: `liquid_us`
- Universe mode: `dynamic`
- Rebalance: `weekly`
- Top N: `50`
- Weighting: `equal`
- Costs: `10 bps`
- Factors: `momentum_12_1`, `reversal_1m`, `low_vol_20`, `gross_profitability`
- Composite: equal factor weights using existing normalized linear aggregation
- Sleeve combination: fixed 25% return-level blend of the four single-factor sleeves

### Key Results
- Composite CAGR / Sharpe / MaxDD / Turnover: `0.1382 / 0.6872 / -0.3624 / 0.1667`
- Sleeve-combo CAGR / Sharpe / MaxDD / Turnover: `0.1243 / 0.6369 / -0.4126 / 4.3917`
- Composite minus sleeve-combo delta:
  - CAGR: `+0.0139`
  - Sharpe: `+0.0503`
  - MaxDD: `+0.0502`
  - Turnover: `-4.2250`

### Initial Interpretation
- Composite outperformed the sleeve combination on both Sharpe and drawdown in the smoke window.
- Sleeve return correlations were high, so the allocator case did not look strong in this short validation run.
- Holdings overlap remained moderate rather than extreme, which suggests some name diversification despite correlated return streams.

### Artifacts
- Results bundle: `results/composite_vs_sleeves_smoke/20260312_054736`
- Latest copies: `results/composite_vs_sleeves_smoke/latest/`
- Full default run remains pending: `PYTHONPATH=src python scripts/run_composite_vs_sleeves.py`

## 2026-03-10 — Phase 3 Full Bulk Ingest (Tiingo + FMP)

### Objective
Run full warehouse expansion with strict validation and promote only if integrity thresholds pass.

### Command
```bash
PYTHONPATH=src python scripts/run_phase3_full_bulk_ingest.py
```

### Run Root
- `data/staging/phase3/20260310_044005/`

### Ingest Summary
- Requested tickers: 2,500
- Tiingo succeeded tickers: 2,500
- FMP fundamentals succeeded tickers: 2,378
- New selected price rows added: 2,500
- Price rows replaced by precedence: 857,013
- New fundamentals rows added: 122,923
- New symbols seen: 0
- ticker_id reuse/new/changed: `3122 / 0 / 0`

### Validation
- Strict thresholds passed (`failed=false`)
- Coverage after ingest:
  - price tickers: 2,528
  - fundamentals-covered price tickers: 2,395
  - coverage ratio: ~94.74%

### Artifacts
- Ingest summary: `results/ingest/phase3/20260310_044005/phase3_ingest_summary.json`
- Validation latest: `results/data_validation/phase3/latest/validation_summary.json`
- Warehouse backup before promotion: `results/ingest/phase3/backups/20260310_044005/`

## 2026-03-10 — Phase 2B Hardening + Expanded Staged Dry-Run

### Objective
Resolve remaining pre-bulk policy gaps and run a stronger staged ingest focused on metadata conflicts and symbol edge cases.

### Commands
```bash
python -m py_compile scripts/build_security_master.py scripts/build_equity_warehouse.py scripts/staged_ingest_tiingo_daily_dry_run.py scripts/staged_ingest_fmp_fundamentals_dry_run.py scripts/run_phase2b_staged_ingest_dry_run.py
PYTHONPATH=src python scripts/run_phase2b_staged_ingest_dry_run.py --cohort-size 120
```

### Policy Changes
- Metadata consolidation rule implemented in security master:
  - latest `last_date` wins
  - tie-break non-`.US` preference, then earliest `first_date`, then lexical
  - conflict flag retained (`active_flag_conflict`)
- Price precedence/versioning policy implemented:
  - all candidates persisted in `equity_prices_daily_versions.parquet`
  - selected row chosen by `source_rank` + deterministic tie-breaks

### Run Root
- `data/staging/phase2b/20260310_042730/`

### Key Results
- Cohort size: 120
- Tiingo staged rows: 581,318 (120/120 tickers with rows)
- FMP staged rows: 10,434 (119/120 tickers with rows; SPY zero fundamentals rows)
- Warehouse strict validation passed (no threshold breaches)
- `ticker_id` stability preserved (`reused=3122, new=0, changed=0`)
- Fundamentals covered tickers improved to 264 (from baseline 225)

### Additional Observations
- Metadata active-flag conflicts remain present in source inputs (202 canonical symbols), now explicitly surfaced and deterministically consolidated.
- Precedence selection materially shifts selected price source mix once full versioning is enabled, which should be expected and reviewed before full bulk ingest.

## 2026-03-10 — Phase 2A Staged Vendor Ingest Dry-Run

### Objective
Validate Tiingo/FMP ingest pipeline on a controlled cohort before full bulk download.

### Commands
```bash
python -m py_compile scripts/staged_ingest_tiingo_daily_dry_run.py scripts/staged_ingest_fmp_fundamentals_dry_run.py scripts/run_phase2a_staged_ingest_dry_run.py
PYTHONPATH=src python scripts/run_phase2a_staged_ingest_dry_run.py --cohort-size 80
```

### Run Root
- `data/staging/phase2a/20260310_040925/`

### Key Results
- Cohort size: 80
- Tiingo rows staged: 367,305 (80/80 tickers with rows)
- FMP fundamentals rows staged: 6,550 (79/80 tickers with rows; SPY returned none)
- Staged warehouse validation passed strict thresholds:
  - duplicates: 0
  - unmatched symbols: 0
  - ticker ID instability: 0
  - critical-null breaches: 0

### Notable Outcome
- Price-row net additions were 0 because staged Tiingo snapshots were already covered by existing date/ticker keys in the baseline price table.
- Fundamentals coverage improved in staged validation (225 -> 259 price-active tickers with fundamentals).
- Pre-bulk issue discovered: 202 canonical symbols in `metadata.parquet` have conflicting `active_flag` values after normalization (e.g., `AAPL` and `AAPL.US` variants), so lifecycle/active-state consolidation needs a deterministic rule before full-scale ingest.

## 2026-03-10 — Warehouse Buildout Phase 1.5 (Identity Hardening)

### Objective
Harden symbol identity and lifecycle integrity before any large external data refresh.

### Commands
```bash
python -m py_compile scripts/build_security_master.py scripts/build_equity_warehouse.py scripts/validate_equity_warehouse.py
PYTHONPATH=src python scripts/build_equity_warehouse.py
PYTHONPATH=src python scripts/validate_equity_warehouse.py
```

### New Warehouse Artifacts
- `data/warehouse/symbol_history.parquet`
- `data/warehouse/ticker_id_stability_report.parquet`
- `data/warehouse/symbol_collision_report.parquet`

### New Validation Artifacts
- `results/data_validation/latest/ticker_id_stability_report.csv`
- `results/data_validation/latest/symbol_collision_report.csv`
- `results/data_validation/latest/symbol_history_coverage.csv`
- `results/data_validation/latest/validation_thresholds.json`

### Key Findings
- `ticker_id` stability check passed (`stability_issue_count=0`).
- Duplicate-key checks passed on all primary warehouse keys.
- Symbol matching remained complete (0 unmatched canonical symbols).
- Fundamentals coverage remains the main limitation (~8.9% of price-active tickers).

## 2026-03-10 — Warehouse Buildout Phase 1 (Scaffolding + Validation)

### Objective
Create initial local warehouse tables and a repeatable validation pipeline using existing local datasets only (no new downloads).

### Commands
```bash
python -m py_compile scripts/build_security_master.py scripts/build_equity_warehouse.py scripts/validate_equity_warehouse.py
PYTHONPATH=src python scripts/build_security_master.py
PYTHONPATH=src python scripts/build_equity_warehouse.py
PYTHONPATH=src python scripts/validate_equity_warehouse.py
```

### Warehouse Outputs
- `data/warehouse/security_master.parquet`
- `data/warehouse/equity_prices_daily.parquet`
- `data/warehouse/universe_membership_daily.parquet`
- `data/warehouse/equity_fundamentals_pit.parquet`
- `data/warehouse/ingestion_audit.parquet`

### Validation Outputs
- `results/data_validation/latest/coverage_by_year.csv`
- `results/data_validation/latest/fundamentals_coverage.csv`
- `results/data_validation/latest/symbol_match_report.csv`
- `results/data_validation/latest/duplicate_key_report.csv`
- `results/data_validation/latest/null_profile.json`
- `results/data_validation/latest/validation_summary.json`

### Key Findings
- Symbol matching to security master is complete for all Phase 1 tables (0 unmatched canonical symbols).
- Duplicate key checks pass for all warehouse primary keys after deterministic price deduplication.
- Fundamentals coverage is still sparse versus price universe (225/2528 price-active tickers; ~8.9%).

## 2026-03-10 — Data Buildout Planning Checkpoint (Tiingo/FMP)

### Objective
Define a planning-only roadmap to maximize long-term local research value before Tiingo/FMP subscription expiry.

### Scope
- Audited current local datasets, ingestion utilities, and warehouse conventions.
- No data downloads executed.
- No strategy logic or factor logic changed.

### Key Findings
- Core parquet equity tables exist and are actively used by the backtest runner.
- Price history breadth is materially ahead of fundamentals breadth.
- Legacy per-ticker caches are large and useful but not yet governed as a single warehouse.
- Symbol normalization logic exists but should be elevated into explicit security master + symbol history tables.

### Planned Next Build Steps
1. Freeze full daily OHLCV and reference metadata snapshots from Tiingo.
2. Build full PIT fundamentals parquet from FMP (and optional Tiingo fallback) with availability-date governance.
3. Add security master, symbol history, and delisting lifecycle tables.
4. Add corporate actions tables and post-load validation reports (coverage, gaps, duplicates, PIT checks).

## 2026-03-10 — Fixed SP500 vs Dynamic liquid_us (Benchmark Stack)

### Objective
Run a clean apples-to-apples benchmark comparison using:
- fixed historical `sp500` universe path (parquet-backed membership alignment)
- `liquid_us` with `--universe_mode dynamic`

### Commands
```bash
PYTHONPATH=src python run.py backtest \
  --start 2005-01-01 \
  --end 2024-12-31 \
  --universe sp500 \
  --factor momentum_12_1,reversal_1m,low_vol_20,gross_profitability \
  --factor_weights 0.25,0.25,0.25,0.25 \
  --top_n 50 \
  --rebalance weekly \
  --costs_bps 10 \
  --max_tickers 2000

PYTHONPATH=src python run.py backtest \
  --start 2005-01-01 \
  --end 2024-12-31 \
  --universe liquid_us \
  --universe_mode dynamic \
  --factor momentum_12_1,reversal_1m,low_vol_20,gross_profitability \
  --factor_weights 0.25,0.25,0.25,0.25 \
  --top_n 50 \
  --rebalance weekly \
  --costs_bps 10 \
  --max_tickers 2000
```

### Artifacts
- SP500: `results/20260310_005138_887570`
- Dynamic liquid_us: `results/20260310_010034_995510`

### Summary Metrics
| Variant | CAGR | Vol | Sharpe | MaxDD | AnnualTurnover* |
|---|---:|---:|---:|---:|---:|
| SP500 (fixed historical membership) | 0.1106 | 0.1747 | 0.6881 | -0.4173 | 11.0784 |
| liquid_us (dynamic) | 0.1445 | 0.1947 | 0.7909 | -0.4384 | 14.3870 |

\* AnnualTurnover computed from `equity.csv` turnover series as `mean(Turnover) * 252`.

### Eligibility / Universe Diagnostics
- SP500:
  - `HistoricalMembershipSource`: `parquet_auto`
  - `MembershipUniqueTickers`: `814`
  - `MembershipPriceOverlapNormalized`: `256`
  - `EligibilityFilteredOnRebalanceMedian/Min/Max`: `172.5 / 156 / 220`
  - `RebalanceSkippedCount`: `5`
- liquid_us dynamic:
  - `HistoricalMembershipSource`: `none`
  - `EligibilityFilteredOnRebalanceMedian/Min/Max`: `822 / 0 / 1281`
  - `FinalTradableOnRebalanceMedian`: `254`
  - `RebalanceSkippedCount`: `62`

### Notes
- SP500 path uses date-aligned historical membership from parquet, reducing universe broadening risk.
- Dynamic liquid_us achieved higher return and Sharpe but with higher volatility, deeper drawdown, higher turnover, and many skipped rebalances.

## 2026-03-10 — Post-Ingest Benchmark Reproducibility Check

### Objective
Verify whether Phase 3 bulk ingest changed previously reported benchmark conclusions.

### Commands
```bash
PYTHONPATH=src python run.py backtest \
  --start 2005-01-01 \
  --end 2024-12-31 \
  --universe sp500 \
  --factor momentum_12_1,reversal_1m,low_vol_20,gross_profitability \
  --factor_weights 0.25,0.25,0.25,0.25 \
  --top_n 50 \
  --rebalance weekly \
  --costs_bps 10 \
  --max_tickers 2000

PYTHONPATH=src python run.py backtest \
  --start 2005-01-01 \
  --end 2024-12-31 \
  --universe liquid_us \
  --universe_mode dynamic \
  --factor momentum_12_1,reversal_1m,low_vol_20,gross_profitability \
  --factor_weights 0.25,0.25,0.25,0.25 \
  --top_n 50 \
  --rebalance weekly \
  --costs_bps 10 \
  --max_tickers 2000
```

### Artifacts
- SP500: `results/20260310_051426_373261`
- Dynamic liquid_us: `results/20260310_051948_618974`

### Summary Metrics
| Variant | CAGR | Vol | Sharpe | MaxDD | AnnualTurnover* |
|---|---:|---:|---:|---:|---:|
| SP500 (post-ingest) | 0.1106 | 0.1747 | 0.6881 | -0.4173 | 11.0784 |
| liquid_us dynamic (post-ingest) | 0.1445 | 0.1947 | 0.7909 | -0.4384 | 14.3870 |

\* AnnualTurnover computed from `equity.csv` turnover series as `mean(Turnover) * 252`.

### Delta vs Pre-Ingest Reference
- SP500:
  - `ΔCAGR`: `-0.0000165`
  - `ΔSharpe`: `-0.0000095`
  - `ΔMaxDD`: `-0.0000093`
  - `ΔAnnualTurnover`: `-0.0000134`
  - `ΔFinalTradableOnRebalanceMedian`: `0.0`
- liquid_us dynamic:
  - `ΔCAGR`: `-0.0000081`
  - `ΔSharpe`: `+0.0000128`
  - `ΔMaxDD`: `+0.0000021`
  - `ΔAnnualTurnover`: `-0.0000181`
  - `ΔFinalTradableOnRebalanceMedian`: `0.0`

### Interpretation
- No material change vs pre-ingest benchmark conclusions; differences are rounding/noise-level only.
- Post-ingest warehouse promotion appears reproducible for this benchmark configuration.

## 2026-03-11 — Portfolio Breadth Sweep Runtime Audit

### Objective
Audit why `scripts/run_portfolio_breadth_sweep.py` is slow across the 8-run `sp500` / `liquid_us` by `top_n` sweep.

### Evidence
- Sweep bundle: `results/portfolio_breadth_sweep/20260311_213108`
- Per-run timings taken from each run's `summary.json`

### Findings
- Internal timed runtime across the 8 backtests summed to approximately `44.7` minutes.
- The dominant bucket was `TimingFactorComputeSeconds`: approximately `2499` seconds total (`41.7` minutes), about `93%` of total timed runtime.
- `TimingDataLoadSeconds` summed to approximately `135` seconds (`2.25` minutes).
- `TimingPortfolioBacktestSeconds` summed to approximately `17.8` seconds.
- `TimingReportWriteSeconds` summed to approximately `25.2` seconds.

### Cache Behavior
- OHLCV parquet loading was reused in-process after the first run of each universe:
  - first `sp500` run: `CacheFetchHit=0`
  - later `sp500` runs: `CacheFetchHit=1`
  - first `liquid_us` run: `CacheFetchHit=0`
  - later `liquid_us` runs: `CacheFetchHit=1`
- Fundamentals augmentation was also reused in-process after the first run of each universe:
  - first run per universe: `CacheFundamentalsHit=0`
  - later runs in same universe: `CacheFundamentalsHit=1`
- Raw and normalized factor scores were recomputed every run:
  - all 8 runs: `CacheFactorRawHit=0`
  - all 8 runs: `CacheFactorNormHit=0`

### Root Cause
- The main inefficiency is repeated factor pipeline work across `top_n` values even though factor scores do not depend on `top_n`.
- In `runner.py`, the factor raw-score cache key includes `top_n`, which prevents reuse across breadth variants.
- Because normalized-score caching keys off the raw-score cache key, normalized factor panels are also recomputed for every `top_n`.
- Dynamic `liquid_us` eligibility is rebuilt each run, but this is materially smaller than the factor pipeline cost.

### Optimization Implemented
- Removed `top_n` from the raw factor cache key in `runner.py`.
- Normalized factor cache reuse now also works across breadth variants because it depends on the raw factor cache key.

### Validation
- Reference sweep: `results/portfolio_breadth_sweep/20260311_213108`
- Optimized sweep: `results/portfolio_breadth_sweep/20260311_223158`
- Result equivalence:
  - `CAGR`, `Vol`, `Sharpe`, `MaxDD`, `EligMedian`, `TradableMedian`: exact match in saved CSVs
  - `RebalanceSkipped`: exact match
  - `AnnualTurnover`: remained `NaN` in both outputs

### Runtime Improvement
- Old internal timed runtime across 8 runs: approximately `2680.19s` (`44.7m`)
- New internal timed runtime across 8 runs: approximately `822.16s` (`13.7m`)
- Improvement: approximately `1858.03s` faster (`69.3%` reduction)

### Cache Effect After Fix
- First run per universe still computes factors normally.
- Later `top_n` runs in the same universe now hit both caches:
  - `CacheFactorRawHit=1`
  - `CacheFactorNormHit=1`
- Factor compute time on cached breadth variants dropped from roughly `188–444s` per run to approximately `0.0003–0.0017s` per run.

### Engineering Conclusion
- Reusing factor panels across breadth variants is safe and materially reduces sweep runtime without changing research results.
- The next safe optimization target is universe eligibility caching, especially for repeated `liquid_us` breadth variants.

## 2026-03-11 — Breadth Sweep Turnover Fix

### Objective
Populate `AnnualTurnover` in breadth sweep outputs without changing backtest logic.

### Root Cause
- The breadth sweep collector read `AnnualTurnover` directly from `summary.json`.
- This backtest path does not currently populate `summary["AnnualTurnover"]`, even though per-run `equity.csv` includes a daily `Turnover` series.

### Fix
- Added a sweep-level helper that:
  - uses `summary["AnnualTurnover"]` when present
  - otherwise derives annual turnover from `equity.csv` as `mean(Turnover) * 252`

### Validation
- Reference breadth bundle without turnover: `results/portfolio_breadth_sweep/20260311_223158`
- Corrected breadth bundle: `results/portfolio_breadth_sweep/20260311_225556`
- `CAGR`, `Vol`, `Sharpe`, `MaxDD`, `EligMedian`, `TradableMedian`, `RebalanceSkipped`: unchanged
- `AnnualTurnover` now populated:
  - `sp500`: `15.93`, `11.08`, `6.47`, `0.26` for `top_n=20/50/100/200`
  - `liquid_us`: `18.75`, `14.39`, `9.55`, `3.38` for `top_n=20/50/100/200`

## 2026-03-11 — Benchmark Factor-Weight Sensitivity Sweep

### Objective
Test alternative factor weights at the now-validated `top_n=50` breadth setting.

### Configuration
- Factors:
  - `momentum_12_1`
  - `reversal_1m`
  - `low_vol_20`
  - `gross_profitability`
- Universes:
  - `sp500`
  - `liquid_us` with `universe_mode=dynamic`
- Fixed settings:
  - `2005-01-01` to `2024-12-31`
  - `top_n=50`
  - weekly rebalance
  - `costs_bps=10`
  - `max_tickers=2000`

### Artifacts
- Results bundle: `results/factor_weight_sweep/20260311_231044`

### Best by Sharpe
- `sp500`:
  - best: `0.30,0.20,0.10,0.40`
  - Sharpe `0.7229`
  - CAGR `0.1231`
  - MaxDD `-0.4426`
  - AnnualTurnover `7.17`
- `liquid_us`:
  - best: `0.20,0.30,0.10,0.40`
  - Sharpe `0.8016`
  - CAGR `0.1547`
  - MaxDD `-0.4693`
  - AnnualTurnover `13.25`

### Comparison vs Equal Weight
- Equal weight baseline (`0.25,0.25,0.25,0.25`) remained strong but was not best in either universe.
- `sp500` improved from Sharpe `0.6881` to `0.7229`.
- `liquid_us` improved from Sharpe `0.7909` to `0.8016`.
- In both universes, heavier `gross_profitability` weight (`0.40`) remained competitive or superior across most tested variants.

### Interpretation
- The benchmark appears more sensitive to factor weights than to further increasing breadth beyond `top_n=50`.
- Gross profitability likely deserves a larger weight than equal-weight benchmarking assigns today.
- The best weight mix is not identical across universes, so any benchmark promotion should be framed as universe-specific unless a common robust set is preferred.

## 2026-03-12 — Subperiod Robustness Test for Benchmark Weight Candidates

### Objective
Test whether the improved factor-weight candidates remain stable across four market subperiods.

### Configuration
- Factors:
  - `momentum_12_1`
  - `reversal_1m`
  - `low_vol_20`
  - `gross_profitability`
- Weight sets tested:
  - baseline: `0.25,0.25,0.25,0.25`
  - SP500 candidate: `0.30,0.20,0.10,0.40`
  - liquid_us candidate: `0.20,0.30,0.10,0.40`
- Universes:
  - `sp500`
  - `liquid_us` with `universe_mode=dynamic`
- Fixed settings:
  - `top_n=50`
  - weekly rebalance
  - `costs_bps=10`
  - `max_tickers=2000`
- Subperiods:
  - `2005-01-01` to `2010-12-31`
  - `2010-01-01` to `2015-12-31`
  - `2015-01-01` to `2020-12-31`
  - `2020-01-01` to `2024-12-31`

### Artifacts
- Results bundle: `results/subperiod_robustness/20260312_000555`

### Runtime
- Full 24-run sweep wall time: approximately `758.0s` (`12.6m`)

### Best Sharpe by Period
- `sp500`
  - `2005-01-01→2010-12-31`: `0.30,0.20,0.10,0.40` (Sharpe `0.5532`)
  - `2010-01-01→2015-12-31`: baseline `0.25,0.25,0.25,0.25` (Sharpe `1.0764`)
  - `2015-01-01→2020-12-31`: `0.20,0.30,0.10,0.40` (Sharpe `0.8192`)
  - `2020-01-01→2024-12-31`: `0.20,0.30,0.10,0.40` (Sharpe `0.5014`)
- `liquid_us`
  - `2005-01-01→2010-12-31`: baseline `0.25,0.25,0.25,0.25` (Sharpe `0.3675`)
  - `2010-01-01→2015-12-31`: baseline `0.25,0.25,0.25,0.25` (Sharpe `0.9892`)
  - `2015-01-01→2020-12-31`: `0.20,0.30,0.10,0.40` (Sharpe `0.9473`)
  - `2020-01-01→2024-12-31`: baseline `0.25,0.25,0.25,0.25` (Sharpe `0.6059`)

### Stability Readout
- `sp500`
  - no single weight set won all four periods
  - average Sharpe by set:
    - `0.20,0.30,0.10,0.40`: `0.7257`
    - `0.30,0.20,0.10,0.40`: `0.7209`
    - baseline: `0.7012`
- `liquid_us`
  - baseline won `3/4` periods
  - average Sharpe by set:
    - baseline: `0.7066`
    - `0.20,0.30,0.10,0.40`: `0.6931`
    - `0.30,0.20,0.10,0.40`: `0.6895`

### Interpretation
- The weight improvements from the full-period sweep are not uniformly stable across regimes.
- `sp500` shows some evidence that non-baseline weights may help, but the winner changes by subperiod and the average Sharpe edge versus baseline is modest.
- `liquid_us` does not support replacing equal weights: the baseline won most subperiods and had the best average Sharpe.

### Recommendation
- Do not update benchmark weights yet.
- Keep equal weights as the default benchmark pending either:
  - stronger cross-period evidence in `sp500`, or
  - a consistent common weight set that holds up in both universes.

## 2026-03-12 — Rebalance Frequency Sweep for Current Benchmark

### Objective
Measure how performance and turnover change as rebalance frequency moves from weekly to biweekly to monthly.

### Configuration
- Factors:
  - `momentum_12_1`
  - `reversal_1m`
  - `low_vol_20`
  - `gross_profitability`
- Factor weights:
  - `0.25,0.25,0.25,0.25`
- Fixed settings:
  - `top_n=50`
  - `costs_bps=10`
  - `max_tickers=2000`
  - `2005-01-01` to `2024-12-31`
- Universes:
  - `sp500`
  - `liquid_us` with `universe_mode=dynamic`
- Frequencies tested:
  - `weekly`
  - `biweekly`
  - `monthly`

### Artifacts
- Results bundle: `results/rebalance_frequency_sweep/20260312_002537`

### Runtime
- Full 6-run sweep wall time: approximately `792.9s` (`13.2m`)

### Results
- `sp500`
  - weekly: Sharpe `0.6881`, CAGR `0.1106`, MaxDD `-0.4173`, AnnualTurnover `11.08`
  - biweekly: Sharpe `0.7025`, CAGR `0.1135`, MaxDD `-0.4214`, AnnualTurnover `7.80`
  - monthly: Sharpe `0.6784`, CAGR `0.1100`, MaxDD `-0.4531`, AnnualTurnover `5.14`
- `liquid_us`
  - weekly: Sharpe `0.7909`, CAGR `0.1445`, MaxDD `-0.4384`, AnnualTurnover `14.39`
  - biweekly: Sharpe `0.7944`, CAGR `0.1442`, MaxDD `-0.4624`, AnnualTurnover `9.86`
  - monthly: Sharpe `0.7553`, CAGR `0.1355`, MaxDD `-0.4862`, AnnualTurnover `6.37`

### Sharpe Pivot
- weekly:
  - `sp500`: `0.6881`
  - `liquid_us`: `0.7909`
- biweekly:
  - `sp500`: `0.7025`
  - `liquid_us`: `0.7944`
- monthly:
  - `sp500`: `0.6784`
  - `liquid_us`: `0.7553`

### Interpretation
- Monthly rebalancing is clearly weaker than weekly/biweekly in both universes.
- Biweekly modestly improves Sharpe versus weekly in both universes while materially reducing turnover.
- The tradeoff is slightly worse max drawdown versus weekly in both universes.

### Recommendation
- Weekly should remain the benchmark default for now.
- Biweekly is a credible research variant because it preserves return/Sharpe while lowering turnover, but the drawdown tradeoff is not clearly better and the improvement is small.

## 2026-03-12 — Benchmark Factor Return Decomposition

### Objective
Estimate how each benchmark factor behaves as its own standalone sleeve and measure how correlated those sleeves are through time in:
- `sp500`
- `liquid_us` with `universe_mode=dynamic`

using fixed benchmark settings:
- `2005-01-01` to `2024-12-31`
- `top_n=50`
- equal weight
- weekly rebalance
- `costs_bps=10`
- `max_tickers=2000`

### Command
```bash
PYTHONPATH=src python scripts/run_factor_decomposition.py
```

### Artifacts
- Results bundle: `results/factor_decomposition/20260312_012153`
- Consolidated results:
  - `results/factor_decomposition/20260312_012153/factor_decomposition_results.csv`
  - `results/factor_decomposition/factor_decomposition_results_latest.csv`
- Correlations:
  - `results/factor_decomposition/20260312_012153/factor_correlation_sp500.csv`
  - `results/factor_decomposition/20260312_012153/factor_correlation_liquid_us.csv`
  - `results/factor_decomposition/factor_correlation_sp500_latest.csv`
  - `results/factor_decomposition/factor_correlation_liquid_us_latest.csv`
- Rolling 252-day average correlations:
  - `results/factor_decomposition/20260312_012153/factor_correlation_rolling252_avg_sp500.csv`
  - `results/factor_decomposition/20260312_012153/factor_correlation_rolling252_avg_liquid_us.csv`
- Per-sleeve copied artifacts:
  - `results/factor_decomposition/20260312_012153/sleeves/sp500/<factor>/`
  - `results/factor_decomposition/20260312_012153/sleeves/liquid_us/<factor>/`

### Runtime
- Full 8-run decomposition wall time: approximately `884.9s` (`14.7m`)

### Standalone Sleeve Results
- `sp500`
  - `gross_profitability`: Sharpe `0.7358`, CAGR `0.1283`, MaxDD `-0.4579`, AnnualTurnover `1.14`
  - `momentum_12_1`: Sharpe `0.5925`, CAGR `0.0979`, MaxDD `-0.4388`, AnnualTurnover `4.81`
  - `low_vol_20`: Sharpe `0.5866`, CAGR `0.0749`, MaxDD `-0.3890`, AnnualTurnover `9.44`
  - `reversal_1m`: Sharpe `0.5622`, CAGR `0.1107`, MaxDD `-0.5824`, AnnualTurnover `17.14`
- `liquid_us`
  - `reversal_1m`: Sharpe `0.8877`, CAGR `0.2504`, MaxDD `-0.5740`, AnnualTurnover `23.22`
  - `gross_profitability`: Sharpe `0.8276`, CAGR `0.1634`, MaxDD `-0.4947`, AnnualTurnover `1.26`
  - `momentum_12_1`: Sharpe `0.6950`, CAGR `0.1518`, MaxDD `-0.5311`, AnnualTurnover `6.78`
  - `low_vol_20`: Sharpe `0.4507`, CAGR `0.0439`, MaxDD `-0.4375`, AnnualTurnover `11.62`

### Correlation Summary
- `sp500`
  - average pairwise correlation: `0.9019`
  - most correlated pair: `reversal_1m` vs `gross_profitability` (`0.9287`)
  - least correlated pair: `momentum_12_1` vs `reversal_1m` (`0.8661`)
- `liquid_us`
  - average pairwise correlation: `0.7574`
  - most correlated pair: `low_vol_20` vs `gross_profitability` (`0.8194`)
  - least correlated pair: `low_vol_20` vs `reversal_1m` (`0.6814`)

### Contribution Summary
- `sp500`
  - best standalone Sharpe: `gross_profitability` (`0.7358`)
  - best standalone CAGR: `gross_profitability` (`0.1283`)
  - lowest standalone drawdown: `low_vol_20` (`-0.3890`)
- `liquid_us`
  - best standalone Sharpe: `reversal_1m` (`0.8877`)
  - best standalone CAGR: `reversal_1m` (`0.2504`)
  - lowest standalone drawdown: `low_vol_20` (`-0.4375`)

### Interpretation
- `gross_profitability` is the strongest standalone sleeve in `sp500` on both Sharpe and CAGR, while `reversal_1m` is strongest in dynamic `liquid_us`.
- `low_vol_20` is the weakest standalone sleeve in both universes on Sharpe and CAGR, though it does provide the mildest drawdown.
- The sleeves are highly correlated in `sp500` and still fairly correlated in `liquid_us`; there is less diversification than the factor list might suggest.
- The benchmark does not look cleanly diversified across independent sleeves. It appears meaningfully driven by a smaller subset of stronger factors, especially `gross_profitability` and `reversal_1m`.

### Recommendation
- Future research should move toward explicit factor-sleeve analysis, but not because the current benchmark shows strong sleeve diversification.
- The next step should be sleeve-aware allocation research focused on whether a smaller set of high-conviction sleeves can preserve returns while improving robustness, turnover control, or drawdown.
- `low_vol_20` should be treated as a defensive diversifier candidate rather than assumed alpha engine, and any continued inclusion should be justified by portfolio-level effects rather than standalone strength.

## 2026-03-12 — Reduced-Factor Benchmark Comparison

### Objective
Test whether a smaller factor set can match or improve the current benchmark across:
- `sp500`
- `liquid_us` with `universe_mode=dynamic`

using fixed settings:
- `2005-01-01` to `2024-12-31`
- `top_n=50`
- weekly rebalance
- equal weight within each factor set
- `costs_bps=10`
- `max_tickers=2000`

### Command
```bash
PYTHONPATH=src python scripts/run_reduced_factor_comparison.py
```

### Factor Sets Tested
1. `gross_profitability,reversal_1m`
2. `gross_profitability,momentum_12_1`
3. `gross_profitability,reversal_1m,momentum_12_1`
4. `momentum_12_1,reversal_1m,low_vol_20,gross_profitability`

### Artifacts
- Results bundle: `results/reduced_factor_comparison/20260312_020959`
- Consolidated results:
  - `results/reduced_factor_comparison/20260312_020959/reduced_factor_results.csv`
  - `results/reduced_factor_comparison/reduced_factor_results_latest.csv`
- Sharpe pivot:
  - `results/reduced_factor_comparison/20260312_020959/reduced_factor_summary.csv`
  - `results/reduced_factor_comparison/reduced_factor_summary_latest.csv`

### Runtime
- Full 8-run comparison wall time: approximately `2066.3s` (`34.4m`)

### Results
- `sp500`
  - `gross_profitability,reversal_1m`: Sharpe `0.6796`, CAGR `0.1245`, MaxDD `-0.4984`
  - `gross_profitability,momentum_12_1`: Sharpe `0.6737`, CAGR `0.1135`, MaxDD `-0.4611`
  - `gross_profitability,reversal_1m,momentum_12_1`: Sharpe `0.6502`, CAGR `0.1152`, MaxDD `-0.4972`
  - benchmark `4-factor`: Sharpe `0.6881`, CAGR `0.1106`, MaxDD `-0.4173`
- `liquid_us`
  - `gross_profitability,reversal_1m`: Sharpe `0.7624`, CAGR `0.1538`, MaxDD `-0.5150`
  - `gross_profitability,momentum_12_1`: Sharpe `0.7043`, CAGR `0.1279`, MaxDD `-0.4385`
  - `gross_profitability,reversal_1m,momentum_12_1`: Sharpe `0.7713`, CAGR `0.1478`, MaxDD `-0.4705`
  - benchmark `4-factor`: Sharpe `0.7909`, CAGR `0.1445`, MaxDD `-0.4384`

### Sharpe Pivot
- `gross_profitability,reversal_1m`
  - `sp500`: `0.6796`
  - `liquid_us`: `0.7624`
- `gross_profitability,momentum_12_1`
  - `sp500`: `0.6737`
  - `liquid_us`: `0.7043`
- `gross_profitability,reversal_1m,momentum_12_1`
  - `sp500`: `0.6502`
  - `liquid_us`: `0.7713`
- benchmark `4-factor`
  - `sp500`: `0.6881`
  - `liquid_us`: `0.7909`

### Interpretation
- The benchmark cannot be cleanly simplified on this evidence. None of the reduced sets beat the current 4-factor model on Sharpe in either universe.
- `low_vol_20` still looks worth keeping in the benchmark despite weak standalone performance. At the portfolio level, the 4-factor benchmark remains the best Sharpe configuration in both universes and materially improves drawdown versus the strongest reduced alternatives in `sp500`.
- If forced to simplify, a 2-factor model is preferable to the tested 3-factor model. The best 2-factor sets stay closer to benchmark Sharpe than the tested 3-factor set, especially in `sp500`.
- The closest simplification candidates are:
  - `sp500`: `gross_profitability,reversal_1m`
  - `liquid_us`: `gross_profitability,reversal_1m,momentum_12_1`

### Recommendation
- Keep the 4-factor benchmark unchanged for now.
- Treat reduced-factor sets as research variants, not replacements.
- If future simplification work continues, prioritize 2-factor stress tests and low-vol inclusion/exclusion tests rather than adopting the current 3-factor variant.

## 2026-03-12 — Reversal 1M Stratification in liquid_us

### Objective
Determine where the `reversal_1m` signal is strongest inside the dynamic `liquid_us` universe by stratifying the candidate pool into terciles on:
- size
- realized volatility
- liquidity

using fixed settings:
- `2005-01-01` to `2024-12-31`
- `top_n=50`
- weekly rebalance
- `costs_bps=10`
- `max_tickers=2000`

### Command
```bash
PYTHONPATH=src python scripts/run_reversal_stratification.py
```

### Method
- Universe construction remained `liquid_us` dynamic.
- Factor logic remained `reversal_1m`.
- Candidate pools were restricted only by bucket membership on each rebalance date.
- Size used `close * shares_outstanding` when available, with fallback proxy `close * ADV20`.
- Volatility used 20-day realized volatility.
- Liquidity used 20-day average dollar volume.

### Artifacts
- Results bundle: `results/reversal_stratification/20260312_023048`
- Bucket tables:
  - `results/reversal_stratification/20260312_023048/reversal_size_buckets.csv`
  - `results/reversal_stratification/20260312_023048/reversal_volatility_buckets.csv`
  - `results/reversal_stratification/20260312_023048/reversal_liquidity_buckets.csv`
- Latest copies:
  - `results/reversal_stratification/reversal_size_buckets_latest.csv`
  - `results/reversal_stratification/reversal_volatility_buckets_latest.csv`
  - `results/reversal_stratification/reversal_liquidity_buckets_latest.csv`

### Runtime
- Full stratification wall time: approximately `289.2s` (`4.8m`)

### Results
- Size buckets
  - `small`: Sharpe `1.4921`, CAGR `0.4266`, MaxDD `-0.5007`, AvgEligible `274.8`
  - `mid`: Sharpe `0.4874`, CAGR `0.1087`, MaxDD `-0.6625`, AvgEligible `275.1`
  - `large`: Sharpe `0.5077`, CAGR `0.1034`, MaxDD `-0.6008`, AvgEligible `275.5`
- Volatility buckets
  - `low_vol`: Sharpe `0.5755`, CAGR `0.0830`, MaxDD `-0.5498`, AvgEligible `275.0`
  - `mid_vol`: Sharpe `0.6890`, CAGR `0.1395`, MaxDD `-0.6365`, AvgEligible `275.3`
  - `high_vol`: Sharpe `1.0379`, CAGR `0.3283`, MaxDD `-0.5397`, AvgEligible `275.6`
- Liquidity buckets
  - `low_liq`: Sharpe `1.5422`, CAGR `0.4104`, MaxDD `-0.4920`, AvgEligible `275.0`
  - `mid_liq`: Sharpe `0.5567`, CAGR `0.1325`, MaxDD `-0.6257`, AvgEligible `275.3`
  - `high_liq`: Sharpe `0.4808`, CAGR `0.1007`, MaxDD `-0.6315`, AvgEligible `275.6`

### Interpretation
- Reversal is strongest in `small` stocks, `high_vol` stocks, and especially `low_liq` stocks.
- The signal appears meaningfully dependent on small caps. `small` Sharpe (`1.49`) is far above `mid` and `large` (both near `0.50`).
- The signal also depends strongly on volatility. `high_vol` is materially stronger than `mid_vol` and `low_vol`.
- The strongest dependence is on liquidity. `low_liq` is dramatically better than `mid_liq` and `high_liq`, suggesting much of the reversal edge lives away from the most liquid names.

### Recommendation
- Treat `reversal_1m` as a niche signal concentrated in smaller, less liquid, higher-volatility names rather than a broad large-cap/core-universe effect.
- Future reversal research should focus on implementation feasibility inside those buckets, especially whether the gross edge survives stricter liquidity and trading-friction controls.

## 2026-03-12 — VectorVest-Style V1 Prototype

### Objective
Build a first-pass VectorVest-inspired screened stock strategy in `quant_lab` using transparent, non-proprietary factors and simple dynamic entry/exit logic.

### Command
```bash
PYTHONPATH=src python scripts/run_vectorvest_style_v1.py
```

### Universe And Timing
- Universe: `liquid_us` with dynamic eligibility
- Window: `2005-01-01` to `2024-12-31`
- Annual reconstitution:
  - first weekly evaluation date of each calendar year
  - stock selection uses prior trading day's screen values
- MA timing rule:
  - evaluated weekly
  - hold a selected name only if prior close > prior 50-day SMA
  - if not, that slot goes to cash
  - no replacement names until the next annual reconstitution

### Implemented Screen
- Implemented fields:
  - `gross_profitability`
  - `momentum_12_1`
  - `revenue_growth`
- Screen construction:
  - each implemented field converted to a cross-sectional normalized score
  - equal weight across implemented fields
  - top 50 names selected at each annual reconstitution
- `revenue_growth` definition in v1:
  - PIT-aligned revenue divided by PIT-aligned revenue from 252 trading days earlier, minus 1
  - this is a simple daily-aligned annual growth proxy, not a quarter-over-quarter or filing-aware bespoke factor

### Artifacts
- Results bundle: `results/vectorvest_style_v1/20260312_025001`
- Core outputs:
  - `results/vectorvest_style_v1/20260312_025001/summary.json`
  - `results/vectorvest_style_v1/20260312_025001/equity.csv`
  - `results/vectorvest_style_v1/20260312_025001/run_config.json`
  - `results/vectorvest_style_v1/20260312_025001/strategy_notes.json`

### Runtime
- Full runtime: approximately `106.1s` (`1.8m`)

### Results
- CAGR: `0.0985`
- Vol: `0.2064`
- Sharpe: `0.5591`
- MaxDD: `-0.5818`
- AnnualTurnover: `11.48`
- AvgActiveHoldings: `26.7`
- AvgCashWeight: `0.1103`

### Interpretation
- The v1 prototype is implementable and transparent, but the first-pass performance is only moderate.
- The strategy spends a meaningful portion of time partially in cash because of the 50-day SMA rule, with average active holdings materially below 50.
- Relative to the current benchmark research track, this v1 does not yet look compelling enough to replace the benchmark. It is better treated as a new research branch than as a production candidate.

### Limitations
- The screen is intentionally simple and may be too coarse relative to the original inspiration.
- `revenue_growth` uses a straightforward PIT-aligned 252-trading-day proxy rather than a more refined fundamental-growth construction.
- Annual reconstitution is very sparse and may leave stale selections in place for too long.
- The MA rule is slot-level and binary; it does not attempt re-entry optimization, ranking refresh, or replacement-name logic within the year.

### Recommended V2 Improvements
- Compare annual versus quarterly reconstitution while keeping the same MA rule.
- Test a cleaner growth component, for example more explicit quarterly or trailing-twelve-month revenue growth if PIT handling is robust enough.
- Add simple rank-refresh logic for vacant slots at scheduled reconstitution subdates, if the design goal moves away from strict annual selection.
- Test alternative exit rules such as 100-day SMA or dual-MA confirmation before adding any more complex features.

## 2026-03-12 — VectorVest-Style V2 Prototype

### Objective
Improve the VectorVest-inspired prototype by:
- changing reconstitution from annual to quarterly
- reducing cash drag via weekly replacement from the frozen quarterly rank list

while keeping the strategy lag-safe, transparent, and non-proprietary.

### Command
```bash
PYTHONPATH=src python scripts/run_vectorvest_style_v2.py
```

### Strategy Definition
- Universe: `liquid_us` dynamic
- Screen fields implemented:
  - `gross_profitability`
  - `momentum_12_1`
  - `revenue_growth`
- `revenue_growth` definition:
  - PIT-aligned revenue divided by PIT-aligned revenue from 252 trading days earlier, minus 1
- Screen construction:
  - existing cross-sectional preprocessing pipeline
  - composite score = simple average of normalized component scores
  - equal weight across implemented components
- Reconstitution schedule:
  - first weekly evaluation date of each calendar quarter
  - quarterly selection uses prior trading day's composite scores
- Timing rule:
  - weekly evaluation
  - hold only if prior close > prior 50-day SMA
  - lagged values only, no lookahead
- Replacement rule:
  - after removing held names that fail timing or eligibility, refill vacancies from the frozen quarterly ranking list
  - replacement must be:
    - not already held
    - currently eligible
    - passing the lagged 50-day SMA rule
  - if no valid replacement exists, slot remains cash

### Artifacts
- Results bundle: `results/vectorvest_style_v2/20260312_025929`
- Core outputs:
  - `results/vectorvest_style_v2/20260312_025929/summary.json`
  - `results/vectorvest_style_v2/20260312_025929/equity.csv`
  - `results/vectorvest_style_v2/20260312_025929/run_config.json`
  - `results/vectorvest_style_v2/20260312_025929/strategy_notes.json`

### Runtime
- Full runtime: approximately `106.0s` (`1.8m`)

### Results
- CAGR: `0.0874`
- Vol: `0.1984`
- Sharpe: `0.5213`
- MaxDD: `-0.5696`
- AnnualTurnover: `8.82`
- AvgActiveHoldings: `45.60`
- AvgCashWeight: `0.0624`

### Comparison Vs V1
- CAGR delta: `-0.0111`
- Sharpe delta: `-0.0378`
- MaxDD delta: `+0.0121` (less severe drawdown)
- AnnualTurnover delta: `-2.6608`
- AvgActiveHoldings delta: `+18.90`
- AvgCashWeight delta: `-0.0479`

### Interpretation
- V2 achieved the intended mechanical improvements:
  - much higher portfolio occupancy
  - lower cash drag
  - lower turnover
  - slightly better max drawdown
- But those implementation improvements did not translate into better CAGR or Sharpe.
- The quarterly refresh plus frozen-rank replacement logic seems to have sacrificed some alpha capture relative to the simpler V1 design.

### Recommendation
- V2 is not strong enough to replace V1 or justify a major new strategy track in its current form.
- It is still strong enough to justify limited further iteration, because the mechanics behaved correctly and the occupancy/cash improvements were substantial.
- The next iteration should focus on preserving alpha rather than further increasing mechanical complexity.

## 2026-03-10 — Benchmark Factor Attribution / Ablation Sweep

### Objective
Attribute benchmark performance drivers across:
- fixed historical `sp500`
- dynamic `liquid_us`

using the same baseline settings (`2005-01-01` to `2024-12-31`, `top_n=50`, weekly rebalance, `costs_bps=10`, `max_tickers=2000`).

### Factor Sets Tested
1. `momentum_12_1,reversal_1m,low_vol_20,gross_profitability`
2. `momentum_12_1,reversal_1m,low_vol_20`
3. `momentum_12_1,reversal_1m,gross_profitability`
4. `momentum_12_1,low_vol_20,gross_profitability`
5. `reversal_1m,low_vol_20,gross_profitability`
6. `momentum_12_1`
7. `gross_profitability`
8. `low_vol_20`
9. `reversal_1m`

### Artifacts
- Consolidated results:
  - `results/factor_ablation_benchmark/20260310_073650/factor_ablation_results.csv`
  - `results/factor_ablation_benchmark/factor_ablation_results_latest.csv`
- Gross profitability marginal table:
  - `results/factor_ablation_benchmark/20260310_073650/gross_profitability_marginal_effect.csv`
  - `results/factor_ablation_benchmark/gross_profitability_marginal_effect_latest.csv`

### Notes
- A runner artifact-serialization bug appeared for single-factor `gross_profitability` runs (`DataFrame` in `FactorParams` JSON serialization).
- Minimal fix applied in `src/quant_lab/engine/runner.py`:
  - add `default=str` to two legacy `json.dumps(...)` calls used for `FactorParams`.
- Strategy logic and factor calculations were unchanged.

## 2026-03-10 — liquid_us Dynamic Skip-Rebalance Root Cause Audit

### Objective
Explain why `liquid_us` dynamic runs repeatedly report:
- `RebalanceSkippedCount = 62`
- skip reason: zero eligible names

for benchmark settings (`2005-01-01` to `2024-12-31`, `top_n=50`, weekly).

### Findings
- Problematic rebalance dates count: `62`
- Zero-eligible dates count: `62`
- Collapse stage by requested pipeline audit:
  - `history_filtered`: `62`
  - all other stages: `0`

### Root Cause
- The dynamic universe uses `effective_min_history_days = 300`.
- Early sample period (from 2005-01-03 through 2006-03-06, weekly rebalance dates) has insufficient in-sample history for all names, so eligibility collapses to zero before factor/ranking stages.
- This is a deterministic warmup/eligibility effect, not a random data outage.

### Artifacts
- `results/universe_diagnostics/liquid_us_skip_analysis.csv`
- `results/universe_diagnostics/liquid_us_skip_analysis_with_stage.csv`
- `results/universe_diagnostics/liquid_us_skip_analysis_summary.json`

## 2026-03-12 — Quality Momentum Comparison

### Objective
Test whether the current 4-factor benchmark adds enough value beyond a simpler quality-momentum portfolio to justify its extra complexity.

Universes:
- `sp500`
- `liquid_us` with `universe_mode=dynamic`

Shared settings:
- `start=2005-01-01`
- `end=2024-12-31`
- `top_n=50`
- weekly rebalance
- `costs_bps=10`
- `max_tickers=2000`

### Strategies Tested
1. `quality_momentum`
   - `gross_profitability`
   - `momentum_12_1`
2. `quality_momentum_pullback`
   - `gross_profitability`
   - `momentum_12_1`
   - `reversal_1m`
3. `benchmark_4factor`
   - `gross_profitability`
   - `momentum_12_1`
   - `reversal_1m`
   - `low_vol_20`

All factor sets used equal internal weights.

### Artifacts
- `results/quality_momentum_comparison/20260312_043003/quality_momentum_results.csv`
- `results/quality_momentum_comparison/20260312_043003/quality_momentum_summary.csv`
- `results/quality_momentum_comparison/quality_momentum_results_latest.csv`
- `results/quality_momentum_comparison/quality_momentum_summary_latest.csv`

### Runtime
- `1670.85s` (`27.8m`)

### Results
| universe | strategy_name | factor_set | CAGR | Vol | Sharpe | MaxDD | AnnualTurnover | EligMedian | TradableMedian | RebalanceSkipped |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| liquid_us | quality_momentum | gross_profitability,momentum_12_1 | 0.1279 | 0.1993 | 0.7043 | -0.4385 | 4.4231 | 822.0 | 254.0 | 62 |
| sp500 | quality_momentum | gross_profitability,momentum_12_1 | 0.1135 | 0.1851 | 0.6737 | -0.4611 | 3.7249 | 172.5 | 172.5 | 6 |
| liquid_us | quality_momentum_pullback | gross_profitability,momentum_12_1,reversal_1m | 0.1478 | 0.2065 | 0.7713 | -0.4705 | 14.0545 | 822.0 | 254.0 | 62 |
| sp500 | quality_momentum_pullback | gross_profitability,momentum_12_1,reversal_1m | 0.1152 | 0.1979 | 0.6502 | -0.4972 | 10.9592 | 172.5 | 172.5 | 5 |
| liquid_us | benchmark_4factor | gross_profitability,momentum_12_1,reversal_1m,low_vol_20 | 0.1445 | 0.1947 | 0.7909 | -0.4384 | 14.3870 | 822.0 | 254.0 | 62 |
| sp500 | benchmark_4factor | gross_profitability,momentum_12_1,reversal_1m,low_vol_20 | 0.1105 | 0.1747 | 0.6878 | -0.4173 | 11.0774 | 172.5 | 172.5 | 5 |

### Sharpe Pivot
| strategy_name | liquid_us | sp500 |
|---|---:|---:|
| quality_momentum | 0.7043 | 0.6737 |
| quality_momentum_pullback | 0.7713 | 0.6502 |
| benchmark_4factor | 0.7909 | 0.6878 |

### Interpretation
- The benchmark is not mostly just a quality-momentum strategy. The 2-factor version is clearly weaker in both universes, especially in `liquid_us`.
- `reversal_1m` adds meaningful value in `liquid_us`:
  - Sharpe improves from `0.7043` to `0.7713`
  - CAGR improves from `0.1279` to `0.1478`
  but it hurts `sp500`:
  - Sharpe falls from `0.6737` to `0.6502`
  - drawdown worsens from `-0.4611` to `-0.4972`
- `low_vol_20` adds modest but consistent benchmark value at the portfolio level:
  - in `liquid_us`, Sharpe improves from `0.7713` to `0.7909` and drawdown improves from `-0.4705` to `-0.4384`
  - in `sp500`, Sharpe improves from `0.6502` to `0.6878` and drawdown improves from `-0.4972` to `-0.4173`
- Best benchmark candidate from this test remains `benchmark_4factor`. It is the only version that is top-Sharpe in both universes, and in `sp500` it also has the best drawdown profile of the three.

### Recommendation
- Keep the current 4-factor benchmark as the default benchmark candidate.
- Future simplification work should focus on whether `reversal_1m` should be treated differently by universe rather than removed globally.


## 2026-03-12 — Composite vs Independent Factor Sleeves

### Objective
Compare the default composite benchmark against an equal-weight combination of independent factor sleeves to isolate cross-factor agreement versus sleeve diversification.

### Command
```bash
PYTHONPATH=src python scripts/run_composite_vs_sleeves.py
```

### Setup
- Universe: `liquid_us`
- Universe mode: `dynamic`
- Dates: `2000-01-01` to `2024-12-31`
- Rebalance: `weekly`
- Top N: `50`
- Weighting: `equal`
- Costs: `10 bps`
- Factors: `momentum_12_1`, `reversal_1m`, `low_vol_20`, `gross_profitability`
- Composite: equal factor weights, existing normalized linear aggregation
- Sleeve combination: fixed 25% return-level combination of the four single-factor sleeves

### Key Results
- Composite CAGR / Sharpe / MaxDD / Turnover: `0.1501` / `0.8380` / `-0.4404` / `13.7905`
- Sleeve-combo CAGR / Sharpe / MaxDD / Turnover: `0.1715` / `0.9374` / `-0.4863` / `10.7718`
- Delta composite minus sleeve-combo:
  - CAGR: `-0.0215`
  - Sharpe: `-0.0994`
  - MaxDD: `0.0459`
  - Turnover: `3.0187`

### Initial Interpretation
- Composite underperformed the sleeve combination on Sharpe.
- Sleeve combination did not improve drawdown relative to the composite.
- Results bundle: `results/composite_vs_sleeves/20260312_055425`


## 2026-03-12 — Sleeve Allocation Sweep

### Objective
Test whether simple capital-allocation tilts across the four benchmark factor sleeves improve the equal-weight sleeve portfolio.

### Command
```bash
PYTHONPATH=src python scripts/run_sleeve_allocation_sweep.py --start 2018-01-01 --end 2020-12-31 --max_tickers 300 --results_root results/sleeve_allocation_sweep_smoke
```

### Setup
- Universe: `liquid_us`
- Universe mode: `dynamic`
- Dates: `2018-01-01` to `2020-12-31`
- Rebalance: `weekly`
- Top N: `50`
- Weighting: `equal`
- Costs: `10 bps`
- Max tickers: `300`
- Sleeves: `momentum_12_1`, `reversal_1m`, `low_vol_20`, `gross_profitability`

### Key Results
- Best allocation by Sharpe: `rev_tilt_2` (`Sharpe=0.7196`, `CAGR=0.1553`, `MaxDD=-0.4228`)
- Equal allocation baseline: `Sharpe=0.6369`, `CAGR=0.1243`, `MaxDD=-0.4126`
- Best minus equal delta:
  - CAGR: `0.0310`
  - Sharpe: `0.0827`
  - MaxDD: `-0.0102`
  - Turnover: `0.4713`

### Initial Interpretation
- Allocation tilts improved Sharpe versus equal sleeves in this run.
- Best result came from `rev_tilt_2`.
- Results bundle: `results/sleeve_allocation_sweep_smoke/20260312_141257`


## 2026-03-12 — Sleeve Allocation Sweep

### Objective
Test whether simple capital-allocation tilts across the four benchmark factor sleeves improve the equal-weight sleeve portfolio.

### Command
```bash
PYTHONPATH=src python scripts/run_sleeve_allocation_sweep.py
```

### Setup
- Universe: `liquid_us`
- Universe mode: `dynamic`
- Dates: `2000-01-01` to `2024-12-31`
- Rebalance: `weekly`
- Top N: `50`
- Weighting: `equal`
- Costs: `10 bps`
- Max tickers: `2000`
- Sleeves: `momentum_12_1`, `reversal_1m`, `low_vol_20`, `gross_profitability`

### Key Results
- Best allocation by Sharpe: `rev_tilt_2` (`Sharpe=0.9849`, `CAGR=0.2069`, `MaxDD=-0.5079`)
- Equal allocation baseline: `Sharpe=0.9374`, `CAGR=0.1715`, `MaxDD=-0.4863`
- Best minus equal delta:
  - CAGR: `0.0354`
  - Sharpe: `0.0476`
  - MaxDD: `-0.0216`
  - Turnover: `1.1757`

### Initial Interpretation
- Allocation tilts improved Sharpe versus equal sleeves in this run.
- Best result came from `rev_tilt_2`.
- Results bundle: `results/sleeve_allocation_sweep/20260312_141634`
