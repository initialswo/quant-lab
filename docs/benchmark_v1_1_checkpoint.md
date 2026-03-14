# Benchmark v1.1 Checkpoint

## 1) Project Purpose
`quant_lab` is a Python quantitative research platform focused on:
- bias-safe backtesting
- systematic factor research
- robust experiment design
- research-first development

## 2) Benchmark v1.1 Configuration
Current lead research model (Benchmark v1.1):

Signals:
- `momentum_12_1`
- `reversal_1m`
- `low_vol_20`
- `gross_profitability`

Portfolio construction:
- `factor_aggregation_method = geometric_rank`
- `top_n = 50`
- `rank_buffer = 20`
- monthly rebalance
- equal weight

Risk layer:
- `dynamic_factor_weights = True`
- `target_vol = 0.14`
- `bear_exposure_scale = 1.00`

Filters:
- `min_price = 5`
- `min_avg_dollar_volume = 5e6`

Universe:
- SP500 is the production-safe baseline
- broader-universe research exists, but benchmark remains SP500-based unless explicitly running a research variant

## 3) Benchmark v1.1 Validation Summary
Key established validation outcomes (from saved artifacts):

- Full-period benchmark promotion (`geometric_rank` vs prior `linear` aggregation):
  - `linear`: Sharpe `0.9858`, MaxDD `-0.3093`, StabilityScore `0.8262`
  - `geometric_rank`: Sharpe `1.0308`, MaxDD `-0.2918`, StabilityScore `0.8608`
  - Result: `geometric_rank` promoted.

- Walk-forward OOS (expanding train / 2-year test windows):
  - `geometric_rank` OOS Sharpe summary: avg `1.1055`, median `1.0278`, worst `0.4093`, best `1.8046`
  - `linear` OOS Sharpe summary: avg `1.0545`, median `1.0409`, worst `0.3803`, best `1.6424`

- Execution-delay robustness (`0/1/3/5` trading-day delays):
  - `geometric_rank` remained stable across tested delays, with Sharpe around `1.03` and MaxDD near `-0.29`.

- Transaction-cost robustness:
  - Base reference (`10 bps` costs, `0 bps` slippage): Sharpe `0.9858`, MaxDD `-0.3093`.
  - Performance degrades as friction increases, but remains positive through the tested grid.

- Contribution concentration:
  - Top-20 contribution share (absolute): `0.3253`
  - Herfindahl (absolute): `0.00965`
  - Effective number of contributors: `~103.66`
  - Interpretation: returns are not dominated by only a handful of names.

## 4) Tested Ideas That Did NOT Improve the Benchmark
Rejected / non-promoted as default benchmark replacements:

- `earnings_yield`: usable coverage after data rebuild, but diagnostics did not show sufficient standalone signal quality for admission.
- `roa`: diagnostics did not clear admission threshold versus existing stack.
- sector-neutral ranking: did not improve lead candidate risk-adjusted results.
- factor orthogonalization: removed overlap but did not improve total portfolio outcomes.
- multi-horizon momentum stack: added complexity without clear benchmark improvement.
- multi-sleeve portfolio architecture: diversification benefits were not strong enough to replace single-composite benchmark.
- volatility-scaled position sizing: no clear net improvement versus equal-weight Top-N in the lead config.
- factor neutralization variants (`beta`, `sector`, `size` combinations): useful as research controls, not promoted as new default benchmark behavior.

## 5) Strategy 2 Status
Cross-asset trend track status:
- implemented as a valid research strategy track
- remains research-only and not promoted as a co-equal benchmark
- combination tests with Benchmark v1.1 improved drawdown profile but did not improve Sharpe versus 100% factor benchmark
- keep active as an optional diversifier candidate pending further refinement

## 6) Data / Research Infrastructure Status
Current infrastructure baseline:
- local equities parquet database
- local FMP fundamentals parquet database (including PIT-safe fields used by quality/value research)
- PIT-safe fundamentals alignment by `available_date`
- sweep execution improvements: resume/checkpoint/cache and variant-status tracking
- factor diagnostics workflow
- contribution concentration analysis workflow
- walk-forward validation workflows
- local cross-asset ETF data support for Strategy 2 research

## 7) Recommended Next Research Directions
1. Prioritize Strategy 2 refinement (cross-asset trend quality and robustness) over incremental Benchmark v1.1 tuning.
2. Develop/validate a stronger low-correlation second strategy source rather than over-optimizing the equity factor stack.
3. Evaluate drawdown-aware combination policies between Benchmark v1.1 and Strategy 2 candidates.
4. Add true historical SP1000/SP1500 membership datasets when available to improve universe-expansion realism.

## 8) Artifact References
Primary result bundles to anchor this checkpoint:

- Rank aggregation promotion:
  - `results/rank_aggregation_test/rank_aggregation_comparison.csv`

- Geometric-rank robustness suite (walk-forward, delay, costs):
  - `results/geometric_rank_robustness_20260309_042452/robustness_summary.json`

- Contribution concentration:
  - `results/contribution_concentration_20260308_235039/contribution_summary.json`

- Cost robustness full grid:
  - `results/cost_robustness_lead_sweep_20260309_012749/cost_robustness_full.csv`
  - `results/cost_robustness_lead_sweep_20260309_012749/cost_robustness_summary.json`

- Cross-asset benchmark + combination analysis:
  - `results/cross_asset_trend_benchmark/cross_asset_trend_summary.json`
  - `results/cross_asset_strategy_combination_test/20260309_050937/strategy_correlation.csv`
  - `results/cross_asset_strategy_combination_test/20260309_050937/portfolio_summary.csv`
