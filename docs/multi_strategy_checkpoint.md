# Multi-Strategy Checkpoint

## 1) Project Context

`quant_lab` is a bias-safe quantitative research engine used to develop systematic trading strategies.

Core development principles:
- Avoid lookahead bias.
- Avoid survivorship bias.
- Prioritize research integrity.
- Build infrastructure before strategy discovery.

## 2) Strategy 1 — Benchmark v1.1

Current benchmark configuration:
- Factors:
  - `momentum_12_1`
  - `reversal_1m`
  - `low_vol_20`
  - `gross_profitability`
- Aggregation method: `geometric_rank`
- Construction parameters:
  - `top_n = 50`
  - `rank_buffer = 20`
  - `rebalance = monthly`
  - `dynamic_factor_weights = True`
  - `target_vol = 0.14`
  - `bear_exposure_scale = 1.0`

Validation summary:
- Sharpe: approximately `1.03`.
- MaxDD: approximately `-29%`.

Validation steps completed:
- Walk-forward OOS testing.
- Delay robustness testing.
- Cost robustness testing.
- Contribution concentration analysis.

## 3) Strategy 2 — Cross-Asset Trend v2

Best configuration from the experiment grid:
- Assets:
  - `SPY`
  - `TLT`
  - `GLD`
  - `DBC`
  - `VNQ`
  - `SHY`
- Signal: dual momentum.
- Lookback: `126` trading days.
- Construction:
  - `top_n = 4`
  - inverse volatility weighting
  - monthly rebalance
  - `1`-day signal lag

Performance summary:
- Sharpe: approximately `0.64`.
- MaxDD: approximately `-13%`.
- Vol: approximately `6.8%`.

Interpretation:
- This sleeve is primarily a diversification component, not a standalone return maximizer.

## 4) Portfolio Construction Experiment

Blended portfolios tested:
- `100% Benchmark`
- `90/10`
- `80/20`
- `70/30`
- `60/40`
- `100% CrossAsset`

Key result:
- Full-period correlation between Benchmark and Cross-Asset v2 is approximately `0.46`.

Best balanced blend:
- `70/30` Benchmark/CrossAsset.
- Metrics (approx.):
  - CAGR: `11.0%`
  - Vol: `11.1%`
  - Sharpe: `1.00`
  - MaxDD: `-20.6%`

Interpretation:
- Diversification improves Sharpe and reduces drawdown versus 100% Benchmark.

## 5) Portfolio Risk Layer Experiment

Volatility targeting test on the `70/30` blend:
- `target_vol = 12%`
- `63`-day realized volatility estimator
- leverage cap: `0.25` to `2.0`

Results (vol-targeted 70/30):
- CAGR: approximately `12.4%`
- Sharpe: approximately `1.02`
- MaxDD: approximately `-23.6%`

Interpretation:
- Vol targeting improves Sharpe and CAGR, but increases drawdown depth.

Current positioning:
- Static `70/30`: conservative baseline.
- Vol-targeted `70/30`: higher-risk growth variant.

## 6) Strategies Tested But Not Promoted

- Reversal-only equity strategy.
  - Reason: very high correlation with Benchmark (approximately `0.92`).
- Sector rotation sleeve.
  - Reason: high drawdown overlap and weak diversification contribution.

## 7) Current Architecture

Strategy Layer:
- Equity factor composite (Benchmark v1.1).
- Cross-Asset Trend v2.

Portfolio Layer:
- Static blend candidate (`70/30`).

Optional Risk Layer:
- Portfolio-level volatility targeting.

## 8) Next Research Directions

- Dynamic allocation between strategies.
- Additional cross-asset signals.
- Expand cross-asset universe.
- Execution and capacity analysis.
- Live trading simulation.

## 9) Artifact References

- `results/cross_asset_trend_v2_grid_*`
- `results/cross_asset_v2_strategy_combination_test`
- `results/portfolio_vol_target_test`
