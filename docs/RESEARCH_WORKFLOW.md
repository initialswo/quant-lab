# Research Workflow

## Backtest vs Walkforward

- `backtest` runs one contiguous sample over the full date range.
- `walkforward` splits history into rolling train/test windows and reports out-of-sample (OOS) results.

### Backtest example

```bash
python run.py backtest \
  --start 2010-01-01 --end 2024-01-01 \
  --top_n 20 --rebalance weekly --costs_bps 10
```

### Walkforward example

```bash
python run.py walkforward \
  --start 2005-01-01 --end 2024-01-01 \
  --train_years 5 --test_years 2 \
  --top_n 20 --rebalance weekly
```

## Running a Sweep

```bash
python run.py sweep \
  --start 2012-01-01 --end 2024-01-01 \
  --top_n 10,20 \
  --rebalance weekly,monthly \
  --costs_bps 5,10 \
  --weighting equal,score \
  --trend_filter 0,1
```

## Risk Controls

- `--max_weight`: single-name cap (rebalance-date weight clipping + redistribution).
- `--sector_cap` + `--sector_map`: optional sector limit. `sector_map` CSV must have `Ticker,Sector`.
- `--target_vol`: portfolio vol target (annualized), using trailing realized vol.
- `--port_vol_lookback`: lookback window for vol targeting.
- `--max_leverage`: leverage ceiling for vol targeting scaling.
- `--slippage_bps`: static slippage add-on.
- `--slippage_vol_mult`: slippage multiplier on trailing asset-vol proxy.
- `--slippage_vol_lookback`: lookback for slippage vol proxy.

## Diagnostics

Walkforward `walkforward_summary.csv` includes:

- `RebalanceCount`: calendar rebalance count in the test window.
- `RiskOnPct`: trend-filter risk-on fraction (1.0 if filter disabled).
- `InvestedPct`: fraction of test days with non-zero gross exposure.
- `AvgHoldings`: average active position count.
- `TurnoverAvg`: average turnover on rebalance days.
- `MaxNameWeight`: max single-name weight seen in the test window.
- `MaxSectorWeight`: max sector weight (NaN if sector caps disabled/no map).
- `EffectiveHoldingsAvg`: average `1 / sum(w^2)` concentration metric.

## Outputs

Each run writes under `results/<run_tag>/`:

- `run_config.json`
- `summary.json` (canonical summary)
- `summary.txt` (compat text view)
- `equity.csv` or `equity_oos.csv`
- `walkforward_summary.csv` (walkforward only)

Global registry appends to `results/registry.csv`.
