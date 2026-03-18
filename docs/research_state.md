# Quant Lab Data Status (Authoritative Summary)

## Current Status

Canonical warehouse path:

`data/warehouse/`

Current promotion candidate:

`data/staging/phase3_tiingo_manifest/20260315_phase3_tiingo_priority_overlay1/warehouse/`

Layer status:

- Canonical price layer: ready
- Security identity layer: ready
- Security classification layer: incomplete

Interpretation:

- The canonical price layer is promotion-safe and should not be blocked by incomplete `sector` / `industry` coverage in `security_master`.
- The security identity layer is also promotion-safe: `ticker_id`, `canonical_symbol`, and symbol-history mappings are validated and stable.
- The security classification layer is not yet complete enough to treat sector- or industry-aware research outputs as production-grade defaults.

## Promotion Readiness

Promotion-safe now:

- `equity_prices_daily.parquet`
- `equity_prices_daily_versions.parquet`
- `equity_fundamentals_pit.parquet`
- `security_master.parquet`
- `symbol_history.parquet`
- `universe_membership_daily.parquet`
- `ticker_id_stability_report.parquet`
- `symbol_collision_report.parquet`
- `ingestion_audit.parquet`

Non-blocking limitation for this promotion:

- `security_master` classification metadata remains incomplete, especially `sector` and `industry`.
- This does not block price-layer promotion.
- It does block treating sector-neutral and industry-neutral research as fully production-ready.

## Validation Snapshot

Required promotion gates are satisfied for the current candidate:

- duplicate price keys: passed
- unmatched canonical symbols: passed
- ticker identity instability: passed
- current S&P 500 price coverage through latest date: passed
- adjusted-price semantics: passed

Additional metadata audit now reported in validation outputs:

- `security_master_metadata_completeness.csv`
- null counts for `name`, `exchange`, `sector`, `industry`

Promotion step to run now:

```bash
PYTHONPATH=src python scripts/promote_staged_warehouse.py \
  --staged-warehouse-root data/staging/phase3_tiingo_manifest/20260315_phase3_tiingo_priority_overlay1/warehouse \
  --warehouse-root data/warehouse \
  --results-root results/ingest/phase3 \
  --validation-summary-path results/data_validation/phase3_tiingo_priority_overlay1/latest/validation_summary.json \
  --metadata-audit-path results/data_validation/phase3_tiingo_priority_overlay1/latest/security_master_metadata_completeness.csv \
  --timestamp 20260315_phase3_price_promotion
```

## Research Guidance

Until classification coverage improves:

- sector-neutral research should remain disabled or clearly flagged as experimental
- industry-neutral research should remain disabled or clearly flagged as experimental
- any benchmark or portfolio constraint that depends on `sector` or `industry` should be treated as provisional

## Canonical Warehouse Tables

`data/warehouse/`

- `security_master.parquet`
- `symbol_history.parquet`
- `equity_prices_daily.parquet`
- `equity_prices_daily_versions.parquet`
- `equity_fundamentals_pit.parquet`
- `universe_membership_daily.parquet`
- `ingestion_audit.parquet`

## Operational Rule

Promote validated warehouse outputs when the price layer and identity layer are ready.
Do not hold price promotion hostage to incomplete classification metadata.
Instead, track classification completeness as a separate quality dimension and keep classification-dependent research guarded until coverage improves.
