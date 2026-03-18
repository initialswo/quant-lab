"""Promote a validated staged warehouse snapshot into the canonical warehouse."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

PROMOTION_FILES = [
    "security_master.parquet",
    "symbol_history.parquet",
    "ticker_id_stability_report.parquet",
    "symbol_collision_report.parquet",
    "equity_prices_daily.parquet",
    "equity_prices_daily_versions.parquet",
    "equity_fundamentals_pit.parquet",
    "universe_membership_daily.parquet",
    "ingestion_audit.parquet",
]


def _load_validation_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"validation summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_validation_passed(summary: dict, path: Path) -> None:
    failed = bool(summary.get("thresholds", {}).get("failed", True))
    if failed:
        raise RuntimeError(f"validation failed; refusing promotion: {path}")


def _security_metadata_snapshot(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    rows = []
    for line in path.read_text(encoding="utf-8").strip().splitlines()[1:]:
        column, row_count, null_count, non_null_count, null_frac, sample_values = line.split(",", 5)
        rows.append(
            {
                "column": column,
                "row_count": int(row_count),
                "null_count": int(null_count),
                "non_null_count": int(non_null_count),
                "null_frac": float(null_frac),
                "sample_values": sample_values,
            }
        )
    return {
        "artifact": str(path),
        "columns": rows,
        "classification_is_informational_only": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Promote a validated staged warehouse snapshot.")
    p.add_argument("--staged-warehouse-root", required=True)
    p.add_argument("--warehouse-root", default="data/warehouse")
    p.add_argument("--results-root", default="results/ingest/phase3")
    p.add_argument("--validation-summary-path", required=True)
    p.add_argument(
        "--metadata-audit-path",
        default="",
        help="Optional security_master_metadata_completeness.csv path for promotion summary only.",
    )
    p.add_argument(
        "--timestamp",
        default="",
        help="Optional promotion timestamp label. Defaults to current UTC time.",
    )
    args = p.parse_args()

    staged_root = Path(args.staged_warehouse_root)
    warehouse_root = Path(args.warehouse_root)
    results_root = Path(args.results_root)
    validation_summary_path = Path(args.validation_summary_path)
    metadata_audit_path = Path(args.metadata_audit_path) if args.metadata_audit_path else None
    ts = str(args.timestamp).strip() or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    if not staged_root.exists():
        raise FileNotFoundError(f"staged warehouse root not found: {staged_root}")

    validation_summary = _load_validation_summary(validation_summary_path)
    _ensure_validation_passed(validation_summary, validation_summary_path)

    missing = [name for name in PROMOTION_FILES if not (staged_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"staged warehouse missing required files: {', '.join(missing)}")

    warehouse_root.mkdir(parents=True, exist_ok=True)
    backup_root = results_root / "backups" / ts
    backup_root.mkdir(parents=True, exist_ok=True)

    backed_up: list[str] = []
    promoted: list[str] = []
    for name in PROMOTION_FILES:
        current = warehouse_root / name
        if current.exists():
            shutil.copy2(current, backup_root / name)
            backed_up.append(name)

    for name in PROMOTION_FILES:
        shutil.copy2(staged_root / name, warehouse_root / name)
        promoted.append(name)

    promotion_summary = {
        "timestamp": ts,
        "staged_warehouse_root": str(staged_root),
        "warehouse_root": str(warehouse_root),
        "validation_summary_path": str(validation_summary_path),
        "validation_failed": False,
        "classification_metadata_audit": _security_metadata_snapshot(metadata_audit_path) if metadata_audit_path else {},
        "classification_metadata_blocks_price_promotion": False,
        "backed_up_files": backed_up,
        "promoted_files": promoted,
    }
    summary_path = backup_root / "promotion_summary.json"
    summary_path.write_text(json.dumps(promotion_summary, indent=2), encoding="utf-8")
    print(json.dumps(promotion_summary, indent=2))


if __name__ == "__main__":
    main()
