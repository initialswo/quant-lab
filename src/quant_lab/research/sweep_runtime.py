"""Shared runtime helpers for resilient research sweeps."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return out or "variant"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


REGISTRY_COLUMNS = [
    "variant",
    "variant_slug",
    "parameters_json",
    "status",
    "start_time",
    "end_time",
    "duration_seconds",
    "output_rows",
    "artifact_path",
    "error_message",
]


@dataclass
class SweepRunResult:
    completed: int
    failed: int
    skipped: int
    elapsed_seconds: float


class SweepState:
    """Durable sweep checkpoint + status registry manager."""

    def __init__(self, results_dir: str | Path, overwrite: bool = False) -> None:
        self.results_dir = Path(results_dir)
        self.overwrite = bool(overwrite)
        self.registry_path = self.results_dir / "variant_status.csv"
        self.artifacts_dir = self.results_dir / "variants"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._records = self._load_registry()
        if self.overwrite:
            for record in self._records.values():
                record["status"] = "pending"
                record["error_message"] = ""
            self._save_registry()

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        if not self.registry_path.exists():
            return {}
        df = pd.read_csv(self.registry_path)
        records: dict[str, dict[str, Any]] = {}
        for row in df.to_dict(orient="records"):
            name = str(row.get("variant", "")).strip()
            if not name:
                continue
            rec = {k: row.get(k, "") for k in REGISTRY_COLUMNS}
            for col in ("duration_seconds", "output_rows"):
                val = rec.get(col, "")
                if pd.isna(val):
                    rec[col] = ""
            records[name] = rec
        return records

    def _save_registry(self) -> None:
        rows = [self._records[k] for k in sorted(self._records.keys())]
        df = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
        _atomic_write_text(self.registry_path, df.to_csv(index=False))

    def ensure_variant(self, variant: str, params: dict[str, Any]) -> dict[str, Any]:
        if variant not in self._records:
            self._records[variant] = {
                "variant": variant,
                "variant_slug": slugify(variant),
                "parameters_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
                "status": "pending",
                "start_time": "",
                "end_time": "",
                "duration_seconds": "",
                "output_rows": "",
                "artifact_path": "",
                "error_message": "",
            }
            self._save_registry()
        return self._records[variant]

    def variant_artifact_path(self, variant: str) -> Path:
        slug = str(self._records.get(variant, {}).get("variant_slug") or slugify(variant))
        return self.artifacts_dir / f"{slug}.json"

    def has_completed_artifact(self, variant: str) -> bool:
        record = self._records.get(variant, {})
        status = str(record.get("status", "")).lower()
        if status not in {"completed", "skipped"}:
            return False
        artifact = self.variant_artifact_path(variant)
        return artifact.exists()

    def mark_running(self, variant: str, start_time: str) -> None:
        rec = self._records[variant]
        rec["status"] = "running"
        rec["start_time"] = start_time
        rec["end_time"] = ""
        rec["duration_seconds"] = ""
        rec["error_message"] = ""
        self._save_registry()

    def mark_completed(self, variant: str, end_time: str, duration_seconds: float, output_rows: int) -> None:
        rec = self._records[variant]
        rec["status"] = "completed"
        rec["end_time"] = end_time
        rec["duration_seconds"] = f"{float(duration_seconds):.6f}"
        rec["output_rows"] = int(output_rows)
        rec["artifact_path"] = str(self.variant_artifact_path(variant).relative_to(self.results_dir))
        rec["error_message"] = ""
        self._save_registry()

    def mark_failed(self, variant: str, end_time: str, duration_seconds: float, error_message: str) -> None:
        rec = self._records[variant]
        rec["status"] = "failed"
        rec["end_time"] = end_time
        rec["duration_seconds"] = f"{float(duration_seconds):.6f}"
        rec["output_rows"] = ""
        rec["error_message"] = str(error_message)[:4000]
        self._save_registry()

    def mark_skipped(self, variant: str) -> None:
        rec = self._records[variant]
        rec["status"] = "skipped"
        rec["end_time"] = utc_now_iso()
        rec["error_message"] = ""
        self._save_registry()

    def write_variant_payload(self, variant: str, payload: dict[str, Any]) -> None:
        target = self.variant_artifact_path(variant)
        _atomic_write_text(target, json.dumps(payload, indent=2, sort_keys=True))

    def load_all_payloads(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for variant in sorted(self._records.keys()):
            if not self.has_completed_artifact(variant):
                continue
            path = self.variant_artifact_path(variant)
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads


def _fmt_seconds(seconds: float) -> str:
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, mm = divmod(m, 60)
    return f"{h}h{mm:02d}m{sec:02d}s"


def run_sweep_variants(
    *,
    variants: list[dict[str, Any]],
    state: SweepState,
    run_variant: Callable[[dict[str, Any]], dict[str, Any]],
    fail_fast: bool = False,
) -> SweepRunResult:
    start_all = time.perf_counter()
    total = len(variants)
    completed = 0
    failed = 0
    skipped = 0

    for idx, variant in enumerate(variants, start=1):
        name = str(variant["name"])
        params = dict(variant.get("params") or {})
        state.ensure_variant(name, params)

        if not state.overwrite and state.has_completed_artifact(name):
            skipped += 1
            state.mark_skipped(name)
            print(f"[{idx}/{total}] SKIP {name} (already completed)")
            continue

        started_iso = utc_now_iso()
        started_t = time.perf_counter()
        state.mark_running(name, started_iso)
        remaining = total - idx + 1
        elapsed = time.perf_counter() - start_all
        print(
            f"[{idx}/{total}] RUN {name} | completed={completed} failed={failed} "
            f"remaining={remaining} elapsed={_fmt_seconds(elapsed)}"
        )

        try:
            payload = run_variant(variant)
            if not isinstance(payload, dict):
                raise TypeError("run_variant must return a dict payload")
            state.write_variant_payload(name, payload)
            duration = time.perf_counter() - started_t
            output_rows = int(payload.get("output_rows", 0))
            state.mark_completed(
                name,
                end_time=utc_now_iso(),
                duration_seconds=duration,
                output_rows=output_rows,
            )
            completed += 1
            done = completed + failed + skipped
            avg = (time.perf_counter() - start_all) / done if done > 0 else 0.0
            eta = avg * max(0, total - done)
            print(f"[{idx}/{total}] DONE {name} in {_fmt_seconds(duration)} | eta={_fmt_seconds(eta)}")
        except Exception as exc:
            duration = time.perf_counter() - started_t
            state.mark_failed(
                name,
                end_time=utc_now_iso(),
                duration_seconds=duration,
                error_message=f"{type(exc).__name__}: {exc}",
            )
            failed += 1
            print(f"[{idx}/{total}] FAIL {name} in {_fmt_seconds(duration)} | {type(exc).__name__}: {exc}")
            if fail_fast:
                break

    elapsed_total = time.perf_counter() - start_all
    print(
        "SWEEP SUMMARY "
        f"total={total} completed={completed} skipped={skipped} failed={failed} "
        f"elapsed={_fmt_seconds(elapsed_total)}"
    )
    return SweepRunResult(
        completed=completed,
        failed=failed,
        skipped=skipped,
        elapsed_seconds=elapsed_total,
    )
