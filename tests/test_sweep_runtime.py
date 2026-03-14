from __future__ import annotations

import pandas as pd

from quant_lab.research.sweep_runtime import SweepState, run_sweep_variants


def _variants() -> list[dict[str, object]]:
    return [
        {"name": "v1", "params": {"x": 1}},
        {"name": "v2", "params": {"x": 2}},
    ]


def test_resume_skips_completed_variants(tmp_path) -> None:
    results_dir = tmp_path / "sweep"
    state = SweepState(results_dir=results_dir)
    calls: list[str] = []

    def _run_variant(v: dict[str, object]) -> dict[str, object]:
        calls.append(str(v["name"]))
        return {"variant": v["name"], "full_row": {"Variant": v["name"]}, "sub_rows": [], "output_rows": 1}

    res1 = run_sweep_variants(variants=_variants(), state=state, run_variant=_run_variant)
    assert res1.completed == 2
    assert calls == ["v1", "v2"]

    state2 = SweepState(results_dir=results_dir)
    res2 = run_sweep_variants(variants=_variants(), state=state2, run_variant=_run_variant)
    assert res2.completed == 0
    assert res2.skipped == 2
    assert calls == ["v1", "v2"]


def test_failed_variant_is_retried_on_next_run(tmp_path) -> None:
    results_dir = tmp_path / "sweep"
    state = SweepState(results_dir=results_dir)
    attempts = {"v2": 0}

    def _run_variant(v: dict[str, object]) -> dict[str, object]:
        name = str(v["name"])
        if name == "v2" and attempts["v2"] == 0:
            attempts["v2"] += 1
            raise RuntimeError("boom")
        return {"variant": name, "full_row": {"Variant": name}, "sub_rows": [], "output_rows": 1}

    res1 = run_sweep_variants(variants=_variants(), state=state, run_variant=_run_variant)
    assert res1.completed == 1
    assert res1.failed == 1

    state2 = SweepState(results_dir=results_dir)
    res2 = run_sweep_variants(variants=_variants(), state=state2, run_variant=_run_variant)
    assert res2.completed == 1
    assert res2.failed == 0

    df = pd.read_csv(results_dir / "variant_status.csv")
    v2 = df.loc[df["variant"] == "v2"].iloc[0]
    assert v2["status"] in {"completed", "skipped"}


def test_registry_contains_expected_columns(tmp_path) -> None:
    results_dir = tmp_path / "sweep"
    state = SweepState(results_dir=results_dir)

    def _run_variant(v: dict[str, object]) -> dict[str, object]:
        return {
            "variant": v["name"],
            "full_row": {"Variant": v["name"]},
            "sub_rows": [{"Variant": v["name"], "Start": "2005-01-01", "End": "2009-12-31"}],
            "output_rows": 2,
        }

    run_sweep_variants(variants=_variants(), state=state, run_variant=_run_variant)
    df = pd.read_csv(results_dir / "variant_status.csv")
    required = {
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
    }
    assert required.issubset(set(df.columns))
    assert set(df["status"]).issubset({"pending", "running", "completed", "failed", "skipped"})
