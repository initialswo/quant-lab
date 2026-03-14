from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.data.universe_dataset import (
    build_point_in_time_universe,
    load_universe_dataset,
    save_universe_dataset,
    summarize_universe_membership,
)
from quant_lab.engine.runner import _align_universe_membership


def test_build_point_in_time_universe_basic() -> None:
    idx = pd.date_range("2024-01-01", periods=12, freq="B")
    close = pd.DataFrame(index=idx)
    close["A"] = 10.0 + np.arange(len(idx), dtype=float)
    close["B"] = np.nan
    close.loc[idx[5]:, "B"] = 20.0 + np.arange(len(idx) - 5, dtype=float)
    close["C"] = 30.0
    close.loc[idx[[2, 4, 6, 8, 10]], "C"] = np.nan

    membership = build_point_in_time_universe(
        close=close,
        min_history_days=5,
        valid_lookback=5,
        min_valid_frac=0.8,
        min_price=1.0,
    )

    assert bool(membership.loc[idx[8], "A"])
    assert not bool(membership.loc[idx[8], "B"])
    assert not bool(membership.loc[idx[10], "C"])


def test_summarize_universe_membership_added_removed() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    membership = pd.DataFrame(
        [[1, 0, 1], [1, 1, 0], [0, 1, 0]],
        index=idx,
        columns=["A", "B", "C"],
        dtype=bool,
    )
    summary = summarize_universe_membership(membership)

    assert summary["eligible_count"].tolist() == [2, 2, 1]
    assert summary["added_count"].tolist() == [2, 1, 0]
    assert summary["removed_count"].tolist() == [0, 1, 1]


def test_save_and_load_universe_dataset(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    membership = pd.DataFrame(
        [[True, False], [True, True], [False, True], [False, False]],
        index=idx,
        columns=["A", "B"],
    )
    summary = summarize_universe_membership(membership)
    paths = save_universe_dataset(membership=membership, summary=summary, outdir_or_path=str(tmp_path))
    assert (tmp_path / "universe_membership.csv").exists()
    assert (tmp_path / "universe_summary.csv").exists()
    assert paths["membership_path"].endswith("universe_membership.csv")
    assert paths["summary_path"].endswith("universe_summary.csv")

    loaded = load_universe_dataset(paths["membership_path"])
    expected = membership.sort_index().reindex(sorted(membership.columns), axis=1).astype(bool)
    assert loaded.equals(expected)


def test_runner_can_use_built_membership(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="B")
    close = pd.DataFrame(
        {
            "A": 100.0 + np.arange(len(idx), dtype=float),
            "B": 200.0 + np.arange(len(idx), dtype=float),
            "SPY": 300.0 + np.arange(len(idx), dtype=float),
        },
        index=idx,
    )
    built = build_point_in_time_universe(
        close=close[["A", "B"]],
        min_history_days=2,
        valid_lookback=2,
        min_valid_frac=1.0,
        min_price=1.0,
    )
    save_paths = save_universe_dataset(
        membership=built,
        summary=summarize_universe_membership(built),
        outdir_or_path=str(tmp_path),
    )
    loaded = load_universe_dataset(save_paths["membership_path"])
    aligned = _align_universe_membership(
        membership=loaded,
        close=close,
        exempt={"SPY"},
    )

    assert aligned.index.equals(close.index)
    assert aligned.columns.tolist() == close.columns.tolist()
    assert bool(aligned["SPY"].all())
