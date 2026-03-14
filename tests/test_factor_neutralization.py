from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.factors.neutralize import neutralize_scores_cs


def test_sector_neutralization_reduces_sector_mean_spread() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    cols = ["A", "B", "C", "D"]
    # Sector effect: Tech high, Util low.
    scores = pd.DataFrame(
        [[2.0, 2.2, -1.8, -2.0], [1.8, 2.1, -1.6, -1.9]],
        index=idx,
        columns=cols,
    )
    sector = {"A": "TECH", "B": "TECH", "C": "UTIL", "D": "UTIL"}
    out = neutralize_scores_cs(
        scores,
        sector_by_ticker=sector,
        log_market_cap_by_ticker=None,
        use_sector_neutralization=True,
        use_size_neutralization=False,
    )
    for dt in idx:
        tech_mean = float(out.loc[dt, ["A", "B"]].mean())
        util_mean = float(out.loc[dt, ["C", "D"]].mean())
        assert abs(tech_mean - util_mean) < 1e-8


def test_size_neutralization_reduces_size_correlation() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    cols = ["A", "B", "C", "D", "E"]
    log_cap = pd.Series({"A": 10.0, "B": 11.0, "C": 12.0, "D": 13.0, "E": 14.0})
    base = np.tile(log_cap.to_numpy(), (len(idx), 1))
    noise = np.array([[0.1, -0.2, 0.1, 0.0, -0.1], [0.0, 0.2, -0.2, 0.0, 0.0], [-0.1, 0.0, 0.1, -0.2, 0.2]])
    scores = pd.DataFrame(base + noise, index=idx, columns=cols)
    out = neutralize_scores_cs(
        scores,
        sector_by_ticker=None,
        log_market_cap_by_ticker=log_cap.to_dict(),
        use_sector_neutralization=False,
        use_size_neutralization=True,
    )
    for dt in idx:
        x = pd.Series(log_cap, index=cols)
        y = out.loc[dt]
        corr = float(x.corr(y))
        assert abs(corr) < 1e-8


def test_neutralization_preserves_nan_structure_and_no_covariate_noop() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    cols = ["A", "B", "C"]
    scores = pd.DataFrame([[1.0, np.nan, 3.0], [2.0, 1.0, np.nan]], index=idx, columns=cols)
    out = neutralize_scores_cs(
        scores,
        sector_by_ticker=None,
        log_market_cap_by_ticker=None,
        use_sector_neutralization=True,
        use_size_neutralization=True,
    )
    assert np.allclose(out.to_numpy(), scores.to_numpy(), equal_nan=True)


def test_beta_neutralization_reduces_beta_correlation() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    cols = ["A", "B", "C", "D", "E"]
    beta_row = pd.Series({"A": 0.5, "B": 0.8, "C": 1.0, "D": 1.2, "E": 1.5})
    scores = pd.DataFrame(
        np.tile(beta_row.to_numpy(dtype=float), (len(idx), 1))
        + np.array(
            [
                [0.2, -0.1, 0.0, 0.1, -0.2],
                [0.0, 0.1, -0.2, 0.2, -0.1],
                [-0.1, 0.0, 0.2, -0.2, 0.1],
            ]
        ),
        index=idx,
        columns=cols,
    )
    beta_exposure = pd.DataFrame(np.tile(beta_row.to_numpy(), (len(idx), 1)), index=idx, columns=cols)
    out = neutralize_scores_cs(
        scores=scores,
        beta_exposure=beta_exposure,
        use_beta_neutralization=True,
        use_sector_neutralization=False,
        use_size_neutralization=False,
    )
    for dt in idx:
        corr = float(out.loc[dt].corr(beta_row))
        assert abs(corr) < 1e-8


def test_explicit_disable_all_neutralization_is_noop() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    cols = ["A", "B", "C"]
    scores = pd.DataFrame([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], index=idx, columns=cols)
    out = neutralize_scores_cs(
        scores=scores,
        beta_exposure=None,
        use_beta_neutralization=False,
        use_sector_neutralization=False,
        use_size_neutralization=False,
    )
    assert np.allclose(out.to_numpy(), scores.to_numpy(), equal_nan=True)
