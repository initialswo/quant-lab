from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.research.growth_leader_equity import (
    apply_growth_screen,
    build_growth_scores,
    run_growth_leader_backtest,
)


def _sample_data(periods: int = 90) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    idx = pd.date_range("2024-01-01", periods=periods, freq="B")
    cols = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(
        {
            "A": np.linspace(50.0, 85.0, len(idx)),
            "B": np.linspace(40.0, 70.0, len(idx)),
            "C": np.linspace(30.0, 35.0, len(idx)),
            "D": np.linspace(20.0, 22.0, len(idx)),
            "E": np.linspace(8.0, 9.0, len(idx)),
        },
        index=idx,
    )
    volume = pd.DataFrame(2_000_000.0, index=idx, columns=cols)
    fundamentals_aligned = {
        "gross_profit": pd.DataFrame(
            {
                "A": 100.0,
                "B": 90.0,
                "C": 10.0,
                "D": -5.0,
                "E": 2.0,
            },
            index=idx,
        ),
        "total_assets": pd.DataFrame(200.0, index=idx, columns=cols),
        "net_income": pd.DataFrame(
            {
                "A": 15.0,
                "B": 12.0,
                "C": 2.0,
                "D": -1.0,
                "E": 0.1,
            },
            index=idx,
        ),
        "shares_outstanding": pd.DataFrame(100.0, index=idx, columns=cols),
    }
    return close, volume, fundamentals_aligned


def test_screening_logic_returns_subset() -> None:
    close, volume, fundamentals = _sample_data(periods=60)
    mask = apply_growth_screen(
        fundamentals_aligned=fundamentals,
        prices=close,
        volume=volume,
        min_price=10.0,
        min_avg_dollar_volume=5_000_000.0,
    )
    assert mask.shape == close.shape
    # E fails min_price, D fails profitability/earnings.
    assert not bool(mask["E"].any())
    assert not bool(mask["D"].any())
    assert bool(mask["A"].any())


def test_ranking_produces_expected_shape() -> None:
    close, volume, fundamentals = _sample_data(periods=70)
    mask = apply_growth_screen(
        fundamentals_aligned=fundamentals,
        prices=close,
        volume=volume,
    )
    scores = build_growth_scores(
        prices=close,
        fundamentals_aligned=fundamentals,
        screen_mask=mask,
    )
    assert scores.shape == close.shape
    assert scores.index.equals(close.index)
    assert scores.columns.tolist() == close.columns.tolist()


def test_backtest_weights_sum_to_one_when_invested() -> None:
    close, volume, fundamentals = _sample_data(periods=80)
    mask = apply_growth_screen(
        fundamentals_aligned=fundamentals,
        prices=close,
        volume=volume,
    )
    scores = build_growth_scores(
        prices=close,
        fundamentals_aligned=fundamentals,
        screen_mask=mask,
    )
    sim, weights, _ = run_growth_leader_backtest(
        score_panel=scores,
        close_prices=close,
        top_n=2,
        rebalance="weekly",
        weighting="equal",
        costs_bps=10.0,
    )
    invested = weights.sum(axis=1) > 0.0
    assert invested.any()
    assert np.allclose(weights.sum(axis=1).loc[invested].to_numpy(), 1.0, atol=1e-10)
    assert {"Equity", "DailyReturn", "Turnover"}.issubset(set(sim.columns))


def test_end_to_end_runs() -> None:
    close, volume, fundamentals = _sample_data(periods=100)
    mask = apply_growth_screen(
        fundamentals_aligned=fundamentals,
        prices=close,
        volume=volume,
    )
    scores = build_growth_scores(
        prices=close,
        fundamentals_aligned=fundamentals,
        screen_mask=mask,
    )
    sim, weights, summary = run_growth_leader_backtest(
        score_panel=scores,
        close_prices=close,
        top_n=3,
        rebalance="weekly",
        weighting="inv_vol",
        costs_bps=10.0,
    )
    assert len(sim) == len(close)
    assert len(weights) == len(close)
    for k in ["CAGR", "Vol", "Sharpe", "MaxDD", "AnnualTurnover"]:
        assert k in summary
