"""Measure the interaction between profitability and short-term reversal."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.engine import runner
from quant_lab.factors.registry import compute_factors
from quant_lab.research.cs_factor_diagnostics import compute_forward_returns
from quant_lab.strategies.topn import rebalance_mask
from quant_lab.universe.liquid_us import build_liquid_us_universe


START = "2000-01-01"
END = "2024-12-31"
RESULTS_ROOT = Path("results") / "factor_interaction_heatmap"
DATA_CACHE_DIR = "data/equities"
FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
QUANTILES = 5
FORWARD_HORIZON = 21
MAX_TICKERS = 2000


def _format_percent(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x) * 100.0:.2f}%"


def _load_panels(start: str, end: str, max_tickers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = runner._load_universe_seed_tickers(
        universe="liquid_us",
        max_tickers=int(max_tickers),
        data_cache_dir=DATA_CACHE_DIR,
    )
    ohlcv_map, _ = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(start),
        end=str(end),
        cache_dir=DATA_CACHE_DIR,
        data_source="parquet",
        refresh=False,
        bulk_prepare=False,
    )
    close_cols, used_tickers, missing_tickers, rejected_tickers, _ = runner._collect_close_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=tickers,
    )
    if not close_cols:
        raise ValueError(
            "No valid OHLCV data found. "
            f"missing={len(missing_tickers)} rejected={len(rejected_tickers)}"
        )
    close = pd.concat(close_cols, axis=1, join="outer")
    close = runner._prepare_close_panel(close_raw=close, price_fill_mode="ffill")
    close = close.sort_index()
    volume = runner._collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Volume",
    ).reindex(index=close.index, columns=close.columns)
    return close.astype(float), volume.astype(float)


def _compute_factor_scores(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    factor_params = runner._augment_factor_params_with_fundamentals(
        factor_names=["reversal_1m", "gross_profitability"],
        factor_params_map={},
        close=close,
        fundamentals_path=FUNDAMENTALS_PATH,
        fundamentals_fallback_lag_days=60,
    )
    return compute_factors(
        factor_names=["reversal_1m", "gross_profitability"],
        close=close,
        factor_params=factor_params,
    )


def _compute_interaction_matrix(
    reversal: pd.DataFrame,
    profitability: pd.DataFrame,
    forward_returns: pd.DataFrame,
    eligibility: pd.DataFrame,
    rebalance: str = "weekly",
) -> pd.DataFrame:
    labels = [f"Q{i}" for i in range(1, QUANTILES + 1)]
    rb = rebalance_mask(pd.DatetimeIndex(reversal.index), rebalance)
    rb_dates = pd.DatetimeIndex(reversal.index[rb])

    by_date: list[pd.DataFrame] = []
    for dt in rb_dates:
        rev_row = reversal.loc[dt]
        prof_row = profitability.loc[dt]
        fwd_row = forward_returns.loc[dt]
        elig_row = eligibility.loc[dt]
        mask = rev_row.notna() & prof_row.notna() & fwd_row.notna() & elig_row.fillna(False).astype(bool)
        if int(mask.sum()) < QUANTILES * QUANTILES:
            continue
        rev_valid = rev_row[mask]
        prof_valid = prof_row[mask]
        fwd_valid = fwd_row[mask]
        try:
            rev_bins = pd.qcut(rev_valid.rank(method="first"), q=QUANTILES, labels=labels)
            prof_bins = pd.qcut(prof_valid.rank(method="first"), q=QUANTILES, labels=labels)
        except ValueError:
            continue
        cell = (
            pd.DataFrame(
                {
                    "profitability_q": prof_bins.astype(str),
                    "reversal_q": rev_bins.astype(str),
                    "fwd_ret": fwd_valid.astype(float),
                }
            )
            .groupby(["profitability_q", "reversal_q"], observed=False)["fwd_ret"]
            .mean()
            .unstack("reversal_q")
            .reindex(index=labels, columns=labels)
        )
        by_date.append(cell.astype(float))

    if not by_date:
        return pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    stacked = np.stack([frame.to_numpy(dtype=float) for frame in by_date], axis=0)
    return pd.DataFrame(
        np.nanmean(stacked, axis=0),
        index=labels,
        columns=labels,
        dtype=float,
    )


def _write_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), labels=list(matrix.columns))
    ax.set_yticks(range(matrix.shape[0]), labels=list(matrix.index))
    ax.set_xlabel("Reversal")
    ax.set_ylabel("Profitability")
    ax.set_title("Profitability / Reversal Interaction")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, _format_percent(data[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Avg next-month return")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--max_tickers", type=int, default=MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    results_root = Path(str(args.results_root))
    results_root.mkdir(parents=True, exist_ok=True)

    close, volume = _load_panels(
        start=str(args.start),
        end=str(args.end),
        max_tickers=int(args.max_tickers),
    )
    scores = _compute_factor_scores(close=close)
    eligibility = build_liquid_us_universe(
        prices=close,
        volumes=volume,
        min_price=1.0,
        min_avg_dollar_volume=0.0,
        adv_window=20,
        min_history=300,
    )
    forward_returns = compute_forward_returns(close=close, horizon=FORWARD_HORIZON)
    matrix = _compute_interaction_matrix(
        reversal=scores["reversal_1m"],
        profitability=scores["gross_profitability"],
        forward_returns=forward_returns,
        eligibility=eligibility,
        rebalance="weekly",
    )

    matrix_path = results_root / "interaction_matrix.csv"
    plot_path = results_root / "interaction_heatmap.png"
    matrix.to_csv(matrix_path, float_format="%.10g")
    _write_heatmap(matrix=matrix, path=plot_path)

    display = matrix.copy()
    for col in display.columns:
        display[col] = display[col].map(_format_percent)
    print("Profitability ↓ / Reversal →")
    print("")
    print(display.to_string())
    print("")
    print(f"Saved: {matrix_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
