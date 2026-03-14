"""Research helpers for a simple ETF sector-rotation sleeve."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from quant_lab.engine.metrics import compute_daily_mark_to_market, compute_metrics
from quant_lab.strategies.topn import rebalance_mask


DEFAULT_SECTOR_UNIVERSE: list[str] = [
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]


def _read_close_series(path: Path, symbol: str) -> pd.Series:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path).copy()
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path).copy()
    else:
        raise ValueError(f"Unsupported file type: {path}")

    cols = {str(c).strip().lower(): c for c in frame.columns}
    date_col = cols.get("date")
    close_col = cols.get("close")
    if date_col is None:
        date_col = cols.get("datetime") or cols.get("timestamp")
    if close_col is None:
        close_col = cols.get("adj close") or cols.get("adj_close")
    if date_col is None or close_col is None:
        raise ValueError(f"Missing date/close columns in {path}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_col], errors="coerce"),
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    if out.empty:
        raise ValueError(f"No valid rows after parsing {path}")
    return pd.Series(
        out["close"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(out["date"]),
        name=symbol,
        dtype=float,
    )


def load_sector_prices(
    universe: Sequence[str] | None = None,
    data_roots: Sequence[str | Path] = ("data/sector_etfs", "data/cross_asset"),
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Load local close-price panel for SPDR sector ETFs.

    The loader tries each symbol across provided roots and supports either
    parquet or CSV files. Missing symbols are skipped gracefully.
    """
    symbols = [str(x).strip().upper() for x in (universe or DEFAULT_SECTOR_UNIVERSE) if str(x).strip()]
    roots = [Path(r) for r in data_roots]
    series_map: dict[str, pd.Series] = {}
    for sym in symbols:
        found_path: Path | None = None
        for root in roots:
            for cand in (
                root / f"{sym}.parquet",
                root / f"{sym}.csv",
                root / f"{sym}.US.parquet",
                root / f"{sym}.US.csv",
                root / f"{sym.lower()}.us.csv",
            ):
                if cand.exists():
                    found_path = cand
                    break
            if found_path is not None:
                break
        if found_path is None:
            continue
        try:
            series_map[sym] = _read_close_series(path=found_path, symbol=sym)
        except Exception:
            continue

    if not series_map:
        return pd.DataFrame(columns=symbols, dtype=float)

    close = pd.concat(series_map.values(), axis=1, join="outer").sort_index()
    close.columns = list(series_map.keys())
    if start is not None:
        close = close.loc[close.index >= pd.Timestamp(start)]
    if end is not None:
        close = close.loc[close.index <= pd.Timestamp(end)]
    return close.astype(float)


def compute_sector_momentum_signals(
    close: pd.DataFrame,
    lookback: int,
    signal_type: str = "absolute",
) -> pd.DataFrame:
    """
    Compute lag-safe sector momentum scores.

    For both `relative` and `absolute` v1, score is trailing lookback total return.
    The series is shifted by one day to keep downstream portfolio construction causal.
    """
    if int(lookback) <= 0:
        raise ValueError("lookback must be > 0")
    mode = str(signal_type).lower().strip()
    if mode not in {"relative", "absolute"}:
        raise ValueError("signal_type must be 'relative' or 'absolute'")
    c = close.astype(float).sort_index()
    return c.pct_change(int(lookback)).shift(1).astype(float)


def build_sector_rotation_weights(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    top_n: int,
    rebalance: str = "monthly",
    weighting: str = "equal",
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """Build long-only sector rotation weights, held constant between rebalances."""
    if int(top_n) <= 0:
        raise ValueError("top_n must be > 0")
    if int(vol_lookback) <= 0:
        raise ValueError("vol_lookback must be > 0")
    method = str(weighting).lower().strip()
    if method not in {"equal", "inv_vol"}:
        raise ValueError("weighting must be 'equal' or 'inv_vol'")

    s = scores.astype(float).sort_index()
    px = close.astype(float).reindex(index=s.index, columns=s.columns)
    vol = px.pct_change().rolling(int(vol_lookback)).std(ddof=0).shift(1)
    rb = rebalance_mask(pd.DatetimeIndex(s.index), rebalance)
    w = pd.DataFrame(0.0, index=s.index, columns=s.columns, dtype=float)
    current = pd.Series(0.0, index=s.columns, dtype=float)

    for dt in s.index:
        if bool(rb.loc[dt]):
            row = s.loc[dt].dropna().sort_values(ascending=False, kind="mergesort")
            selected = row.index.tolist()[: int(top_n)]
            nxt = pd.Series(0.0, index=s.columns, dtype=float)
            if selected:
                if method == "equal":
                    nxt.loc[selected] = 1.0 / float(len(selected))
                else:
                    vol_row = pd.to_numeric(vol.loc[dt, selected], errors="coerce")
                    valid = vol_row[(vol_row > 0.0) & vol_row.notna()]
                    if valid.empty:
                        nxt.loc[selected] = 1.0 / float(len(selected))
                    else:
                        inv = 1.0 / valid
                        inv = inv / float(inv.sum())
                        nxt.loc[inv.index] = inv.to_numpy(dtype=float)
            current = nxt
        w.loc[dt] = current
    return w.astype(float)


def annual_turnover(weights: pd.DataFrame, rebalance: str = "monthly") -> float:
    """Approximate annual turnover from rebalance-day average turnover."""
    w = weights.astype(float).sort_index()
    td = 0.5 * w.diff().abs().sum(axis=1).fillna(0.0)
    rb = rebalance_mask(pd.DatetimeIndex(w.index), rebalance)
    trb = td.loc[rb]
    if trb.empty:
        return float("nan")
    per_reb = float(trb.mean())
    reb_per_year = 12.0 if str(rebalance).lower() == "monthly" else 52.0
    if str(rebalance).lower() == "daily":
        reb_per_year = 252.0
    return float(per_reb * reb_per_year)


def run_sector_rotation_backtest(
    close: pd.DataFrame,
    lookback: int,
    signal_type: str,
    top_n: int,
    weighting: str,
    rebalance: str = "monthly",
    costs_bps: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run sector rotation backtest and return simulation frame + summary metrics."""
    scores = compute_sector_momentum_signals(close=close, lookback=int(lookback), signal_type=signal_type)
    weights = build_sector_rotation_weights(
        scores=scores,
        close=close,
        top_n=int(top_n),
        rebalance=rebalance,
        weighting=weighting,
        vol_lookback=20,
    )
    rb = rebalance_mask(pd.DatetimeIndex(weights.index), rebalance)
    rb_dates = pd.DatetimeIndex(weights.index[rb])
    equity, daily_ret, weights_daily = compute_daily_mark_to_market(
        close=close.astype(float).reindex(index=weights.index, columns=weights.columns),
        weights_rebal=weights,
        rebalance_dates=rb_dates,
        costs_bps=float(costs_bps),
        slippage_bps=0.0,
    )
    turnover = (0.5 * weights_daily.diff().abs().sum(axis=1)).fillna(0.0).rename("Turnover")
    sim = pd.concat([equity.rename("Equity"), daily_ret.rename("DailyReturn"), turnover], axis=1)

    m = compute_metrics(sim["DailyReturn"])
    summary = {
        "CAGR": float(m.get("CAGR", np.nan)),
        "Vol": float(m.get("Vol", np.nan)),
        "Sharpe": float(m.get("Sharpe", np.nan)),
        "MaxDD": float(m.get("MaxDD", np.nan)),
        "AnnualTurnover": float(annual_turnover(weights_daily, rebalance=rebalance)),
        "AvgAssetsHeld": float((weights_daily > 0.0).sum(axis=1).mean()),
        "Lookback": float(lookback),
        "TopN": float(top_n),
        "CostsBps": float(costs_bps),
    }
    return sim.astype(float), summary
