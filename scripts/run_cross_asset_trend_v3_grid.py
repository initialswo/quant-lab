"""Research grid for Cross-Asset Trend v3."""

from __future__ import annotations

import itertools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_daily_mark_to_market, compute_metrics
from quant_lab.research.cross_asset_trend import annual_turnover
from quant_lab.strategies.topn import rebalance_mask


ASSETS: list[str] = [
    "SPY",
    "TLT",
    "GLD",
    "DBC",
    "VNQ",
    "SHY",
    "EFA",
    "EEM",
    "IEF",
    "LQD",
    "TIP",
    "UUP",
    "FXE",
]
LOOKBACKS: list[int] = [63, 126, 252]
SIGNAL_TYPES: list[str] = ["dual", "relative_only"]
TOP_N_VALUES: list[int] = [1, 2, 3, 4]
WEIGHTING_METHODS: list[str] = ["equal", "inverse_vol"]
REBALANCES: list[str] = ["weekly", "monthly"]
ABS_MOM_FILTER_VALUES: list[bool] = [True, False]
VOL_TARGETS: list[float | None] = [None, 0.10, 0.15]
START = "2005-01-01"
END = "2024-12-31"
SIGNAL_SKIP_DAYS = 21
SIGNAL_LAG_DAYS = 1
SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]


def _load_close_panel(root: str = "data/cross_asset") -> pd.DataFrame:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"cross-asset directory missing: {root}")
    panel: dict[str, pd.Series] = {}
    for sym in ASSETS:
        p = base / f"{sym}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing cross-asset parquet: {p}")
        df = pd.read_parquet(p, columns=["date", "close"]).copy()
        if df.empty:
            raise ValueError(f"empty cross-asset dataset: {p}")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
        if df.empty:
            raise ValueError(f"no valid rows after parsing: {p}")
        panel[sym] = pd.Series(df["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name=sym)
    close = pd.concat(panel.values(), axis=1, join="outer").sort_index()
    close.columns = list(panel.keys())
    close = close.loc[(close.index >= pd.Timestamp(START)) & (close.index <= pd.Timestamp(END))]
    if close.empty:
        raise ValueError("No usable close panel for selected window.")
    return close.astype(float)


def _raw_momentum_signal(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if int(lookback) <= SIGNAL_SKIP_DAYS:
        raise ValueError("lookback must be > SIGNAL_SKIP_DAYS")
    c = close.astype(float).sort_index()
    return c.shift(SIGNAL_SKIP_DAYS) / c.shift(int(lookback)) - 1.0


def _build_score_signal(raw_signal: pd.DataFrame, signal_type: str) -> pd.DataFrame:
    mode = str(signal_type).strip().lower()
    if mode == "relative_only":
        return raw_signal.astype(float)
    if mode == "dual":
        if "SHY" not in raw_signal.columns:
            raise ValueError("SHY is required for dual signal.")
        shy = raw_signal["SHY"]
        return raw_signal.sub(shy, axis=0).astype(float)
    raise ValueError(f"unknown signal_type: {signal_type}")


def _inverse_vol_weights(vol_row: pd.Series, selected: list[str], budget: float) -> pd.Series:
    out = pd.Series(0.0, index=vol_row.index, dtype=float)
    if not selected or float(budget) <= 0.0:
        return out
    vol_sel = pd.to_numeric(vol_row.reindex(selected), errors="coerce")
    valid = vol_sel[(vol_sel > 0.0) & vol_sel.notna()]
    if valid.empty:
        out.loc[selected] = float(budget) / float(len(selected))
        return out
    inv = 1.0 / valid
    inv = inv / float(inv.sum())
    out.loc[inv.index] = inv.to_numpy(dtype=float) * float(budget)
    return out


def _build_weights(
    raw_signal: pd.DataFrame,
    rank_signal: pd.DataFrame,
    vol: pd.DataFrame,
    top_n: int,
    weighting: str,
    rebalance: str,
    absolute_momentum_filter: bool,
) -> pd.DataFrame:
    if "SHY" not in raw_signal.columns:
        raise ValueError("SHY is required for filter fallback.")
    raw = raw_signal.astype(float).sort_index()
    rank_sig = rank_signal.astype(float).reindex(index=raw.index, columns=raw.columns)
    vol = vol.astype(float).reindex(index=raw.index, columns=raw.columns)
    raw_lag = raw.shift(SIGNAL_LAG_DAYS)
    rank_lag = rank_sig.shift(SIGNAL_LAG_DAYS)
    rb = rebalance_mask(pd.DatetimeIndex(raw.index), rebalance)
    w = pd.DataFrame(0.0, index=raw.index, columns=raw.columns, dtype=float)
    current = pd.Series(0.0, index=raw.columns, dtype=float)

    for dt in raw.index:
        if bool(rb.loc[dt]):
            rank_row = rank_lag.loc[dt]
            ranked = rank_row.dropna().sort_values(ascending=False)
            selected = ranked.index.tolist()[: int(top_n)]
            if bool(absolute_momentum_filter):
                abs_row = raw_lag.loc[dt]
                selected = [a for a in selected if bool(pd.notna(abs_row.get(a))) and float(abs_row.get(a)) > 0.0]
            missing_slots = max(0, int(top_n) - len(selected))
            shy_floor = float(missing_slots) / float(top_n)

            nxt = pd.Series(0.0, index=raw.columns, dtype=float)
            if str(weighting).lower() == "equal":
                if selected:
                    nxt.loc[selected] += 1.0 / float(top_n)
                if missing_slots > 0:
                    nxt.loc["SHY"] += shy_floor
            elif str(weighting).lower() == "inverse_vol":
                remaining = 1.0 - shy_floor
                nxt = _inverse_vol_weights(vol_row=vol.loc[dt], selected=selected, budget=remaining)
                if missing_slots > 0:
                    nxt.loc["SHY"] += shy_floor
            else:
                raise ValueError(f"unknown weighting: {weighting}")
            current = nxt
        w.loc[dt] = current
    return w.astype(float)


def _compute_stability(daily_returns: pd.Series) -> float:
    sharpes: list[float] = []
    r = daily_returns.astype(float)
    for start, end in SUBPERIODS:
        s = r.loc[(r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]
        m = compute_metrics(s)
        sharpes.append(float(m.get("Sharpe", float("nan"))))
    sharpe_series = pd.Series(sharpes, dtype=float)
    return float(sharpe_series.mean() - 0.5 * sharpe_series.std(ddof=0))


def _apply_vol_target(daily_ret: pd.Series, vol_target: float | None) -> tuple[pd.Series, pd.Series]:
    r = daily_ret.astype(float)
    if vol_target is None:
        lev = pd.Series(1.0, index=r.index, dtype=float)
        return r, lev
    rv63 = r.rolling(63).std(ddof=0) * (252.0**0.5)
    lev_raw = float(vol_target) / rv63.replace(0.0, pd.NA)
    lev = pd.to_numeric(lev_raw, errors="coerce").clip(lower=0.25, upper=2.0).shift(1).fillna(1.0)
    return (r * lev).astype(float), lev.astype(float)


def main() -> None:
    close = _load_close_panel(root="data/cross_asset")
    vol20 = close.pct_change().rolling(20).std(ddof=0).shift(1)

    combos = list(
        itertools.product(
            LOOKBACKS,
            SIGNAL_TYPES,
            TOP_N_VALUES,
            WEIGHTING_METHODS,
            REBALANCES,
            ABS_MOM_FILTER_VALUES,
            VOL_TARGETS,
        )
    )
    rows: list[dict[str, Any]] = []
    for lookback, signal_type, top_n, weighting, rebalance, abs_filter, vol_target in combos:
        raw_sig = _raw_momentum_signal(close=close, lookback=int(lookback))
        rank_sig = _build_score_signal(raw_signal=raw_sig, signal_type=str(signal_type))
        w_rebal = _build_weights(
            raw_signal=raw_sig,
            rank_signal=rank_sig,
            vol=vol20,
            top_n=int(top_n),
            weighting=str(weighting),
            rebalance=str(rebalance),
            absolute_momentum_filter=bool(abs_filter),
        )
        rb = rebalance_mask(pd.DatetimeIndex(w_rebal.index), str(rebalance))
        rb_dates = pd.DatetimeIndex(w_rebal.index[rb])
        _, daily_ret, weights_daily = compute_daily_mark_to_market(
            close=close,
            weights_rebal=w_rebal,
            rebalance_dates=rb_dates,
            costs_bps=0.0,
            slippage_bps=0.0,
        )
        daily_ret_vt, lev = _apply_vol_target(daily_ret, vol_target=vol_target)
        m = compute_metrics(daily_ret_vt)
        row = {
            "VariantName": (
                f"lb{lookback}_{signal_type}_top{top_n}_{weighting}_{rebalance}"
                f"_abs{int(bool(abs_filter))}_vt{('none' if vol_target is None else f'{float(vol_target):.2f}')}"
            ),
            "lookback": int(lookback),
            "signal_type": str(signal_type),
            "top_n": int(top_n),
            "weighting": str(weighting),
            "rebalance": str(rebalance),
            "absolute_momentum_filter": bool(abs_filter),
            "vol_target": "None" if vol_target is None else float(vol_target),
            "CAGR": float(m.get("CAGR", float("nan"))),
            "Vol": float(m.get("Vol", float("nan"))),
            "Sharpe": float(m.get("Sharpe", float("nan"))),
            "MaxDD": float(m.get("MaxDD", float("nan"))),
            "AnnualTurnover": float(annual_turnover(weights_daily, rebalance=str(rebalance))),
            "StabilityScore": float(_compute_stability(daily_ret_vt)),
            "AvgLeverage": float(lev.mean()),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No variants were evaluated.")

    df = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(
        drop=True
    )
    top_sharpe = df.head(10).copy()
    top_cagr = df.sort_values(["CAGR", "Sharpe"], ascending=[False, False], kind="mergesort").head(10).copy()

    best_by_sharpe = df.iloc[0]
    best_by_cagr = df.sort_values(["CAGR", "Sharpe"], ascending=[False, False], kind="mergesort").iloc[0]
    best_by_maxdd = df.sort_values(["MaxDD", "Sharpe"], ascending=[False, False], kind="mergesort").iloc[0]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path("results/cross_asset_trend_v3_grid")
    outdir = root / ts
    outdir.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    main_csv = outdir / "cross_asset_trend_v3_grid.csv"
    top_sharpe_csv = outdir / "top_variants_by_sharpe.csv"
    top_cagr_csv = outdir / "top_variants_by_cagr.csv"

    df.to_csv(main_csv, index=False, float_format="%.10g")
    top_sharpe.to_csv(top_sharpe_csv, index=False, float_format="%.10g")
    top_cagr.to_csv(top_cagr_csv, index=False, float_format="%.10g")

    df.to_csv(Path("results/cross_asset_trend_v3_grid_latest.csv"), index=False, float_format="%.10g")

    print("CROSS-ASSET TREND V3 GRID")
    print("-------------------------")
    print(f"Variants: {len(df)}")
    print("\nTop 10 by Sharpe:")
    print(top_sharpe.to_string(index=False))
    print("\nBest by Sharpe:")
    print(pd.DataFrame([best_by_sharpe]).to_string(index=False))
    print("\nBest by CAGR:")
    print(pd.DataFrame([best_by_cagr]).to_string(index=False))
    print("\nBest by Least Severe Drawdown:")
    print(pd.DataFrame([best_by_maxdd]).to_string(index=False))
    print(f"\nSaved: {main_csv}")
    print(f"Saved: {top_sharpe_csv}")
    print(f"Saved: {top_cagr_csv}")
    print("Saved: results/cross_asset_trend_v3_grid_latest.csv")


if __name__ == "__main__":
    main()
