"""Research-only experiment grid for Cross-Asset Trend v2 configurations."""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_daily_mark_to_market, compute_metrics
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
SIGNAL_TYPES: list[str] = ["absolute", "dual"]
TOP_N_VALUES: list[int] = [2, 3, 4]
WEIGHTING_METHODS: list[str] = ["equal", "inverse_vol"]
START = "2005-01-01"
END = "2024-12-31"
REBALANCE = "monthly"
SIGNAL_SKIP_DAYS = 21
SIGNAL_LAG_DAYS = 1
SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]


def _load_close_panel(root: str = "data/cross_asset") -> pd.DataFrame:
    r = Path(root)
    if not r.exists():
        raise FileNotFoundError(f"cross-asset directory missing: {root}")
    close_map: dict[str, pd.Series] = {}
    for sym in ASSETS:
        p = r / f"{sym}.parquet"
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
        close_map[sym] = pd.Series(df["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name=sym)
    close = pd.concat(close_map.values(), axis=1, join="outer").sort_index()
    close.columns = list(close_map.keys())
    close = close.loc[(close.index >= pd.Timestamp(START)) & (close.index <= pd.Timestamp(END))]
    if close.empty:
        raise ValueError("No usable cross-asset price history for selected window.")
    return close.astype(float)


def _compute_signal(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if int(lookback) <= SIGNAL_SKIP_DAYS:
        raise ValueError("lookback must be greater than 21")
    c = close.astype(float).sort_index()
    return c.shift(SIGNAL_SKIP_DAYS) / c.shift(int(lookback)) - 1.0


def _inverse_vol_weights(vol_row: pd.Series, selected: list[str]) -> pd.Series:
    out = pd.Series(0.0, index=vol_row.index, dtype=float)
    if not selected:
        return out
    vol_sel = pd.to_numeric(vol_row.reindex(selected), errors="coerce")
    valid = vol_sel[(vol_sel > 0.0) & vol_sel.notna()]
    if valid.empty:
        out.loc[selected] = 1.0 / float(len(selected))
        return out
    inv = 1.0 / valid
    inv = inv / float(inv.sum())
    out.loc[inv.index] = inv.to_numpy(dtype=float)
    return out


def _build_weights(
    signal: pd.DataFrame,
    vol: pd.DataFrame,
    signal_type: str,
    top_n: int,
    weighting: str,
) -> pd.DataFrame:
    if "SHY" not in signal.columns:
        raise ValueError("SHY is required for absolute-momentum fallback.")
    sig = signal.astype(float).sort_index()
    vol = vol.astype(float).reindex(sig.index)
    sig_lag = sig.shift(SIGNAL_LAG_DAYS)
    rb = rebalance_mask(pd.DatetimeIndex(sig.index), REBALANCE)
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns, dtype=float)
    current = pd.Series(0.0, index=sig.columns, dtype=float)

    for dt in sig.index:
        if bool(rb.loc[dt]):
            row = sig_lag.loc[dt]
            ranked = row.dropna().sort_values(ascending=False)
            if signal_type == "absolute":
                selected = ranked[ranked > 0.0].index.tolist()[: int(top_n)]
                if not selected:
                    selected = ["SHY"]
            elif signal_type == "dual":
                selected = ranked.index.tolist()[: int(top_n)]
            else:
                raise ValueError(f"unknown signal_type: {signal_type}")

            if weighting == "equal":
                nxt = pd.Series(0.0, index=sig.columns, dtype=float)
                if selected:
                    nxt.loc[selected] = 1.0 / float(len(selected))
            elif weighting == "inverse_vol":
                nxt = _inverse_vol_weights(vol_row=vol.loc[dt], selected=selected)
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


def main() -> None:
    close = _load_close_panel(root="data/cross_asset")
    returns = close.pct_change()
    vol20 = returns.rolling(20).std(ddof=0).shift(1)

    combos = list(itertools.product(LOOKBACKS, SIGNAL_TYPES, TOP_N_VALUES, WEIGHTING_METHODS))
    rows: list[dict[str, Any]] = []
    best_variant_payload: dict[str, Any] | None = None
    best_returns: pd.Series | None = None

    for lookback, signal_type, top_n, weighting in combos:
        signal = _compute_signal(close=close, lookback=int(lookback))
        weights_rebal = _build_weights(
            signal=signal,
            vol=vol20,
            signal_type=str(signal_type),
            top_n=int(top_n),
            weighting=str(weighting),
        )
        rb = rebalance_mask(pd.DatetimeIndex(weights_rebal.index), REBALANCE)
        rb_dates = pd.DatetimeIndex(weights_rebal.index[rb])
        _, daily_ret, weights_daily = compute_daily_mark_to_market(
            close=close,
            weights_rebal=weights_rebal,
            rebalance_dates=rb_dates,
            costs_bps=0.0,
            slippage_bps=0.0,
        )
        m = compute_metrics(daily_ret)
        stability = _compute_stability(daily_ret)
        variant_name = f"lb{lookback}_{signal_type}_top{top_n}_{weighting}"
        row = {
            "VariantName": variant_name,
            "lookback": int(lookback),
            "signal_type": str(signal_type),
            "top_n": int(top_n),
            "weighting": str(weighting),
            "CAGR": float(m.get("CAGR", float("nan"))),
            "Vol": float(m.get("Vol", float("nan"))),
            "Sharpe": float(m.get("Sharpe", float("nan"))),
            "MaxDD": float(m.get("MaxDD", float("nan"))),
            "StabilityScore": float(stability),
        }
        rows.append(row)

        if best_variant_payload is None:
            best_variant_payload = {
                "variant": row,
                "weights_daily": weights_daily.copy(),
                "signal": signal.copy(),
            }
            best_returns = daily_ret.copy()
        else:
            curr_stability = float(row["StabilityScore"])
            best_stability = float(best_variant_payload["variant"]["StabilityScore"])
            curr_sharpe = float(row["Sharpe"])
            best_sharpe = float(best_variant_payload["variant"]["Sharpe"])
            if (curr_stability > best_stability) or (
                curr_stability == best_stability and curr_sharpe > best_sharpe
            ):
                best_variant_payload = {
                    "variant": row,
                    "weights_daily": weights_daily.copy(),
                    "signal": signal.copy(),
                }
                best_returns = daily_ret.copy()

    if best_variant_payload is None or best_returns is None:
        raise RuntimeError("No variants were evaluated.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["StabilityScore", "Sharpe"], ascending=[False, False]).reset_index(drop=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)

    out_csv = results_root / f"cross_asset_trend_v2_grid_{ts}.csv"
    out_dir = results_root / f"cross_asset_trend_v2_grid_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False, float_format="%.10g")

    best_variant = dict(best_variant_payload["variant"])
    combined = pd.DataFrame(
        {
            "date": best_returns.index,
            "returns": best_returns.to_numpy(dtype=float),
            "equity": (1.0 + best_returns.astype(float).fillna(0.0)).cumprod().to_numpy(dtype=float),
            "assets_held": (best_variant_payload["weights_daily"] > 0.0).sum(axis=1).to_numpy(dtype=int),
        }
    )
    combined.to_csv(out_dir / "combined_returns.csv", index=False, float_format="%.10g")

    summary_payload = {
        "grid": {
            "lookbacks": LOOKBACKS,
            "signal_types": SIGNAL_TYPES,
            "top_n_values": TOP_N_VALUES,
            "weighting_methods": WEIGHTING_METHODS,
            "rebalance": REBALANCE,
            "signal_formula": "close[t-21] / close[t-lookback] - 1",
            "signal_lag_days": SIGNAL_LAG_DAYS,
            "fallback_rule_absolute": "if zero positive assets, allocate 100% SHY",
            "assets": ASSETS,
            "variant_count": int(len(rows)),
        },
        "best_variant": best_variant,
        "artifacts": {
            "grid_results_csv": str(out_csv),
            "combined_returns": str(out_dir / "combined_returns.csv"),
            "parameter_summary": str(out_dir / "parameter_summary.json"),
        },
    }
    (out_dir / "parameter_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    latest_csv = results_root / "cross_asset_trend_v2_grid_latest.csv"
    latest_dir = results_root / "cross_asset_trend_v2_grid_latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(latest_csv, index=False, float_format="%.10g")
    combined.to_csv(latest_dir / "combined_returns.csv", index=False, float_format="%.10g")
    (latest_dir / "parameter_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("CROSS-ASSET TREND V2 GRID")
    print("-------------------------")
    print(f"Variants: {len(rows)}")
    print("\nTop 10 variants by StabilityScore:")
    print(df.head(10).to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_dir / 'combined_returns.csv'}")
    print(f"Saved: {out_dir / 'parameter_summary.json'}")


if __name__ == "__main__":
    main()
