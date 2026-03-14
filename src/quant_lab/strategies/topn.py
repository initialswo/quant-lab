"""Top-N portfolio strategy."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from quant_lab.engine.metrics import compute_daily_mark_to_market

_WARNED_CAP_IMPOSSIBLE = False


def rebalance_mask(index: pd.DatetimeIndex, rebalance: str) -> pd.Series:
    """Return True on rebalance dates for daily, weekly, biweekly, or monthly schedules."""
    reb = rebalance.lower()
    dates = pd.DatetimeIndex(index)

    if reb == "daily":
        mask = np.ones(len(dates), dtype=bool)
    elif reb == "weekly":
        periods = pd.Series(dates.to_period("W-FRI"), index=dates)
        mask = (periods != periods.shift(1)).fillna(True).to_numpy()
    elif reb == "biweekly":
        periods = pd.Series(dates.to_period("W-FRI"), index=dates)
        week_codes = pd.Series(pd.factorize(periods)[0], index=dates)
        period_start = (periods != periods.shift(1)).fillna(True)
        mask = (period_start & ((week_codes % 2) == 0)).to_numpy()
    elif reb == "monthly":
        periods = pd.Series(dates.to_period("M"), index=dates)
        mask = (periods != periods.shift(1)).fillna(True).to_numpy()
    else:
        raise ValueError(f"Unsupported rebalance frequency: {rebalance}")

    return pd.Series(mask, index=dates)


def _apply_max_weight_cap(weights: pd.Series, max_weight: float, eps: float) -> pd.Series:
    """Cap single-name weights and redistribute to uncapped names."""
    global _WARNED_CAP_IMPOSSIBLE

    w = weights.copy().astype(float)
    k = int((w > 0.0).sum())
    if k == 0:
        return w
    if k * max_weight < 1.0 - eps:
        if not _WARNED_CAP_IMPOSSIBLE:
            warnings.warn(
                f"Skipping max_weight cap: selected_count*max_weight={k*max_weight:.4f} < 1.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_CAP_IMPOSSIBLE = True
        return w
    if (w <= max_weight + eps).all():
        return w

    uncapped = w.index.tolist()
    fixed = pd.Series(0.0, index=w.index, dtype=float)
    remaining = 1.0

    while True:
        if not uncapped:
            break
        subtotal = float(w.loc[uncapped].sum())
        if subtotal <= eps:
            fixed.loc[uncapped] = remaining / len(uncapped)
            break

        scaled = w.loc[uncapped] * (remaining / subtotal)
        over = scaled[scaled > max_weight + eps]
        if over.empty:
            fixed.loc[uncapped] = scaled
            break

        for name in over.index:
            fixed.loc[name] = max_weight
            remaining -= max_weight
            uncapped.remove(name)
        if remaining <= eps:
            break

    total = float(fixed.sum())
    if total > eps:
        fixed = fixed / total
    return fixed.clip(lower=0.0)


def _apply_sector_caps(
    weights: pd.Series,
    sector_by_ticker: dict[str, str] | None,
    sector_cap: float,
    eps: float,
    max_iters: int = 20,
) -> pd.Series:
    """Iteratively trim sector overweights and redistribute to uncapped sectors."""
    if sector_cap <= 0.0 or sector_by_ticker is None:
        return weights

    w = weights.copy().astype(float).clip(lower=0.0)
    if float(w.sum()) <= eps:
        return w

    sectors = pd.Series({t: sector_by_ticker.get(t, "UNKNOWN") for t in w.index}, dtype=object)

    for _ in range(max_iters):
        sec_w = w.groupby(sectors).sum()
        over = sec_w[sec_w > sector_cap + eps]
        if over.empty:
            break

        for sec, sec_sum in over.items():
            members = sectors[sectors == sec].index
            if sec_sum > eps:
                scale = sector_cap / float(sec_sum)
                w.loc[members] = w.loc[members] * scale

        total = float(w.sum())
        if total >= 1.0 - eps:
            w = w / total
            continue

        free = 1.0 - total
        sec_w = w.groupby(sectors).sum()
        headroom = (sector_cap - sec_w).clip(lower=0.0)
        eligible_secs = headroom[headroom > eps].index
        if len(eligible_secs) == 0:
            break

        eligible = sectors[sectors.isin(eligible_secs)].index
        if len(eligible) == 0:
            break

        base = w.loc[eligible].copy()
        if float(base.sum()) <= eps:
            base[:] = 1.0 / len(base)
        else:
            base = base / float(base.sum())

        w.loc[eligible] = w.loc[eligible] + free * base
        total = float(w.sum())
        if total > eps:
            w = w / total

    total = float(w.sum())
    if total > eps:
        w = w / total
    return w.clip(lower=0.0)


def _sector_neutral_rank_row(
    row: pd.Series,
    sector_by_ticker: dict[str, str] | None,
) -> pd.Series:
    """Convert one cross-section into within-sector percentile ranks."""
    if sector_by_ticker is None:
        return row
    out = pd.Series(np.nan, index=row.index, dtype=float)
    finite_mask = np.isfinite(row.to_numpy(dtype=float))
    valid = row.iloc[finite_mask].astype(float)
    if valid.empty:
        return out
    sector_series = pd.Series(
        {ticker: str(sector_by_ticker.get(ticker, "UNKNOWN")) for ticker in valid.index},
        dtype=object,
    )
    for sector_name, members in sector_series.groupby(sector_series):
        idx = members.index
        out.loc[idx] = valid.loc[idx].rank(method="average", pct=True).astype(float)
    return out


def build_topn_weights(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    top_n: int,
    rebalance: str,
    weighting: str = "equal",
    vol_lookback: int = 20,
    min_vol: float = 1e-6,
    score_clip: float = 5.0,
    score_floor: float = 0.0,
    eps: float = 1e-12,
    max_weight: float = 0.15,
    sector_cap: float = 0.0,
    sector_by_ticker: dict[str, str] | None = None,
    sector_neutral: bool = False,
    rank_buffer: int = 0,
    volatility_scaled_weights: bool = False,
) -> pd.DataFrame:
    """Build Top-N portfolio weights and hold between rebalance dates."""
    weighting = weighting.lower()
    if top_n <= 0:
        raise ValueError("top_n must be > 0")
    if weighting not in {"equal", "inv_vol", "score", "score_inv_vol"}:
        raise ValueError("weighting must be one of: equal, inv_vol, score, score_inv_vol")
    if vol_lookback <= 0:
        raise ValueError("vol_lookback must be > 0")
    if min_vol <= 0:
        raise ValueError("min_vol must be > 0")
    if score_clip <= 0:
        raise ValueError("score_clip must be > 0")
    if score_floor < 0:
        raise ValueError("score_floor must be >= 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if max_weight <= 0:
        raise ValueError("max_weight must be > 0")
    if int(rank_buffer) < 0:
        raise ValueError("rank_buffer must be >= 0")

    scores = scores.astype(float)
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    rmask = rebalance_mask(scores.index, rebalance)
    close = close.astype(float).reindex(index=scores.index, columns=scores.columns)
    rets = close.pct_change()
    vol = rets.rolling(vol_lookback).std().shift(1)
    rank_scores = scores.shift(1)

    current = pd.Series(0.0, index=scores.columns)
    for dt in scores.index:
        if bool(rmask.loc[dt]):
            row_base = rank_scores.loc[dt] if weighting in {"score", "score_inv_vol"} else scores.loc[dt]
            row = (
                _sector_neutral_rank_row(row_base, sector_by_ticker=sector_by_ticker)
                if bool(sector_neutral)
                else row_base
            )
            finite_mask = np.isfinite(row.to_numpy())
            valid = row.iloc[finite_mask]
            if valid.empty:
                if current.sum() == 0.0:
                    current = pd.Series(0.0, index=scores.columns)
            else:
                ordered = valid.sort_values(ascending=False, kind="mergesort")
                top_buy = ordered.iloc[:top_n].index
                if int(rank_buffer) > 0:
                    hold_cut = int(top_n + int(rank_buffer))
                    hold_band = set(ordered.iloc[:hold_cut].index)
                    existing = set(current[current > 0.0].index)
                    keep_existing = existing.intersection(hold_band)
                    picked = [t for t in ordered.index if (t in set(top_buy) or t in keep_existing)]
                else:
                    picked = list(top_buy)
                new_w = pd.Series(0.0, index=scores.columns)

                if weighting == "equal":
                    if bool(volatility_scaled_weights):
                        score_row = row.loc[picked].astype(float).clip(lower=0.0)
                        vol_row = vol.loc[dt, picked].astype(float).clip(lower=min_vol)
                        ok = np.isfinite(score_row.to_numpy()) & np.isfinite(vol_row.to_numpy())
                        score_ok = score_row.iloc[ok]
                        vol_ok = vol_row.iloc[ok]
                        if score_ok.empty or float(score_ok.sum()) <= eps:
                            base = pd.Series(1.0 / len(picked), index=picked, dtype=float)
                        else:
                            raw = score_ok / vol_ok
                            raw_sum = float(raw.sum())
                            if raw_sum > eps:
                                base = pd.Series(0.0, index=picked, dtype=float)
                                base.loc[raw.index] = raw / raw_sum
                            else:
                                base = pd.Series(1.0 / len(picked), index=picked, dtype=float)
                    else:
                        base = pd.Series(1.0 / len(picked), index=picked, dtype=float)
                elif weighting == "inv_vol":
                    vol_row = vol.loc[dt, picked]
                    if vol_row.isna().any() or (not np.isfinite(vol_row.to_numpy()).all()):
                        base = pd.Series(1.0 / len(picked), index=picked, dtype=float)
                    else:
                        inv = 1.0 / np.maximum(vol_row.to_numpy(dtype=float), float(min_vol))
                        inv_sum = float(inv.sum())
                        if inv_sum <= eps:
                            base = pd.Series(1.0 / len(picked), index=picked, dtype=float)
                        else:
                            base = pd.Series(inv / inv_sum, index=picked, dtype=float)
                else:
                    s = row.loc[picked].astype(float).clip(lower=-score_clip, upper=score_clip)
                    s_pos = s - float(s.min())
                    if score_floor > 0.0:
                        kept = s_pos[s_pos > score_floor]
                        if len(kept) >= 2:
                            picked = kept.index.tolist()
                            s_pos = kept
                        else:
                            s_pos = pd.Series(dtype=float)
                    if float(s_pos.sum()) <= eps:
                        base = pd.Series(1.0 / len(picked), index=picked, dtype=float)
                    else:
                        base = s_pos / float(s_pos.sum())
                    if weighting == "score_inv_vol":
                        vol_row = vol.loc[dt, picked]
                        if vol_row.isna().any() or (not np.isfinite(vol_row.to_numpy()).all()):
                            base = pd.Series(1.0 / len(picked), index=picked, dtype=float)
                        else:
                            base = base / vol_row.astype(float).clip(lower=min_vol)
                            base_sum = float(base.sum())
                            if base_sum > eps:
                                base = base / base_sum
                            else:
                                base = pd.Series(1.0 / len(picked), index=picked, dtype=float)

                base = _apply_max_weight_cap(base, max_weight=max_weight, eps=eps)
                base = _apply_sector_caps(
                    base,
                    sector_by_ticker=sector_by_ticker,
                    sector_cap=sector_cap,
                    eps=eps,
                )
                base_sum = float(base.sum())
                if base_sum > eps:
                    base = base / base_sum
                else:
                    base = pd.Series(1.0 / len(picked), index=picked, dtype=float)

                new_w.loc[picked] = base.reindex(picked).fillna(0.0).to_numpy(dtype=float)
                current = new_w
        weights.loc[dt] = current

    return weights


def build_multi_sleeve_weights(
    sleeve_scores: dict[str, pd.DataFrame],
    sleeve_allocations: dict[str, float],
    sleeve_top_n: dict[str, int],
    close: pd.DataFrame,
    rebalance: str,
    weighting: str = "equal",
    vol_lookback: int = 20,
    min_vol: float = 1e-6,
    score_clip: float = 5.0,
    score_floor: float = 0.0,
    eps: float = 1e-12,
    max_weight: float = 0.15,
    sector_cap: float = 0.0,
    sector_by_ticker: dict[str, str] | None = None,
    sector_neutral: bool = False,
    rank_buffer: int = 0,
    volatility_scaled_weights: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build and aggregate multiple Top-N sleeves into one portfolio."""
    if not sleeve_scores:
        raise ValueError("sleeve_scores must be non-empty")
    score_keys = set(sleeve_scores.keys())
    if score_keys != set(sleeve_allocations.keys()) or score_keys != set(sleeve_top_n.keys()):
        raise ValueError("sleeve_scores/sleeve_allocations/sleeve_top_n must have identical keys")

    alloc_raw = pd.Series({k: float(v) for k, v in sleeve_allocations.items()}, dtype=float)
    if not np.isfinite(alloc_raw.to_numpy()).all():
        raise ValueError("sleeve_allocations must be finite")
    if (alloc_raw < 0.0).any():
        raise ValueError("sleeve_allocations must be >= 0")
    alloc_sum = float(alloc_raw.sum())
    if alloc_sum <= eps:
        raise ValueError("sleeve_allocations sum must be > 0")
    alloc = alloc_raw / alloc_sum

    aligned_close = close.astype(float)
    for sleeve_name, panel in sleeve_scores.items():
        if not isinstance(panel, pd.DataFrame):
            raise ValueError(f"sleeve_scores[{sleeve_name!r}] must be a DataFrame")
        aligned_close = aligned_close.reindex(index=panel.index.union(aligned_close.index))
        aligned_close = aligned_close.reindex(columns=panel.columns.union(aligned_close.columns))
    aligned_close = aligned_close.sort_index()
    aligned_close = aligned_close.reindex(columns=sorted(aligned_close.columns))

    sleeve_weights: dict[str, pd.DataFrame] = {}
    agg = pd.DataFrame(0.0, index=aligned_close.index, columns=aligned_close.columns, dtype=float)
    for sleeve_name in sorted(sleeve_scores.keys()):
        score_panel = (
            sleeve_scores[sleeve_name]
            .astype(float)
            .reindex(index=aligned_close.index, columns=aligned_close.columns)
        )
        top_n = int(sleeve_top_n[sleeve_name])
        if top_n <= 0:
            raise ValueError(f"sleeve_top_n[{sleeve_name!r}] must be > 0")
        sw = build_topn_weights(
            scores=score_panel,
            close=aligned_close,
            top_n=top_n,
            rebalance=rebalance,
            weighting=weighting,
            vol_lookback=vol_lookback,
            min_vol=min_vol,
            score_clip=score_clip,
            score_floor=score_floor,
            eps=eps,
            max_weight=max_weight,
            sector_cap=sector_cap,
            sector_by_ticker=sector_by_ticker,
            sector_neutral=bool(sector_neutral),
            rank_buffer=int(rank_buffer),
            volatility_scaled_weights=bool(volatility_scaled_weights),
        )
        sleeve_weights[sleeve_name] = sw
        agg = agg.add(sw * float(alloc.loc[sleeve_name]), fill_value=0.0)

    row_sum = agg.sum(axis=1).replace(0.0, np.nan)
    agg = agg.div(row_sum, axis=0).fillna(0.0)
    return agg, sleeve_weights


def simulate_portfolio(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    costs_bps: float,
    slippage_bps: float = 0.0,
    slippage_vol_mult: float = 0.0,
    slippage_vol_lookback: int = 20,
    rebalance_dates: pd.DatetimeIndex | None = None,
    execution_delay_days: int = 0,
) -> pd.DataFrame:
    """
    Simulate a portfolio with daily mark-to-market and rebalance-day costs.

    Causal alignment:
    portfolio_return[t] = sum_i(weights[t-1, i] * close_ret[t, i]) - cost[t]
    """
    close = close.astype(float).sort_index()
    if int(execution_delay_days) < 0:
        raise ValueError("execution_delay_days must be >= 0")
    if rebalance_dates is None:
        rebalance_dates = pd.DatetimeIndex(weights.index)
    else:
        rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    weights_impl = weights.astype(float).reindex(index=close.index, columns=close.columns).ffill().fillna(0.0)
    rebalance_dates_impl = rebalance_dates
    delay = int(execution_delay_days)
    if delay > 0:
        weights_impl = weights_impl.shift(delay).fillna(0.0)
        pos = {dt: i for i, dt in enumerate(close.index)}
        delayed_dates: list[pd.Timestamp] = []
        for dt in pd.DatetimeIndex(rebalance_dates):
            i = pos.get(pd.Timestamp(dt))
            if i is None:
                continue
            j = i + delay
            if j < len(close.index):
                delayed_dates.append(pd.Timestamp(close.index[j]))
        rebalance_dates_impl = pd.DatetimeIndex(sorted(set(delayed_dates)))

    if slippage_bps == 0.0 and slippage_vol_mult == 0.0:
        equity, net_daily, weights_daily = compute_daily_mark_to_market(
            close=close,
            weights_rebal=weights_impl,
            rebalance_dates=rebalance_dates_impl,
            costs_bps=costs_bps,
            slippage_bps=0.0,
        )
        turnover = 0.5 * weights_daily.diff().abs().sum(axis=1).fillna(0.0)
        effective_cost_bps = pd.Series(float(costs_bps), index=close.index)
    else:
        weights_daily = weights_impl
        asset_ret = close.pct_change().fillna(0.0)
        w_prev = weights_daily.shift(1).fillna(0.0)
        gross_daily = (w_prev * asset_ret).sum(axis=1, min_count=1).fillna(0.0)
        if not gross_daily.empty:
            gross_daily.iloc[0] = 0.0
        turnover = 0.5 * weights_daily.diff().abs().sum(axis=1).fillna(0.0)
        rebalance_set = set(rebalance_dates_impl)
        rebalance_mask = pd.Series(weights_daily.index.isin(rebalance_set), index=weights_daily.index)
        if slippage_vol_lookback <= 0:
            raise ValueError("slippage_vol_lookback must be > 0")
        asset_vol_bps = close.pct_change().rolling(slippage_vol_lookback).std(ddof=0) * 10000.0
        gross = weights_daily.abs().sum(axis=1).replace(0.0, np.nan)
        weighted_asset_vol_bps = (
            (weights_daily.abs() * asset_vol_bps).sum(axis=1).div(gross).fillna(0.0)
        )
        effective_cost_bps = costs_bps + slippage_bps + slippage_vol_mult * weighted_asset_vol_bps
        costs = pd.Series(0.0, index=close.index, dtype=float)
        costs.loc[rebalance_mask] = turnover.loc[rebalance_mask] * (
            effective_cost_bps.loc[rebalance_mask] / 10000.0
        )
        net_daily = gross_daily - costs
        equity = (1.0 + net_daily).cumprod()

    return pd.DataFrame(
        {
            "Equity": equity,
            "DailyReturn": net_daily,
            "Turnover": turnover,
            "EffectiveCostBps": effective_cost_bps,
        },
        index=close.index,
    )


def _sanity_check_topn_weights() -> None:
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    cols = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(
        {
            "A": np.linspace(100, 160, len(idx)),
            "B": np.linspace(100, 140, len(idx)),
            "C": np.linspace(100, 130, len(idx)),
            "D": np.linspace(100, 125, len(idx)),
            "E": np.linspace(100, 120, len(idx)),
        },
        index=idx,
    )
    scores = close.pct_change(21) - close.pct_change(5)
    for mode in ["equal", "inv_vol", "score", "score_inv_vol"]:
        w = build_topn_weights(
            scores=scores,
            close=close,
            top_n=3,
            rebalance="weekly",
            weighting=mode,
            max_weight=0.5,
            sector_cap=0.7,
            sector_by_ticker={"A": "Tech", "B": "Tech", "C": "Fin", "D": "Fin", "E": "HC"},
        )
        gross = w.sum(axis=1)
        invested = gross > 0
        if invested.any():
            assert np.allclose(gross.loc[invested].to_numpy(), 1.0, atol=1e-8)
        assert (w.fillna(0.0) >= -1e-12).all().all()


if __name__ == "__main__":
    _sanity_check_topn_weights()
