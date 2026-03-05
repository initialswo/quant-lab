import numpy as np
import pandas as pd


def _apply_weight_floor(weights: pd.Series, weight_floor: float) -> pd.Series:
    floored = weights.copy()
    floored[floored < weight_floor] = 0.0
    total = float(floored.sum())
    if total > 0.0:
        return floored / total

    fallback = pd.Series(0.0, index=weights.index, dtype=float)
    fallback.loc[weights.idxmax()] = 1.0
    return fallback


def _covariance_weights(returns_window: pd.DataFrame, selected: list[str]) -> pd.Series:
    block = returns_window[selected].dropna(how="any")
    if len(block) < 2:
        return pd.Series(1.0 / len(selected), index=selected, dtype=float)

    cov = block.cov().values.astype(float)
    if not np.all(np.isfinite(cov)):
        return pd.Series(1.0 / len(selected), index=selected, dtype=float)

    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return pd.Series(1.0 / len(selected), index=selected, dtype=float)

    ones = np.ones(len(selected), dtype=float)
    raw = inv @ ones
    raw = np.clip(raw, 0.0, None)
    denom = float(raw.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return pd.Series(1.0 / len(selected), index=selected, dtype=float)

    return pd.Series(raw / denom, index=selected, dtype=float)


def _inv_vol_weights(
    returns_window: pd.DataFrame,
    selected: list[str],
    vol_floor: float = 1e-6,
) -> pd.Series:
    block = returns_window[selected].dropna(how="any")
    if len(block) < 2:
        return pd.Series(1.0 / len(selected), index=selected, dtype=float)

    vol = block.std() * np.sqrt(252.0)
    vol = vol.clip(lower=float(vol_floor))
    raw = 1.0 / vol
    denom = float(raw.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return pd.Series(1.0 / len(selected), index=selected, dtype=float)
    return (raw / denom).astype(float)


def compute_momentum(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return prices / prices.shift(lookback) - 1


def dual_momentum_alloc(
    prices: pd.DataFrame,
    momentum: pd.DataFrame,
    rebalance: str = "ME",
    smoothing_alpha: float = 1.0,
    weight_floor: float = 1e-4,
    top_k: int = 1,
    use_cov: bool = False,
    roll_vol: int = 20,
    weight_method: str | None = None,
    cash_ticker: str = "SGOV",
    abs_mom_threshold: float = 0.0,
) -> pd.DataFrame:
    if cash_ticker not in prices.columns:
        raise ValueError(f"cash_ticker '{cash_ticker}' not found in price columns.")

    risky_cols = [c for c in prices.columns if c != cash_ticker]
    if not risky_cols:
        raise ValueError("No risky assets available after excluding cash_ticker.")

    monthly_mom = momentum.resample(rebalance).last().dropna(how="all")
    daily_returns = prices.pct_change()
    k = max(1, int(top_k))
    method = (weight_method or ("inv_cov" if use_cov else "equal")).lower()
    if method not in {"equal", "inv_vol", "inv_cov"}:
        raise ValueError(f"Unsupported weight_method '{method}'. Expected one of: equal, inv_vol, inv_cov.")

    monthly_signal = pd.DataFrame(0.0, index=monthly_mom.index, columns=prices.columns)
    for dt in monthly_signal.index:
        risky_row = monthly_mom.loc[dt, risky_cols].dropna()
        if risky_row.empty:
            monthly_signal.loc[dt, cash_ticker] = 1.0
            continue

        best_momentum = float(risky_row.max())
        if best_momentum <= float(abs_mom_threshold):
            monthly_signal.loc[dt, cash_ticker] = 1.0
            continue

        eligible = risky_row[risky_row > 0.0].sort_values(ascending=False)
        if eligible.empty:
            monthly_signal.loc[dt, cash_ticker] = 1.0
            continue

        selected = eligible.head(k).index.tolist()
        if len(selected) == 1:
            sel_weights = pd.Series(1.0, index=selected, dtype=float)
        else:
            window = daily_returns.loc[:dt].tail(max(2, int(roll_vol)))
            if method == "equal":
                sel_weights = pd.Series(1.0 / len(selected), index=selected, dtype=float)
            elif method == "inv_vol":
                sel_weights = _inv_vol_weights(window, selected)
            else:
                sel_weights = _covariance_weights(window, selected)
        monthly_signal.loc[dt, sel_weights.index] = sel_weights.values

    alpha = float(smoothing_alpha)
    alpha = max(0.0, min(1.0, alpha))
    floor = max(0.0, float(weight_floor))

    smoothed = pd.DataFrame(0.0, index=monthly_signal.index, columns=monthly_signal.columns)
    prev = pd.Series(0.0, index=monthly_signal.columns, dtype=float)
    for dt in monthly_signal.index:
        target = monthly_signal.loc[dt].astype(float)
        current = alpha * target + (1.0 - alpha) * prev

        # Keep cash transitions intact; normalize only invested-to-invested blends.
        if target.sum() > 0.0 and prev.sum() > 0.0:
            total = float(current.sum())
            if total > 0.0:
                current = current / total

        current = current.clip(lower=0.0)
        # Apply floor as the final step before weights are stored for this rebalance.
        current = _apply_weight_floor(current, floor)

        smoothed.loc[dt] = current
        prev = current

    return smoothed.reindex(prices.index, method="ffill").fillna(0.0)


def compute_trend_strength(prices: pd.DataFrame, ma_trend: int) -> pd.Series:
    spy_ma = prices["SPY"].rolling(ma_trend).mean()
    return (prices["SPY"] / spy_ma) - 1.0


def compute_vol_score(returns: pd.DataFrame, roll_vol: int) -> pd.Series:
    spy_vol = returns["SPY"].rolling(roll_vol).std() * np.sqrt(252)
    vol_base = spy_vol.rolling(252).median()
    vol_ratio = (spy_vol / vol_base) - 1.0

    vol_scale = 0.25
    return 1.0 / (1.0 + np.exp(-(vol_ratio / vol_scale)))
