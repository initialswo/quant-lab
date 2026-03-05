import numpy as np
import pandas as pd

LEVERAGE_HARD_CAP = 1.5
CASH_VOL_FLOOR = 0.03


def sleeve_returns(prices_returns: pd.DataFrame, allocation: pd.DataFrame) -> pd.Series:
    return (allocation.shift(1) * prices_returns).sum(axis=1)


def rolling_portfolio_vol_from_cov(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    window: int,
) -> pd.Series:
    idx = returns_df.index
    out = pd.Series(index=idx, dtype=float)

    for i in range(window, len(idx)):
        w = weights_df.iloc[i].values.astype(float)
        if np.allclose(w.sum(), 0.0):
            out.iloc[i] = np.nan
            continue

        block = returns_df.iloc[i - window + 1 : i + 1]
        cov = block.cov().values
        port_var = float(w.T @ (cov * 252.0) @ w)
        out.iloc[i] = np.sqrt(port_var) if port_var >= 0 else np.nan

    return out


def apply_vol_target(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    target_vol: float | pd.Series,
    roll_vol: int,
    gross_cap: pd.Series,
    use_cov: bool = True,
    leverage_hard_cap: float = LEVERAGE_HARD_CAP,
    cash_vol_floor: float = CASH_VOL_FLOOR,
) -> pd.Series:
    blend_returns = (weights.shift(1) * returns).sum(axis=1)

    if use_cov:
        port_vol = rolling_portfolio_vol_from_cov(returns, weights, window=roll_vol)
    else:
        port_vol = blend_returns.rolling(roll_vol).std() * np.sqrt(252)

    port_vol = port_vol.clip(lower=cash_vol_floor)
    if isinstance(target_vol, pd.Series):
        target = target_vol.reindex(weights.index).astype(float)
    else:
        target = pd.Series(float(target_vol), index=weights.index, dtype=float)

    scale = (target / port_vol).clip(lower=0.0, upper=leverage_hard_cap)
    scale = np.minimum(scale, gross_cap)
    scale = pd.Series(scale, index=weights.index).clip(lower=0.0, upper=leverage_hard_cap)

    return (blend_returns * scale.shift(1)).dropna()
