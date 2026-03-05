import numpy as np
import pandas as pd


def perf_stats(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    td = 252
    ann_ret = float(s.mean() * td)
    ann_vol = float(s.std() * np.sqrt(td))
    sharpe = float(ann_ret / ann_vol) if ann_vol != 0 else np.nan
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe}


def max_drawdown(series: pd.Series) -> float:
    s = series.dropna()
    curve = (1 + s).cumprod()
    dd = curve / curve.cummax() - 1
    return float(dd.min())
