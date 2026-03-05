import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str],
    period: str,
    interval: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        close = raw["Close"]
    else:
        close = raw.rename("Close").to_frame()

    return close


def clean_prices(prices: pd.DataFrame, cash_ticker: str | None = None) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()

    cleaned = prices.copy()
    if cash_ticker and cash_ticker in cleaned.columns:
        risky_cols = [c for c in cleaned.columns if c != cash_ticker]
        if risky_cols:
            return cleaned.dropna(subset=risky_cols, how="any")

    return cleaned.dropna().copy()
