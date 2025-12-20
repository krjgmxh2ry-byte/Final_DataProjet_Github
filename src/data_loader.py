"""Data loading and preprocessing."""

import yfinance as yf
import pandas as pd


def load_data_assets(
    database: str, start: str = "2013-01-01", end: str = "2023-12-31"
) -> pd.DataFrame:
    """Load assets data from Yahoo Finance from start to end dates."""
    assets = yf.download(database, start=start, end=end)
    assets.columns = assets.columns.get_level_values(0)
    return assets


def load_benchmark(
    tickers: str = "^GSPC", start: str = "2013-01-01", end: str = "2023-12-31"
) -> pd.DataFrame:
    """Load market benchmark data from Yahoo Finance from start to end dates."""
    bench = yf.download(tickers=tickers, start=start, end=end, interval="1d")
    bench.columns = bench.columns.get_level_values(0)
    return bench


def compute_daily_return(df: pd.DataFrame) -> pd.Series:
    """Compute daily returns from closing prices."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    else:
        return (df["Close"] / df["Close"].shift(1)) - 1


def compute_portfolio_returns(stocks: dict) -> pd.DataFrame:
    """Compute daily returns for multiple stocks and build a portfolio DataFrame.

    Args:
        stocks: Dictionary of stock_name -> DataFrame with 'Close' prices

    Returns:
        DataFrame containing individual stock returns and portfolio average
    """
    rendements = {name: compute_daily_return(df) for name, df in stocks.items()}
    portefeuille = pd.concat(
        [pd.DataFrame(r) for r in rendements.values()], axis=1
    ).reindex(next(iter(rendements.values())).index)
    portefeuille.columns = list(rendements.keys())
    portefeuille["moyenne"] = portefeuille.mean(axis=1)
    return portefeuille
