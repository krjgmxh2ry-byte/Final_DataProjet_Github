"""Data loading and preprocessing."""
# This file contains utility functions used to download and prepare
# all the financial data needed for the project.

import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
# Splits data into train/test sets

TICKERS = ["MSFT", "AMZN", "AAPL"]
# Core tickers used to build the portfolio; change here to reuse the pipeline.


def load_data_assets(tickers: list, start="2013-01-01", end="2023-12-31") -> dict:
    """
    Download price data for several stocks.

    Returns:
        dict: ticker -> DataFrame
    """
    assets = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end)
        # Flatten multi-index columns so names stay simple

        df.columns = df.columns.get_level_values(0)
        assets[ticker] = df
    return assets


def load_benchmark(
    tickers: str = "^GSPC", start: str = "2013-01-01", end: str = "2023-12-31"
) -> pd.DataFrame:
    """Load benchmark index data (default: S&P500 (^GSPC))."""
    
    bench = yf.download(tickers=tickers, start=start, end=end, interval="1d")
    bench.columns = bench.columns.get_level_values(0)
    return bench


def load_risk_free(ticker="^TNX", start="2013-01-01", end="2023-12-31") -> pd.Series:
    """Load risk-free rate and smooth it over 20 days."""
    df_rf = yf.download(ticker, start=start, end=end)["Close"]
    # 20-day smoothing + convert to daily rate (~250 trading days)
    return df_rf.rolling(window=20).mean() / 250  # 250 trading days


def compute_daily_return(df: pd.DataFrame) -> pd.Series:
    """Compute daily returns from closing prices."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    else:
        # (P_t / P_{t-1}) - 1 : Return percentage change between days.
        return (df["Close"] / df["Close"].shift(1)) - 1


def compute_portfolio_returns(stocks: dict) -> pd.DataFrame:
    """Compute daily returns for multiple stocks and build a portfolio DataFrame.

    Args:
        stocks: Dictionary of stock_name -> DataFrame with 'Close' prices

    Returns:
        DataFrame containing individual stock returns and portfolio average
    """

    rendements = {name: compute_daily_return(df) for name, df in stocks.items()}
    # Compute returns for each stock

    portefeuille = pd.concat(
        [pd.DataFrame(r) for r in rendements.values()], axis=1
    ).reindex(next(iter(rendements.values())).index)  
    # Put all returns into one DataFrame (align dates)

    portefeuille.columns = list(rendements.keys())
    # Name the columns with the stock names

    
    portefeuille["moyenne"] = portefeuille.mean(axis=1)
     # Add portfolio mean return (simple unweighted average)
    return portefeuille


def compute_portfolio(df_assets: dict) -> pd.DataFrame:
    """Constructs a normalized portfolio DataFrame and computes its mean.

    Args:
        df_assets: Dictionary of ticker -> DataFrame containing stock data with 'Close' column.

    Returns:
        pd.DataFrame: DataFrame with normalized columns for each asset and an additional column 'moyenne' representing the portfolio average.
    """
    norm_assets = {
        ticker: normalize_data(df, "Close") for ticker, df in df_assets.items()
    }
    portefeuille_norm = pd.concat(norm_assets, axis=1)
    portefeuille_norm["moyenne"] = portefeuille_norm.mean(axis=1)
    return portefeuille_norm


def normalize_data(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Normalize a DataFrame column by its first value.

    Args:
        df : input pd.DataFrame
        column : column to normalize

    Returns:
        Normalized column
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    series = df[column].dropna()

    if series.empty:
        raise ValueError("Selected column contains only NaNs")

    return df[column] / series.iloc[0]


def block_return(block: pd.Series) -> float:
    """Compute the return over a block of prices."""
    if len(block) < 2 or block.isna().all():
        return np.nan
    return (block.iloc[-1] - block.iloc[0]) / block.iloc[0]


def build_final_df_clean(
    sp500: pd.DataFrame, df_assets: dict, rf: pd.Series, window: int = 20
) -> pd.DataFrame:
    """
    Build a clean final DataFrame with:
        - Rolling coefficient of variation (CV)
        - Rolling Sharpe ratios
        - Rolling covariance
        - Binary indicator if portfolio outperforms S&P500

    Args:
        sp500 (pd.DataFrame): Benchmark data with 'Close' column
        df_assets (dict): Dictionary of ticker -> DataFrame with 'Close' prices
        rf (pd.Series): Risk-free rate (aligned on same dates as portfolio)
        window (int): Rolling window size

    Returns:
        pd.DataFrame: Clean final DataFrame ready for analysis
    """
   # Assemble the closing prices of each asset in the portfolio.
    portefeuille = pd.concat([df["Close"] for df in df_assets.values()], axis=1)
    portefeuille.columns = df_assets.keys()
    # Normalize the S&P500 to compare relative dynamics.
    sp500_norm = normalize_data(sp500, "Close").reindex(portefeuille.index)

    # Rolling CV to quantify relative volatility (portfolio vs. S&P500).
    port_rolling_var = portefeuille.rolling(window=window).var().mean(axis=1)
    port_rolling_mean = portefeuille.rolling(window=window).mean().mean(axis=1)
    cv_rolling_port = port_rolling_var / port_rolling_mean

    sp_rolling_var = sp500_norm.rolling(window=window).var()
    sp_rolling_mean = sp500_norm.rolling(window=window).mean()
    cv_rolling_sp = sp_rolling_var / sp_rolling_mean

    def block_return(block):
        return (block.iloc[-1] - block.iloc[0]) / block.iloc[0]

    # Average return over the sliding window for the portfolio and the benchmark.
    r_port = portefeuille.rolling(window=window).apply(block_return)
    r_port_mean = r_port.mean(axis=1)

    r_sp500 = sp500["Close"].rolling(window=window).apply(block_return)

    # Sharpe based on the 10Y rate as a proxy for the risk-free rate (aligned by date).
    sharpe_port = (r_port_mean - rf["^TNX"]) / cv_rolling_port
    sharpe_sp500 = (r_sp500 - rf["^TNX"]) / cv_rolling_sp

    portefeuille_returns = compute_portfolio_returns(df_assets)
    sp500_returns = pd.DataFrame(compute_daily_return(sp500), columns=["Close"])
    df_rend = pd.concat(
        [portefeuille_returns["moyenne"], sp500_returns["Close"]], axis=1
    )

    df_rend.columns = ["moyenne", "Close"]
    cov_by_block = df_rend["moyenne"].rolling(window=window).cov(df_rend["Close"])

    r_port_mean_21 = r_port.mean(axis=1)
    r_sp500_21 = sp500["Close"].rolling(window=window + 1).apply(block_return)

    # Binary target: 1 if the portfolio outperforms the S&P500 on the sliding block.
    df_final = pd.DataFrame(
        {
            "cv_port": cv_rolling_port,
            "cv_sp500": cv_rolling_sp,
            "sharpe_port": sharpe_port,
            "sharpe_sp500": sharpe_sp500,
            "covariance": cov_by_block,
        }
    )

    df_final["result"] = (r_port_mean_21 > r_sp500_21).astype(int)
    return df_final.dropna()


def load_and_split(test_size: float = 0.2, random_state: int = 42):
    """Build final dataframe using build_final_df_clean, split features and target.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df_assets = load_data_assets(tickers=TICKERS)
    sp500 = load_benchmark()
    rf = load_risk_free()

    df = build_final_df_clean(sp500, df_assets, rf)

    X = df.drop(columns=["result"])
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, df
