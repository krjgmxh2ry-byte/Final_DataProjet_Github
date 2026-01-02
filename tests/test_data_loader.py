import pandas as pd
from src.data_loader import load_data_assets


def test_load_data_assets_returns_dataframe():
    tickers = ["AAPL", "AMZN", "MSFT"]

    df_assets = load_data_assets(tickers)

    # Should return a dictionary
    assert isinstance(df_assets, dict)

    # Each value should be a DataFrame
    for t in tickers:
        assert isinstance(df_assets[t], pd.DataFrame)
        assert len(df_assets[t]) > 0
        assert "Close" in df_assets[t].columns