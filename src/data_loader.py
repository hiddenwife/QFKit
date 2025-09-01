import yfinance as yf
import numpy as np
import pandas as pd

def get_stock_data(ticker, start="2020-01-01", end="2025-01-01"):
    """
    Downloads historical stock data using Yahoo Finance.
    Adds log returns to the DataFrame.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.columns = [col[0] for col in df.columns] 
    return df.dropna()
