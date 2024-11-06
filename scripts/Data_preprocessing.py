import pandas as pd
import yfinance as yf  # Assuming you're fetching financial data with Yahoo Finance

# Example function definitions

# Fetch data from Yahoo Finance
def fetch_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start_date, end=end_date)
    return data

# Data summary function
def data_summary(data):
    summaries = {}
    for ticker, df in data.items():
        summaries[ticker] = df.describe()
    return summaries

# Check data types function
def check_data_types(data):
    data_types = {}
    for ticker, df in data.items():
        data_types[ticker] = df.dtypes
    return data_types

# Clean data (example: removing NaN values)
def clean_data(data):
    for ticker, df in data.items():
        data[ticker] = df.dropna()
    return data

# Normalize data (example: normalizing closing prices)
def normalize_data(data):
    for ticker, df in data.items():
        df['Close'] = (df['Close'] - df['Close'].min()) / (df['Close'].max() - df['Close'].min())
    return data

# Calculate daily return
def calculate_daily_return(data):
    for ticker, df in data.items():
        df['Daily Return'] = df['Close'].pct_change()
    return data

# Rolling statistics: Mean and Standard Deviation
def rolling_statistics(data, window=30):
    for ticker, df in data.items():
        df['Rolling Mean'] = df['Close'].rolling(window).mean()
        df['Rolling Std'] = df['Close'].rolling(window).std()
    return data

# Seasonal decomposition (using statsmodels or another method)
def seasonal_decomposition(data, period=365):
    from statsmodels.tsa.seasonal import seasonal_decompose
    decompositions = {}
    for ticker, df in data.items():
        decompositions[ticker] = seasonal_decompose(df['Close'], period=period)
    return decompositions

# Outlier detection (example: using 3 standard deviations from the mean)
def detect_outliers(data):
    outliers = {}
    for ticker, df in data.items():
        mean = df['Close'].mean()
        std = df['Close'].std()
        outliers[ticker] = df[(df['Close'] > mean + 3*std) | (df['Close'] < mean - 3*std)]
    return outliers

# Risk metrics: Sharpe Ratio and Value at Risk (VaR)
def calculate_risk_metrics(data):
    risk_metrics = {}
    for ticker, df in data.items():
        # Example: Calculate Sharpe Ratio and 95% Value at Risk
        daily_returns = df['Daily Return'].dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252**0.5)  # Assuming 252 trading days
        var_95 = daily_returns.quantile(0.05)
        risk_metrics[ticker] = {'Sharpe Ratio': sharpe_ratio, 'VaR (95%)': var_95}
    return risk_metrics
