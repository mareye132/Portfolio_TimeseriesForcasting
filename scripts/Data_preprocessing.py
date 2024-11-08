import pandas as pd
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose

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

# Clean data function (removes rows with missing values)
def clean_data(data):
    for ticker, df in data.items():
        data[ticker] = df.dropna()
    return data

# Handle missing values
def handle_missing_values(data):
    for ticker, df in data.items():
        df.interpolate(method='linear', inplace=True)
        df.fillna(method='bfill', inplace=True)
    return data

# Normalize data (normalize all numeric columns)
def normalize_data(data):
    for ticker, df in data.items():
        for column in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if column in df.columns:
                df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
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

# Seasonal decomposition with the additive model
def seasonal_decomposition(data, period=365, model='additive'):
    decompositions = {}
    for ticker, df in data.items():
        if 'Close' in df.columns and len(df['Close']) >= period:
            decomposition = seasonal_decompose(df['Close'], period=period, model=model)
            decompositions[ticker] = decomposition
            decomposition.plot()
    return decompositions

# Outlier detection
def detect_outliers(data):
    outliers = {}
    for ticker, df in data.items():
        mean = df['Daily Return'].mean()
        std = df['Daily Return'].std()
        high_outliers = df[df['Daily Return'] > mean + 3 * std]
        low_outliers = df[df['Daily Return'] < mean - 3 * std]
        outliers[ticker] = {'High Returns': high_outliers, 'Low Returns': low_outliers}
    return outliers

# Log days with unusually high or low returns
def log_unusual_returns(outliers):
    for ticker, logs in outliers.items():
        print(f"\nTicker: {ticker}")
        print("High Return Days:")
        print(logs['High Returns'][['Daily Return']])
        print("Low Return Days:")
        print(logs['Low Returns'][['Daily Return']])

# Risk metrics: Sharpe Ratio and Value at Risk (VaR)
def calculate_risk_metrics(data):
    risk_metrics = {}
    for ticker, df in data.items():
        daily_returns = df['Daily Return'].dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252**0.5)  # 252 trading days per year
        var_95 = daily_returns.quantile(0.05)
        risk_metrics[ticker] = {'Sharpe Ratio': sharpe_ratio, 'VaR (95%)': var_95}
    return risk_metrics

# Main script to run all functions
if __name__ == "__main__":
    tickers = ['TSLA', 'BND', 'SPY']
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    # Step 1: Fetch and prepare data
    data = fetch_data(tickers, start_date, end_date)
    data = clean_data(data)
    data = handle_missing_values(data)
    data = normalize_data(data)
    
    # Step 2: Data summary and data type check
    summaries = data_summary(data)
    data_types = check_data_types(data)
    
    # Step 3: Calculate daily returns and rolling statistics
    data = calculate_daily_return(data)
    data = rolling_statistics(data)
    
    # Step 4: Perform seasonal decomposition and detect outliers
    decompositions = seasonal_decomposition(data)
    outliers = detect_outliers(data)
    log_unusual_returns(outliers)
    
    # Step 5: Calculate risk metrics
    risk_metrics = calculate_risk_metrics(data)
    
    # Print results
    print("\nData Summary:")
    for ticker, summary in summaries.items():
        print(f"\nSummary for {ticker}:\n", summary)
    
    print("\nRisk Metrics:")
    for ticker, metrics in risk_metrics.items():
        print(f"\nRisk Metrics for {ticker}:\n", metrics)
