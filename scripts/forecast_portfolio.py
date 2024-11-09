import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt

# Set data directory path
data_dir = 'C:/Users/user/Desktop/Github/Portfolio_TimeseriesForcasting/notebooks/data'

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)

# Load historical data for TSLA, BND, SPY
assets = ['TSLA', 'BND', 'SPY']
historical_data = yf.download(assets, start='2018-01-01', end='2024-01-01')['Close']

# Print the column names to verify the data structure
print("Columns in historical data:", historical_data.columns)

# Forecasted data: Replace with actual forecast logic or load forecast data
forecasted_data = historical_data.tail(30)

# Calculate daily returns
daily_returns = forecasted_data.pct_change().dropna()

# Compute average annual return
annual_returns = daily_returns.mean() * 252  # Assuming 252 trading days per year

# Compute covariance matrix
cov_matrix = daily_returns.cov() * 252  # Annualized covariance

# Portfolio optimization functions
def portfolio_performance(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def negative_sharpe_ratio(weights, returns, cov_matrix):
    return -portfolio_performance(weights, returns, cov_matrix)[2]

# Constraints and bounds
num_assets = len(assets)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Sum of weights must be 1
bounds = tuple((0, 1) for asset in range(num_assets))  # Weights between 0 and 1

# Initial guess for weights
initial_weights = num_assets * [1. / num_assets]

# Optimize portfolio for maximum Sharpe ratio
optimized = minimize(negative_sharpe_ratio, initial_weights, args=(annual_returns, cov_matrix),
                     method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = optimized.x

# Calculate performance of the optimized portfolio
opt_return, opt_volatility, opt_sharpe = portfolio_performance(optimal_weights, annual_returns, cov_matrix)

# Print results
print("Optimized Portfolio Weights:", optimal_weights)
print("Expected Return:", opt_return)
print("Expected Volatility:", opt_volatility)
print("Sharpe Ratio:", opt_sharpe)

# Save results to CSV in the specified path
results_df = pd.DataFrame({
    'Asset': assets,
    'Weight': optimal_weights
})
results_df_path = os.path.join(data_dir, 'optimized_portfolio_weights.csv')
results_df.to_csv(results_df_path, index=False)
print(f"Portfolio weights saved to {results_df_path}")

# Compute cumulative returns for each asset
cumulative_returns = (1 + historical_data.pct_change()).cumprod()

# Check the available assets in cumulative_returns
print("Assets available for cumulative returns:", cumulative_returns.columns)

# Create a DataFrame for asset weights (for plotting later)
weights_df = pd.DataFrame({
    'Asset': assets,
    'Weight': optimal_weights
})

# Plot cumulative returns
plt.figure(figsize=(14, 7))
for asset in weights_df['Asset']:
    if asset in cumulative_returns.columns:  # Ensure the asset exists in the data
        plt.plot(cumulative_returns.index, cumulative_returns[asset], label=asset)
    else:
        print(f"Warning: {asset} not found in the historical data!")

plt.title('Cumulative Returns of Portfolio Assets')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()
