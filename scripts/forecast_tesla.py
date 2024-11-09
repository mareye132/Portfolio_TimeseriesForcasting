import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import os

# Set data directory path
data_dir = 'C:/Users/user/Desktop/Github/Portfolio_TimeseriesForcasting/notebooks/data'
os.makedirs(data_dir, exist_ok=True)

# Download Tesla's historical stock data
tesla_data = yf.download('TSLA', start='2010-01-01', end='2024-01-01')

# Save the historical data to CSV
tesla_data_path = os.path.join(data_dir, 'tesla_historical_data.csv')
tesla_data.to_csv(tesla_data_path)
print(f"Tesla historical data saved to {tesla_data_path}")

# Use adjusted closing prices for analysis
tesla_close = tesla_data['Adj Close']
tesla_close.index = pd.to_datetime(tesla_close.index)

# Split data into training and test sets (e.g., use the last 6 months for testing)
train_data = tesla_close[:-180]  # Adjust this split as needed
test_data = tesla_close[-180:]

# Fit an ARIMA model
model = ARIMA(train_data, order=(5, 1, 0))  # You can adjust the (p, d, q) parameters as needed
model_fit = model.fit()

# Forecast for the next 6-12 months (e.g., 180 days)
forecast_steps = 180  # 6 months
forecast_result = model_fit.get_forecast(steps=forecast_steps)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval

# Save forecast to CSV
forecast_df = pd.DataFrame({
    'Date': pd.date_range(start=test_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B'),
    'Forecast': forecast,
    'Lower CI': conf_int.iloc[:, 0],
    'Upper CI': conf_int.iloc[:, 1]
})
forecast_df_path = os.path.join(data_dir, 'tesla_forecast.csv')
forecast_df.to_csv(forecast_df_path, index=False)
print(f"Tesla forecast saved to {forecast_df_path}")

# Plot forecast vs actuals
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data, label='Training Data')
plt.plot(test_data.index, test_data, label='Actual Prices', color='green')
plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='red')
plt.fill_between(forecast_df['Date'], forecast_df['Lower CI'], forecast_df['Upper CI'], color='pink', alpha=0.3, label='95% Confidence Interval')
plt.title('Tesla Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
