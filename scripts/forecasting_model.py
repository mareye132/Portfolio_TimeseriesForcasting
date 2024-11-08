import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os  # Importing os for checking current directory

# Check the current working directory
print(f"Current working directory: {os.getcwd()}")

# Step 1: Fetch data from Yahoo Finance
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Date'] = data.index
    data['Close'] = data['Close'].fillna(method='ffill')
    return data[['Date', 'Close']]

# Step 2: Split data into training and testing sets
def split_data(data, train_size=0.8):
    train_size = int(len(data) * train_size)
    train, test = data[:train_size], data[train_size:]
    return train, test

# Step 3: Train SARIMA model
def train_sarima_model(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(train_data['Close'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit

# Step 4: Forecast and evaluate the model
def forecast_and_evaluate(model_fit, test_data):
    forecast = model_fit.forecast(steps=len(test_data))
    test_data['Forecast'] = forecast.values
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(test_data['Close'], test_data['Forecast'])
    rmse = np.sqrt(mean_squared_error(test_data['Close'], test_data['Forecast']))
    mape = mean_absolute_percentage_error(test_data['Close'], test_data['Forecast'])
    
    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    return test_data, metrics

# Main execution function
def main():
    ticker = 'TSLA'
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    # Step 1: Fetch and prepare data
    data = fetch_data(ticker, start_date, end_date)
    
    # Step 2: Split the data
    train, test = split_data(data)
    
    # Step 3: Train and forecast using SARIMA model
    model_fit = train_sarima_model(train)
    test_data, metrics = forecast_and_evaluate(model_fit, test)
    
    # Print metrics
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")
    
    # Save results
    output_path = 'C:/Users/user/Desktop/Github/Portfolio_TimeseriesForcasting/notebooks/data/forecast_results.csv'  # Relative path to 'notebooks' directory
    
    # Check if the directory exists, and create it if it doesn't
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the forecasted data to CSV
    test_data.to_csv(output_path, index=False)
    print(f"Forecast results saved to '{output_path}'")

if __name__ == "__main__":
    main()
