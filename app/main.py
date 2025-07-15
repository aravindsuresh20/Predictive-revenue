import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import timedelta

def run_forecast(predicted_days):
    # Load dataset
    df = pd.read_csv('data/deutsche_bank_financial_performance.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)

    # Use first column for forecasting if 'Revenue' is not guaranteed
    value_col = df.columns[0]  # auto-detect numeric target
    data_values = df[[value_col]]  # keep as DataFrame

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_values)

    # Create sequences
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(x), np.array(y)

    seq_length = 30
    X, y = create_sequences(scaled_data, seq_length)

    # Reshape and train model
    X_train = X.reshape(X.shape[0], -1)
    dtrain = xgb.DMatrix(X_train, label=y)
    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'max_depth': 6, 'eta': 0.1}
    model = xgb.train(params, dtrain, num_boost_round=50)

    # Forecast from last date in original data
    last_date = data_values.index[-1]
    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=predicted_days)

    input_seq = scaled_data[-seq_length:]
    forecast = []

    for _ in range(predicted_days):
        dmatrix = xgb.DMatrix(input_seq.reshape(1, -1))
        pred = model.predict(dmatrix)[0]
        forecast.append(pred)
        input_seq = np.append(input_seq[1:], [[pred]], axis=0)

    forecast = np.array(forecast)
    forecasted_prices = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({'Revenue': forecasted_prices}, index=forecast_dates)
    forecast_df.index.name = 'Date'

    # Append to Excel file
    os.makedirs('output', exist_ok=True)
    excel_path = 'output/forecasted_financial_data.xlsx'
    if os.path.exists(excel_path):
        existing_df = pd.read_excel(excel_path, engine='openpyxl', index_col='Date')
        combined_df = pd.concat([existing_df, forecast_df])
    else:
        combined_df = forecast_df

    combined_df.to_excel(excel_path, engine='openpyxl')

    # Save forecast plot
    os.makedirs('static', exist_ok=True)
    chart_path = 'static/financial_forecast.png'
    plt.figure(figsize=(12, 6))
    try:
        plt.plot(data_values.index[-30:], data_values[value_col].values[-30:], label='Actual (last 30)')
    except:
        plt.plot(data_values.index, data_values[value_col].values, label='Actual')
    plt.plot(forecast_df.index, forecast_df['Revenue'], label='Forecast', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel(value_col)
    plt.title('Financial Forecast')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(chart_path)

    # Perform EDA and save summary
    eda_summary_path = 'output/eda_summary.txt'
    with open(eda_summary_path, 'w') as f:
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write(f"\nMissing Values:\n{df.isnull().sum()}\n")
        f.write(f"\nSummary Statistics:\n{df.describe()}\n")

    return chart_path, forecast_df
