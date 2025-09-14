import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from yfinance.exceptions import YFRateLimitError
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date, max_retries=5):
    stock = yf.Ticker(ticker)
    for attempt in range(max_retries):
        try:
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                raise ValueError("No data fetched—check dates or ticker.")
            return df[['Close', ]]
        except YFRateLimitError:
            wait_time = (2 ** attempt) + np.random.random()  # Exponential backoff
            print(f"Rate limited on attempt {attempt + 1}/{max_retries}. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Other error: {e}")
            if attempt == max_retries - 1:
                raise
    raise YFRateLimitError("Max retries exceeded.")

def prepare_data(df, look_back=60):
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])  # Previous 'look_back' days
        y.append(scaled_data[i, 0])              # Next day's price
    X, y = np.array(X), np.array(y)
    
    # Reshape X for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split into train and test sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

def build_lstm_model(look_back):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))  # Prevent overfitting
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer for price prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_predict(ticker='AAPL', start_date='2023-01-01', end_date='2025-09-13', look_back=60, epochs=30):
    # Cache data locally
    cache_file = f"{ticker}_stock_data.csv"
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
    else:
        df = fetch_stock_data(ticker, start_date, end_date)
        df.to_csv(cache_file)
        print(f"Saved data to {cache_file}.")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, look_back)

    # Build and train model
    model = build_lstm_model(look_back)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform predictions to original scale
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform([y_test])
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[look_back:look_back+len(train_predict)], train_predict, label='Train Predictions')
    plt.plot(df.index[look_back:look_back+len(train_predict)], y_train_inv.T, label='Actual Train Prices')
    plt.plot(df.index[look_back+len(train_predict):look_back+len(train_predict)+len(test_predict)], test_predict, label='Test Predictions')
    plt.plot(df.index[look_back+len(train_predict):look_back+len(train_predict)+len(test_predict)], y_test_inv.T, label='Actual Test Prices')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('stock_plot.png')
    plt.show()
    
    train_rmse = np.sqrt(np.mean((train_predict - y_train_inv.T) ** 2))
    test_rmse = np.sqrt(np.mean((test_predict - y_test_inv.T) ** 2))
    
    with open('rmse_results.txt', 'w') as f:
        f.write(f'Train RMSE: {train_rmse:.2f}\n')
        f.write(f'Test RMSE: {test_rmse:.2f}\n')
    print(f'Train RMSE: {train_rmse:.2f}')
    print(f'Test RMSE: {test_rmse:.2f}')

    last_sequence = df['Close'].values[-look_back:]
    last_sequence = scaler.transform(last_sequence.reshape(-1, 1))
    last_sequence = np.reshape(last_sequence, (1, look_back, 1))
    next_price = model.predict(last_sequence)
    next_price = scaler.inverse_transform(next_price)[0][0]
    print(f'Predicted next day price for {ticker}: ${next_price:.2f}')

if __name__ == "__main__":
    train_and_predict(ticker='AAPL', start_date='2023-01-01', end_date='2025-09-13', look_back=60, epochs=30)

    input("Press Enter to exit...")  # Keeps terminal open to view output
