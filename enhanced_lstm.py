import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from yfinance.exceptions import YFRateLimitError
import matplotlib.pyplot as plt

# Step 1: Fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date, max_retries=5):
    stock = yf.Ticker(ticker)
    for attempt in range(max_retries):
        try:
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                raise ValueError("No data fetched—check dates or ticker.")
            return df[['Close', 'Volume']]
        except YFRateLimitError:
            wait_time = (2 ** attempt) + np.random.random()  # Exponential backoff
            print(f"Rate limited on attempt {attempt + 1}/{max_retries}. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Other error: {e}")
            if attempt == max_retries - 1:
                raise
    raise YFRateLimitError("Max retries exceeded.")

# Step 2: Compute technical indicators
def compute_indicators(df):
    # Simple Moving Average (20 days)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Exponential Moving Average (50 days)
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Relative Strength Index (14 days)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    # Drop rows with NaN values from indicators
    df = df.dropna()
    return df

# Step 3: Prepare data for LSTM (multi-feature and multi-output)
def prepare_data(df, look_back=60):
    features = ['Close', 'Volume', 'SMA_20', 'EMA_50', 'RSI_14', 'MACD']
    
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    # Create sequences
    X = []
    y_price = []
    y_trend = []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, :])  # All features for look_back days
        y_price.append(scaled_data[i, 0])  # Next day's scaled Close
    X = np.array(X)
    y_price = np.array(y_price)
    
    # Trend: 1 if next Close > current Close, else 0 (binary)
    trend = (df['Close'].shift(-1) > df['Close']).astype(int).values[look_back:]
    y_trend = np.array(trend)
    
    # Split into train and test sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_price_train, y_price_test = y_price[:train_size], y_price[train_size:]
    y_trend_train, y_trend_test = y_trend[:train_size], y_trend[train_size:]
    
    return X_train, X_test, y_price_train, y_price_test, y_trend_train, y_trend_test, scaler, df, features

# Step 4: Build deeper multi-output LSTM model
def build_lstm_model(look_back, num_features):
    input_layer = Input(shape=(look_back, num_features))
    lstm1 = LSTM(units=100, return_sequences=True)(input_layer)
    dropout1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(units=100, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.2)(lstm2)
    lstm3 = LSTM(units=100)(dropout2)
    dropout3 = Dropout(0.2)(lstm3)
    
    # Outputs: Price (regression) and Trend (classification)
    price_output = Dense(1, name='price')(dropout3)
    trend_output = Dense(1, activation='sigmoid', name='trend')(dropout3)
    
    model = Model(inputs=input_layer, outputs=[price_output, trend_output])
    model.compile(optimizer='adam', 
                  loss={'price': 'mean_squared_error', 'trend': 'binary_crossentropy'},
                  metrics={'price': 'mae', 'trend': 'accuracy'})
    return model

# Step 5: Train and evaluate the model
def train_and_predict(ticker='AAPL', start_date='2020-01-01', end_date='2025-09-13', look_back=60, epochs=50):
    # Cache data locally
    cache_file = f"{ticker}_stock_data_extended.csv"
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
    else:
        df = fetch_stock_data(ticker, start_date, end_date)
        df.to_csv(cache_file)
        print(f"Saved data to {cache_file}.")
    
    # Compute indicators
    df = compute_indicators(df)
    
    # Prepare data
    X_train, X_test, y_price_train, y_price_test, y_trend_train, y_trend_test, scaler, df, features = prepare_data(df, look_back)
    
    # Build and train model with early stopping
    model = build_lstm_model(look_back, len(features))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, [y_price_train, y_trend_train], 
                        epochs=epochs, batch_size=32, validation_split=0.2, 
                        callbacks=[early_stopping], verbose=1)
    
    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Total Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Make predictions
    train_predict_price, train_predict_trend = model.predict(X_train)
    test_predict_price, test_predict_trend = model.predict(X_test)
    
    # Inverse transform price predictions (handle multi-feature scaler)
    def inverse_price(pred_price):
        dummy = np.zeros((len(pred_price), scaler.n_features_in_))
        dummy[:, 0] = pred_price.flatten()  # Close is first feature
        return scaler.inverse_transform(dummy)[:, 0]
    
    train_predict_price_inv = inverse_price(train_predict_price)
    y_price_train_inv = inverse_price(y_price_train.reshape(-1, 1))
    test_predict_price_inv = inverse_price(test_predict_price)
    y_price_test_inv = inverse_price(y_price_test.reshape(-1, 1))
    
    # Plot price results
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[look_back:look_back + len(train_predict_price_inv)], train_predict_price_inv, label='Train Price Predictions')
    plt.plot(df.index[look_back:look_back + len(train_predict_price_inv)], y_price_train_inv, label='Actual Train Prices')
    plt.plot(df.index[look_back + len(train_predict_price_inv):], test_predict_price_inv, label='Test Price Predictions')
    plt.plot(df.index[look_back + len(train_predict_price_inv):], y_price_test_inv, label='Actual Test Prices')
    plt.title(f'{ticker} Stock Price Prediction (Enhanced Model)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()
    
    # Calculate Price RMSE
    train_rmse = np.sqrt(np.mean((train_predict_price_inv - y_price_train_inv) ** 2))
    test_rmse = np.sqrt(np.mean((test_predict_price_inv - y_price_test_inv) ** 2))
    print(f'Train Price RMSE: {train_rmse:.2f}')
    print(f'Test Price RMSE: {test_rmse:.2f}')
    
    # Calculate Trend Accuracy
    train_trend_pred = (train_predict_trend > 0.5).astype(int).flatten()
    test_trend_pred = (test_predict_trend > 0.5).astype(int).flatten()
    train_trend_acc = np.mean(train_trend_pred == y_trend_train)
    test_trend_acc = np.mean(test_trend_pred == y_trend_test)
    print(f'Train Trend Accuracy: {train_trend_acc:.2f}')
    print(f'Test Trend Accuracy: {test_trend_acc:.2f}')
    
    # Predict next day's price and trend
    last_sequence = scaler.transform(df[features].values[-look_back:])
    last_sequence = np.reshape(last_sequence, (1, look_back, len(features)))
    next_price_pred, next_trend_pred = model.predict(last_sequence)
    next_price_inv = inverse_price(next_price_pred)
    next_trend = 'Up' if next_trend_pred[0][0] > 0.5 else 'Down'
    print(f'Predicted next day price for {ticker}: ${next_price_inv[0]:.2f}')
    print(f'Predicted next day trend for {ticker}: {next_trend}')

# Run the model
if __name__ == "__main__":
    train_and_predict(ticker='AAPL', start_date='2020-01-01', end_date='2025-09-13', look_back=60, epochs=50)
    input("Press Enter to exit...")