import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import shutil
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from yfinance.exceptions import YFRateLimitError
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date, max_retries=5):
    stock = yf.Ticker(ticker)
    for attempt in range(max_retries):
        try:
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                raise ValueError("No data fetched—check dates or ticker.")
            return df[['Close', 'Open', 'High', 'Low', 'Volume']]
        except YFRateLimitError:
            wait_time = (2 ** attempt) + np.random.random()
            print(f"Rate limited on attempt {attempt + 1}/{max_retries}. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Other error: {e}")
            if attempt == max_retries - 1:
                raise
    raise YFRateLimitError("Max retries exceeded.")

def compute_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = rolling_mean + (rolling_std * 2)
    df['BB_lower'] = rolling_mean - (rolling_std * 2)
    
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['ATR_14'] = tr.rolling(window=14).mean()
    
    df = df.dropna()
    return df

def prepare_data(df, look_back=120):  # Updated to match your call
    features = ['Close', 'Volume', 'SMA_20', 'EMA_50', 'RSI_14', 'MACD', 'BB_upper', 'BB_lower', 'Stochastic_K', 'ATR_14']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    X = []
    y_price = []
    y_trend = []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, :])
        y_price.append(scaled_data[i, 0])  # Next day's scaled Close
    X = np.array(X)
    y_price = np.array(y_price)
    
    trend = (df['Close'].shift(-1) > df['Close']).astype(int).values[look_back:]
    y_trend = np.array(trend)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_price_train, y_price_test = y_price[:train_size], y_price[train_size:]
    y_trend_train, y_trend_test = y_trend[:train_size], y_trend[train_size:]
    
    return X_train, X_test, y_price_train, y_price_test, y_trend_train, y_trend_test, scaler, df, features

def build_lstm_model(hp):
    num_features = hp.Fixed('num_features', 10)
    look_back = hp.Fixed('look_back', 120)  # Updated to match your call
    input_layer = Input(shape=(look_back, num_features))
    
    x = input_layer
    num_layers = hp.Int('num_layers', min_value=2, max_value=4, step=1)
    for i in range(num_layers):
        x = LSTM(units=hp.Int(f'units_{i}', min_value=50, max_value=150, step=50), return_sequences=(i < num_layers - 1))(x)
        x = Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.3, step=0.1))(x)
    
    # Price output (regression)
    price_output = Dense(1, name='price_output')(x)
    
    # Trend output (classification with deeper branch)
    trend_hidden = Dense(10, activation='relu')(x)
    trend_output = Dense(1, activation='sigmoid', name='trend_output')(trend_hidden)
    
    model = Model(inputs=input_layer, outputs=[price_output, trend_output])
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss={'price_output': 'mean_squared_error', 'trend_output': 'binary_crossentropy'},
                  loss_weights={'price_output': 0.5, 'trend_output': 0.5},  # Balanced weights
                  metrics={'price_output': 'mae', 'trend_output': 'accuracy'})
    return model

def train_and_predict(ticker='AAPL', start_date='2020-01-01', end_date='2025-09-13', look_back=120, max_epochs=50, tuner_trials=10):
    tuner_dir = 'tuner_dir'
    if os.path.exists(tuner_dir):
        shutil.rmtree(tuner_dir)
        print(f"Cleared tuner directory {tuner_dir} for fresh search.")
    
    cache_file = f"{ticker}_stock_data_extended.csv"
    required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
        if not all(col in df.columns for col in required_columns):
            print(f"Cached file {cache_file} missing required columns. Refetching data...")
            os.remove(cache_file)
        else:
            print(f"Loading cached data from {cache_file}...")
            df = df[required_columns]
    if not os.path.exists(cache_file):
        df = fetch_stock_data(ticker, start_date, end_date)
        df.to_csv(cache_file)
        print(f"Saved data to {cache_file}.")
    
    df = compute_indicators(df)
    X_train, X_test, y_price_train, y_price_test, y_trend_train, y_trend_test, scaler, df, features = prepare_data(df, look_back)

    trend_class_weight = {0: 1.0, 1: 1.5}  #Adjusting for trend imbalance - can be adjusted if needed
    
    tuner = kt.RandomSearch(build_lstm_model,
                            objective='val_loss',
                            max_trials=tuner_trials,
                            executions_per_trial=1,
                            directory='tuner_dir',
                            project_name='lstm_stock_tuning')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tuner.search(X_train, [y_price_train, y_trend_train], epochs=20, validation_split=0.2, callbacks=[early_stopping])
    
    # Get top 3 hyperparameteres for ensemble
    top_hps = tuner.get_best_hyperparameters(num_trials=3)
    print("Top 3 Hyperparameters for Ensemble:")
    for j, hps in enumerate(top_hps):
        print(f"Model {j+1}:")
        print(f"Number of Layers: {hps.get('num_layers')}")
        for i in range(hps.get('num_layers')):
            print(f"Layer {i+1} Units: {hps.get(f'units_{i}')}, Dropout: {hps.get(f'dropout_{i}')}")
        print(f"Learning Rate: {hps.get('learning_rate')}")

    # Train ensemble models with top 3 hyperparameters 
    ensemble_price_train_preds = []
    ensemble_price_test_preds = []
    ensemble_trend_train_preds = []
    ensemble_trend_test_preds = []
    ensemble_next_price_preds = []
    ensemble_next_trend_preds = []
    histories = []
    
      for hps in top_hps:
        model = tuner.hypermodel.build(hps)
        history = model.fit(X_train, [y_price_train, y_trend_train], epochs=max_epochs, batch_size=32,
                            validation_split=0.2, callbacks=[early_stopping], verbose=1, class_weight={'trend_output': trend_class_weight})

        #Ensemble predictions 
        train_predict_price, train_predict_trend = model.predict(X_train)
        test_predict_price, test_predict_trend = model.predict(X_test)
        ensemble_price_train_preds.append(train_predict_price)
        ensemble_price_test_preds.append(test_predict_price)
        ensemble_trend_train_preds.append(train_predict_trend)
        ensemble_trend_test_preds.append(test_predict_trend)

        #Next day predictions
        last_sequence = scaler.transform(df[features].values[-look_back:])
        last_sequence = np.reshape(last_sequence, (1, look_back, len(features)))
        next_price_pred, next_trend_pred = model.predict(last_sequence)
        ensemble_next_price_preds.append(next_price_pred)
        ensemble_next_trend_preds.append(next_trend_pred)

    # Debug: Print history lengths
    print("History lengths:", [len(h.history['loss']) for h in histories])
    
    # Truncate histories to minimum length for averaging
    min_epochs = min(len(h.history['loss']) for h in histories)
    truncated_losses = [h.history['loss'][:min_epochs] for h in histories]
    truncated_val_losses = [h.history['val_loss'][:min_epochs] for h in histories]
    avg_loss = np.mean(truncated_losses, axis=0)
    avg_val_loss = np.mean(truncated_val_losses, axis=0)


    #Updated plotting to include ensemble predictions
    plt.figure(figsize=(8, 4))
    plt.plot(avg_loss, label='Average Total Loss')
    plt.plot(avg_val_loss, label='Average Validation Loss')
    plt.title('Ensemble Average Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
   #Ensemble predictions (average for price, majority vote for trend)
    avg_train_price = np.mean(ensemble_price_train_preds, axis=0)
    avg_test_price = np.mean(ensemble_price_test_preds, axis=0)
    train_trend_votes = np.round(np.mean(ensemble_trend_train_preds, axis=0)).astype(int).flatten()
    test_trend_votes = np.round(np.mean(ensemble_trend_test_preds, axis=0)).astype(int).flatten()
    
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.min_, price_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
    def inverse_price(pred_price):
        return price_scaler.inverse_transform(pred_price.reshape(-1, 1)).flatten()
    
    train_predict_price_inv = inverse_price(avg_train_price)
    y_price_train_inv = inverse_price(y_price_train.reshape(-1, 1))
    test_predict_price_inv = inverse_price(avg_test_price)
    y_price_test_inv = inverse_price(y_price_test.reshape(-1, 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[look_back:look_back + len(train_predict_price_inv)], train_predict_price_inv, label='Ensemble Train Price Predictions')
    plt.plot(df.index[look_back:look_back + len(train_predict_price_inv)], y_price_train_inv, label='Actual Train Prices')
    plt.plot(df.index[look_back + len(train_predict_price_inv):], test_predict_price_inv, label='Ensemble Test Price Predictions')
    plt.plot(df.index[look_back + len(train_predict_price_inv):], y_price_test_inv, label='Actual Test Prices')
    plt.title(f'{ticker} Stock Price Prediction (Ensemble Tuned Model)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()
    
    train_rmse = np.sqrt(np.mean((train_predict_price_inv - y_price_train_inv) ** 2))
    test_rmse = np.sqrt(np.mean((test_predict_price_inv - y_price_test_inv) ** 2))
    print(f'Ensemble Train Price RMSE: {train_rmse:.2f}')
    print(f'Ensemble Test Price RMSE: {test_rmse:.2f}')
    
    train_trend_acc = np.mean(train_trend_votes == y_trend_train)
    test_trend_acc = np.mean(test_trend_votes == y_trend_test)
    print(f'Ensemble Train Trend Accuracy: {train_trend_acc:.2f}')
    print(f'Ensemble Test Trend Accuracy: {test_trend_acc:.2f}')
    
    avg_next_price_pred = np.mean(ensemble_next_price_preds, axis=0)
    avg_next_trend_pred = np.mean(ensemble_next_trend_preds, axis=0)
    next_price_inv = inverse_price(avg_next_price_pred)
    next_trend = 'Up' if avg_next_trend_pred[0][0] > 0.5 else 'Down'
    print(f'Ensemble Predicted next day price for {ticker}: ${next_price_inv[0]:.2f}')
    print(f'Ensemble Predicted next day trend for {ticker}: {next_trend}')

if __name__ == "__main__":
    train_and_predict(ticker='AAPL', start_date='2020-01-01', end_date='2025-09-13', look_back=120, max_epochs=50, tuner_trials=20)
    input("Press Enter to exit...")



