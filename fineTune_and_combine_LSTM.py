import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import product
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from data_fetcher import get_historical_data
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def preprocess_data(df: pd.DataFrame,
                    sel_columns:list,
                    desig_columns:str,
                    backcandles=60):
    scaled_data = MinMaxScaler().fit_transform(df[sel_columns])
    X, y = [], []
    for i in range(backcandles, len(scaled_data)):
        X.append(scaled_data[i - backcandles:i])
        y.append(scaled_data[i, list(sel_columns).index(desig_columns)])
    return np.array(X), np.array(y)

def build_model(df, input_shape):
    model = Sequential()
    model.add(LSTM(max(16, min(256, int(len(df) / 100))), return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    closeness = np.mean(np.abs(y_true - y_pred))
    return mae, rmse, closeness

def aggregate_data_from_logs(log_path='model_logs.csv', backcandles=60):
    log_df = pd.read_csv(log_path)
    X_total, y_total = [], []

    for idx, row in log_df.iterrows():
        try:
            macd_params = {
                'fast': int(row['macd_fast']),
                'slow': int(row['macd_slow']),
                'signal': int(row['macd_signal'])
            }
            rsi_params = {
                'length': int(row['rsi_length']),
                'scalar': int(row['rsi_scalar']),
                'drift': int(row['rsi_drift'])
            }
            train_ratio = float(row['train_ratio'])

            df = get_historical_data(ticker, macd_params, rsi_params)
            X, y = preprocess_data(df, backcandles=backcandles)
            split = int(len(X) * train_ratio)

            X_total.append(X[:split])
            y_total.append(y[:split])

        except Exception as e:
            print(f"⚠️ Skipped row {idx} due to error: {e}")
            continue

    return np.concatenate(X_total), np.concatenate(y_total)

def execute_v2(ticker:str,units:int,backcandles:int=60,batch_size:int=15):
    # Parameter grid
    macd_params_grid = list(product([8, 12], [17, 26], [5, 9]))  # (fast, slow, signal)
    rsi_params_grid = list(product([10, 14], [100], [1, 2]))     # (length, scalar, drift)
    train_ratios = [0.8, 0.7, 0.6]

    # Paths
    model_dir = 'saved_models'
    log_file = 'model_logs.csv'
    os.makedirs(model_dir, exist_ok=True)

    # Create CSV and write header (if not already exists)
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
'train_ratio', 'val_split', 
'macd_fast', 'macd_slow', 'macd_signal', 
'rsi_length', 'rsi_scalar', 'rsi_drift',
'MAE', 'RMSE', 'Closeness', 'model_filename'
])

    # Iterate and log
    for train_ratio in train_ratios:
        val_split = 1 - train_ratio

        for (fast, slow, signal), (length, scalar, drift) in product(macd_params_grid, rsi_params_grid):
            macd_kwargs = {'fast': fast, 'slow': slow, 'signal': signal}
            rsi_kwargs = {'length': length, 'scalar': scalar, 'drift': drift}

            df = get_historical_data(ticker, macd_kwargs, rsi_kwargs)
            X, y = preprocess_data(df, df.columns, 'Close', backcandles=backcandles)
            split = int(len(X) * train_ratio)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Create filename
            filename = (
                f"ratio_{int(train_ratio * 100)}"
                f"__macd_{fast}_{slow}_{signal}"
                f"__rsi_{length}_{scalar}_{drift}.h5"
            )
            model_path = os.path.join(model_dir, filename)

            # Load or build model
            if os.path.exists(model_path):
                model = load_model(model_path)
                print(f"🔁 Fine-tuning existing model: {filename}")
                epochs = 5
            else:
                model = build_model(df,(X.shape[1], X.shape[2]))
                print(f"🚀 Training new model: {filename}")
                epochs = 10

            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                      validation_split=val_split, shuffle=False, verbose=0)

            # Evaluate
            y_pred = model.predict(X_test)
            mae, rmse, closeness = evaluate_model(y_test, y_pred)

            # Save model
            model.save(model_path)
            print(f"✅ Saved: {filename} | MAE={mae:.5f}, RMSE={rmse:.5f}, Closeness={closeness:.5f}")

        # Log results
    with open(log_file, mode='a', newline='') as f:
writer = csv.writer(f)
writer.writerow([
train_ratio, val_split,
fast, slow, signal,
length, scalar, drift,
mae, rmse, closeness,
filename
])
    # Load aggregated training data
    X_train, y_train = aggregate_data_from_logs()
    print(f"✅ Aggregated data shape: {X_train.shape}, {y_train.shape}")

    # Load existing model (e.g., best one or your choice)
    model_path = 'saved_models/ratio_80__macd_12_26_9__rsi_14_100_2.h5'  # replace as needed
    model = load_model(model_path)

    # Fine-tune on full aggregated training set
    model.fit(
X_train, y_train,
epochs=10,
batch_size=batch_size,
validation_split=0.2,
shuffle=False,
callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
verbose=1)
    model.save('fine_tuned_model.h5')
    print("✅ Fine-tuned model saved as fine_tuned_model.h5")