import os
import csv
import numpy as np
import pandas as pd
from itertools import product
from pandas_ta import macd, rsi
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def preprocess_data(df: pd.DataFrame, backcandles=60):
    scaled_data = MinMaxScaler().fit_transform(df[df.columns])
    
    X, y = [], []
    for i in range(backcandles, len(scaled_data)):
        X.append(scaled_data[i-backcandles:i])
        y.append(scaled_data[i, list(df.columns).index('Close')])
    
    return np.array(X), np.array(y)

def build_and_train_model(X_train, y_train, input_shape, val_split,units:int=50):
    model = Sequential()
    model.add(LSTM(units, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=15, validation_split=val_split, shuffle=False, verbose=0)
    return model

def execute(df, units:int=50):
    # Parameter grids
    macd_params_grid = list(product([8, 12], #fast
                                    [17, 26], #slow
                                    [5, 9]))  #signal
    rsi_params_grid = list(product([10, 14], [100], [1, 2])) 
    train_ratios = (0.8, 0.7, 0.6)
    best_models = {}

    for train_ratio in train_ratios:
        best_rmse = float('inf')
        best_model = None
        best_config = None

        for (fast, slow, signal), (length, scalar, drift) in product(macd_params_grid, rsi_params_grid):
            macd_kwargs = {'fast': fast, 'slow': slow, 'signal': signal}
            rsi_kwargs = {'length': length, 'scalar': scalar, 'drift': drift}
            
            X, y = preprocess_data(df)
            split = int(len(X) * train_ratio)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_and_train_model(X_train, y_train, (X.shape[1], X.shape[2]), val_split=1 - train_ratio, units)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"Ratio {train_ratio:.2f} | MACD({fast},{slow},{signal}) RSI({length},{scalar},{drift}) → MAE={mae:.5f}, RMSE={rmse:.5f}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_config = (macd_kwargs, rsi_kwargs)
                # Save model
                model.save(f"best_model_ratio_{int(train_ratio*100)}.h5")

        best_models[train_ratio] = {'rmse': best_rmse, 'model': best_model, 'config': best_config}
        print(f"✅ Best for train_ratio {train_ratio}: RMSE={best_rmse:.5f} with config {best_config}")