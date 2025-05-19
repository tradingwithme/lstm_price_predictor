import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

def data_preprocessing(df : DataFrame, backcandles=60):
    X_data = []
    for i in range(len(df.columns[:-1])):
        X_data.append([])
        for j in range(backcandles, scaled_df.shape[0]): X_data[i].append(scaled_df[j-backcandles:j, i])
    X_data = np.moveaxis(X_data, [0], [2])
    print('X refined shape:',X_data.shape)
    X_data, y_data = np.array(X_data), scaled_df[backcandles:,-1]
    print('X shape:',X_data.shape, 'Y shape:',y_data.shape)
    y_data = np.reshape(y_data,(len(y_data),1))
    print('Y reshaped:',y_data.shape)
    return X_data, y_data
    
def split_data(df : DataFrame, train_ratio:float=0.8,backcandles=60):
    X, y = data_preprocessing()
    splitLimit = int(len(X)*train_ratio)
    X_train, X_test = X[:splitLimit], X[splitLimit:]
    y_train, y_test = y[:splitLimit], y[splitLimit:]
    return X_train, X_test, y_train, y_test

def build_model(df:DataFrame,
                units:int=50,
                backcandles:int=60,
                train_ratio:float=0.8):
    X_train, X_test, y_train, y_test = split_data(df,train_ratio=train_ratio,backcandles=backcandles)
    lstm_input = Input(shape=(60, len(df.columns[:-1])), name="lstm_input")
    inputs = LSTM(units, name="first_layer")(lstm_input)
    inputs = Dense(1, name = 'dense_layer')(inputs)
    output = Activation('linear', name = "output")(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam()
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model, X_train, X_test, y_train, y_test