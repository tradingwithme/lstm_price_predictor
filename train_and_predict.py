from numpy import sqrt
from IPython.display import display
from pandas import concat, DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_and_predict(ticker,batch_size:int=15, epochs:int=10,
                      shuffle:bool=True):
    y_pred = model.predict(X_test)
    model.fit(x=X_train,y=y_train,batch_size=batch_size,epochs=epochs,shuffle=shuffle,validation_split=abs(1-train_ratio))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    plot_df = concat([DataFrame(y_test,columns=['test']), 
DataFrame(y_pred,columns=['pred'])],axis=1)
    ax = plot_df.plot(figsize=(10, 6), title=f'{ticker} Actual vs Predicted')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.grid(True)
    plt.show();
    filename=f'{ticker}_prediction_plot.png'
    plt.savefig(filename)
    print(f'Saved output as {filename}')
    model_filename = f'LSTM_predicting_model_for_{ticker}.h5'
    model.save(model_filename)
    print(f'LSTM model saved under file name: {model_filename}')
