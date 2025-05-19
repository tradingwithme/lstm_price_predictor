from warnings import filterwarnings
from model_builder import build_model
from bestLSTModelFinder import execute
from data_fetcher import get_historical_data
global model, X_train, X_test, y_train, y_test
from train_and_predict import train_and_predict
from fineTune_and_combine_LSTM import execute_v2

def main(ticker:str,
         units=None,
         epochs:int=10,
         batch_size:int=15,
         shuffle:bool=True,
         model_name:str='simpleLSTMmodel',  
         train_ratio:float=0.8,backcandles:int=60):
    """
    Main function to build, train, and evaluate LSTM models on financial time series data.

    Parameters:
    -----------
    ticker : str
        The ticker symbol of the financial instrument to be analyzed (e.g., "AAPL", "BTC-USD").
    
    units : int, optional
        Number of LSTM units in the hidden layers. Default dynamically calculated based on dataset size.
    
    epochs : int, optional
        Number of epochs to train the model. Default is 10.
    
    batch_size : int, optional
        Number of samples per gradient update. Default is 15.
    
    shuffle : bool, optional
        Whether to shuffle the training data before each epoch. Default is True.
    
    model_name : str, optional
        The type of model to use. Options:
            - 'simpleLSTMmodel': Builds and trains a basic LSTM model.
            - 'bestLSTMFinder': Runs a hyperparameter search to find the best model.
            - 'fineTuneCombineLSTM': Fine-tunes and combines multiple LSTM models. Default is 'simpleLSTMmodel'.
    
    train_ratio : float, optional
        Proportion of the dataset to use for training. Default is 0.8 (80%).
    
    backcandles : int, optional
        Number of previous time steps (candles) to consider for each prediction window. Default is 60.

    Notes:
    ------
    This function relies on external modules:
        - model_builder
        - train_and_predict
        - bestLSTModelFinder
        - fineTune_and_combine_LSTM
        - data_fetcher

    """
    filterwarnings('ignore')
    df = get_historical_data(df,backcandles=backcandles)
    if units is None: units = max(16, min(256, int(len(df) / 100)))
    if model_name=='simpleLSTMmodel':
        df['TargetNextClose'] = df.Close.shift(-1)
        model, X_train, X_test, y_train, y_test = build_model(df, units=units, backcandles=backcandles, train_ratio=train_ratio)
        train_and_predict(ticker,batch_size=batch_size, epochs=epochs,shuffle=shuffle)
    elif model_name=='bestLSTMFinder': execute(df,units=units)
    elif model_name=='fineTuneCombineLSTM': execute_v2(ticker,units,backcandles=backcandles,batch_size=batch_size)

if __name__ == '__main__':
    main(
        ticker='AAPL',
        units=None,
        epochs=20,
        batch_size=32,
        model_name='simpleLSTMmodel',
        train_ratio=0.8,
        backcandles=60
    )