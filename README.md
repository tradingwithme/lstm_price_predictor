# LSTM-based Financial Time Series Prediction

A modular framework for training and evaluating LSTM models on financial time series data using historical prices. This project supports model building, training, tuning, and prediction for a variety of LSTM-based architectures.

## 🚀 Features

- Download and preprocess financial data
- Build and train simple LSTM models
- Automatically search for the best LSTM configuration
- Fine-tune and combine multiple LSTM models for better accuracy
- Supports model selection via `model_name` parameter

## 🧩 Project Structure

```

project/
│
├── main\_script.py                 # Contains the main() function
├── model\_builder.py              # Builds the LSTM model
├── train\_and\_predict.py          # Trains and evaluates models
├── bestLSTModelFinder.py         # Finds optimal LSTM configs
├── fineTune\_and\_combine\_LSTM.py  # Combines and fine-tunes models
└── data\_fetcher.py               # Fetches historical data

````

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lstm-finance-predictor.git
   cd lstm-finance-predictor
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Usage

```python
from main_script import main

# Run with a simple LSTM model
main(
    ticker="AAPL",
    epochs=20,
    batch_size=32,
    model_name="simpleLSTMmodel",
    train_ratio=0.8,
    backcandles=60
)

# Or use advanced modes:
main(ticker="AAPL", model_name="bestLSTMFinder")
main(ticker="BTC-USD", model_name="fineTuneCombineLSTM")
```

## 📝 Parameters

| Parameter     | Type  | Default                          | Description                                                      |
| ------------- | ----- | -------------------------------- | ---------------------------------------------------------------- |
| `ticker`      | str   | —                                | Stock or crypto ticker symbol                                    |
| `units`       | int   | `max(16, min(256, len(df)/100))` | LSTM units in the model                                          |
| `epochs`      | int   | 10                               | Number of training epochs                                        |
| `batch_size`  | int   | 15                               | Size of training batches                                         |
| `shuffle`     | bool  | True                             | Shuffle training data                                            |
| `model_name`  | str   | 'simpleLSTMmodel'                | Model type: simpleLSTMmodel, bestLSTMFinder, fineTuneCombineLSTM |
| `train_ratio` | float | 0.8                              | Ratio of training vs testing data                                |
| `backcandles` | int   | 60                               | Number of time steps for LSTM windowing                          |

## 🧪 Models Supported

* **simpleLSTMmodel**: Basic LSTM for regression of next close price.
* **bestLSTMFinder**: Grid or random search to find best-performing hyperparameters.
* **fineTuneCombineLSTM**: Advanced model combining and fine-tuning multiple LSTM variants.

## 📈 Output

* Model training logs
* Prediction vs actual price plots
* Saved models (optional, can be added)

## 🛠️ TODO

* Add logging and metrics export
* Add support for saving models and results
* Integrate more model types (GRU, Transformer)

## 📄 License

MIT License

---

*Developed for academic and educational purposes. Not intended for live trading.*

```

---

Let me know if you'd like help generating the `requirements.txt`, a CLI wrapper, or visualization add-ons.
```
