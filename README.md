# LSTM-Based Financial Time Series Prediction + Portfolio Optimization

A modular framework for training and evaluating LSTM models on financial time series data using historical prices. This project now also supports **portfolio optimization using predicted returns**, with a **quantum-inspired hook** for future enhancements.

## 🚀 Features

### 📊 LSTM Modeling
- Download and preprocess financial data
- Build and train simple LSTM models
- Automatically search for the best LSTM configuration
- Fine-tune and combine multiple LSTM models
- Supports model selection via the `model_name` parameter

### 💼 Portfolio Optimization
- Use LSTM-predicted prices to estimate returns
- Compute expected returns and risk (covariance)
- Optimize portfolio weights via Mean-Variance Optimization (MVO)
- Generate efficient frontier plots with Sharpe ratios
- Export optimal weights as `.json`

### 🔮 Quantum-Inspired Hook
- `quantum_enabled = False` flag included
- Placeholder for future quantum optimization (QAOA, Ising, Simulated Annealing)

---

## 🧩 Project Structure

```

project/
│
├── main.py                         # Entrypoint for LSTM model execution
├── model\_builder.py                # Builds the LSTM model
├── train\_and\_predict.py            # Trains and evaluates models
├── bestLSTModelFinder.py           # Finds optimal LSTM configs
├── fineTune\_and\_combine\_LSTM.py    # Combines and fine-tunes LSTM models
├── data\_fetcher.py                 # Downloads historical data
│
├── optimizer.py                    # Runs classical portfolio optimization
├── utils.py                        # Contains core logic for risk/return, plotting
├── sample\_input.csv                # Dummy or LSTM-predicted prices (for testing)
├── output/
│   ├── weights.json                # Optimal portfolio allocation
│   └── efficient\_frontier.png      # Portfolio visualization

````

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/tradingwithme/lstm_price_predictor.git
   cd lstm_price_predictor
   ````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Usage

### Run LSTM model:

```python
from main import main

# Run simple LSTM model
main(
    ticker="AAPL",
    units=50,  # or None for auto-tune
    epochs=20,
    batch_size=32,
    model_name="simpleLSTMmodel",
    train_ratio=0.8,
    backcandles=60
)
```

### Run optimizer (after LSTM predictions):

```python
from optimizer import run_optimizer

# Assume df_pred contains predicted prices from LSTM
run_optimizer(df_pred, ticker_list=df_pred.columns.tolist())
```

---

## 🧮 Parameters

| Parameter     | Type  | Default                          | Description                                                            |
| ------------- | ----- | -------------------------------- | ---------------------------------------------------------------------- |
| `ticker`      | str   | —                                | Stock or crypto ticker symbol                                          |
| `units`       | int   | `max(16, min(256, len(df)/100))` | LSTM units in the model                                                |
| `epochs`      | int   | 10                               | Number of training epochs                                              |
| `batch_size`  | int   | 15                               | Size of training batches                                               |
| `shuffle`     | bool  | True                             | Shuffle training data                                                  |
| `model_name`  | str   | "simpleLSTMmodel"                | Model type: `simpleLSTMmodel`, `bestLSTMFinder`, `fineTuneCombineLSTM` |
| `train_ratio` | float | 0.8                              | Ratio of training vs testing data                                      |
| `backcandles` | int   | 60                               | Number of time steps for LSTM windowing                                |

---

## 📈 Output

* Model training logs
* LSTM prediction vs actual plots
* Portfolio efficient frontier (`output/efficient_frontier.png`)
* Optimal weights (`output/weights.json`)
* Optional: Sharpe ratio, max drawdown, Conditional VaR

---

## 📊 Bonus Metrics

* Max Drawdown (planned) ✅
* Conditional Value at Risk (CVaR) ✅
* Annotated Efficient Frontier (Optimal Portfolio) ✅

---

## 🔮 Quantum-Inspired Extension

The module includes a placeholder (`quantum_enabled = False`) to support future development of quantum optimization strategies:

* Simulated Annealing
* Quantum Approximate Optimization Algorithm (QAOA)
* Ising model solvers

---

## 🛠️ TODO

* Add live model export and loading
* Add dashboard (e.g., Streamlit)
* Integrate other model types (GRU, Transformer)
* Implement real quantum-inspired solvers

---

## 📄 License

MIT License

> *Developed for academic and educational purposes. Not intended for live trading.*
