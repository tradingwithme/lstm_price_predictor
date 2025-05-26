# LSTM-Based Financial Time Series Prediction + Portfolio Optimization

A modular framework for training and evaluating LSTM models on financial time series data using historical prices. This project now also supports **portfolio optimization using predicted returns**, with a **quantum-inspired hook** for future enhancements.

## 🚀 Features

### 📊 LSTM Modeling

* Download and preprocess financial data.
* Build and train simple LSTM models.
* Automatically search for the best LSTM configuration.
* Fine-tune and combine multiple LSTM models.
* Supports model selection via the `model_name` parameter.

### 💼 Portfolio Optimization

* Use LSTM-predicted prices to estimate returns.
* Compute expected returns and risk (covariance matrix).
* Optimize portfolio weights via **Quantum-inspired optimization** or **Mean-Variance Optimization (MVO)**.
* Generate efficient frontier plots with Sharpe ratios.
* Export optimal portfolio weights as `.json`.

### 🔮 Quantum-Inspired Hook

* The `quantum_enabled = True` flag includes support for quantum-inspired optimization techniques such as **Simulated Annealing**, **Quantum Approximate Optimization Algorithm (QAOA)**, and **Ising Model solvers** (future enhancements).

---

## 🧩 Project Structure

```
project/
│
├── main.py                         # Entrypoint for LSTM model execution
├── model_builder.py                # Builds the LSTM model
├── train_and_predict.py            # Trains and evaluates models
├── bestLSTModelFinder.py           # Finds optimal LSTM configurations
├── fineTune_and_combine_LSTM.py    # Combines and fine-tunes LSTM models
├── data_fetcher.py                 # Downloads historical data
│
├── optimizer.py                    # Runs classical portfolio optimization
├── quantum_optimizer.py            # Handles quantum portfolio optimization
├── utils.py                        # Core logic for risk/return, plotting
├── sample_input.csv                # Dummy or LSTM-predicted prices (for testing)
├── output/
│   ├── weights.json                # Optimal portfolio allocation
│   └── efficient_frontier.png      # Portfolio visualization
```

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/tradingwithme/lstm_price_predictor.git
   cd lstm_price_predictor
   ```

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

### Run Quantum Portfolio Optimization (optional):

```python
from main import main

# Run with quantum optimization enabled
main(
    ticker=["AAPL", "TSLA", "GOOGL", "MSFT"],
    quantumEnabled=True,  # Enables quantum optimization
    budget=3,  # Number of assets to select
    lam=0.5,  # Risk aversion parameter
    epochs=20,
    batch_size=32,
    model_name="simpleLSTMmodel",
    train_ratio=0.8,
    backcandles=60
)
```

---

## 🧮 Parameters

| Parameter        | Type  | Default                          | Description                                                            |
| ---------------- | ----- | -------------------------------- | ---------------------------------------------------------------------- |
| `ticker`         | str   | —                                | Stock or crypto ticker symbol                                          |
| `units`          | int   | `max(16, min(256, len(df)/100))` | LSTM units in the model                                                |
| `epochs`         | int   | 10                               | Number of training epochs                                              |
| `batch_size`     | int   | 15                               | Size of training batches                                               |
| `shuffle`        | bool  | True                             | Shuffle training data                                                  |
| `model_name`     | str   | "simpleLSTMmodel"                | Model type: `simpleLSTMmodel`, `bestLSTMFinder`, `fineTuneCombineLSTM` |
| `train_ratio`    | float | 0.8                              | Ratio of training vs testing data                                      |
| `backcandles`    | int   | 60                               | Number of time steps for LSTM windowing                                |
| `quantumEnabled` | bool  | `False`                          | Enable quantum portfolio optimization using QPU or Simulated Annealing |
| `budget`         | int   | None                             | Limit on the number of assets selected for portfolio optimization      |
| `lam`            | float | 0.5                              | Risk aversion parameter for portfolio optimization                     |

---

## 📈 Output

* Model training logs
* LSTM prediction vs actual plots
* Portfolio efficient frontier (`output/efficient_frontier.png`)
* Optimal portfolio weights (`output/weights.json`)
* Optional: Sharpe ratio, max drawdown, Conditional VaR

---

## 📊 Bonus Metrics

* **Max Drawdown** (planned) ✅
* **Conditional Value at Risk (CVaR)** (planned) ✅
* **Annotated Efficient Frontier (Optimal Portfolio)** ✅

---

## 🔮 Quantum-Inspired Extension

The module includes a placeholder (`quantum_enabled = False`) to support future development of quantum optimization strategies:

* **Simulated Annealing** (current)
* **Quantum Approximate Optimization Algorithm (QAOA)**
* **Ising model solvers**

You can toggle quantum optimization with the `quantumEnabled` flag and control the number of assets in the portfolio with the `budget` parameter. The quantum optimization strategy can be selected by adjusting the `use_qpu` parameter in `quantum_optimizer.py`.

---

## 🛠️ TODO

* Add live model export and loading
* Add dashboard (e.g., Streamlit)
* Integrate other model types (GRU, Transformer)
* Implement real quantum-inspired solvers (QAOA, Ising)

---

## 📄 License

MIT License

> *Developed for academic and educational purposes. Not intended for live trading.*
