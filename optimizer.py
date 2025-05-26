import pandas as pd
import numpy as np
import json

# Add at the bottom of optimizer.py
def run_optimizer(df_pred: pd.DataFrame, ticker_list: list):
    from utils import (
        compute_statistics,
        mean_variance_optimization,
        simulate_frontier,
        plot_efficient_frontier,
        quantum_enabled
    )
    import json

    RISK_FREE_RATE = 0.02
    OUTPUT_WEIGHT_FILE = 'output/weights.json'
    OUTPUT_PLOT_FILE = 'output/efficient_frontier.png'

    returns = df_pred[ticker_list].pct_change().dropna()
    expected_returns, cov_matrix = compute_statistics(returns)
    weights, sharpe = mean_variance_optimization(expected_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE)

    # Save weights
    weights_dict = dict(zip(df_pred.columns, weights))
    with open(OUTPUT_WEIGHT_FILE, 'w') as f: json.dump(weights_dict, f, indent=4)
    print(f"Weights saved to {OUTPUT_WEIGHT_FILE}")
    print(f"Expected Returns: {expected_returns}")
    frontier_df = simulate_frontier(expected_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE)
    plot_efficient_frontier(frontier_df, expected_returns, cov_matrix, weights, OUTPUT_PLOT_FILE)
    print(f"Portfolio optimization complete. Sharpe Ratio: {sharpe:.4f}")

