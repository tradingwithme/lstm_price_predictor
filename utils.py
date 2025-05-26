import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

quantum_enabled = False # The purpose of this variable is to serve as a future placeholder 

def compute_statistics(returns: pd.DataFrame):
    mu = returns.mean().values * 252
    sigma = returns.cov().values * 252
    return mu, sigma

def mean_variance_optimization(expected_returns, cov_matrix, risk_free_rate=0.02, target_return=None):
    expected_returns = np.array(expected_returns)
    n_assets = len(expected_returns)
    w = cp.Variable(n_assets)
    if target_return is None: target_return = expected_returns.mean()
    risk = cp.quad_form(w, cov_matrix)
    constraints = [cp.sum(w) == 1, w >= 0, expected_returns @ w >= target_return]
    problem = cp.Problem(cp.Minimize(risk), constraints)
    problem.solve()
    weights = w.value
    sharpe = (expected_returns @ weights - risk_free_rate) / np.sqrt(weights.T @ cov_matrix @ weights)
    return weights, sharpe

def simulate_frontier(expected_returns, cov_matrix, n_portfolios=5000, risk_free_rate=0.02):
    results = {'return': [], 'volatility': [], 'sharpe': []}
    for _ in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(len(expected_returns)))
        ret = np.dot(weights, expected_returns)
        vol = np.sqrt(np.dot(weights.T, cov_matrix @ weights))
        sharpe = (ret - risk_free_rate) / vol
        results['return'].append(ret)
        results['volatility'].append(vol)
        results['sharpe'].append(sharpe)
    return pd.DataFrame(results)

def plot_efficient_frontier(df, expected_returns, cov_matrix, weights, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['volatility'], df['return'], c=df['sharpe'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='sharpe ratio')
    opt_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    opt_ret = expected_returns @ weights
    plt.scatter(opt_vol, opt_ret, c='red', s=100, marker='*', label='Optimal Portfolio')
    plt.title("efficient frontier")
    plt.xlabel("volatility based on standard deviation)")
    plt.ylabel("expected return")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()