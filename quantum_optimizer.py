import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
from neal import SimulatedAnnealingSampler

def quantum_portfolio_optimization(expected_returns, cov_matrix, budget=None, 
                                   lam:float=0.5, # Risk aversion parameter; can tune this
                                   use_qpu=False):
    """
    Solve portfolio optimization as a QUBO using quantum annealing.

    Parameters:
    - expected_returns: np.ndarray of expected returns.
    - cov_matrix: np.ndarray of covariance matrix.
    - budget: Optional limit on number of assets in portfolio.
    - use_qpu: Whether to use actual QPU (True) or simulated annealing (False)

    Returns:
    - Selected asset indices (binary list)
    """

    num_assets = len(expected_returns)
    Q = {}

    # QUBO formulation: maximize (mu^T x - lambda * x^T Σ x)
    # Convert to minimization: -mu^T x + lambda * x^T Σ x

    for i in range(num_assets):
        for j in range(num_assets):
            Q[(i, j)] = lam * cov_matrix[i, j]
        Q[(i, i)] -= expected_returns[i]

    # Budget constraint penalty: enforce sum(x) = budget
    if budget:
        A = 4.0  # penalty factor
        for i in range(num_assets):
            Q[(i, i)] += A * (1 - 2 * budget)
            for j in range(i + 1, num_assets):
                Q[(i, j)] += 2 * A

    sampler = EmbeddingComposite(DWaveSampler()) if use_qpu else SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=100)
    best_sample = sampleset.first.sample
    return np.array([best_sample[i] for i in range(num_assets)])