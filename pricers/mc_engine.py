# monte_carlo_pricer.py

import numpy as np
from scipy.stats import norm

"""

          paths
            ↓
      raw payoffs per path
            ↓                       ← VARIANCE REDUCTION ( during of after simulation phase )
      discounted cashflows
            ↓
      sample mean → avg price
      sample var → CI
      M ⟶ ∞ → convergence 

"""


def simulate_gbm_paths(S0, r, sigma, T, N, M, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dt = T / N

    Z = np.random.randn(M, N)

    S = np.zeros((M, N + 1))

    S[:, 0] = S0

    for t in range(1, N + 1):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:, t-1])
        
    # return all simulated paths
    return S



# -------- Antithetic variates -------------
"""Use negative correlation to offset variances : 

- generate M/2 paths
- for each path , simulates also its antithetic twin (-Z, 1-U,..)
- compute 2 payoffs -> avg pairwise -> treat each as 1 observation / realization

"""

def GBM_simulation_antithetic(S0, r, sigma, T, N, M, seed=None):

    if seed is not None:
        np.random.seed(seed)
        assert M % 2 == 0, "M must be even for antithetic variates"


    dt = T / N
    half_M = M//2
    Z = np.random.randn(half_M, N)
    Z_antithetic = -Z # since guassian

    Z_full = np.vstack((Z, Z_antithetic))
    S = np.zeros((M, N + 1))

    S[:, 0] = S0

    for t in range(1, N + 1):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt+ sigma * np.sqrt(dt) * Z_full[:, t-1])

    return S


def simulate_local_vol_paths(S0, r, LV, T, N, M, seed=None):
    """
    LV : function sigma = LV(t, S) returning scalar local vol
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    Z = np.random.randn(M, N)

    S = np.zeros((M, N+1))
    S[:,0] = S0

    for t in range(1, N+1):
        current_t = t*dt                         # continuous time
        for i in range(M):
            sigma = LV(current_t, S[i,t-1])[0,0]  # <-- Local vol from surface
            S[i,t] = S[i,t-1] * np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[i,t-1])

    return S





# a dispatcher :
def simulate_paths(
        method,
        S0,
        r,
        sigma,
        T,
        N,
        M,
        LV=None,
        seed=None
    ):
    """
    Central dispatcher for Monte Carlo path generation.

    This function selects and executes the appropriate stochastic
    path simulation method (e.g. GBM, antithetic GBM, local volatility)
    based on the `method` argument. It provides a single entry point
    for all path generation logic, ensuring that pricing code remains
    agnostic to the underlying simulation scheme.

    All model-specific requirements (such as a local volatility
    surface for local-vol simulations) are validated here, making this
    function the sole authority responsible for simulation selection
    and configuration.
    """
    if method == "plain":
        return simulate_gbm_paths(S0, r, sigma, T, N, M, seed)
    elif method == "antithetic":
        return GBM_simulation_antithetic(S0, r, sigma, T, N, M, seed)
    elif method == "local_vol":
        if LV is None:
            raise ValueError("LV must be provided for local vol simulation")
        return simulate_local_vol_paths(S0, r, LV, T, N, M, seed)
    else:
        raise ValueError(f"Unknown simulation method: {method}")








    

