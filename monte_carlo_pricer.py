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


def GBM_simulation(S0, r, sigma, T, N, M, seed=None):
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


def local_vol_simulation(S0, r, LV, T, N, M, seed=None):
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


# -------- Control variates -------------

def control_variate_correction(X,Y, EY, beta=None):

    cov_XY = np.cov(X, Y, ddof=1)[0, 1] # 2x2 -> get the scalar 
    var_Y = np.var(Y, ddof=1)
    if beta is None:
        # optimal beta that make X_adj with the lower variance
        beta = cov_XY / var_Y
    
    X_adj = X + beta*(EY - Y)
    return X_adj , beta


# a dispatcher :
def simulate_paths(method, *args, **kwargs):
    if method == "plain":
        return GBM_simulation(*args, **kwargs)
    elif method == "antithetic":
        return GBM_simulation_antithetic(*args, **kwargs)
    elif method == "local_vol":
        return local_vol_simulation(*args, **kwargs) 
    else:
        raise ValueError("Unknown simulation method")

    

def mc_estimate(discounted_payoffs, alpha = .05):

    m = len(discounted_payoffs)
    mean = np.mean(discounted_payoffs)
    std = np.std(discounted_payoffs, ddof = 1)

    # confidence interval:
    stderr = std / np.sqrt(m)
    z = norm.ppf(1 - alpha / 2) 

    ci_low = mean - z * stderr
    ci_high = mean + z * stderr

    return mean, (ci_low, ci_high), stderr


def mc_pricer(
            payoff_fn,
            payoff_args,
            S0, r, sigma, T,
            N, M,
            sim_method="plain",
            use_control=False,
            alpha=0.05,
            LV=None, 
            seed=None
        ):
    # simulate
    if sim_method == "local_vol":
        if LV is None:
            raise ValueError("LV must be provided when using local_vol simulation")
        paths = local_vol_simulation(S0, r, LV, T, N, M, seed)
    else:
        paths = simulate_paths(sim_method, S0, r, sigma, T, N, M, seed)

    # raw payoffs
    payoffs = np.array([payoff_fn(path, *payoff_args) for path in paths ])

    # discount
    X = np.exp(-r * T) * payoffs

    # control variate (optional)
    if use_control:
        # Discounted stock (control variable Y)
        S_T = paths[:,-1]
        Y = np.exp(-r * T) * S_T
        EY = S0 # known exactly under the risk-neutral measure 

        X , beta = control_variate_correction(X,Y,EY)
    else:
        beta = None

    # estimate
    price, ci, stderr = mc_estimate(X, alpha)

    return price, ci, stderr, beta






    

