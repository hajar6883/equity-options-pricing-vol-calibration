import numpy as np 
from scipy.stats import norm
from mc_engine import simulate_paths
from control_variates import control_variate_correction




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