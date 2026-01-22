import numpy as np

def mc_gbm_terminal(S0, r, q, sigma, T, Z):

    drift = (r-q-0.5*sigma**2)*T
    diff = sigma * np.sqrt(T)*Z

    return S0 *np.exp(drift + diff )

def price_call_mc(S0, K, r, q, sigma, T, n_paths=200_000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = mc_gbm_terminal(S0, r, q, sigma, T, Z)
    payoff = np.maximum(ST-K, 0.0)
    disc = np.exp(-r*T)
    return disc* payoff.mean(), Z #for CRN reuse (same Z for bumped and unbumped runs to reduce variance)


############# method 1 : Finite differences (bump-and-revalue)
def delta_FD_call( S0, K, r, q, sigma, T, Z, h= 1e-2):

    ST_up   = mc_gbm_terminal(S0 + h, r, q, sigma, T, Z)
    ST_down = mc_gbm_terminal(S0 - h, r, q, sigma, T, Z)
    disc = np.exp(-r * T)
    V_up = disc * np.maximum(ST_up - K, 0.0).mean()
    V_dn = disc * np.maximum(ST_down - K, 0.0).mean()
    return (V_up - V_dn) / (2*h)

def vega_fd_call(S0, K, r, q, sigma, T, Z, h=1e-4):
    ST_up   = mc_gbm_terminal(S0, r, q, sigma + h, T, Z)
    ST_down = mc_gbm_terminal(S0, r, q, sigma - h, T, Z)
    disc = np.exp(-r * T)
    V_up = disc * np.maximum(ST_up - K, 0.0).mean()
    V_dn = disc * np.maximum(ST_down - K, 0.0).mean()
    return (V_up - V_dn) / (2*h)


##### method 2 : pathwise derivatives (infinitesimal perturbation)
# Catch: requires payoff to be differentiable “enough”

def delta_pathwise_call(S0, K, r, q, sigma, T, Z):
    """Pathwise delta for European call under GBM exact simulation."""
    ST = mc_gbm_terminal(S0, r, q, sigma, T, Z)
    indicator = (ST > K).astype(float)
    dST_dS0 = ST / S0
    disc = np.exp(-r * T)
    return disc * np.mean(indicator * dST_dS0)


# (Does NOT work well for discontinuous payoffs (digital, barrier indicators) because derivative involves a Dirac delta)


#method 3 : Likelihood function 
def delta_lrm_call(S0, K, r, q, sigma, T, Z):
    """
    Likelihood ratio delta for European payoff under GBM.
    Uses log ST ~ N(m, s^2). Score wrt m then chain to S0.
    """
    ST = mc_gbm_terminal(S0, r, q, sigma, T, Z)
    payoff = np.maximum(ST - K, 0.0)
    disc = np.exp(-r * T)

    # X = log ST = m + s Z, with m depends on log S0
    # Score wrt m: (X - m)/s^2 = Z / s
    s = sigma * np.sqrt(T)
    score_m = Z / s  # since X - m = sZ
    dm_dS0 = 1.0 / S0  # m includes log S0
    score_S0 = score_m * dm_dS0

    return disc * np.mean(payoff * score_S0)

def vega_lrm_call(S0, K, r, q, sigma, T, Z):
    """
    Likelihood ratio vega for GBM.
    Differentiate log-likelihood of X wrt sigma.
    X = m(sigma) + s(sigma) Z, with m = logS0 + (r-q-0.5 sigma^2)T and s=sigma sqrt(T).
    """
    ST = mc_gbm_terminal(S0, r, q, sigma, T, Z)
    payoff = np.maximum(ST - K, 0.0)
    disc = np.exp(-r * T)

    s = sigma * np.sqrt(T)

    # For X ~ N(m, s^2):
    # d/dsigma log f = (X-m)/s^2 * (dm/dsigma) + [ ((X-m)^2)/s^3 - 1/s ] * (ds/dsigma)
    # Here X-m = sZ
    dm_dsigma = -sigma * T
    ds_dsigma = np.sqrt(T)

    score = ( (s*Z)/ (s**2) ) * dm_dsigma + ( ((s*Z)**2)/(s**3) - 1.0/s ) * ds_dsigma
    # simplify: (Z/s)*dm_dsigma + ( (Z^2)/s - 1/s)*ds_dsigma

    return disc * np.mean(payoff * score)