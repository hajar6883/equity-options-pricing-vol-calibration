
import numpy as np
from scipy.stats import norm as Normal


def _d1_d2(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return np.nan, np.nan
    srt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / srt
    d2 = d1 - srt
    return d1, d2


# -------- Call Greeks --------

def bs_delta_call(S, K, T, model):
    d1, _ = _d1_d2(S, K, T, model.r, model.q, model.sigma)
    return np.exp(-model.q * T) * Normal.cdf(d1)


def bs_gamma(S, K, T, model):
    d1, _ = _d1_d2(S, K, T, model.r, model.q, model.sigma)
    return (
        np.exp(-model.q * T)
        * Normal.pdf(d1)
        / (S * model.sigma * np.sqrt(T))
    )


def bs_vega(S, K, T, model):
    d1, _ = _d1_d2(S, K, T, model.r, model.q, model.sigma)
    return S * np.exp(-model.q * T) * Normal.pdf(d1) * np.sqrt(T)


def bs_theta_call(S, K, T, model):
    d1, d2 = _d1_d2(S, K, T, model.r, model.q, model.sigma)

    term1 = -(
        S * np.exp(-model.q * T)
        * Normal.pdf(d1)
        * model.sigma
        / (2 * np.sqrt(T))
    )
    term2 = -model.q * S * np.exp(-model.q * T) * Normal.cdf(d1)
    term3 = model.r * K * np.exp(-model.r * T) * Normal.cdf(d2)

    return term1 + term2 + term3


def bs_rho_call(S, K, T, model):
    _, d2 = _d1_d2(S, K, T, model.r, model.q, model.sigma)
    return K * T * np.exp(-model.r * T) * Normal.cdf(d2)


# -------- Put Greeks --------

def bs_delta_put(S, K, T, model):
    d1, _ = _d1_d2(S, K, T, model.r, model.q, model.sigma)
    return np.exp(-model.q * T) * (Normal.cdf(d1) - 1)


def bs_theta_put(S, K, T, model):
    d1, d2 = _d1_d2(S, K, T, model.r, model.q, model.sigma)

    term1 = -(
        S * np.exp(-model.q * T)
        * Normal.pdf(d1)
        * model.sigma
        / (2 * np.sqrt(T))
    )
    term2 = model.q * S * np.exp(-model.q * T) * Normal.cdf(-d1)
    term3 = -model.r * K * np.exp(-model.r * T) * Normal.cdf(-d2)

    return term1 + term2 + term3


def bs_rho_put(S, K, T, model):
    _, d2 = _d1_d2(S, K, T, model.r, model.q, model.sigma)
    return -K * T * np.exp(-model.r * T) * Normal.cdf(-d2)
