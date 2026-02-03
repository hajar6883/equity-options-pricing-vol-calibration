# weighting scheme:

import numpy as np 

def moneyness_weights(m_grid, sigma_m=0.08):
    """
    Peak at ATM, decay in wings.
    """
    w = np.exp(-0.5 * ((m_grid - 1.0) / sigma_m)**2)
    return w / w.mean()

def maturity_weights(maturities):
    """
    Downweight very short maturities.
    """
    w = np.sqrt(maturities)
    return w / w.mean()


def build_weight_matrix(m_grid, maturities):
    wm = moneyness_weights(m_grid)[None, :]
    wT = maturity_weights(maturities)[:, None]
    W = wT * wm
    return W / W.mean()