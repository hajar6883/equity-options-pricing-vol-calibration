
import numpy as np
from scipy.interpolate import RectBivariateSpline
from models.black_scholes import BlackScholesModel


def build_dupire_local_vol_surface(spot, m_grid, maturities, Z, r=0.03):
    """
    Build a Dupire local vol surface from a moneyness-normalized IV surface.

    Inputs match the output of build_market_iv_surface_moneyness:
        spot      : current spot price
        m_grid    : moneyness grid (K/spot), shape (n_K,)
        maturities: time to expiry in years, shape (n_T,) — must be increasing
        Z         : IV surface, shape (n_T, n_K) — NaNs allowed

    Steps:
        1. Convert moneyness to strikes: K = m * spot
        2. Interpolate IV(K,T) with bivariate spline (fills NaNs implicitly)
        3. Compute C(K,T) via Black-Scholes on the interpolated grid
        4. Differentiate numerically: C_T, C_K, C_KK
        5. Apply Dupire: σ²(K,T) = 2*(C_T + r·K·C_K) / (K²·C_KK)

    Returns:
        LV_interp : callable LV(t, k) → local vol scalar
    """
    S0 = spot
    K = (m_grid * spot).astype(float)
    T = np.asarray(maturities, dtype=float)
    IV_grid = np.where(np.isfinite(Z), Z, 0.0)  # spline needs no NaNs

    # Interpolate IV(K,T)
    IV_interp = RectBivariateSpline(T, K, IV_grid)

    # convert IV-grid → Call Price grid
    C = np.zeros_like(IV_grid)
    for i, t in enumerate(T):
        for j, k in enumerate(K):
            sigma = float(np.squeeze(IV_interp(t, k)))
            model = BlackScholesModel(r=r, sigma=sigma)
            C[i, j] = model.call_price(S0, k, t + 1e-6)  # avoid T=0

    # partial derivatives
    C_T  = np.gradient(C, T, axis=0)
    C_K  = np.gradient(C, K, axis=1)
    C_KK = np.gradient(C_K, K, axis=1)

    # Dupire formula: σ²(K,T) = 2*(C_T + r·K·C_K) / (K² · C_KK)
    numerator   = 2.0 * (C_T + r * K[np.newaxis, :] * C_K)
    denominator = K[np.newaxis, :] ** 2 * C_KK
    local_vol = np.sqrt(np.maximum(numerator / denominator, 0))

    # Return interpolation function
    LV_interp = RectBivariateSpline(T, K, local_vol)

    return LV_interp      # callable: LV(K,T) → σ_loc
