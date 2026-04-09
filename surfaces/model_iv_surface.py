
import numpy as np
from scipy.optimize import brentq
from models.heston import heston_call_price_cf, heston_call_price_carr_madan, heston_price_mc_euler
from models.black76 import black76_price
from utils.root_finding import implied_vol_from_price



"""Heston model (CF or MC)
        ↓
Option price
        ↓
Black-76 pricing function
        ↓
Generic Brent inverter
        ↓
Implied volatility"""


def heston_iv_surface_on_m_grid(
        spot,
        m_grid,
        maturities,
        r, q,
        heston_params,  # (v0, kappa, theta, sigma, rho)
        u_max=100.0,
        n_u=2001,
        cp="C",
        debug=False,
    ):

    v0, kappa, theta, sigma, rho = heston_params
    IV = np.full((len(maturities), len(m_grid)), np.nan, dtype=float)

    for i, T in enumerate(maturities):
        df = np.exp(-r * T)
        F = spot * np.exp((r - q) * T)

        for j, m in enumerate(m_grid):
            K = m * F

            price = heston_call_price_cf(
                S0=spot, K=K, v0=v0, r=r, q=q, T=T,
                kappa=kappa, theta=theta, sigma=sigma, rho=rho,
                u_max=u_max, n_u=n_u,
            )

            # skip IV inversion for near-zero prices (deep OTM, IV inversion unreliable)
            if price < df * 1e-6:
                IV[i, j] = np.nan
                continue

            def price_fn(vol):
                return black76_price(F, K, T, df, vol, cp)

            try:
                iv = implied_vol_from_price(price_fn=price_fn, market_price=price)
            except (ValueError, RuntimeError):
                iv = np.nan

            IV[i, j] = iv

        if debug:
            print(f"T={T:.4f}: finite={np.isfinite(IV[i]).sum()}/{len(m_grid)}")

    return IV



def heston_price_surface_mc_euler(
    spot, m_grid, maturities, r, q,
    heston_params,
    Zs,                  # FIXED random nbs
    n_steps=400,
):
    """
    Returns model CALL PRICE surface [T x m]
    """

    v0, kappa, theta, sigma, rho = heston_params
    P = np.full((len(maturities), len(m_grid)), np.nan)

    for i, T in enumerate(maturities):
        df = np.exp(-r * T)
        F  = spot * np.exp((r - q) * T)

        Z1, Z2 = Zs[i]

        ST = heston_price_mc_euler(
            spot, v0, r, q, T,
            kappa, theta, sigma, rho,
            n_steps=n_steps,
            Z1=Z1, Z2=Z2
        )

        for j, m in enumerate(m_grid):
            K = m * F
            P[i, j] = df * np.maximum(ST - K, 0.0).mean()

    return P




