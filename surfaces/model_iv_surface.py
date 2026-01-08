
import numpy as np
from scipy.optimize import brentq
from models.heston import heston_call_price_cf, heston_mc_terminal_prices
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

def implied_vol_from_price(
    price_fn,
    market_price,
    vol_lo=1e-8,
    vol_hi=5.0,
    tol=1e-6
):
    """
    Generic implied volatility inversion using Brent's method.

    Parameters
    ----------
    price_fn : callable
        Function of volatility: price_fn(vol) -> price
    market_price : float
        Observed option price
    """
    def f(vol):
        return price_fn(vol) - market_price

    return brentq(f, vol_lo, vol_hi, xtol=tol)


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

            # Heston CF gives call; if you later want puts, use parity
            price = heston_call_price_cf(
                S0=spot, K=K, v0=v0, r=r, q=q, T=T,
                kappa=kappa, theta=theta, sigma=sigma, rho=rho,
                u_max=u_max, n_u=n_u
            )

            # Map price -> Black-76 implied vol
            def price_fn(vol):
                return black76_price(F, K, T, df, vol, cp)

            try:
                iv = implied_vol_from_price(
                    price_fn=price_fn,
                    market_price=price
                )
            except ValueError:
                iv = np.nan

            IV[i, j] = iv

        if debug:
            print(f"T={T:.4f}: finite={np.isfinite(IV[i]).sum()}/{len(m_grid)}")

    return IV




def heston_iv_surface_on_m_grid_mc_euler(
    spot,
    m_grid,
    maturities,
    r, q,
    heston_params,  # (v0, kappa, theta, sigma, rho)
    n_steps=1000,
    n_paths=10000,
    seed=1234,
    cp="C",
    debug=False,
):
    v0, kappa, theta, sigma, rho = heston_params
    IV = np.full((len(maturities), len(m_grid)), np.nan, dtype=float)

    rng = np.random.default_rng(seed)

    for i, T in enumerate(maturities):
        df = np.exp(-r * T)
        F = spot * np.exp((r - q) * T)

        # Common random numbers for this maturity (reduces smile noise)
        Z1 = rng.standard_normal((n_steps, n_paths))
        Z2 = rng.standard_normal((n_steps, n_paths))

        # Simulate once per maturity
        ST = heston_mc_terminal_prices(
            S0=spot, v0=v0, r=r, q=q, T=T,
            kappa=kappa, theta=theta, sigma=sigma, rho=rho,
            Z1=Z1, Z2=Z2
        )

        for j, m in enumerate(m_grid):
            K = m * F

            price = df * np.maximum(ST - K, 0.0).mean() # call price under MC
            def price_fn(vol):
                return black76_price(F, K, T, df, vol, cp)

            try:
                iv = implied_vol_from_price(
                    price_fn=price_fn,
                    market_price=price
                )
            except ValueError:
                iv = np.nan

            IV[i, j] = iv

        if debug:
            print(f"T={T:.4f}: finite={np.isfinite(IV[i]).sum()}/{len(m_grid)}")

    return IV




