
import numpy as np
from scipy.interpolate import RectBivariateSpline
from core import BlackScholesModel


def prepare_IV_grid(market_surface):
    """
    Converts market_surface dict into strike/maturity grids.
    
    Input:
        market_surface : {expiry: {strike: IV}}
    
    Output:
        K  - sorted strikes (1D array)
        T  - maturity index (1D array)
        Z  - IV grid [T x K] with NaNs where missing
        maturities - list of expiries in original order
    """
    
    maturities = list(market_surface.keys())
    strikes = sorted(list(set(k for d in market_surface.values() for k in d.keys())))
    
    T = np.arange(len(maturities))                 # later convert to year fraction
    K = np.array(strikes)

    Z = np.zeros((len(T), len(K)))

    for i,m in enumerate(maturities):
        for j,s in enumerate(K):
            Z[i,j] = market_surface[m].get(s, np.nan)

    return K, T, Z, maturities


def build_local_vol_surface(S0, market_surface, r=0.03):
    """Build IV_surface(K,T)
    Interpolate smoothly (bivariate spline)
    Compute option prices C(K,T) via Black-Scholes
    Compute partial derivatives C_T, C_K, C_KK
    Compute σ_loc(K,T) using Dupire formula
    Store as 2D table or interpolation function"""
    K, T, IV_grid, _ = prepare_IV_grid(market_surface)

    print('IV_grid:\n', IV_grid )

    # Interpolate IV(K,T)
    IV_interp = RectBivariateSpline(T, K, IV_grid)

    # convert IV-grid → Call Price grid
    model = BlackScholesModel(r=r)   # sigma not needed now

    C = np.zeros_like(IV_grid)
    for i,t in enumerate(T):
        for j,k in enumerate(K):
            # sigma = float(IV_interp(t,k))
            # sigma = IV_interp(t, k)[0,0]      
            sigma = np.squeeze(IV_interp(t,k)) 

            print(sigma)
            
            C[i,j] = model.call_price(S0, k, t+1e-6, sigma) # avoid T=0

    # partial derivatives
    C_T  = np.gradient(C, T, axis=0)
    C_K  = np.gradient(C, K, axis=1)
    C_KK = np.gradient(C_K, K, axis=1)

    # Dupire formula
    local_vol = np.sqrt( np.maximum( 
        2*C_T / (K[np.newaxis,:]**2 * C_KK + 2*K[np.newaxis,:]*C_K), 
        0 )
    )

    # Return interpolation function
    LV_interp = RectBivariateSpline(T, K, local_vol)

    return LV_interp      # callable: LV(K,T) → σ_loc
