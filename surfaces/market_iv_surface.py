import numpy as np 
import yfinance as yf
import pandas as pd 
from scipy.interpolate import PchipInterpolator, CubicSpline



def prepare_IV_grid(market_surface):
    """
    Converts market_surface dict into strike/maturity grids.
    
    In
        market_surface : {expiry: {strike: IV}}
    
    Outs:
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


def build_market_iv_surface_moneyness( 
    ticker_symbol: str,
    m_grid = np.linspace(0.8, 1.2, 21),   # moneyness grid
    min_oi: int = 5,
    max_expiries: int = 50,
    interp_kind: str = "linear",          # "linear", "pchip", "cubic"
    use_forward: bool = True,             # m = K/F if True, else K/S
    r: float = 0.0,
    q: float = 0.0,
):
    """
    Build a market implied-vol surface on a common moneyness grid.

    For each expiry, IVs are interpolated onto the same grid, making
    surfaces comparable across maturities.

    interp_kind:
        - "linear": np.interp
        - "pchip" : shape-preserving cubic
        - "cubic" : cubic spline (may overshoot)

    use_forward:
        - True : m = K / F(T) (normalized by the forward prce at the carry rate)
        - False: m = K / S0  (spot money can be misleading ATM today is not ATM at maturity when there is carry)
    """
    ticker = yf.Ticker(ticker_symbol)
    spot = float(ticker.info["currentPrice"])
    expiries = list(ticker.options)[:max_expiries]

    val_date = pd.Timestamp.now(tz="UTC").normalize()

    maturities = []
    Z = []

    for expiry in expiries:
        exp_dt = pd.to_datetime(expiry).tz_localize("UTC")
        T = (exp_dt - val_date).total_seconds() / (365.0 * 24 * 3600)
        if T <= 1/365:
            continue

        chain = ticker.option_chain(expiry)
        calls = chain.calls.copy()

        calls = calls[
            calls["impliedVolatility"].notna()
            & (calls["openInterest"] >= min_oi)
        ]
        if calls.empty:
            continue

        strikes = calls["strike"].to_numpy(dtype=float)
        ivs = calls["impliedVolatility"].to_numpy(dtype=float)

        # forward or spot moneyness
        if use_forward:
            F = spot * np.exp((r - q) * T)
            m = strikes / F
        else:
            m = strikes / spot

        # OTM calls only
        otm_mask = m >= 1.0
        m = m[otm_mask]
        ivs = ivs[otm_mask]

        if len(m) < 5:
            continue

        # sort by moneyness
        order = np.argsort(m)
        m_sorted = m[order]
        iv_sorted = ivs[order]

        # interpolate
        if interp_kind == "linear":
            iv_interp = np.interp(
                m_grid, m_sorted, iv_sorted,
                left=np.nan, right=np.nan
            )
        elif interp_kind == "pchip":
            interp = PchipInterpolator(
                m_sorted, iv_sorted, extrapolate=False
            )
            iv_interp = interp(m_grid)
        elif interp_kind == "cubic":
            interp = CubicSpline(
                m_sorted, iv_sorted, bc_type="natural"
            )
            iv_interp = interp(m_grid)
        else:
            raise ValueError("interp_kind must be linear, pchip, or cubic")

        maturities.append(T)
        Z.append(iv_interp)

    maturities = np.array(maturities)
    Z = np.array(Z)

    idx = np.argsort(maturities)
    maturities = maturities[idx]
    Z = Z[idx]

    return spot, m_grid, maturities, Z
