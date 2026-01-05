import numpy as np 
from scipy.optimize import brentq
import yfinance as yf
import pandas as pd 

from heston import heston_call_price_cf

from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator, CubicSpline




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


def black76_price(F, K, T, df, vol, cp="C"):

    """When the underlying is a forward or future or when we already normalized by carry -> Equity: interest − dividends
    FX: domestic − foreign rate"""
    cp = cp.upper()
    if T <= 0 or vol <= 0:
        intrinsic = max(F - K, 0.0) if cp == "C" else max(K - F, 0.0)
        return df * intrinsic

    srt = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol * vol * T) / srt
    d2 = d1 - srt

    if cp == "C":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))



def implied_vol_black76(price, F, K, T, df, cp="C", vol_lo=1e-8, vol_hi=5.0):
    cp = cp.upper()
    intrinsic = df * (max(F - K, 0.0) if cp == "C" else max(K - F, 0.0))
    # loose upper bound
    upper = df * (F if cp == "C" else K)

    if not (intrinsic - 1e-12 <= price <= upper + 1e-12):
        return np.nan

    def f(v):
        return black76_price(F, K, T, df, v, cp) - price

    try:
        return brentq(f, vol_lo, vol_hi, maxiter=200)
    except Exception:
        return np.nan

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
            call_price = heston_call_price_cf(
                S0=spot, K=K, v0=v0, r=r, q=q, T=T,
                kappa=kappa, theta=theta, sigma=sigma, rho=rho,
                u_max=u_max, n_u=n_u
            )

            price = call_price
            iv = implied_vol_black76(price, F, K, T, df, cp=cp)

            IV[i, j] = iv

        if debug:
            print(f"T={T:.4f}: finite={np.isfinite(IV[i]).sum()}/{len(m_grid)}")

    return IV


def iv_error_surfaces(Z_model, Z_mkt):
    diff = Z_model - Z_mkt
    logerr = np.log(Z_model / Z_mkt)
    logerr = np.where(
        np.isfinite(Z_model) & np.isfinite(Z_mkt) & (Z_model > 0) & (Z_mkt > 0),
        logerr,
        np.nan
    )
    return diff, logerr
