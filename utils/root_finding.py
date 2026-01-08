# Root-finding methods to invert the pricing formula.

import numpy as np 
from models.black_scholes import BlackScholesModel
from greeks.bs_greeks import bs_vega
from scipy.optimize import brentq



def find_IV_dichotomic( S,K,T, market_price, r,q, iv_low=1e-6, iv_high=5.0, tol=1e-6, cp="call"):

    def price_diff(iv):
        model = BlackScholesModel(r=r, sigma=iv, q=q)
        price = ( model.call_price(S,K,T) if cp == "call" else model.put_price(S,K,T))
        return price - market_price

    f_low = price_diff(iv_low)
    f_high = price_diff(iv_high)
    # monotonicity precondition check

    assert f_low * f_high < 0,  "Root not bracketed"
    iv_mid = None

    while iv_high - iv_low >tol:

        iv_mid = .5 * (iv_high + iv_low)  

        f_mid = price_diff(iv_mid)

        if abs(f_mid) < tol:
            return iv_mid
        if f_mid < 0:
            iv_low = iv_mid
        else:
            iv_high = iv_mid

    return 0.5 * (iv_low + iv_high)

    

# instead of halfing blindly , we use slope info -> newton_RAPHSON 
def find_IV_newton(S,K,T, market_price, r,q, init_guess,
                           max_iter=50, vol_tol=1e-6, price_tol=1e-8, cp="call"):

    iv = init_guess
    r = 0.05

    for _ in range(max_iter):
        model = BlackScholesModel(r=r, sigma=iv, q=q)

        price = (
            model.call_price(S, K, T)
            if cp == "call"
            else model.put_price(S, K, T)
        )

        diff = price - market_price
        if abs(diff) < price_tol:
            return iv

        if cp != "call":
            raise NotImplementedError("Newton method requires vega; implement put vega if needed")

        vega = bs_vega(S, K, T, model)
        if vega < 1e-10:
            break
        iv_new = iv - diff / vega
        if abs(iv_new - iv) < vol_tol:
            return iv_new

        iv = iv_new

    return iv




