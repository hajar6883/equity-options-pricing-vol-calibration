#Root-finding methods to invert the pricing formula.

import numpy as np 
from core import BlackScholesModel
from greeks import CallGreeks



def find_IV_dichotomic( option , market_price , iv_low,iv_high, tolerance=1e-6):
    # monotonicity precondition check

    r = 0.05
    f_low = option.price(BlackScholesModel(r=r, sigma=iv_low)) - market_price
    f_high = option.price(BlackScholesModel(r=r, sigma=iv_high)) - market_price

    assert f_low * f_high < 0,  "Root not bracketed"
    iv_mid = None

    while iv_high - iv_low >tolerance:

        iv_mid = (iv_high + iv_low)/2

        model = BlackScholesModel(r = r, sigma=iv_mid)
        BS_price = option.price(model)
        f = BS_price - market_price

        if f == 0:
            break
        if f < 0:
            iv_low = iv_mid
        else :
            iv_high = iv_mid

    return iv_mid

    

# instead of halfing blindly , we use slope info -> newton_RAPHSON 
def find_IV_newton_raphson(option, market_price, init_guess,
                           max_iter=100, vol_tol=1e-6, price_tol=1e-8):

    iv = init_guess
    r = 0.05

    for _ in range(max_iter):
        model = BlackScholesModel(r=r, sigma=iv)
        price = option.price(model)
        diff = price - market_price

        if abs(diff) < price_tol:
            return iv

        vega = CallGreeks(option, model).vega()
        if vega < 1e-8:
            break

        iv_new = iv - diff / vega

        if abs(iv_new - iv) < vol_tol:
            return iv_new

        iv = iv_new

    return iv


def find_IV_brent(
    option,
    market_price,
    iv_low,
    iv_high,
    r=0.05,
    vol_tol=1e-6,
    price_tol=1e-8,
    max_iter=100
):
    def f(iv):
        return option.price(BlackScholesModel(r=r, sigma=iv)) - market_price

    a, b = iv_low, iv_high
    fa, fb = f(a), f(b)

    assert fa * fb < 0, "Root not bracketed"

    c, fc = a, fa
    d = e = b - a

    for _ in range(max_iter):

        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol = 2 * np.finfo(float).eps * abs(b) + vol_tol
        m = 0.5 * (c - b)

        # ---- stopping condition ----
        if abs(m) <= tol or abs(fb) < price_tol:
            return b

        # ---- try interpolation ----
        if abs(e) >= tol and abs(fa) > abs(fb):

            s = fb / fa

            if a == c:
                # secant
                p = 2 * m * s
                q = 1 - s
            else:
                # inverse quadratic interpolation
                q = fa / fc
                r_ = fb / fc
                p = s * (2 * m * q * (q - r_) - (b - a) * (r_ - 1))
                q = (q - 1) * (r_ - 1) * (s - 1)

            if p > 0:
                q = -q
            p = abs(p)

            if (2 * p < min(3 * m * q - abs(tol * q), abs(e * q))):
                e = d
                d = p / q
            else:
                d = m
                e = m
        else:
            d = m
            e = m

        a, fa = b, fb
        b = b + d if abs(d) > tol else b + np.sign(m) * tol
        fb = f(b)

        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
            e = d = b - a

    return b
