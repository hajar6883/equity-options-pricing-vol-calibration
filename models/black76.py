from scipy.stats import norm as Normal  # Normal distributions helpers 
import numpy as np

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
        return df * (F * Normal.cdf(d1) - K * Normal.cdf(d2))
    else:
        return df * (K * Normal.cdf(-d2) - F * Normal.cdf(-d1))

