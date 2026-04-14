import numpy as np
from scipy.optimize import minimize


def hagan_SABR(K,F,T, alpha, beta, rho, nu, eps=1e-07):
    """ the hagan approx. for the SABR implied Black vol for european option"""
    if abs(nu) < eps:
        return alpha / ((F * K) ** ((1 - beta) / 2))
    
    one_minus_beta = 1.0 - beta
    FK_beta = (F * K) ** (one_minus_beta / 2.0)

    # Time correction common terms (order-T expansion):

    backbone_convexity = ((one_minus_beta ** 2) * alpha ** 2) / (24 * (F * K) ** one_minus_beta)
    skew_interaction = (rho * beta * nu * alpha) / (4 * FK_beta)
    smile_curvature = ((2 - 3 * rho ** 2) * nu ** 2) / 24

    #ATM case :
    if abs(F - K) < eps:
        vol_atm = (
            alpha / (F ** one_minus_beta)
            * (1 + (backbone_convexity + skew_interaction + smile_curvature) * T)
        )
        return vol_atm
    # Non-ATM (general formula)

    # base(backbone) volatility level (CEV scaling)
    
    log_FK = np.log(F/K)

    z = (nu / alpha) * FK_beta * log_FK    # log-moneyness scalling var:
    xz = np.log((np.sqrt(1.0 - 2.0*rho*z + z*z) + z - rho) / (1.0 - rho))     # skew_smile_adj
    ratio = 1.0 if abs(z) < eps else z / xz

    # sigma_imp(F,K)
    base = alpha / FK_beta
    vol = base * ratio * (1 + (backbone_convexity + skew_interaction + smile_curvature) * T)
    return vol


def calibrate_sabr(
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    F: float,
    T: float,
    beta: float = 1.0,
) -> dict:
    """
    Calibrate SABR (alpha, rho, nu) to a single maturity smile by minimizing
    IV RMSE. Beta is fixed (typically 1.0 for equity/index).

    Parameters
    ----------
    strikes    : array of strikes
    market_ivs : array of market implied vols (same length)
    F          : forward price at maturity T
    T          : time to maturity in years
    beta       : CEV exponent, fixed (default 1.0)

    Returns
    -------
    dict with keys: alpha, beta, rho, nu, rmse
    """
    strikes    = np.asarray(strikes,    dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)

    mask = np.isfinite(market_ivs) & (market_ivs > 0)
    strikes, market_ivs = strikes[mask], market_ivs[mask]

    if len(strikes) < 3:
        raise ValueError("Need at least 3 finite IV points to calibrate SABR.")

    atm_iv = np.interp(F, strikes, market_ivs)

    def _loss(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 0.999:
            return 1e10
        model_ivs = np.array([hagan_SABR(K, F, T, alpha, beta, rho, nu) for K in strikes])
        if not np.all(np.isfinite(model_ivs)):
            return 1e10
        return float(np.mean((model_ivs - market_ivs) ** 2))

    # initial guess: alpha ~ ATM IV (for beta=1), rho ~ -0.3, nu ~ 0.4
    x0     = [atm_iv, -0.3, 0.4]
    bounds = [(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)]

    res = minimize(_loss, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-12})

    alpha, rho, nu = res.x
    model_ivs = np.array([hagan_SABR(K, F, T, alpha, beta, rho, nu) for K in strikes])
    rmse = float(np.sqrt(np.mean((model_ivs - market_ivs) ** 2)))

    return {"alpha": alpha, "beta": beta, "rho": rho, "nu": nu, "rmse": rmse}





    

