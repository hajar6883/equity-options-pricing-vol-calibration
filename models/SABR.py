import numpy as np 

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