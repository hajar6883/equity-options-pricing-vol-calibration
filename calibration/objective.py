from models.SABR import hagan_SABR
import numpy as np 



def CalibrateSABR_AllIn(params, F, T, strikes, mkt_ivs, beta=.5 ):
    """
    :param params: model params to calibrate
    :param strikes: Description
    :param mkt_ivs: Dict{strike: corresp. market implied vol}
    :param beta: preset_beta
    """
    alpha, rho, nu = params
    sse = 0.0

    for K in strikes:
        if K not in mkt_ivs:
            continue

        sabr_vol = hagan_SABR(
            K, F, T,
            alpha,
            beta,
            rho,
            nu
        )

        diff = sabr_vol - mkt_ivs[K]
        sse += diff * diff

    return sse


        
