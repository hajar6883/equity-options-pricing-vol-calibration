import numpy as np 

def iv_error_surfaces(Z_model, Z_mkt):
    diff = Z_model - Z_mkt
    logerr = np.log(Z_model / Z_mkt)
    logerr = np.where(
        np.isfinite(Z_model) & np.isfinite(Z_mkt) & (Z_model > 0) & (Z_mkt > 0),
        logerr,
        np.nan
    )
    return diff, logerr