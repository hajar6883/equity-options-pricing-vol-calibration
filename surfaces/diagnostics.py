import numpy as np 



def common_mask(Z_model, Z_mkt):
    """
    Returns a boolean mask where both surfaces are finite and positive.
    """
    return (np.isfinite(Z_model) & np.isfinite(Z_mkt) & (Z_mkt > 0.0) & (Z_model > 0.0))


def iv_error_surfaces(Z_model, Z_mkt):

    mask = common_mask(Z_model= Z_model, Z_mkt=Z_mkt)

    diff   = np.full_like(Z_model, np.nan)
    rel    = np.full_like(Z_model, np.nan)
    logerr = np.full_like(Z_model, np.nan)

    diff[mask]   = Z_model[mask] - Z_mkt[mask]
    rel[mask]    = diff[mask] / Z_mkt[mask]
    logerr[mask] = np.log(Z_model[mask] / Z_mkt[mask])

    return diff, rel, logerr

def bucket_errors(
    err_surface,
    m_grid,
    maturities,
    m_bins=((0.8,0.9),(0.9,0.97),(0.97,1.03),(1.03,1.1),(1.1,1.2)),
    T_bins=((0.0,0.25),(0.25,0.75),(0.75,2.0),(2.0,5.0))
):
    """
    Aggregate errors by moneyness and maturity buckets.
    Returns dict[(m_bin, T_bin)] = mean abs error
    """
    results = {}

    for mb in m_bins:
        m_mask = (m_grid >= mb[0]) & (m_grid < mb[1])

        for Tb in T_bins:
            T_mask = (maturities >= Tb[0]) & (maturities < Tb[1])

            bucket = err_surface[np.ix_(T_mask, m_mask)]
            val = np.nanmean(np.abs(bucket))

            results[(mb, Tb)] = val

    return results

def maturity_weighted_rmse(err_surface, maturities):
    """
    RMSE per maturity slice.
    """
    rmse = []
    for i, T in enumerate(maturities):
        e = err_surface[i]
        rmse.append(np.sqrt(np.nanmean(e**2)))
    return np.array(rmse)
