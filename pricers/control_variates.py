import numpy as np 
# -------- Control variates -------------

def control_variate_correction(X,Y, EY, beta=None):

    cov_XY = np.cov(X, Y, ddof=1)[0, 1] # 2x2 -> get the scalar 
    var_Y = np.var(Y, ddof=1)
    if beta is None:
        # optimal beta that make X_adj with the lower variance
        beta = cov_XY / var_Y
    
    X_adj = X + beta*(EY - Y)
    return X_adj , beta