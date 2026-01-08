import numpy as np

#def heston_cf(u, S0, v0, r, q, T, kappa, theta, sigma, rho):   
#     """
#     Returns Characteristic function of X_T = ln(S_T):
#     phi(u) = E[ exp(i u ln S_T) ] under risk-neutral measure ( using affine/Riccati ODEs , 
#     Feynman–Kac theorem to link PDE to an expectation of a stochastic process)
#     """
#     if np.any(np.isclose(sigma, 0.0)):
#         raise ValueError(f"sigma too small / zero: {sigma}")

#     u = np.asarray(u, dtype=np.complex128)
#     x0 = np.log(S0)
#     a = kappa* theta
#     b = kappa

#     iu = 1j*u
#     d = np.sqrt((rho * sigma * iu - b) ** 2 + sigma**2 * (iu + u**2))
#     # g = (b - rho * sigma * iu - d) / (b - rho * sigma * iu + d)
#     g = (b - rho * sigma * iu + d) / (b - rho * sigma * iu - d)


#     exp_neg_dT = np.exp(-d * T)

#     # Avoid problem if (1 - g*exp(-dT)) is near zero
#     one_minus_g_exp = 1.0 - g * exp_neg_dT
#     one_minus_g = 1.0 - g

#     D = ((b - rho * sigma * iu - d) / sigma**2) * ((1.0 - exp_neg_dT) / one_minus_g_exp)

#     C = (r - q) * iu * T + (a / sigma**2) * (
#         (b - rho * sigma * iu - d) * T - 2.0 * np.log(one_minus_g_exp / one_minus_g)
#     )

#     return np.exp(C + D * v0 + iu * x0)

#a stable version 

# ????????????
import numpy as np

def heston_cf(u, S0, v0, r, q, T, kappa, theta, sigma, rho):
    u = np.asarray(u, dtype=np.complex128)
    x0 = np.log(S0)
    a = kappa * theta
    b = kappa
    iu = 1j * u

    out = np.empty_like(u, dtype=np.complex128)
    mask0 = np.isclose(u, 0.0)
    if np.any(mask0):
        out[mask0] = 1.0 + 0.0j

    um = ~mask0
    if not np.any(um):
        return out

    uu = u[um]
    iuu = iu[um]

    d = np.sqrt((rho * sigma * iuu - b)**2 + sigma**2 * (iuu + uu**2))

    g = (b - rho * sigma * iuu + d) / (b - rho * sigma * iuu - d)

    # enforce |g| < 1 (stabilization)
    flip = np.abs(g) > 1.0
    g[flip] = 1.0 / g[flip]

    exp_neg_dT = np.exp(-d * T)

    one_minus_g = 1.0 - g
    one_minus_g_exp = 1.0 - g * exp_neg_dT

    eps = 1e-14
    one_minus_g = np.where(np.abs(one_minus_g) < eps, eps + 0j, one_minus_g)
    one_minus_g_exp = np.where(np.abs(one_minus_g_exp) < eps, eps + 0j, one_minus_g_exp)

    C = (r - q) * iuu * T + (a / sigma**2) * (
        (b - rho * sigma * iuu + d) * T
        - 2.0 * np.log(one_minus_g_exp / one_minus_g)
    )

    D = ((b - rho * sigma * iuu + d) / sigma**2) * ((1.0 - exp_neg_dT) / one_minus_g_exp)

    expo = C + D * v0 + iuu * x0

    # ---- overflow guard: clip real part of exponent ----
    # exp(700) is near float overflow; keep margin
    re = np.real(expo)
    re_clip = np.clip(re, -700.0, 700.0)
    expo = re_clip + 1j * np.imag(expo)

    out[um] = np.exp(expo)
    return out


def _simpson(y: np.ndarray, x: np.ndarray) -> float:
    """
    Simpson's rule for evenly spaced grid.
    Requires odd number of points.
    """
    n = len(x)
    if n < 3 or (n % 2 == 0):
        raise ValueError("Simpson requires an odd number of points >= 3.")
    h = x[1] - x[0]
    return (h / 3.0) * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-2:2].sum())


def heston_call_price_cf(
    S0: float,
    K: float,
    v0: float,
    r: float,
    q: float,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    u_max: float = 200.0,
    n_u: int = 4001,
) -> float:
    """
    European call via Heston semi-closed form:
        C = S0*e^{-qT}*P1 - K*e^{-rT}*P2
    with P1, P2 computed by Fourier integrals ( see heston_model.ipynb)

    Integration is truncated at u_max and evaluated with Simpson.
    """
    if n_u % 2 == 0:
        n_u += 1  # make it odd for Simpson

    u = np.linspace(1e-6, u_max, n_u) # avoid division by zero
    lnK = np.log(K)

    # P2 integrand uses phi(u)

    phi_u = heston_cf(u, S0, v0, r, q, T, kappa, theta, sigma, rho)
    
    integrand_P2 = np.real(np.exp(-1j * u * lnK) * phi_u / (1j * u) )
    integrand_P2 = np.nan_to_num(integrand_P2)
    
    phi_um_i = heston_cf(u - 1j, S0, v0, r, q, T, kappa, theta, sigma, rho)

    phi_minus_i = S0 * np.exp((r - q) * T)  # <-- do this, not CF(-i)
    
    # print("phi_u finite?", np.isfinite(phi_u).all())
    # print("phi_um_i finite?", np.isfinite(phi_um_i).all())
    # print("phi_minus_i =", phi_minus_i, "finite?", np.isfinite(phi_minus_i))
        



    integrand_P1 = np.real(np.exp(-1j * u * lnK) * (phi_um_i / phi_minus_i) / (1j * u))

    P2 = 0.5 + (1.0 / np.pi) * _simpson(integrand_P2, u)
    P1 = 0.5 + (1.0 / np.pi) * _simpson(integrand_P1, u)

    call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    return float(np.real(call))




def heston_mc_terminal_prices(
    S0, v0, r, q, T, kappa, theta, sigma, rho,
    n_steps=1000, n_paths=100000, Z1=None, Z2=None, seed=None):
    """
    - Simulate Heston dynamics using Euler discretisation scheme
    - Simulate log price (not price) and returns only terminal prices S_T
    - Enforces positivity via truncation
    Z1, Z2 shape: (n_steps, n_paths)
    """

    if seed is not None:
        np.random.seed(seed)

    if Z1 is None or Z2 is None:
        Z1 = np.random.randn(n_steps, n_paths)
        Z2 = np.random.randn(n_steps, n_paths)


    n_steps, n_paths = Z1.shape
    dt = T / n_steps
    sqdt = np.sqrt(dt)

    x = np.full(n_paths, np.log(S0), dtype=np.float64)
    v = np.full(n_paths, v0, dtype=np.float64)

    sqrt_1mr2 = np.sqrt(max(1.0 - rho * rho, 0.0))

    for t in range(n_steps):
        z1 = Z1[t]
        z2 = Z2[t]

        dWv = sqdt * z1
        dWs = sqdt * (rho * z1 + sqrt_1mr2 * z2)

        v_pos = np.maximum(v, 0.0)

        v = v + kappa * (theta - v_pos) * dt + sigma * np.sqrt(v_pos) * dWv
        v = np.maximum(v, 0.0)

        x = x + (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dWs

    return np.exp(x)








      