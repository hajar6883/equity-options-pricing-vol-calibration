"""
Heston synthetic calibration test.

Generates a noiseless IV surface from known parameters using the CF pricer,
then calibrates to recover them. This is the minimum sanity check:
if calibration can't recover its own model's prices, it won't work on
real market data either.

Run from the project root:
    python experiments/scripts/heston_synthetic_calibration.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from scipy.optimize import differential_evolution, minimize

from models.heston import feller_satisfied
from surfaces.model_iv_surface import heston_iv_surface_on_m_grid
from surfaces.diagnostics import iv_error_surfaces, maturity_weighted_rmse
from utils.weights import build_weight_matrix
from utils.plots import plot_iv_slices, plot_rmse_by_maturity


# ── Synthetic market setup ─────────────────────────────────────────────────────
# sigma=0.3 so Feller holds: 2*2*0.04=0.16 > 0.09=0.3²
# Narrower moneyness grid avoids deep-OTM IV inversion failures
TRUE_PARAMS = (0.04, 2.0, 0.04, 0.3, -0.7)   # v0, kappa, theta, sigma, rho
SPOT        = 100.0
R, Q        = 0.03, 0.01
M_GRID      = np.linspace(0.90, 1.10, 11)     # stay away from deep OTM/ITM
MATURITIES  = np.array([0.25, 0.5, 1.0, 2.0])


def _loss(params, spot, m_grid, maturities, r, q, Z_target, W):
    """
    Weighted IV MSE using CF pricing.
    Feller violation is logged as a warning but not hard-rejected —
    practitioners routinely calibrate Heston without Feller holding.
    """
    v0, kappa, theta, sigma, rho = params

    if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 0.999:
        return 1e10

    Z_model = heston_iv_surface_on_m_grid(
        spot=spot, m_grid=m_grid, maturities=maturities,
        r=r, q=q, heston_params=params,
    )

    mask = np.isfinite(Z_model) & np.isfinite(Z_target)
    if mask.sum() < 5:
        return 1e10

    diff = Z_model[mask] - Z_target[mask]
    return float(np.mean(W[mask] * diff ** 2))


def calibrate(spot, m_grid, maturities, r, q, Z_target, W):
    """
    Two-stage calibration:
      1. Differential evolution  — finds the basin
      2. L-BFGS-B refinement     — polishes within the basin
    """
    bounds = [
        (1e-4, 0.5),    # v0
        (0.1,  15.0),   # kappa
        (1e-4, 0.5),    # theta
        (0.05, 2.0),    # sigma
        (-0.98, 0.98),  # rho
    ]

    loss = lambda p: _loss(p, spot, m_grid, maturities, r, q, Z_target, W)

    print("Stage 1 — differential evolution")
    res1 = differential_evolution(
        loss, bounds,
        maxiter=300, popsize=12, tol=1e-7,
        seed=42, disp=True,
    )
    print(f"  best loss: {res1.fun:.2e}  params: {np.round(res1.x, 4)}")

    print("\nStage 2 — L-BFGS-B refinement")
    res2 = minimize(
        loss, res1.x,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-10},
    )
    print(f"  best loss: {res2.fun:.2e}  params: {np.round(res2.x, 4)}")

    return res2.x if res2.fun < res1.fun else res1.x


if __name__ == "__main__":
    print("=" * 60)
    print("HESTON SYNTHETIC CALIBRATION TEST")
    print("=" * 60)

    v0_t, kappa_t, theta_t, sigma_t, rho_t = TRUE_PARAMS
    print(f"\nTrue params:")
    print(f"  v0={v0_t}, kappa={kappa_t}, theta={theta_t}, sigma={sigma_t}, rho={rho_t}")
    print(f"  Feller condition (2κθ > ξ²): {feller_satisfied(kappa_t, theta_t, sigma_t)}")
    print(f"  2κθ = {2*kappa_t*theta_t:.4f},  ξ² = {sigma_t**2:.4f}")

    # 1. Generate noiseless target surface from true params
    print("\nGenerating synthetic IV surface from true params ...")
    Z_target = heston_iv_surface_on_m_grid(
        spot=SPOT, m_grid=M_GRID, maturities=MATURITIES,
        r=R, q=Q, heston_params=TRUE_PARAMS,
    )
    n_finite = np.isfinite(Z_target).sum()
    print(f"  Finite IVs: {n_finite} / {Z_target.size}")
    if n_finite < Z_target.size * 0.8:
        raise RuntimeError(
            f"Only {n_finite}/{Z_target.size} finite IVs — surface too sparse to identify params. "
            "Try narrowing M_GRID or reducing sigma in TRUE_PARAMS."
        )

    # 2. Build weight matrix (ATM-weighted, sqrt-T maturity weighting)
    W = build_weight_matrix(M_GRID, MATURITIES)

    # 3. Calibrate
    print()
    recovered = calibrate(SPOT, M_GRID, MATURITIES, R, Q, Z_target, W)
    v0_r, kappa_r, theta_r, sigma_r, rho_r = recovered

    # 4. Parameter recovery table
    print("\n" + "=" * 60)
    print(f"{'Param':<10} {'True':>10} {'Recovered':>12} {'Abs Error':>12}")
    print("-" * 46)
    for name, true, rec in zip(
        ["v0", "kappa", "theta", "sigma", "rho"],
        TRUE_PARAMS, recovered
    ):
        print(f"{name:<10} {true:>10.4f} {rec:>12.4f} {abs(true - rec):>12.4f}")

    feller_true = feller_satisfied(kappa_t, theta_t, sigma_t)
    feller_rec  = feller_satisfied(kappa_r, theta_r, sigma_r)
    print(f"\n  Feller satisfied — true: {feller_true}, recovered: {feller_rec}")
    if not feller_rec:
        print("  NOTE: Feller violation is common in calibrated Heston params — not a bug.")

    # 5. Fit quality
    Z_fit = heston_iv_surface_on_m_grid(
        spot=SPOT, m_grid=M_GRID, maturities=MATURITIES,
        r=R, q=Q, heston_params=tuple(recovered),
    )
    diff, _, _ = iv_error_surfaces(Z_fit, Z_target)
    rmse_per_T = maturity_weighted_rmse(diff, MATURITIES)

    print("\nRMSE by maturity:")
    for T, rmse in zip(MATURITIES, rmse_per_T):
        status = "OK" if rmse < 0.002 else "WARN"   # 20bp threshold
        print(f"  T={T:.2f}  RMSE={rmse*100:.4f}%  [{status}]")

    overall = np.sqrt(np.nanmean(diff ** 2))
    print(f"\nOverall IV RMSE: {overall*100:.4f}%")
    print("(Target: < 0.01% on synthetic noiseless data)")

    # 6. Plots
    plot_iv_slices(M_GRID, MATURITIES, Z_target, Z_fit)
    plot_rmse_by_maturity(MATURITIES, rmse_per_T, title="Synthetic calibration RMSE")
