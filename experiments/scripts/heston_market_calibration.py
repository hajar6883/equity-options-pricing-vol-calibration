"""
Heston calibration on real market data.

Fetches the IV surface for a given ticker via yfinance, calibrates
Heston parameters, and reports fit quality.

Run from the project root:
    python experiments/scripts/heston_market_calibration.py --ticker SPY
    python experiments/scripts/heston_market_calibration.py --ticker AAPL --max_expiries 12
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from functools import partial
from scipy.optimize import differential_evolution, minimize

from models.heston import feller_satisfied
from surfaces.market_iv_surface import build_market_iv_surface_moneyness
from surfaces.model_iv_surface import heston_iv_surface_on_m_grid
from surfaces.diagnostics import iv_error_surfaces, maturity_weighted_rmse
from utils.weights import build_weight_matrix
from utils.plots import plot_iv_slices, plot_rmse_by_maturity


# ── default run config ─────────────────────────────────────────────────────────
DEFAULT_TICKER    = "SPY"
M_GRID            = np.linspace(0.85, 1.15, 13)
MAX_EXPIRIES      = 50
R                 = 0.05
Q                 = 0.013   # SPY dividend yield ~1.3%


# ── loss and calibration (same structure as synthetic script) ──────────────────

def _loss(params, spot, m_grid, maturities, r, q, Z_target, W):
    v0, kappa, theta, sigma, rho = params
    if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 0.999:
        return 1e10

    # n_u=501 is fast enough for calibration; use 2001 only for final diagnostics
    Z_model = heston_iv_surface_on_m_grid(
        spot=spot, m_grid=m_grid, maturities=maturities,
        r=r, q=q, heston_params=params,
        u_max=100.0, n_u=501,
    )

    mask = np.isfinite(Z_model) & np.isfinite(Z_target)
    if mask.sum() < 5:
        return 1e10

    diff = Z_model[mask] - Z_target[mask]
    return float(np.mean(W[mask] * diff**2))


def calibrate(spot, m_grid, maturities, r, q, Z_target, W):
    bounds = [
        (1e-4, 0.5),    # v0
        (0.1,  15.0),   # kappa
        (1e-4, 0.5),    # theta
        (0.05, 2.0),    # sigma
        (-0.98, 0.98),  # rho
    ]

    loss = partial(_loss, spot=spot, m_grid=m_grid, maturities=maturities,
                   r=r, q=q, Z_target=Z_target, W=W)

    print("Stage 1 — differential evolution")
    res1 = differential_evolution(
        loss, bounds,
        maxiter=200, popsize=8, tol=1e-6,
        seed=42, disp=True, workers=-1, updating="deferred",
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


# ── main ───────────────────────────────────────────────────────────────────────

def main(ticker, max_expiries, r, q):
    print("=" * 60)
    print(f"HESTON MARKET CALIBRATION  —  {ticker}")
    print("=" * 60)

    # 1. fetch market surface
    print(f"\nFetching IV surface for {ticker} ...")
    spot, m_grid, maturities, Z_mkt = build_market_iv_surface_moneyness(
        ticker_symbol=ticker,
        m_grid=M_GRID,
        max_expiries=max_expiries,
        interp_kind="pchip",
        use_forward=True,
        r=r,
        q=q,
    )
    # drop weekly/very short expiries — Heston needs term structure, not just weeklies
    min_T = 0.08   # ~1 month
    keep = maturities >= min_T
    maturities = maturities[keep]
    Z_mkt = Z_mkt[keep]

    print(f"  Spot: {spot:.2f}")
    print(f"  Maturities ({len(maturities)}, >= {min_T:.2f}y): {np.round(maturities, 3)}")
    n_finite = int(np.isfinite(Z_mkt).sum())
    print(f"  Finite IVs: {n_finite} / {Z_mkt.size}")

    if n_finite < 10:
        raise RuntimeError("Too few finite IVs — try a more liquid ticker or wider m_grid.")

    # 2. build weight matrix
    W = build_weight_matrix(m_grid, maturities)

    # 3. calibrate
    print()
    recovered = calibrate(spot, m_grid, maturities, r, q, Z_mkt, W)
    v0, kappa, theta, sigma, rho = recovered

    # 4. parameter table
    print("\n" + "=" * 60)
    print("CALIBRATED HESTON PARAMETERS")
    print("-" * 40)
    print(f"  v0    = {v0:.4f}   (initial variance, sqrt = {np.sqrt(v0):.2%} vol)")
    print(f"  kappa = {kappa:.4f}   (mean-reversion speed)")
    print(f"  theta = {theta:.4f}   (long-run variance, sqrt = {np.sqrt(theta):.2%} vol)")
    print(f"  sigma = {sigma:.4f}   (vol-of-vol)")
    print(f"  rho   = {rho:.4f}   (spot-vol correlation)")
    print(f"\n  Feller condition (2κθ > ξ²): {feller_satisfied(kappa, theta, sigma)}")
    print(f"  2κθ = {2*kappa*theta:.4f},  ξ² = {sigma**2:.4f}")

    # 5. fit quality
    Z_fit = heston_iv_surface_on_m_grid(
        spot=spot, m_grid=m_grid, maturities=maturities,
        r=r, q=q, heston_params=tuple(recovered),
    )

    diff, _, _ = iv_error_surfaces(Z_fit, Z_mkt)
    rmse_per_T = maturity_weighted_rmse(diff, maturities)
    overall    = float(np.sqrt(np.nanmean(diff**2)))

    print("\nRMSE by maturity:")
    print(f"  {'T (y)':<10} {'RMSE (bp)':>10}  {'status'}")
    print("  " + "-" * 30)
    for T, rmse in zip(maturities, rmse_per_T):
        status = "OK" if rmse < 0.02 else "WARN"
        print(f"  {T:<10.3f} {rmse*100:>10.2f}    [{status}]")

    print(f"\n  Overall IV RMSE: {overall*100:.2f} bp")

    # 6. plots
    plot_iv_slices(m_grid, maturities, Z_mkt, Z_fit)
    plot_rmse_by_maturity(maturities, rmse_per_T,
                          title=f"Heston calibration RMSE — {ticker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",       type=str,   default=DEFAULT_TICKER)
    parser.add_argument("--max_expiries", type=int,   default=MAX_EXPIRIES)
    parser.add_argument("--r",            type=float, default=R)
    parser.add_argument("--q",            type=float, default=Q)
    args = parser.parse_args()

    main(args.ticker, args.max_expiries, args.r, args.q)
