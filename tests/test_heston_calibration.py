"""
Heston unit tests + calibration smoke test.

Run: pytest tests/ -v -s
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from models.heston import heston_call_price_cf, heston_cf, feller_satisfied
from surfaces.model_iv_surface import heston_iv_surface_on_m_grid
from surfaces.diagnostics import iv_error_surfaces
from experiments.scripts.heston_synthetic_calibration import (
    TRUE_PARAMS, SPOT, R, Q, M_GRID, MATURITIES, calibrate
)
from utils.weights import build_weight_matrix


# ── CF unit tests ──────────────────────────────────────────────────────────────

def test_cf_unit_at_zero():
    """phi(0) = 1 by definition."""
    v0, kappa, theta, sigma, rho = TRUE_PARAMS
    phi0 = heston_cf(0.0, SPOT, v0, R, Q, 1.0, kappa, theta, sigma, rho)
    assert abs(phi0 - 1.0) < 1e-10

def test_cf_modulus_bounded():
    """|phi(u)| <= 1 for all real u."""
    v0, kappa, theta, sigma, rho = TRUE_PARAMS
    u = np.linspace(0.1, 100.0, 500)
    phi = heston_cf(u, SPOT, v0, R, Q, 1.0, kappa, theta, sigma, rho)
    assert np.all(np.abs(phi) <= 1.0 + 1e-8)

def test_call_no_arbitrage():
    """max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}."""
    v0, kappa, theta, sigma, rho = TRUE_PARAMS
    T = 1.0
    for K in [80.0, 100.0, 120.0]:
        C = heston_call_price_cf(SPOT, K, v0, R, Q, T, kappa, theta, sigma, rho)
        lower = max(SPOT * np.exp(-Q * T) - K * np.exp(-R * T), 0.0)
        upper = SPOT * np.exp(-Q * T)
        assert lower <= C + 1e-8
        assert C    <= upper + 1e-8

def test_feller_helper():
    assert feller_satisfied(2.0, 0.04, 0.3)
    assert not feller_satisfied(0.5, 0.04, 0.5)


# ── calibration smoke test ─────────────────────────────────────────────────────

def test_synthetic_calibration_rmse():
    """
    Calibration must recover its own model's prices on noiseless data.
    Delegates to heston_synthetic_calibration.calibrate — no logic duplicated.
    Threshold: overall IV RMSE < 0.01%.
    """
    Z_target = heston_iv_surface_on_m_grid(
        spot=SPOT, m_grid=M_GRID, maturities=MATURITIES,
        r=R, q=Q, heston_params=TRUE_PARAMS,
    )
    W = build_weight_matrix(M_GRID, MATURITIES)
    recovered = calibrate(SPOT, M_GRID, MATURITIES, R, Q, Z_target, W)

    Z_fit = heston_iv_surface_on_m_grid(
        spot=SPOT, m_grid=M_GRID, maturities=MATURITIES,
        r=R, q=Q, heston_params=tuple(recovered),
    )
    diff, _, _ = iv_error_surfaces(Z_fit, Z_target)
    rmse = float(np.sqrt(np.nanmean(diff**2)))

    print(f"\n  overall RMSE = {rmse*100:.4f}%")
    print(f"  recovered:   {np.round(recovered, 4)}")

    assert rmse < 1e-4, f"RMSE {rmse*100:.4f}% exceeds 0.01% threshold"
