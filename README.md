# Equity Options: Pricing & Volatility Calibration
## Overview
This project is a modular framework for pricing and calibrating equity options, with a strong focus on volatility modeling.

It combines analytic pricing, Monte Carlo simulation, volatility surface construction, and model calibration, and is designed to be extensible to multiple volatility dynamics (local, stochastic, and econometric).
The codebase evolves from a prototype into a structured research framework, emphasizing:
- clean separation between models, numerics, and data
- reproducible Monte Carlo experiments
- realistic calibration and stress-testing workflows (on-going)

## Core features:
- Black–Scholes analytic pricing
- Monte Carlo pricers with variance-reduction techniques
- Stochastic Volatility models
- Market implied-volatility surface construction
- Model-to-market comparison and surface error diagnostics

## Quick layout 
```text
equity-options-pricing-vol-calibration/
│
├── models/
│   ├── black_scholes.py        # Black–Scholes analytic model
│   ├── black76.py              # Black–76 pricing (forward-based)
│   ├── heston.py               # Heston model (CF + MC simulation)
│   └── local_vol.py            # Dupire local volatility construction
│   └── SABR.py            # Forward-based dynamics (Black–76 framework)

│
├── pricers/
│   └── monte_carlo_pricer.py   # Generic Monte Carlo pricing engine
│
├── greeks/
│   └── bs_greeks.py            # Analytic Greeks (Black–Scholes)
│
├── surfaces/
│   ├── market_iv_surface.py    # Market IV surface construction & interpolation
│   └── model_iv_surface.py     # Model-implied IV surfaces (Heston CF / MC)
│
├── utils/
│   └── root_finding.py         # Generic numerical solvers (Brent, etc.)
│
├── experiments/
│   ├── notebooks/              # Exploratory and validation notebooks
│   └── s