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
equity-options-pricing-vol