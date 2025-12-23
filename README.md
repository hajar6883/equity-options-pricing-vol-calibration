This is a pricing library with modules for valuing European vanilla options using the Black–Scholes–Merton closed-form formula. We then extend the framework to price exotic derivatives using Monte Carlo or automatic differentiation.

This module is essential because it provides a benchmark price for Monte Carlo convergence, analytical Greeks to compare with numerical Greeks, and a control variate for Monte Carlo variance reduction.


## Quick overview 
```text
pricing_engine/
├── core/
│   └── Black–Scholes model (closed-form pricing)
│       └── No-arbitrage bounds
├── greeks/
│   └── analytical_greeks (Delta, Gamma, Vega, …)
├── implied_vol/
│   └── root-finding (Bisection, Newton, Brent)
├── monte_carlo_pricer/
│   ├── GBM path simulator
│   ├── Variance reduction (Antithetic, Control variate)
│   └── Monte Carlo pricer + diagnostics
└── notebooks/
    └── demos
```

