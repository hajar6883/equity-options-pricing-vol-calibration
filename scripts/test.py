
from models.black_scholes import BlackScholesModel
from greeks.bs_greeks import bs_delta_call, bs_vega

model = BlackScholesModel(r=0.05, sigma=0.2, q=0.01)

delta = bs_delta_call(S=100, K=100, T=1.0, model=model)
vega  = bs_vega(S=100, K=100, T=1.0, model=model)

print(delta)
