from scipy.stats import norm as Normal  # Normal distributions helpers 
import numpy as np
from math import sqrt


class BlackScholesModel:
    def __init__(self, r,  sigma, q=0):
        self.r = r
        self.q = q # continuous dividend yield for divendent paying assets
        self.sigma = sigma

    def d1(self, S, K, T ):
        return (np.log(S/K) + (self.r - self.q + 0.5*self.sigma**2)*T) / (self.sigma*sqrt(T))
    
    def d2(self, S, K, T ):
        return self.d1(S, K, T)  - self.sigma*sqrt(T)
    
    # helper for greeks ( to avoid duplicating logic)
    def d1d2(self, option):
        d1 = self.d1(option.S, option.K, option.T)
        d2 = d1 - self.sigma * np.sqrt(option.T)
        return d1, d2
        
        

class Option:
    def __init__(self, S, K, T):
        self.S = S # current stock price
        self.K = K # strike
        self.T = T # maturity 

    def price(self, model):
        raise NotImplementedError
    
    @staticmethod
    def check_bounds(option_type , S, K, r, q, T, price):
        "no-arbitrage bounds for European options"
        disc_S = S* np.exp(-q*T)
        disc_K = K *np.exp(-r*T)
        

        if option_type== "call":
            upper = disc_S
            lower = max(0, disc_S - disc_K)
        if option_type== "put":
            upper = disc_K
            lower = max(0, disc_K -  disc_S)

        return lower <= price <= upper, lower, upper     

class EuropeanCall(Option):
    def price(self, model):
        d1 = model.d1(self.S, self.K, self.T)
        d2 = model.d2(self.S, self.K, self.T)

        return (
            np.exp(-model.q * self.T) * self.S * Normal.cdf(d1) 
            - np.exp(-model.r * self.T) * self.K * Normal.cdf(d2)
        )

    
    
class EuropeanPut(Option):
    def price(self, model):
        d1 = model.d1(self.S, self.K, self.T)
        d2 = model.d2(self.S, self.K, self.T)

        return (
            np.exp(-model.r * self.T) * self.K * Normal.cdf(-d2)
            -np.exp(-model.q * self.T) * self.S * Normal.cdf(-d1) 
            
        )
    


def main():
    model = BlackScholesModel(r=0.05, sigma=0.2, q=0.0)

    call = EuropeanCall(S=100, K=100, T=1.0)
    print("Call price:", call.price(model))
    price = call.price(model)
    ok, low, up = call. check_bounds("call", 100, 100, 0.05, 0, 1, price)
    print("Bounds OK:", ok, "  [", low, ",", up, "]")

    put  = EuropeanPut(S=100, K=100, T=1.0)
    print("Put price :", put.price(model))

if __name__ == "__main__":
    main()