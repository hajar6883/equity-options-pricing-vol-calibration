#core.py
from scipy.stats import norm as Normal  # Normal distributions helpers 
import numpy as np
from math import sqrt


class BlackScholesModel:
    def __init__(self, r, q=0):
        self.r = r
        self.q = q

    # -----------------------------------
    # d1, d2 now always depend on sigma
    # -----------------------------------
    def d1(self, S, K, T, sigma):
        return (np.log(S/K) + (self.r - self.q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    def d2(self, S, K, T, sigma):
        return self.d1(S, K, T, sigma) - sigma*np.sqrt(T)


    # some pricing methods temporal just for the Local Vol workflow (will update the pricing scheme once implemented the one exotic ones ..??)
    
    def call_price(self, S, K, T, sigma):
        if T <= 0: return max(S-K,0)
        d1 = self.d1(S,K,T,sigma); d2 = self.d2(S,K,T,sigma)
        return np.exp(-self.q*T)*S*Normal.cdf(d1) - np.exp(-self.r*T)*K*Normal.cdf(d2)

    def put_price(self, S, K, T, sigma):
        if T <= 0: return max(K-S,0)
        d1 = self.d1(S,K,T,sigma); d2 = self.d2(S,K,T,sigma)
        return np.exp(-self.r*T)*K*Normal.cdf(-d2) - np.exp(-self.q*T)*S*Normal.cdf(-d1)

    #  greeks helper
    def d1d2(self, S,K,T,sigma):
        return self.d1(S,K,T,sigma), self.d2(S,K,T,sigma)

    
    
        
        

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