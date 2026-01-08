
from scipy.stats import norm as Normal  # Normal distributions helpers 
import numpy as np


class BlackScholesModel:
    def __init__(self, r, sigma, q=0):
        self.r = r
        self.q = q
        self.sigma = sigma

    
    def call_price(self, S, K, T):
        if T <= 0: return max(S-K,0)
        d1 = self.d1(S,K,T,self.sigma); d2 = self.d2(S,K,T,self.sigma)
        return np.exp(-self.q*T)*S*Normal.cdf(d1) - np.exp(-self.r*T)*K*Normal.cdf(d2)

    def put_price(self, S, K, T):
        if T <= 0: return max(K-S,0)
        d1 = self.d1(S,K,T,self.sigma); d2 = self.d2(S,K,T,self.sigma)
        return np.exp(-self.r*T)*K*Normal.cdf(-d2) - np.exp(-self.q*T)*S*Normal.cdf(-d1)

   

    
    
        
        


    

