from scipy.stats import norm as Normal  # Normal distributions helpers 
import numpy as np
from math import sqrt




class AnalyticGreeks:
    "Greeks depend on (option, model) — but model contains σ , Greeks object must never cache d1/d2"
    def __init__(self, option, model):
        self.o = option
        self.m = model

    # def _d1d2(self):
    #     d1 = self.m.d1(self.o.S, self.o.K, self.o.T)
    #     d2 = self.m.d2(self.o.S, self.o.K, self.o.T)
    #     return d1, d2
    def _d1d2(self):
       return self.m.d1d2(self.o)



class CallGreeks(AnalyticGreeks):
    def delta(self):
        d1, _ = self._d1d2()
        return np.exp(-self.m.q * self.o.T) * Normal.cdf(d1)

    def gamma(self):
        d1, _ = self._d1d2()
        return np.exp(-self.m.q * self.o.T) * Normal.pdf(d1) / (self.o.S * self.m.sigma * np.sqrt(self.o.T))

    def vega(self):
        d1, _ = self._d1d2()
        return self.o.S * np.exp(-self.m.q * self.o.T) * Normal.pdf(d1) * np.sqrt(self.o.T)

    def theta(self):
        " ! This is annualised theta. In practice, we often quote **per-day** theta"
        d1, d2 = self._d1d2()
        term1 = - (self.o.S * np.exp(-self.m.q * self.o.T) * Normal.pdf(d1) * self.m.sigma) / (2 * np.sqrt(self.o.T))
        term2 = - self.m.q * self.o.S * np.exp(-self.m.q * self.o.T) * Normal.cdf(d1)
        term3 = self.m.r * self.o.K * np.exp(-self.m.r * self.o.T) * Normal.cdf(d2)
        return term1 + term2 + term3

    def rho(self):
        " ! This return ∂Price / ∂r where r is in absolute units , Merket rho is usually per 1% change ( so divide by 100 :)"
        _, d2 = self._d1d2()
        return self.o.K * self.o.T * np.exp(-self.m.r * self.o.T) * Normal.cdf(d2)


class PutGreeks(AnalyticGreeks):
    def delta(self):
        d1, _ = self._d1d2()
        return np.exp(-self.m.q * self.o.T) * (Normal.cdf(d1) - 1)

    def gamma(self):
        d1, _ = self._d1d2()
        return np.exp(-self.m.q * self.o.T) * Normal.pdf(d1) / (self.o.S * self.m.sigma * np.sqrt(self.o.T))

    def vega(self):
        d1, _ = self._d1d2()
        return self.o.S * np.exp(-self.m.q * self.o.T) * Normal.pdf(d1) * np.sqrt(self.o.T)

    def theta(self):
        d1, d2 = self._d1d2()
        term1 = - (self.o.S * np.exp(-self.m.q * self.o.T) * Normal.pdf(d1) * self.m.sigma) / (2 * np.sqrt(self.o.T))
        term2 = + self.m.q * self.o.S * np.exp(-self.m.q * self.o.T) * Normal.cdf(-d1)
        term3 = - self.m.r * self.o.K * np.exp(-self.m.r * self.o.T) * Normal.cdf(-d2)
        return term1 + term2 + term3

    def rho(self):
        _, d2 = self._d1d2()
        return - self.o.K * self.o.T * np.exp(-self.m.r * self.o.T) * Normal.cdf(-d2)


