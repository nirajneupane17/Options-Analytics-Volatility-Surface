"""
black_scholes.py
================
Black-Scholes European options pricing and put-call parity.
Author : Niraj Neupane | github.com/nirajneupane17
"""
import numpy as np
from scipy.stats import norm


def bs_call(S, K, T, r, sigma):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0: return max(S-K, 0)
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0: return max(K-S, 0)
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def put_call_parity(S, K, T, r, call_price=None, put_price=None):
    """Verify put-call parity: C - P = S - K*exp(-rT)."""
    pv_k = K*np.exp(-r*T)
    if call_price and put_price:
        lhs = call_price - put_price
        rhs = S - pv_k
        return {'LHS (C-P)': round(lhs,4), 'RHS (S-PV(K))': round(rhs,4),
                'difference': round(abs(lhs-rhs),6), 'parity_holds': abs(lhs-rhs) < 0.01}
    return {'PV_strike': round(pv_k,4)}


def pricing_table(S, r, sigma, strikes, maturities):
    """Generate options pricing table across strikes and maturities."""
    import pandas as pd
    rows = []
    for K in strikes:
        for T in maturities:
            rows.append({'Strike': K, 'Maturity': round(T,2),
                         'Call': round(bs_call(S,K,T,r,sigma),4),
                         'Put':  round(bs_put(S,K,T,r,sigma),4),
                         'Moneyness': 'ITM' if S>K else ('ATM' if S==K else 'OTM')})
    return pd.DataFrame(rows)
