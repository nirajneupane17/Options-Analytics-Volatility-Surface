"""
implied_vol.py
==============
Implied volatility extraction and volatility surface construction.
Author : Niraj Neupane | github.com/nirajneupane17
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from black_scholes import bs_call, bs_put


def implied_vol(price, S, K, T, r, option_type='call'):
    """Extract implied vol via Brent method."""
    try:
        f = lambda s: (bs_call(S,K,T,r,s) if option_type=='call' else bs_put(S,K,T,r,s)) - price
        return brentq(f, 1e-6, 10.0)
    except:
        return np.nan


def vol_surface_model(K, S, T, atm_vol=0.20, skew=-0.12, kurt=0.06):
    """Parametric equity skew volatility surface."""
    m = np.log(K/S) / np.sqrt(T)
    return np.clip((atm_vol+0.03*np.exp(-T)) + (skew/np.sqrt(T))*m + (kurt/T)*m**2, 0.05, 0.80)


def build_vol_surface(S, strikes, maturities, r):
    """Build volatility surface DataFrame from model."""
    rows = []
    for T in maturities:
        for K in strikes:
            vol = vol_surface_model(K, S, T)
            rows.append({'Maturity':round(T,2), 'Strike':K,
                         'ImpliedVol':round(vol*100,3),
                         'CallPrice': round(bs_call(S,K,T,r,vol),4),
                         'Moneyness': round(np.log(K/S)/np.sqrt(T),4)})
    return pd.DataFrame(rows)


def vol_risk_premium(realised_vol, implied_vol_est):
    """Compute volatility risk premium = implied - realised."""
    vrp = implied_vol_est - realised_vol
    return {'mean_vrp': round(vrp.mean(),4), 'std_vrp': round(vrp.std(),4),
            'pct_positive': round((vrp>0).mean()*100,1), 'series': vrp}
