"""
greeks.py
=========
Black-Scholes Greeks computation and delta hedging simulation.
Author : Niraj Neupane | github.com/nirajneupane17
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def greeks(S, K, T, r, sigma):
    """Compute all BS Greeks for a European call."""
    if T <= 0:
        return {'delta':1.0,'gamma':0.0,'vega':0.0,'theta':0.0,'rho':0.0}
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return {
        'delta': norm.cdf(d1),
        'gamma': norm.pdf(d1)/(S*sigma*np.sqrt(T)),
        'vega':  S*norm.pdf(d1)*np.sqrt(T),
        'theta': (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)),
        'rho':   K*T*np.exp(-r*T)*norm.cdf(d2)
    }


def greeks_surface(S, strikes, maturities, r, sigma):
    """Compute Greeks surface across all strikes and maturities."""
    rows = []
    for K in strikes:
        for T in maturities:
            g = greeks(S, K, T, r, sigma)
            rows.append({'Strike':K,'Maturity':T,**{k:round(v,6) for k,v in g.items()}})
    return pd.DataFrame(rows)


def delta_hedge_simulation(S0, K, T, r, sigma, n_days=252, seed=42):
    """
    Simulate daily delta hedging over T years.

    Returns
    -------
    dict with stock path, deltas, option values, hedge P&L
    """
    np.random.seed(seed)
    dt = T/n_days
    Z  = np.random.randn(n_days)

    S = np.zeros(n_days+1); S[0] = S0
    for i in range(n_days):
        S[i+1] = S[i]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[i])

    times   = np.linspace(T, 0, n_days+1)
    from black_scholes import bs_call
    deltas  = np.array([greeks(S[i],K,max(times[i],1e-6),r,sigma)['delta'] for i in range(n_days+1)])
    opt_val = np.array([bs_call(S[i],K,max(times[i],1e-6),r,sigma) for i in range(n_days+1)])
    hedge   = np.array([(opt_val[i+1]-opt_val[i]) - deltas[i]*(S[i+1]-S[i]) for i in range(n_days)])

    return {'stock_path': S, 'deltas': deltas, 'option_values': opt_val,
            'hedge_pnl': hedge, 'cum_pnl': np.cumsum(hedge),
            'total_pnl': hedge.sum()}
