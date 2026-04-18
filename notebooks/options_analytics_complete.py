"""
Options Analytics & Volatility Surface Construction
=====================================================
Complete options analytics pipeline in one file:
  - Black-Scholes pricing
  - Greeks computation and heatmaps
  - Delta hedging simulation
  - Implied volatility extraction
  - Volatility smile and 3D surface
  - ATM term structure and risk-reversal
  - Implied vs realised volatility analysis

Author : Niraj Neupane | github.com/nirajneupane17
Project: Options-Analytics-Volatility-Surface
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Dark professional style ───────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor':  '#0d1117',
    'axes.edgecolor':   '#30363d', 'axes.labelcolor': '#c9d1d9',
    'xtick.color':      '#8b949e', 'ytick.color':     '#8b949e',
    'text.color':       '#c9d1d9', 'grid.color':      '#21262d',
    'grid.linewidth':   0.6,       'font.size':        11,
    'axes.titlesize':   13,        'axes.titleweight': 'bold',
    'axes.titlecolor':  '#f0f6fc', 'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d', 'legend.fontsize':  9,
})
COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657', '#79c0ff']

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: BLACK-SCHOLES CORE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def bs_call(S, K, T, r, sigma):
    """Black-Scholes European call option price."""
    if T <= 0 or sigma <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    """Black-Scholes European put option price."""
    if T <= 0 or sigma <= 0: return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def greeks(S, K, T, r, sigma):
    """
    Compute all Black-Scholes Greeks for a European call.

    Returns
    -------
    dict with Delta, Gamma, Vega, Theta, Rho
    """
    if T <= 0:
        return {'delta': 1.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return {
        'delta': norm.cdf(d1),
        'gamma': norm.pdf(d1) / (S*sigma*np.sqrt(T)),
        'vega':  S*norm.pdf(d1)*np.sqrt(T),
        'theta': (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)),
        'rho':   K*T*np.exp(-r*T)*norm.cdf(d2)
    }


def implied_vol(price, S, K, T, r, option_type='call'):
    """
    Extract implied volatility via Brent root-finding method.

    Parameters
    ----------
    price       : observed market option price
    option_type : 'call' or 'put'

    Returns
    -------
    float — implied volatility (or np.nan if not found)
    """
    try:
        f = lambda sigma: (
            bs_call(S, K, T, r, sigma) if option_type == 'call'
            else bs_put(S, K, T, r, sigma)
        ) - price
        return brentq(f, 1e-6, 10.0)
    except Exception:
        return np.nan


def vol_surface_model(K, S, T, atm_vol=0.20, skew=-0.12, kurt=0.06):
    """
    Parametric volatility surface model (equity skew approximation).
    vol(K,T) = atm + skew * moneyness + kurt * moneyness^2
    """
    moneyness = np.log(K/S) / np.sqrt(T)
    atm = atm_vol + 0.03*np.exp(-T)
    s   = skew / np.sqrt(T)
    k   = kurt / T
    return np.clip(atm + s*moneyness + k*moneyness**2, 0.05, 0.80)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: LOAD DATA
# ════════════════════════════════════════════════════════════════════════════

returns = pd.read_csv('data/returns.csv', index_col='Date', parse_dates=True)
spy_ret = returns['SPY']
S0, r, sigma = 100, 0.05, 0.20
print(f"Loaded {len(spy_ret):,} observations | {spy_ret.index[0].date()} to {spy_ret.index[-1].date()}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: PRICING SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

print("\n--- Options Pricing Summary ---")
rows = []
for K in [80, 90, 95, 100, 105, 110, 120]:
    for T in [0.25, 0.5, 1.0, 2.0]:
        c = bs_call(S0, K, T, r, sigma)
        p = bs_put(S0, K, T, r, sigma)
        g = greeks(S0, K, T, r, sigma)
        rows.append({'Strike': K, 'Maturity': T,
                     'Call': round(c, 4), 'Put': round(p, 4),
                     'Delta': round(g['delta'], 4), 'Gamma': round(g['gamma'], 4),
                     'Vega': round(g['vega'], 4),   'Theta': round(g['theta'], 4)})

pricing_df = pd.DataFrame(rows)
pricing_df.to_csv('results/options_pricing_summary.csv', index=False)
print(pricing_df[pricing_df['Maturity'] == 1.0].to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: BS PRICING SURFACE (3D)
# ════════════════════════════════════════════════════════════════════════════

print("\n--- Chart 1: BS Pricing Surface ---")
strikes  = np.linspace(70, 135, 50)
mats     = np.linspace(0.05, 2.0, 50)
K_grid, T_grid = np.meshgrid(strikes, mats)
C_grid = np.vectorize(lambda K, T: bs_call(S0, K, T, r, sigma))(K_grid, T_grid)

fig = plt.figure(figsize=(14, 8), facecolor='#0d1117')
ax  = fig.add_subplot(111, projection='3d', facecolor='#0d1117')
surf = ax.plot_surface(K_grid, T_grid, C_grid, cmap='plasma', alpha=0.92,
                        linewidth=0, antialiased=True)
ax.set_xlabel('Strike (K)', color='#8b949e', labelpad=10)
ax.set_ylabel('Maturity (T)', color='#8b949e', labelpad=10)
ax.set_zlabel('Call Price ($)', color='#8b949e', labelpad=10)
ax.set_title('Black-Scholes Call Price Surface\nS=100 | r=5% | σ=20%', color='#f0f6fc', pad=15)
ax.tick_params(colors='#8b949e')
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.set_edgecolor('#21262d')
ax.grid(True, color='#21262d', linewidth=0.4)
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e')
ax.view_init(elev=28, azim=-55)
plt.tight_layout()
plt.savefig('results/01_bs_pricing_surface.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("   Saved: results/01_bs_pricing_surface.png")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: GREEKS HEATMAPS
# ════════════════════════════════════════════════════════════════════════════

print("\n--- Chart 2: Greeks Heatmaps ---")
strikes_g = np.linspace(70, 135, 60)
mats_g    = np.linspace(0.05, 2.0, 60)
K_g, T_g  = np.meshgrid(strikes_g, mats_g)

delta_g = np.vectorize(lambda K, T: greeks(S0,K,T,r,sigma)['delta'])(K_g, T_g)
gamma_g = np.vectorize(lambda K, T: greeks(S0,K,T,r,sigma)['gamma'])(K_g, T_g)
vega_g  = np.vectorize(lambda K, T: greeks(S0,K,T,r,sigma)['vega'])(K_g, T_g)
theta_g = np.vectorize(lambda K, T: greeks(S0,K,T,r,sigma)['theta'])(K_g, T_g)

fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor='#0d1117')
fig.suptitle('Options Greeks Heatmaps\nS=100 | r=5% | σ=20%', color='#f0f6fc',
             fontsize=14, fontweight='bold', y=1.01)
for ax, data, name, cmap in zip(axes.flat,
                                  [delta_g, gamma_g, vega_g, theta_g],
                                  ['Delta','Gamma','Vega','Theta'],
                                  ['RdYlGn','hot','YlOrRd','RdBu_r']):
    im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap,
                    extent=[strikes_g[0], strikes_g[-1], mats_g[0], mats_g[-1]])
    ax.axvline(S0, color='#f0f6fc', linewidth=1.5, linestyle='--', alpha=0.8, label='ATM')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=8)
    ax.set_title(name, color='#f0f6fc', fontsize=13, fontweight='bold')
    ax.set_xlabel('Strike (K)', color='#8b949e', fontsize=9)
    ax.set_ylabel('Maturity (T, years)', color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig('results/02_greeks_heatmaps.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("   Saved: results/02_greeks_heatmaps.png")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: DELTA HEDGING SIMULATION
# ════════════════════════════════════════════════════════════════════════════

print("\n--- Chart 3: Delta Hedging P&L ---")
n_days = 252; dt = 1/252
S0_h, K_h, T_h = 100, 100, 1.0

Z      = np.random.randn(n_days)
S_path = np.zeros(n_days+1); S_path[0] = S0_h
for i in range(n_days):
    S_path[i+1] = S_path[i] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[i])

times    = np.linspace(T_h, 0, n_days+1)
deltas_h = np.array([greeks(S_path[i], K_h, max(times[i],1e-6), r, sigma)['delta'] for i in range(n_days+1)])
option_v = np.array([bs_call(S_path[i], K_h, max(times[i],1e-6), r, sigma) for i in range(n_days+1)])

hedge_pnl = np.array([
    (option_v[i+1]-option_v[i]) - deltas_h[i]*(S_path[i+1]-S_path[i])
    for i in range(n_days)
])
cum_pnl = np.cumsum(hedge_pnl)
t_axis  = np.arange(n_days)

fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor='#0d1117')
fig.suptitle('Delta Hedging Simulation — Daily Rebalancing\nS₀=100 | K=100 | T=1yr | σ=20%',
             color='#f0f6fc', fontsize=14, fontweight='bold')

axes[0,0].plot(S_path, color='#58a6ff', linewidth=1.5, label='Stock price')
axes[0,0].axhline(K_h, color='#f78166', linewidth=1.2, linestyle='--', alpha=0.8, label=f'K={K_h}')
axes[0,0].fill_between(range(len(S_path)), S_path, K_h, where=S_path>K_h, alpha=0.15, color='#3fb950')
axes[0,0].fill_between(range(len(S_path)), S_path, K_h, where=S_path<=K_h, alpha=0.15, color='#f78166')
axes[0,0].set_title('Simulated Stock Path', color='#f0f6fc')
axes[0,0].set_ylabel('Stock Price ($)', color='#8b949e')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(deltas_h, color='#3fb950', linewidth=1.5)
axes[0,1].fill_between(range(len(deltas_h)), deltas_h, alpha=0.2, color='#3fb950')
axes[0,1].axhline(0.5, color='#8b949e', linewidth=1, linestyle='--', alpha=0.6, label='ATM delta')
axes[0,1].set_title('Delta Over Time', color='#f0f6fc')
axes[0,1].set_ylabel('Delta', color='#8b949e')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].bar(t_axis, hedge_pnl, color=np.where(hedge_pnl>=0,'#3fb950','#f78166'), alpha=0.75, width=0.8)
axes[1,0].axhline(0, color='#8b949e', linewidth=0.8)
axes[1,0].set_title('Daily Hedge P&L', color='#f0f6fc')
axes[1,0].set_ylabel('P&L ($)', color='#8b949e')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(cum_pnl, color='#d2a8ff', linewidth=2)
axes[1,1].fill_between(t_axis, cum_pnl, alpha=0.2, color='#d2a8ff')
axes[1,1].axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
axes[1,1].set_title('Cumulative Hedge P&L', color='#f0f6fc')
axes[1,1].set_ylabel('Cumulative P&L ($)', color='#8b949e')
axes[1,1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel('Trading Days', color='#8b949e')
    ax.tick_params(colors='#8b949e')

plt.tight_layout()
plt.savefig('results/03_delta_hedging_pnl.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("   Saved: results/03_delta_hedging_pnl.png")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7: VOLATILITY SMILE
# ════════════════════════════════════════════════════════════════════════════

print("\n--- Chart 4: Volatility Smile ---")
strikes_smile = np.linspace(75, 130, 40)
mats_smile    = [0.25, 0.5, 1.0, 2.0]
labels_smile  = ['3M', '6M', '1Y', '2Y']

fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='#0d1117')

for T, lbl, c in zip(mats_smile, labels_smile, COLORS):
    smile = vol_surface_model(strikes_smile, S0, T) * 100
    axes[0].plot(strikes_smile, smile, color=c, linewidth=2.5,
                 label=f'{lbl}', marker='o', markersize=3)
axes[0].axvline(S0, color='#8b949e', linewidth=1.2, linestyle='--', alpha=0.7, label='ATM')
axes[0].fill_between(strikes_smile,
                      vol_surface_model(strikes_smile, S0, 0.25)*100,
                      vol_surface_model(strikes_smile, S0, 2.0)*100,
                      alpha=0.08, color='#58a6ff')
axes[0].set_title('Implied Volatility Smile', color='#f0f6fc')
axes[0].set_xlabel('Strike (K)', color='#8b949e')
axes[0].set_ylabel('Implied Vol (%)', color='#8b949e')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

moneyness_ax = np.linspace(-2, 2, 100)
for T, lbl, c in zip(mats_smile, labels_smile, COLORS):
    atm = 0.20 + 0.03*np.exp(-T)
    vol = atm + (-0.12/np.sqrt(T))*moneyness_ax + (0.06/T)*moneyness_ax**2
    axes[1].plot(moneyness_ax, vol*100, color=c, linewidth=2.5, label=lbl)
axes[1].axvline(0, color='#8b949e', linewidth=1.2, linestyle='--', alpha=0.7, label='ATM')
axes[1].set_title('Smile by Log-Moneyness', color='#f0f6fc')
axes[1].set_xlabel('Log-Moneyness', color='#8b949e')
axes[1].set_ylabel('Implied Vol (%)', color='#8b949e')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

for ax in axes: ax.tick_params(colors='#8b949e')
plt.tight_layout()
plt.savefig('results/04_volatility_smile.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("   Saved: results/04_volatility_smile.png")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8: 3D VOLATILITY SURFACE
# ════════════════════════════════════════════════════════════════════════════

print("\n--- Chart 5: 3D Volatility Surface ---")
strikes_3d = np.linspace(70, 135, 50)
mats_3d    = np.linspace(0.1, 2.5, 50)
K3, T3     = np.meshgrid(strikes_3d, mats_3d)
V3         = vol_surface_model(K3, S0, T3) * 100

fig = plt.figure(figsize=(14, 9), facecolor='#0d1117')
ax  = fig.add_subplot(111, projection='3d', facecolor='#0d1117')
surf = ax.plot_surface(K3, T3, V3, cmap='RdYlGn_r', alpha=0.93,
                        linewidth=0, antialiased=True)
atm_vol_ts = vol_surface_model(np.full_like(mats_3d, S0), S0, mats_3d)*100
ax.plot([S0]*len(mats_3d), mats_3d, atm_vol_ts,
        color='#f0f6fc', linewidth=3, label='ATM term structure', zorder=10)
ax.set_xlabel('Strike (K)', color='#8b949e', labelpad=12)
ax.set_ylabel('Maturity (Years)', color='#8b949e', labelpad=12)
ax.set_zlabel('Implied Vol (%)', color='#8b949e', labelpad=12)
ax.set_title('Implied Volatility Surface\nEquity Skew Model', color='#f0f6fc', pad=20)
ax.tick_params(colors='#8b949e', labelsize=8)
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.set_edgecolor('#21262d')
ax.grid(True, color='#21262d', linewidth=0.4)
ax.legend(loc='upper right', fontsize=9)
cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=12, pad=0.1)
cbar.set_label('Implied Vol (%)', color='#8b949e', fontsize=9)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=8)
ax.view_init(elev=22, azim=-50)
plt.tight_layout()
plt.savefig('results/05_volatility_surface_3d.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("   Saved: results/05_volatility_surface_3d.png")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9: TERM STRUCTURE & RISK-REVERSAL
# ════════════════════════════════════════════════════════════════════════════

print("\n--- Chart 6: Term Structure ---")
mats_ts = np.linspace(0.1, 3.0, 100)
atm_ts  = (0.20 + 0.03*np.exp(-mats_ts))*100
rr_25   = (-0.12/np.sqrt(mats_ts))*100
fly_25  = (0.03/mats_ts)*100

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0d1117')
axes[0].plot(mats_ts, atm_ts, color='#58a6ff', linewidth=2.5)
axes[0].fill_between(mats_ts, atm_ts*0.95, atm_ts*1.05, alpha=0.15, color='#58a6ff')
axes[0].set_title('ATM Volatility Term Structure', color='#f0f6fc')
axes[0].set_ylabel('ATM Implied Vol (%)', color='#8b949e')

axes[1].plot(mats_ts, rr_25, color='#f78166', linewidth=2.5)
axes[1].axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
axes[1].fill_between(mats_ts, rr_25, 0, where=rr_25<0, alpha=0.2, color='#f78166')
axes[1].set_title('25Δ Risk Reversal (Skew)', color='#f0f6fc')
axes[1].set_ylabel('Risk Reversal (%)', color='#8b949e')

axes[2].plot(mats_ts, fly_25, color='#3fb950', linewidth=2.5)
axes[2].fill_between(mats_ts, fly_25, alpha=0.15, color='#3fb950')
axes[2].set_title('25Δ Butterfly (Curvature)', color='#f0f6fc')
axes[2].set_ylabel('Butterfly (%)', color='#8b949e')

for ax in axes:
    ax.set_xlabel('Maturity (Years)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/06_vol_term_structure.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("   Saved: results/06_vol_term_structure.png")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10: IMPLIED VS REALISED VOLATILITY
# ════════════════════════════════════════════════════════════════════════════

print("\n--- Chart 7: Implied vs Realised Vol ---")
rv_63  = spy_ret.rolling(63).std() * np.sqrt(252) * 100
iv_est = rv_63 * 1.08 + pd.Series(np.random.normal(0,1.5,len(rv_63)), index=spy_ret.index)
vrp    = (iv_est - rv_63).dropna()

fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor='#0d1117')

axes[0].plot(rv_63.index,  rv_63,  color='#3fb950', linewidth=1.2, label='Realised Vol 63d (%)', alpha=0.9)
axes[0].plot(iv_est.index, iv_est, color='#58a6ff', linewidth=1.2, label='Implied Vol (est.)',   alpha=0.9)
axes[0].fill_between(iv_est.index, rv_63, iv_est, where=iv_est>=rv_63, alpha=0.15, color='#58a6ff', label='Vol Risk Premium')
axes[0].fill_between(iv_est.index, rv_63, iv_est, where=iv_est<rv_63,  alpha=0.15, color='#f78166', label='IV < RV')
axes[0].set_title('Implied vs Realised Volatility', color='#f0f6fc')
axes[0].set_ylabel('Volatility (%)', color='#8b949e')
axes[0].legend(loc='upper right'); axes[0].grid(True, alpha=0.3)

axes[1].fill_between(vrp.index, vrp, 0, where=vrp>=0, color='#3fb950', alpha=0.6, label='Positive VRP')
axes[1].fill_between(vrp.index, vrp, 0, where=vrp<0,  color='#f78166', alpha=0.6, label='Negative VRP')
axes[1].axhline(vrp.mean(), color='#58a6ff', linewidth=1.5, linestyle='--',
                label=f'Mean VRP = {vrp.mean():.1f}%')
axes[1].axhline(0, color='#8b949e', linewidth=0.8)
axes[1].set_title('Volatility Risk Premium', color='#f0f6fc')
axes[1].set_ylabel('VRP (%)', color='#8b949e')
axes[1].legend(loc='upper right'); axes[1].grid(True, alpha=0.3)

for ax in axes:
    ax.tick_params(colors='#8b949e')
plt.tight_layout()
plt.savefig('results/07_implied_vs_realised.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("   Saved: results/07_implied_vs_realised.png")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11: SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  OPTIONS ANALYTICS — COMPLETE SUMMARY")
print("="*55)
atm = greeks(S0, S0, 1.0, r, sigma)
print(f"  ATM Call (T=1yr, σ=20%) : ${bs_call(S0,S0,1,r,sigma):.4f}")
print(f"  ATM Put  (T=1yr, σ=20%) : ${bs_put(S0,S0,1,r,sigma):.4f}")
print(f"  Delta    : {atm['delta']:.4f}")
print(f"  Gamma    : {atm['gamma']:.6f}")
print(f"  Vega     : {atm['vega']:.4f}")
print(f"  Theta    : {atm['theta']:.4f}")
print(f"  Mean VRP : {vrp.mean():.2f}%")
print(f"\n  All 7 charts saved to results/")
print("="*55)
