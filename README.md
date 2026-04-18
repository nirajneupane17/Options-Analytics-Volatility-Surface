# Options Analytics & Volatility Surface Construction

Python-based options analytics framework covering Black-Scholes 
pricing, Greeks computation, implied volatility extraction, 
volatility smile and surface construction, and term structure 
analysis across strikes and maturities.

---

## Overview

Volatility surfaces are the backbone of derivatives desks — 
every options price, hedge ratio, and risk metric depends on 
accurately modeling how implied volatility varies across 
strikes and maturities. This project builds a complete options 
analytics engine from Greeks to full 3D surface construction.

---

## Models and Analytics

**Black-Scholes Pricing**
- European call and put pricing
- Put-call parity verification
- Price sensitivity to all inputs

**Greeks Analysis**
- Delta, Gamma, Vega, Theta, Rho
- Greeks surface across strikes and maturities
- Delta hedging simulation and P&L attribution

**Implied Volatility**
- Bisection and Brent method extraction
- Volatility smile construction per maturity
- Smile skew and curvature analysis

**Volatility Surface**
- Full 3D implied vol surface across strikes and maturities
- ATM term structure
- Risk-reversal and butterfly spreads

**Volatility Regime Analysis**
- Implied vs realised volatility spread
- Volatility risk premium estimation
- Term structure shape classification

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/NumPy-%230288D1.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-%234527A0.svg?style=for-the-badge&logo=pandas&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%231565C0.svg?style=for-the-badge&logo=scipy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23C62828.svg?style=for-the-badge&logo=Matplotlib&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%2300C853.svg?style=for-the-badge&logo=plotly&logoColor=white)

---

## Project Structure

```
Options-Analytics-Volatility-Surface/
│
├── data/
│   ├── returns.csv
│   └── prices.csv
│
├── notebooks/
│   ├── 01_black_scholes_pricing.ipynb
│   ├── 02_greeks_analysis.ipynb
│   ├── 03_implied_volatility.ipynb
│   ├── 04_volatility_surface.ipynb
│   └── 05_vol_regime_analysis.ipynb
│
├── src/
│   ├── black_scholes.py
│   ├── greeks.py
│   └── implied_vol.py
│
├── results/
│   ├── 01_bs_pricing_surface.png
│   ├── 02_greeks_heatmaps.png
│   ├── 03_delta_hedging_pnl.png
│   ├── 04_volatility_smile.png
│   ├── 05_volatility_surface_3d.png
│   ├── 06_vol_term_structure.png
│   ├── 07_implied_vs_realised.png
│   └── options_pricing_summary.csv
│
└── README.md
```

---

## Key Results

- Implied vol surface shows pronounced skew for short maturities
  flattening at longer horizons — consistent with equity markets
- Volatility risk premium averages 3 to 5 vol points across 
  the sample period with spikes during stress events
- Delta hedging simulation achieves P&L within 2% of theoretical
  with daily rebalancing across different market regimes
- ATM term structure shifts from backwardation to contango 
  during low-volatility regimes

---

## References

- Black, F. and Scholes, M. (1973) — The Pricing of Options
- Hull, J. — Options, Futures and Other Derivatives
- Gatheral, J. — The Volatility Surface
