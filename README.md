# Financial Analysis & Simulation Toolkit

A modular Python toolkit for **time-series analysis** and **Monte Carlo simulations** of stocks, ETFs, and indices using Yahoo Finance data, all easily traversable by a GUI.

## Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## GUI
- Tabs for: Load Data, Analysis, Portfolio, Simulation, Compare, Forecast.
- Multi‑select checkbox lists across tabs for intuitive selection.
- Portfolio tab: assign weights next to each ticker, click "Create/Update Portfolio" to:
  - build a portfolio ticker (normalised weighted price series),
  - compute and show correlation/covariance, volatility and variance contributions,
  - expose the portfolio in other tabs (simulate, compare, forecast) as a first‑class item.
- All heavy computations run off the main thread; plotting and UI updates are scheduled on the main thread to keep the UI responsive.

## Features
**Data loading**: 
- Fetches OHLCV data via `yfinance`, computes log returns, handles invalid tickers gracefully.  

**Analysis**: 
- Annualised returns/volatility, Sharpe ratio, CAGR, Value-at-Risk, growth probabilities, rolling stats, moving averages, relative growth, return plots.  

**Portfolio tools**: 
- Quick creation of weighted portfolios from any combination of tickers.
- Automatic normalisation of weights and construction of a portfolio price series.
- Full analytics: covariance/correlation, annualised volatility, variance % contribution, cumulative return plots.
- Use the portfolio like any other instrument — simulate, compare to real price history, or forecast.

**Simulations**:
- Geometric Brownian Motion (GBM) with thousands of paths, probability of finishing above current price, terminal price stats, visualised sample paths.  
- Comparison between simulated data and real data with the mean and median simulated data plotted against real 'Close' data.

**Forecast**:
- ARIMA+GARCH for price forecasting with volatility-adjusted confidence intervals and optional outlier removal.

## Project Structure
finance_project/

├── main.py # Entry point (CLI orchestration)

├── gui.py # GUI for all analysis and plotting

├── src/

│ ├── data_loader.py # Fetch & preprocess data

│ ├── analysis.py # FinancialInstrument & TimeSeriesAnalysis

│ ├── simulation.py # Monte Carlo GBM simulation

│ ├── portfolio.py # Portfolio analytics

├── requirements.txt

└── README.md
