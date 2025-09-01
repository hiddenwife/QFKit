# 📈 Financial Analysis & Simulation Toolkit

A modular Python toolkit for **time-series analysis** and **Monte Carlo simulations** of stocks, ETFs, and indices using Yahoo Finance data.  

## Features
- **Data loading**: Fetches OHLCV data via `yfinance`, computes log returns, handles invalid tickers gracefully.  
- **Analysis**: Annualised returns/volatility, Sharpe ratio, CAGR, Value-at-Risk, growth probabilities, rolling stats, moving averages, relative growth, return plots.  
- **Portfolio tools**: Correlation & covariance matrices, volatility, variance contributions, cumulative growth of weighted portfolios.  
- **Simulations**: Geometric Brownian Motion (GBM) with thousands of paths, probability of finishing above current price, terminal price stats, visualised sample paths.  

## Project Structure
finance_project/

├── main.py # Entry point (CLI orchestration)

├── src/

│ ├── data_loader.py # Fetch & preprocess data

│ ├── analysis.py # FinancialInstrument & TimeSeriesAnalysis

│ ├── simulation.py # Monte Carlo GBM simulation

│ ├── portfolio.py # Portfolio analytics

├── requirements.txt

└── README.md
