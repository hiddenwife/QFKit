📈 Financial Analysis & Simulation Toolkit
This project is a modular Python toolkit for financial time-series analysis and Monte Carlo simulations of stock, ETF, and index performance using Yahoo Finance data.

It lets you:
- Fetch stock/ETF/index data from Yahoo Finance.
- Compute statistics: returns, volatility, Sharpe ratio, CAGR, Value-at-Risk.
- Plot prices, returns, moving averages, and relative growth.
- Run Geometric Brownian Motion (GBM) simulations to estimate possible future paths.
- Compare multiple tickers side by side.


⚙️ Features
✅ Data Loading
Fetches historical OHLCV data via yfinance.
Computes log returns automatically.
Handles invalid tickers gracefully (skips with a warning).


✅ Analysis
Summary statistics (mean, std, min, max returns).
Annualised return & volatility from log returns.
Growth probability over 5, 10, 15 years (lognormal model).
Expected future price + confidence intervals.
Value-at-Risk (VaR) estimation.
Sharpe ratio (risk-adjusted performance).
Rolling volatility & Sharpe ratio (sliding window analysis).
CAGR (Compound Annual Growth Rate).
Relative growth plots (normalized to 1 at start).
Moving averages (e.g. 20-day, 70-day overlays).
Return plots (% daily returns across tickers).


✅ Monte Carlo Simulation
Implements Geometric Brownian Motion (GBM): dS_t = μ S_t dt + σ S_t dW_t where μ = drift (expected return), σ = volatility, W_t = Brownian motion.
Simulates thousands of random price paths into the future.
Provides:
Probability stock finishes above current price.
Distribution of terminal prices (mean, median, std).
Sampled paths for visualization.
Expected (mean) and median trajectories.
Future extension: percentile “fan charts” (10th–90th percentile bands).


📂 Project Structure
finance_project/
│

├── main.py                 # Entry point — user interaction & orchestration

│

├── src/

│   ├── data_loader.py      # Download & prepare Yahoo Finance data

│   ├── analysis.py         # FinancialInstrument & TimeSeriesAnalysis classes

│   ├── simulation.py       # Simulation class (GBM Monte Carlo)

│

├── requirements.txt        # Python dependencies

└── README.md               # You are here 🚀


