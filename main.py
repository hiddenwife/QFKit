"""
About:
  Modular Python toolkit for ETF, fund, and portfolio analytics. 
  Integrates market data (yfinance) with time-series analysis, 
  Monte Carlo (GBM) simulations, risk metrics, and Bayesian forecasting, 
  all accessible via an interactive GUI.

Developer: Christopher Andrews
GitHub: https://github.com/hiddenwife/financial_tools

Read LICENCE for use of any of this code.

"""

from gui import launch_gui

if __name__ == "__main__":
    print("This code will compare funds, ETFs, trackers etc from Yahoo Finance.\n")
    print("Launching Financial Analysis & Simulation Toolkit")
    launch_gui()