import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Portfolio:
    def __init__(self, instruments: dict):
        """
        instruments: dict {ticker: FinancialInstrument}
        Expects each FinancialInstrument to have a DataFrame `df` with a 'Log_Returns' column.
        """
        self.instruments = instruments
        self.returns = pd.DataFrame({t: inst.df['Log_Returns'] for t, inst in instruments.items()})

    def correlation_matrix(self):
        return self.returns.corr()

    def covariance_matrix(self):
        return self.returns.cov()

    def portfolio_volatility(self, weights):
        """
        weights: array-like of weights (sum to 1)
        """
        cov = self.covariance_matrix()
        w = np.asarray(weights)
        return np.sqrt(w.T @ cov.values @ w)

    def portfolio_log_returns(self, weights):
        w = np.asarray(weights)
        return (self.returns * w).sum(axis=1)

    def cumulative_value_series(self, weights):
        """
        Returns portfolio value series starting at 1 using arithmetic returns.
        Assumes self.returns contains log-returns; if it contains simple returns,
        remove np.expm1().
        """
        w = np.asarray(weights)
        # convert log-returns to simple returns per period
        simple = self.returns.apply(np.expm1)
        # portfolio simple return each period
        port_simple = (simple * w).sum(axis=1)
        # cumulative value starting at 1
        return (1 + port_simple).cumprod()


    def plot_cumulative_return(self, weights, title="Portfolio relative growth (Start = 1)"):
        series = self.cumulative_value_series(weights)
        plt.figure()
        series.plot(title=title)
        plt.xlabel("Date")
        plt.ylabel("Relative value (start = 1)")
        plt.grid(True)
        plt.show()


    def variance_contributions(self, weights):
        """
        Returns a pandas Series of each instrument's contribution to portfolio variance (as fraction of total variance).
        """
        cov = self.covariance_matrix().values
        w = np.asarray(weights)
        marginal = cov @ w
        contrib = w * marginal
        total_var = w.T @ cov @ w
        return pd.Series(contrib / total_var, index=self.returns.columns)

