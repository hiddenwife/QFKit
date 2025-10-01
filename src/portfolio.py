import pandas as pd
import numpy as np

class Portfolio:
    """
    Domain class that encapsulates portfolio calculations.

    instruments: dict {ticker: FinancialInstrument-like}
        Each instrument must expose a DataFrame `df` with a 'Log_Returns' column
        and a 'Close' column for time series construction when needed.
    """

    def __init__(self, instruments: dict):
        if not isinstance(instruments, dict):
            raise TypeError("instruments must be a dict of ticker -> instrument")
        if len(instruments) < 2:
            raise ValueError("Portfolio requires at least two instruments")
        self.instruments = instruments
        # build a DataFrame of aligned log-returns
        returns = pd.concat(
            {t: inst.df['Log_Returns'] for t, inst in instruments.items()},
            axis=1
        )
        # instead of dropping NAs, forward-fill so portfolio exists from earliest start
        self.returns = returns.ffill().dropna()

    @classmethod
    def from_weighted_close_series(cls, name: str, instruments: dict, weights: np.ndarray):
        """
        Construct a Portfolio-like time series from weighted Close prices.
        Returns a minimal object suitable for analysis (has .df DataFrame with Close and Log_Returns).
        This can be used by GUI code when a full Portfolio object is not desired.
        """
        if len(instruments) < 2:
            raise ValueError("At least two instruments required")
        if len(weights) != len(instruments):
            raise ValueError("weights length must match instruments count")
        weights = np.asarray(weights, dtype=float)
        if np.any(weights < 0):
            raise ValueError("weights cannot be negative")
        if weights.sum() == 0:
            raise ValueError("sum of weights cannot be zero")
        weights = weights / weights.sum()

        # build aligned close price DataFrame and compute weighted close
        close_prices = pd.concat({t: inst.df['Close'] for t, inst in instruments.items()}, axis=1)

        # forward-fill missing values so early instruments carry the portfolio before later ones start
        close_prices = close_prices.ffill()

        portfolio_close = (close_prices * weights).sum(axis=1)
        df = pd.DataFrame(index=portfolio_close.index)
        df['Close'] = portfolio_close
        df['Log_Returns'] = np.log(df['Close']).diff()
        df.dropna(inplace=True)

        # return a simple object with df attribute
        class TS:
            def __init__(self, name, df):
                self.name = name
                self.df = df
        return TS(name, df)

    def correlation_matrix(self) -> pd.DataFrame:
        """Return correlation matrix of log-returns."""
        return self.returns.corr()

    def covariance_matrix(self) -> pd.DataFrame:
        """Return covariance matrix of log-returns."""
        return self.returns.cov()

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Return portfolio volatility (std dev) given weights.
        weights must align with self.returns columns order.
        """
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != self.returns.shape[1]:
            raise ValueError("weights length must match number of instruments in portfolio")
        cov = self.covariance_matrix().values
        return float(np.sqrt(w.T @ cov @ w))

    def portfolio_log_returns(self, weights: np.ndarray) -> pd.Series:
        """
        Return portfolio log-returns series given weights (aligned to self.returns columns).
        """
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != self.returns.shape[1]:
            raise ValueError("weights length must match number of instruments in portfolio")
        return (self.returns * w).sum(axis=1)

    def cumulative_value_series(self, weights: np.ndarray, start_value: float = 1.0) -> pd.Series:
        """
        Build cumulative value series starting at start_value using log-returns -> simple returns.
        Returns an index-aligned Series of portfolio value.
        """
        # convert log-returns to simple returns per period
        w = np.asarray(weights, dtype=float)
        simple = self.returns.apply(np.expm1)
        port_simple = (simple * w).sum(axis=1)
        return (1.0 + port_simple).cumprod() * start_value

    def variance_contributions(self, weights: np.ndarray) -> pd.Series:
        """
        Return each instrument's contribution to portfolio variance as fractions summing to 1.
        Index matches self.returns.columns.
        """
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != self.returns.shape[1]:
            raise ValueError("weights length must match number of instruments in portfolio")
        cov = self.covariance_matrix().values
        marginal = cov @ w
        contrib = w * marginal
        total_var = float(w.T @ cov @ w)
        if total_var == 0:
            raise ValueError("Total variance is zero; cannot compute contributions")
        return pd.Series(contrib / total_var, index=self.returns.columns)
