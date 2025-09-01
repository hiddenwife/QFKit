import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.analysis import FinancialInstrument

class Simulation(FinancialInstrument):
    def __init__(self, ticker, df):
        super().__init__(ticker, df)

    def _get_mu_sigma(self):
        """
        Return (mu, sigma) annualised. Expects FinancialInstrument to provide
        log returns in self.df['Log_Returns'] or arithmetic in 'Returns'.
        Adjust if your base class differs.
        """
        if 'Log_Returns' in self.df.columns:
            r = self.df['Log_Returns'].dropna()
            # mu: mean log-return per period * periods per year, sigma: std * sqrt(periods)
            periods = 252  # default business days; change if your data freq differs
            mu = r.mean() * periods
            sigma = r.std() * np.sqrt(periods)
        elif 'Returns' in self.df.columns:
            r = self.df['Returns'].dropna()
            periods = 252
            mu = r.mean() * periods
            sigma = r.std() * np.sqrt(periods)
        else:
            # fallback: compute log-returns from Close
            r = np.log(self.df['Close'] / self.df['Close'].shift(1)).dropna()
            periods = 252
            mu = r.mean() * periods
            sigma = r.std() * np.sqrt(periods)
        return mu, sigma

    def gbm_paths(self, mu=None, sigma=None, T=1, steps_per_year=252, n_sims=1000, as_dates=True):
        """
        Simulate Geometric Brownian Motion paths and return a DataFrame.

        - mu, sigma: annualised drift and volatility. If None, estimated from history.
        - T: horizon in years
        - steps_per_year: number of steps per year (e.g. 252)
        - n_sims: number of simulation paths
        - as_dates: if True index result by business days starting from next trading day
        """
        if mu is None or sigma is None:
            mu, sigma = self._get_mu_sigma()

        S0 = float(self.df['Close'].iloc[-1])
        total_steps = int(np.round(T * steps_per_year))
        dt = T / total_steps

        # preallocate and simulate
        paths = np.empty((total_steps + 1, n_sims), dtype=float)
        paths[0, :] = S0
        for t in range(1, total_steps + 1):
            Z = np.random.standard_normal(n_sims)
            paths[t, :] = paths[t-1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        # build DataFrame with columns 0..n_sims-1
        if as_dates:
            # start from next business day after last date in df
            try:
                last_date = pd.to_datetime(self.df.index[-1])
                dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=total_steps + 1)
            except Exception:
                dates = pd.RangeIndex(total_steps + 1)
            df_paths = pd.DataFrame(paths, index=dates)
        else:
            df_paths = pd.DataFrame(paths)

        return df_paths

    def prob_increase_mc(self, horizon_years=10, n_sims=1000, steps_per_year=252):
        """
        Monte Carlo probability that price after horizon_years is above today's price.
        """
        sims = self.gbm_paths(T=horizon_years, n_sims=n_sims, steps_per_year=steps_per_year, as_dates=False)
        S0 = float(self.df['Close'].iloc[-1])
        ST = sims.iloc[-1, :]
        return float((ST > S0).mean())

    def run_simulation(self, horizon_years=10, n_sims=1000, steps_per_year=252, mu=None, sigma=None,
                       plot=True, n_plot_paths=50, seed=None):
        """
        Run GBM sims and return a dict with:
          - 'paths': DataFrame (date indexed) of simulated prices
          - 'prob_increase': probability final > S0
          - 'S0': starting price
          - 'ST_stats': dict with mean, median, std of final prices

        If plot=True, shows up to n_plot_paths sampled paths.
        """
        if seed is not None:
            np.random.seed(seed)

        paths = self.gbm_paths(mu=mu, sigma=sigma, T=horizon_years,
                               steps_per_year=steps_per_year, n_sims=n_sims, as_dates=True)
        S0 = float(self.df['Close'].iloc[-1])
        ST = paths.iloc[-1, :]

        result = {
            'paths': paths,
            'prob_increase': float((ST > S0).mean()),
            'S0': S0,
            'ST_stats': {
                'mean': float(ST.mean()),
                'median': float(ST.median()),
                'std': float(ST.std()),
            }
        }

        if plot:
            sample = paths.iloc[:, :min(n_plot_paths, paths.shape[1])]
            plt.figure(figsize=(10, 6))
            sample.plot(legend=False, alpha=0.6)
            plt.axhline(S0, color='k', linestyle='--', linewidth=1, label='S0')
            plt.title(f"GBM sample paths (n_sims={n_sims}, showing {sample.shape[1]})")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.grid(True)
            plt.show()

        return result
