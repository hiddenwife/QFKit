import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
      """
      if seed is not None:
          np.random.seed(seed)

      # Ensure we always have mu and sigma
      if mu is None or sigma is None:
          mu, sigma = self._get_mu_sigma()

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
          # Ensure paths.index is a DatetimeIndex before plotting
          if not isinstance(paths.index, pd.DatetimeIndex):
              start_date = pd.Timestamp.today()
              paths.index = pd.bdate_range(start=start_date, periods=len(paths.index))

          # Normalize paths for relative growth
          sample = paths.iloc[:, :min(n_plot_paths, paths.shape[1])] / S0

          fig, ax = plt.subplots(figsize=(10, 6))

          # Plot sample paths (relative growth)
          for col in sample.columns:
              ax.plot(sample.index, sample[col], color='gray', alpha=0.4, linewidth=1)

          # Add starting line (always 1)
          s0_line = ax.axhline(1, color='k', linestyle='--', linewidth=1, label='Start (S0)')

          # Expected path (mean, normalized)
          time_grid = np.linspace(0, horizon_years, len(sample.index))
          expected_path = np.exp(mu * time_grid)
          expected_line, = ax.plot(sample.index, expected_path, color="blue", linewidth=2.5, label="Expected (mean)")

          # Median path (normalized)
          median_path = np.exp((mu - 0.5 * sigma**2) * time_grid)
          median_line, = ax.plot(sample.index, median_path, color="red", linewidth=2.0, label="Median")

          info_proxy = Line2D([], [], linestyle='None', label=f"μ={mu:.2%}, std={sigma:.2%}, S0={S0:.2f}, P(end > S0)={result['prob_increase']:.2%}")

          ax.set_title(f"GBM relative growth for {self.ticker} over {horizon_years}y (n_sims={n_sims}, showing {sample.shape[1]})")
          ax.set_xlabel("Date")
          ax.set_ylabel("Relative Growth")
          ax.grid(True)
          ax.legend([expected_line, median_line, s0_line, info_proxy], 
                    [f'Expected (mean) = {expected_path[-1]:.2f}', 
                        f'Median = {median_path[-1]:.2f}',
                        f"P(end > S0) = {result['prob_increase']:.2%}"], 
                    loc='upper left')
          plt.show()

      return result

    def compare_simulation_to_real(self, n_sims=5000, steps_per_year=252, start_date=None,
                                  end_date=None, include_ci=True, ci=0.95, seed=None, plot=True):
        """
        Simulate GBM from the dataset start (or given start_date) up to end_date (or last real date)
        and compare expected/median simulated paths to the real Close series.

        Returns a dict with keys:
          - 'sim_dates': DatetimeIndex used for simulation
          - 'expected': expected price series (mean across sims) indexed by sim_dates
          - 'median': median price series indexed by sim_dates
          - 'ci': (lower, upper) arrays if include_ci True, else None
          - 'paths': DataFrame of all simulated end-to-end paths (may be large)
          - 'real': real Close series aligned to sim_dates (may contain NaN if dates mismatch)
        """
        if seed is not None:
            np.random.seed(seed)

        # Determine simulation date range
        df_index = pd.to_datetime(self.df.index)
        start = pd.to_datetime(start_date) if start_date is not None else df_index[0]
        end = pd.to_datetime(end_date) if end_date is not None else df_index[-1]

        # Use business days for index and compute years horizon
        sim_dates = pd.bdate_range(start=start, end=end)
        total_steps = len(sim_dates) - 1
        if total_steps <= 0:
            raise ValueError("Simulation period must contain at least 1 step (end > start).")

        T = total_steps / steps_per_year
        dt = T / total_steps

        mu, sigma = (self._get_mu_sigma() if (mu := None) is None else (mu, sigma))  # use historical if not provided
        # (above line just forces use of _get_mu_sigma)

        # Starting price: use Close at start (if start equals first index) or interpolate/locate nearest
        try:
            S0 = float(self.df.loc[start, 'Close'])
        except Exception:
            # fall back to first close if exact start not found
            S0 = float(self.df['Close'].iloc[0])

        # Simulate
        paths = np.empty((total_steps + 1, n_sims), dtype=float)
        paths[0, :] = S0
        for t in range(1, total_steps + 1):
            Z = np.random.standard_normal(n_sims)
            paths[t, :] = paths[t-1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        paths_df = pd.DataFrame(paths, index=sim_dates)

        # Compute statistics
        expected = paths_df.mean(axis=1)
        median = paths_df.median(axis=1)
        ci_lower = ci_upper = None
        if include_ci:
            lower_q = (1 - ci) / 2
            upper_q = 1 - lower_q
            ci_lower = paths_df.quantile(lower_q, axis=1)
            ci_upper = paths_df.quantile(upper_q, axis=1)

        # Align real data to sim_dates (reindex with forward/backfill if needed)
        real_close = self.df['Close'].reindex(sim_dates)
        # if real_close is all NaN (e.g., sim started before data), try aligning overlapping portion
        if real_close.isna().all():
            # align to overlapping range
            overlap = sim_dates.intersection(pd.to_datetime(self.df.index))
            real_close = self.df['Close'].reindex(overlap)
            expected = expected.reindex(overlap)
            median = median.reindex(overlap)
            if include_ci:
                ci_lower = ci_lower.reindex(overlap)
                ci_upper = ci_upper.reindex(overlap)
            paths_df = paths_df.reindex(overlap)

        result = {
            'sim_dates': paths_df.index,
            'expected': expected,
            'median': median,
            'ci': (ci_lower, ci_upper) if include_ci else None,
            'paths': paths_df,
            'real': real_close
        }

        if plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            # plot real
            ax.plot(result['real'].index, result['real'].values / result['real'].iloc[0], label=f'Real (norm) {(result['real'].values[-1] / result['real'].iloc[0]):.2f}', color='black', linewidth=2)
            # plot expected & median normalized to same S0
            ax.plot(result['expected'].index, result['expected'].values / result['expected'].iloc[0], label=f'Sim Expected (norm) {(result['expected'].values[-1] / result['expected'].iloc[0]):.2f}', color='blue', linewidth=2)
            ax.plot(result['median'].index, result['median'].values / result['median'].iloc[0], label=f'Sim Median (norm) {(result['median'].values[-1] / result['median'].iloc[0]):.2f}', color='red', linewidth=1.8, linestyle='--')

            if include_ci and result['ci'][0] is not None:
                ax.fill_between(result['sim_dates'], (result['ci'][0] / result['expected'].iloc[0]), (result['ci'][1] / result['expected'].iloc[0]),
                                color='blue', alpha=0.15, label=f'{int(ci*100)}% CI (sim)')

            ax.set_title(f"Simulated expected & median vs Real ({self.ticker})\nSim from {result['sim_dates'][0].date()} to {result['sim_dates'][-1].date()}")
            ax.set_ylabel("Normalized price (start = 1)")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(True)
            plt.show()

        return result

