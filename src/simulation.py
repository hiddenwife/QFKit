# src/simulation.py
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from src.analysis import FinancialInstrument
import pymc as pm

class Simulation(FinancialInstrument):
    def __init__(self, ticker, df):
        super().__init__(ticker, df)

    def _get_mu_sigma(self, window_days: int | None = None):
        """Annualised mu, sigma (log returns); if window_days set, only use recent window."""
        r = np.log(self.df['Close'] / self.df['Close'].shift(1)).dropna()
        if window_days is not None:
            window_days = max(1, int(window_days))
            r = r.iloc[-window_days:]

        periods = 252
        if len(r) == 0:
            return 0.0, 0.0, 0.0

        mu = float(r.mean() * periods)
        med = float(r.median() * periods)
        sigma = float(r.std(ddof=0) * np.sqrt(periods))
        return mu, sigma, med

    # ----------------------------------------------------------------
    # GBM
    # ----------------------------------------------------------------
    def gbm_paths(self, mu, sigma, T, steps_per_year, n_sims, seed=None, jump=False, lambda_j=0.1, mu_j=-0.1, sigma_j=0.3):
        """Standard GBM paths; optional Merton jumps."""
        if seed is not None:
            rng = np.random.default_rng(seed)
            randn = rng.standard_normal
            randp = rng.poisson
        else:
            randn = np.random.standard_normal
            randp = np.random.poisson

        S0 = float(self.df['Close'].iloc[-1])
        total_steps = int(np.round(T * steps_per_year))
        dt = T / total_steps

        paths = np.empty((total_steps + 1, n_sims))
        paths[0, :] = S0

        for t in range(1, total_steps + 1):
            Z = randn(n_sims)
            jump_term = 0.0
            if jump:
                N = randp(lambda_j * dt, n_sims)  # number of jumps
                jump_term = N * (mu_j + sigma_j * randn(n_sims))
            paths[t, :] = paths[t-1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + jump_term)

        dates = pd.bdate_range(start=pd.to_datetime(self.df.index[-1]) + pd.Timedelta(days=1), periods=total_steps + 1)
        return pd.DataFrame(paths, index=dates)

    # ----------------------------------------------------------------
    # Heston model
    # ----------------------------------------------------------------
    def heston_paths(self, mu, v0, kappa, theta, xi, rho, T, steps_per_year, n_sims, seed=None):
        """Heston stochastic volatility model."""
        if seed is not None:
            rng = np.random.default_rng(seed)
            randn = rng.standard_normal
        else:
            randn = np.random.standard_normal

        S0 = float(self.df['Close'].iloc[-1])
        total_steps = int(np.round(T * steps_per_year))
        dt = T / total_steps

        S = np.empty((total_steps + 1, n_sims))
        v = np.empty((total_steps + 1, n_sims))
        S[0, :] = S0
        v[0, :] = v0

        for t in range(1, total_steps + 1):
            Z1 = randn(n_sims)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * randn(n_sims)
            v[t, :] = np.maximum(v[t-1, :] + kappa * (theta - v[t-1, :]) * dt + xi * np.sqrt(np.maximum(v[t-1, :], 0)) * np.sqrt(dt) * Z2, 0)
            S[t, :] = S[t-1, :] * np.exp((mu - 0.5 * v[t-1, :]) * dt + np.sqrt(v[t-1, :] * dt) * Z1)

        dates = pd.bdate_range(start=pd.to_datetime(self.df.index[-1]) + pd.Timedelta(days=1), periods=total_steps + 1)
        return pd.DataFrame(S, index=dates)

    # ----------------------------------------------------------------
    # Run simulation wrapper
    # ----------------------------------------------------------------
    def run_simulation(self, horizon_years=2, n_sims=1000, seed=None, window_days=None, heston=False, jump=False):
        # The time_varying flag is implicitly handled by whether window_days is provided
        time_varying = window_days is not None
        mu, sigma, med = self._get_mu_sigma(window_days=window_days)

        if heston:
            v0 = sigma**2
            kappa = 2.0      # mean reversion speed
            theta = sigma**2 # long term variance
            xi = 0.1         # vol of vol
            rho = -0.7       # correlation
            paths = self.heston_paths(mu, v0, kappa, theta, xi, rho, horizon_years, 252, n_sims, seed)
        else:
            paths = self.gbm_paths(mu, sigma, horizon_years, 252, n_sims, seed, jump=jump)

        S0 = float(self.df['Close'].iloc[-1])
        ST = paths.iloc[-1, :]

        return {
            'ticker': self.ticker,
            'paths': paths,
            'prob_increase': float((ST > S0).mean()),
            'S0': S0,
            'ST_stats': {
              'mean': float(ST.mean()),
              'median': float(ST.median()),
              'std': float(ST.std())},
            'params': {'mu': mu, 'sigma': sigma, 'horizon_years': horizon_years, 'n_sims': n_sims, 'window_days': window_days,
                       'heston': heston, 'jump': jump, 'time_varying': time_varying}
        }

    # ----------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------
    def make_sim_figure(self, res, n_plot_paths=50):
        paths = res['paths']
        params = res['params']
        S0 = res['S0']
        prob_increase = res['prob_increase']

        time_varying = params.get('time_varying', False)
        jump = params.get('jump', False)

        sample = paths.iloc[:, :min(n_plot_paths, paths.shape[1])] / S0
        fig = Figure(figsize=(10, 6))
        ax = fig.subplots()

        ax.plot(sample.index, sample.values, color='gray', alpha=0.4, linewidth=1)
        ax.axhline(1, color='k', linestyle='--', linewidth=1, label=f'Start Price ({S0:.2f})')

        time_grid = np.linspace(0, params['horizon_years'], len(sample.index))
        expected_path = np.exp(params['mu'] * time_grid)
        median_path = np.exp((params['mu'] - 0.5 * params['sigma']**2) * time_grid)
        ax.plot(sample.index, expected_path, color="blue", linewidth=2.0, label=f"Expected (mean) = {expected_path[-1]:.2f}")
        ax.plot(sample.index, median_path, color='red', linestyle='--', linewidth=2, label=f"Median = {(median_path[-1]):.2f}")
        
        ax.plot([], [], ' ', label=f"P(end > S0) = {prob_increase:.2%}")

        if time_varying:
            ax.plot([], [], ' ', label='\u2713' + r" Time-varying $\mu$ & $\sigma$")
        if jump:
            ax.plot([], [], ' ', label='\u2713' + r" Merton Jumps")

        ax.set_title(f"Simulation: {self.ticker} ({params['horizon_years']}y, {params['n_sims']} sims)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Relative Growth (Start = 1)")
        ax.grid(True)
        ax.legend()
        return fig

class SDE_Simulation(Simulation):
    def __init__(self, ticker, df):
        super().__init__(ticker, df)

    def _get_log_series(self):
        close = self.df['Close'].dropna()
        return np.log(close).to_numpy()

    def _get_last_log_price(self):
        return self._get_log_series()[-1]

    def estimate_ou_parameters(self, dt=1/252, draws=1000, tune=500):
        data = self._get_log_series()
        if len(data) < 2:
            raise ValueError("Not enough data for OU estimation.")
        with pm.Model() as model:
            theta = pm.HalfNormal("theta", sigma=5.0)
            mu = pm.Normal("mu", mu=float(data.mean()), sigma=float(data.std()) if data.std() > 0 else 1.0)
            sigma = pm.HalfNormal("sigma", sigma=float(np.std(np.diff(data))) if np.std(np.diff(data)) > 0 else 1.0)
            tau = pm.math.exp(-theta * dt)
            mean = mu + (data[:-1] - mu) * tau
            var = (sigma**2 / (2 * theta)) * (1 - pm.math.exp(-2 * theta * dt))
            pm.Normal("obs", mu=mean, sigma=pm.math.sqrt(var), observed=data[1:])
            trace = pm.sample(draws=draws, tune=tune, target_accept=0.9, return_inferencedata=True, cores=1, progressbar=False)
        return trace

    def run_ou_simulation(self, theta, mu, sigma, T=1.0, n_sims=500, seed=None):
        """Pure computation: simulate OU (on log-price), return numeric paths and params."""
        if seed is not None:
            rng = np.random.default_rng(seed)
            randn = rng.standard_normal
        else:
            randn = np.random.randn

        X0 = self._get_last_log_price()
        dt = 1 / 252
        total_steps = int(round(T / dt))
        exp_neg_theta_dt = np.exp(-theta * dt)
        if theta > 0:
            var_dt = (sigma**2) / (2 * theta) * (1 - np.exp(-2 * theta * dt))
        else:
            var_dt = (sigma**2) * dt

        log_paths = np.empty((total_steps + 1, n_sims), dtype=float)
        log_paths[0, :] = X0
        for t in range(1, total_steps + 1):
            mean_t = mu + (log_paths[t-1, :] - mu) * exp_neg_theta_dt
            Z = randn(n_sims)
            log_paths[t, :] = mean_t + Z * np.sqrt(var_dt)

        paths = np.exp(log_paths)
        dates = pd.bdate_range(start=pd.to_datetime(self.df.index[-1]) + pd.Timedelta(days=1), periods=total_steps + 1)
        return {
            'ticker': self.ticker,
            'paths': pd.DataFrame(paths, index=dates),
            'params': {'theta': theta, 'mu': mu, 'sigma': sigma}
        }

    def make_ou_figure(self, res, n_plot_paths=50):
        paths = res['paths']
        sample = paths.iloc[:, :min(n_plot_paths, paths.shape[1])]
        long_term_price = float(np.exp(res['params']['mu']))

        fig = Figure(figsize=(10, 6))
        ax = fig.subplots()
        ax.plot(sample.index, sample.values, color='gray', alpha=0.4)
        ax.axhline(long_term_price, color='red', linestyle='--', label=f'Reversion Level ≈ exp(μ)={long_term_price:.2f}')
        ax.set_title(f"Log-OU Simulated Price Paths ({self.ticker})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        return fig