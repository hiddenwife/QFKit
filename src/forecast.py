# src/forecast.py

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy.stats import t as student_t

import pymc as pm
import arviz as az


class EpicBayesForecaster:
    """
    A lightweight Bayesian forecaster:
     - AR(p) mean process for returns
     - Student-t likelihood for fat tails
     - ADVI (fast) or NUTS (full MCMC)
    """

    def __init__(self, df: pd.DataFrame, returns_col="Log_Returns"):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")

        self.df = df.sort_index().copy()
        if self.df.index.freq is None:
            try:
                inferred = pd.infer_freq(self.df.index)
                if inferred is not None:
                    self.df = self.df.asfreq(inferred)
                else:
                    self.df = self.df.asfreq("B")
            except Exception:
                self.df = self.df.asfreq("B")

        self.returns_col = returns_col
        if returns_col not in self.df.columns:
            self.df[returns_col] = np.log(self.df["Close"]).diff()
        self.returns = self.df[returns_col].dropna().astype(float)

        self.model: Optional[pm.Model] = None
        self.idata: Optional[az.InferenceData] = None
        self.fitted = False
        self.p = 0

    def fit(self, p=1, draws=1000, tune=1000, chains=4,
            method="advi", target_accept=0.9, cores=1, random_seed=42,
            advi_iter=20000):
        """
        Fit AR(p) + StudentT model. method='advi' (fast) or 'nuts' (full MCMC).
        Returns arviz InferenceData (stored at self.idata).
        """
        self.p = int(p)
        r = self.returns.copy().dropna()
        if len(r) < max(30, self.p + 10):
            raise ValueError("Need >= 30 returns for stable fitting")

        # Prepare design if p>0
        if self.p > 0:
            # X matrix of shape (T-p, p)
            X = np.column_stack([
                r.values[self.p - i - 1: len(r) - i - 1] for i in range(self.p)
            ])
            y = r.values[self.p:]
        else:
            X = None
            y = r.values

        with pm.Model() as m:
            mu0 = pm.Normal("mu0", mu=0.0, sigma=1.0)
            if self.p > 0:
                phi = pm.Normal("phi", mu=0.0, sigma=0.5, shape=self.p)
            else:
                phi = None
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            nu = pm.Exponential("nu", 1/10.0) + 2.0  # df > 2 for well-behaved t

            if self.p > 0:
                mu_t = mu0 + pm.math.dot(X, phi)
            else:
                mu_t = mu0

            pm.StudentT("obs", nu=nu, mu=mu_t, sigma=sigma, observed=y)

            self.model = m

            if method.lower() in ("advi", "meanfield", "fullrank"):
                # ADVI/variational inference -> returns approx then sample draws
                approx = pm.fit(n=advi_iter, method="advi", random_seed=random_seed, progressbar=False)
                self.idata = approx.sample(draws=draws)
            else:
                # NUTS MCMC
                self.idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                                       target_accept=target_accept, random_seed=random_seed, progressbar=False)

            self.fitted = True

        return self.idata

    def _flatten_posterior(self, varname: str):
        """
        Extract a posterior variable from arviz InferenceData and flatten to shape (n_draws, ...)
        Returns numpy array with shape (n_total_draws, *var_shape_after_sample)
        """
        post = self.idata.posterior
        if varname not in post:
            raise KeyError(f"Variable '{varname}' not found in posterior")
        arr = post[varname].values  # shape (chain, draw, *shape)
        # Ensure last dims exist
        flat = arr.reshape((-1,) + arr.shape[2:])
        return flat  # (n_total_draws, ...)

    def forecast(self, steps=30, draws=500, random_seed=42):
        """
        Posterior predictive simulation of future prices.
        Returns DataFrame with median and CI columns indexed by business days.
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() first")

        post = self.idata.posterior
        rng = np.random.default_rng(random_seed)

        n_chains = post.sizes["chain"]
        n_draws = post.sizes["draw"]
        n_total = n_chains * n_draws
        draws = int(min(draws, n_total))

        # select draw indices (flatten chain,draw)
        all_idx = np.arange(n_total)
        sel_idx = rng.choice(all_idx, size=draws, replace=True)

        # Flatten parameters robustly
        mu0_all = self._flatten_posterior("mu0").squeeze()  # shape (n_total,)
        mu0 = mu0_all[sel_idx]

        nu_all = self._flatten_posterior("nu").squeeze()
        nu = nu_all[sel_idx]

        sigma_all = self._flatten_posterior("sigma").squeeze()
        sigma = sigma_all[sel_idx]

        if self.p > 0:
            # phi may be (n_total, p) or (n_total,) if p==1 depending on trace shape.
            phi_all = self._flatten_posterior("phi")
            # ensure it's 2D with shape (n_total, p)
            if phi_all.ndim == 1:
                phi_all = phi_all[:, None]
            phi = phi_all[sel_idx, :]
        else:
            phi = None

        # Simulation loop
        S0 = float(self.df["Close"].iloc[-1])
        prices = np.zeros((draws, steps), dtype=float)

        # last observed returns (for AR initialization)
        if self.p > 0:
            last_r = list(self.returns.values[-self.p:])
        else:
            last_r = []

        for i_draw in range(draws):
            price = S0
            r_hist = list(last_r)
            mu0_i = mu0[i_draw]
            nu_i = float(nu[i_draw])
            sigma_i = float(sigma[i_draw])
            if self.p > 0:
                phi_i = phi[i_draw, :]

            for t in range(steps):
                if self.p > 0:
                    # use most recent p returns (last element is most recent)
                    recent = np.array(r_hist[-self.p:]) if len(r_hist) >= self.p else np.array([0.0] * self.p)
                    mu_t = mu0_i + np.dot(phi_i, recent[::-1]) if recent.size > 0 else mu0_i
                else:
                    mu_t = mu0_i

                # draw Student-t standardized and scale
                t_std = rng.standard_t(df=float(nu_i))
                ret = mu_t + sigma_i * t_std

                # update price
                price = price * np.exp(ret)
                prices[i_draw, t] = price

                # update history
                if self.p > 0:
                    r_hist.append(ret)
                    if len(r_hist) > self.p:
                        # keep only last p
                        r_hist = r_hist[-self.p:]

        fc = {
            "median": np.median(prices, axis=0),
            "lower_95": np.percentile(prices, 2.5, axis=0),
            "upper_95": np.percentile(prices, 97.5, axis=0),
            "lower_80": np.percentile(prices, 10, axis=0),
            "upper_80": np.percentile(prices, 90, axis=0),
        }

        last_date = self.df.index[-1]
        try:
            idx = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)
        except Exception:
            idx = pd.RangeIndex(start=0, stop=steps)

        return pd.DataFrame(fc, index=idx)

    def plot_forecast_matplotlib(self, steps=30, draws=500, history=200):
        fc = self.forecast(steps=steps, draws=draws)
        hist = self.df["Close"].iloc[-history:]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist.index, hist.values, label="History", color="black")
        ax.plot(fc.index, fc["median"], label="Median Forecast", linestyle="--", color="blue")
        ax.fill_between(fc.index, fc["lower_95"], fc["upper_95"],
                        color="skyblue", alpha=0.3, label="95% CI")
        ax.fill_between(fc.index, fc["lower_80"], fc["upper_80"],
                        color="dodgerblue", alpha=0.3, label="80% CI")
        ax.legend()
        ax.set_title("Bayesian Forecast")
        ax.grid(True, linestyle="--", alpha=0.6)
        return fig

    def plot_forecast_interactive(self, steps=30, draws=500, history=200):
        fc = self.forecast(steps=steps, draws=draws)
        hist = self.df["Close"].iloc[-history:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values,
                                 mode="lines", name="History", line=dict(color="black")))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["median"],
                                 mode="lines", name="Forecast Median",
                                 line=dict(color="blue", dash="dash")))
        # 95% CI
        fig.add_trace(go.Scatter(x=fc.index, y=fc["upper_95"],
                                 mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["lower_95"],
                                 mode="lines", fill="tonexty", name="95% CI",
                                 line=dict(width=0), fillcolor="rgba(135,206,250,0.3)"))
        # 80% CI
        fig.add_trace(go.Scatter(x=fc.index, y=fc["upper_80"],
                                 mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["lower_80"],
                                 mode="lines", fill="tonexty", name="80% CI",
                                 line=dict(width=0), fillcolor="rgba(30,144,255,0.25)"))

        fig.update_layout(title="Bayesian Forecast (Interactive)",
                          xaxis_title="Date", yaxis_title="Price",
                          template="plotly_white", hovermode="x unified")
        return fig


