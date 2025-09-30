# src/forecast.py

from typing import Optional

import numpy as np
import pandas as pd

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
            self.df = self.df.asfreq("B")

        self.returns_col = returns_col
        if returns_col not in self.df.columns:
            self.df[returns_col] = np.log(self.df["Close"]).diff()
        self.returns = self.df[self.returns_col].dropna().astype(float)

        self.model: Optional[pm.Model] = None
        self.idata: Optional[az.InferenceData] = None
        self.fitted = False
        self.p = 0

    def fit(self, p=1, draws=1000, tune=1000, chains=4,
            method="advi", target_accept=0.9, cores=1, random_seed=42,
            advi_iter=20000, sigma_prior_std=0.05):
        self.p = int(p)
        r = self.returns.copy().dropna()
        if len(r) < max(30, self.p + 10):
            raise ValueError("Need >= 30 returns for stable fitting")

        X = np.column_stack([r.values[self.p - i - 1: len(r) - i - 1] for i in range(self.p)]) if self.p > 0 else None
        y = r.values[self.p:] if self.p > 0 else r.values

        with pm.Model() as m:
            mu0 = pm.Normal("mu0", mu=0.0, sigma=0.1)
            phi = pm.Normal("phi", mu=0.0, sigma=0.5, shape=self.p) if self.p > 0 else None
            sigma = pm.HalfNormal("sigma", sigma=sigma_prior_std)

            # ==========================================================
            # FIX 1: Tame the tails with a more stable prior for nu
            # A Gamma distribution prevents nu from getting too small, which
            # is the primary cause of the explosive "fat tails".
            # ==========================================================
            nu = pm.Gamma("nu", alpha=2, beta=0.1) + 1.0

            mu_t = mu0 + pm.math.dot(X, phi) if self.p > 0 else mu0
            pm.StudentT("obs", nu=nu, mu=mu_t, sigma=sigma, observed=y)
            self.model = m

            if method.lower().startswith("advi"):
                approx = pm.fit(n=advi_iter, method="advi", random_seed=random_seed, progressbar=False)
                self.idata = approx.sample(draws=draws)
            else:
                self.idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                                       target_accept=target_accept, random_seed=random_seed, progressbar=False)
            self.fitted = True
        return self.idata

    def _flatten_posterior(self, varname: str):
        arr = self.idata.posterior[varname].values
        return arr.reshape((-1,) + arr.shape[2:])

    def forecast(self, steps=30, draws=500, random_seed=42):
        if not self.fitted:
            raise RuntimeError("Call .fit() first")

        rng = np.random.default_rng(random_seed)
        n_total = self.idata.posterior.sizes["chain"] * self.idata.posterior.sizes["draw"]
        draws = int(min(draws, n_total))
        sel_idx = rng.choice(np.arange(n_total), size=draws, replace=True)

        mu0 = self._flatten_posterior("mu0").squeeze()[sel_idx]
        nu = self._flatten_posterior("nu").squeeze()[sel_idx]
        sigma = self._flatten_posterior("sigma").squeeze()[sel_idx]
        phi = self._flatten_posterior("phi")[sel_idx, :] if self.p > 0 else None

        S0 = float(self.df["Close"].iloc[-1])
        prices = np.zeros((draws, steps), dtype=float)
        last_r = list(self.returns.values[-self.p:]) if self.p > 0 else []

        for i_draw in range(draws):
            price = S0
            r_hist = list(last_r)
            for t in range(steps):
                mu_t = mu0[i_draw]
                if self.p > 0:
                    mu_t += np.dot(phi[i_draw, :], r_hist[::-1])

                ret = mu_t + sigma[i_draw] * rng.standard_t(df=nu[i_draw])
                
                # ==========================================================
                # FIX 2: Clip extreme returns to prevent numerical overflow
                # This is a safeguard that stops a single path from exploding
                # to infinity and ruining the plot's scale.
                # ==========================================================
                ret = np.clip(ret, -0.35, 0.35) # Cap daily moves at +/- 35%

                price *= np.exp(ret)
                prices[i_draw, t] = price

                if self.p > 0:
                    r_hist.append(ret)
                    r_hist.pop(0)

        fc = {
            "median": np.nanmedian(prices, axis=0),
            "lower_95": np.nanpercentile(prices, 2.5, axis=0),
            "upper_95": np.nanpercentile(prices, 97.5, axis=0),
            "lower_80": np.nanpercentile(prices, 10, axis=0),
            "upper_80": np.nanpercentile(prices, 90, axis=0),
            "lower_50": np.nanpercentile(prices, 25, axis=0),
            "upper_50": np.nanpercentile(prices, 75, axis=0)
        }

        last_date = self.df.index[-1]
        idx = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)
        return pd.DataFrame(fc, index=idx)