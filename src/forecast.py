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
     - Optional hierarchy: learn an additive bias and multiplicative sigma scaling
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
        self.learn_bias_variance = False

    def fit(self, p=1, draws=1000, tune=1000, chains=4,
            method="advi", target_accept=0.9, cores=1, random_seed=42,
            advi_iter=20000, sigma_prior_std=0.05, learn_bias_variance: bool = False):
        """
        Fit the Bayesian AR(p) model.
        If learn_bias_variance is True, include:
            bias ~ Normal(0, 0.05)
            sigma_scale ~ HalfNormal(0.5)
        that are inferred from the data and used in forecasting.
        """
        self.p = int(p)
        r = self.returns.copy().dropna()
        if len(r) < max(30, self.p + 10):
            raise ValueError("Need >= 30 returns for stable fitting")

        X = np.column_stack([r.values[self.p - i - 1: len(r) - i - 1] for i in range(self.p)]) if self.p > 0 else None
        y = r.values[self.p:] if self.p > 0 else r.values

        self.learn_bias_variance = bool(learn_bias_variance)

        with pm.Model() as m:
            mu0 = pm.Normal("mu0", mu=0.0, sigma=0.1)
            phi = pm.Normal("phi", mu=0.0, sigma=0.5, shape=self.p) if self.p > 0 else None
            sigma = pm.HalfNormal("sigma", sigma=sigma_prior_std)

            # ==========================================================
            # Tame tail parameter prior
            # ==========================================================
            nu = pm.Gamma("nu", alpha=2, beta=0.1) + 1.0

            # Optional hierarchical bias and sigma scaling
            if self.learn_bias_variance:
                # additive bias on returns mean (regularised)
                bias = pm.Normal("bias", mu=0.0, sigma=0.05)
                # multiplicative scale on sigma (regularised, >0)
                sigma_scale = pm.HalfNormal("sigma_scale", sigma=0.5)
            else:
                # Not learning; create deterministic placeholders for trace consistency
                bias = None
                sigma_scale = None

            mu_t = mu0 + pm.math.dot(X, phi) if self.p > 0 else mu0
            if bias is not None:
                mu_t = mu_t + bias

            # use scaled sigma if requested
            sigma_obs = sigma * sigma_scale if sigma_scale is not None else sigma

            pm.StudentT("obs", nu=nu, mu=mu_t, sigma=sigma_obs, observed=y)
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
        """
        Flatten a posterior variable into shape (n_posterior_draws, ...).
        If varname does not exist in the posterior (e.g., bias when not learned),
        return sensible defaults:
          - For scalar: zeros
          - For scale: ones
          - For vector: zeros shaped appropriately
        """
        posterior = self.idata.posterior
        # compute total size (chains * draws)
        n_chains = posterior.sizes.get("chain", 1)
        n_draws = posterior.sizes.get("draw", 1)
        n_total = int(n_chains * n_draws)

        if varname not in posterior.data_vars:
            # Provide defaults:
            # - bias -> zeros
            # - sigma_scale -> ones
            # - phi -> zeros of length self.p
            if varname == "bias":
                return np.zeros((n_total,))
            if varname == "sigma_scale":
                return np.ones((n_total,))
            if varname == "phi":
                if self.p > 0:
                    return np.zeros((n_total, self.p))
                else:
                    return np.zeros((n_total, 0))
            # generic fallback
            return np.zeros((n_total,))
        arr = posterior[varname].values
        # arr has shape (chain, draw, *shape)
        # flatten chain and draw dims into first axis
        newshape = (arr.shape[0] * arr.shape[1],) + arr.shape[2:]
        return arr.reshape(newshape)

    def forecast(self, steps=30, draws=500, random_seed=42):
        if not self.fitted:
            raise RuntimeError("Call .fit() first")

        rng = np.random.default_rng(random_seed)
        n_chains = self.idata.posterior.sizes["chain"]
        n_draws = self.idata.posterior.sizes["draw"]
        n_total = n_chains * n_draws
        draws = int(min(draws, n_total))
        sel_idx = rng.choice(np.arange(n_total), size=draws, replace=True)

        mu0_all = self._flatten_posterior("mu0").squeeze()[sel_idx]
        nu_all = self._flatten_posterior("nu").squeeze()[sel_idx]
        sigma_all = self._flatten_posterior("sigma").squeeze()[sel_idx]
        phi_all = self._flatten_posterior("phi")
        if phi_all.size:
            phi_all = phi_all[sel_idx, :] if self.p > 0 else None
        else:
            phi_all = None

        # bias and sigma_scale defaults handled in _flatten_posterior
        bias_all = self._flatten_posterior("bias").squeeze()[sel_idx]
        sigma_scale_all = self._flatten_posterior("sigma_scale").squeeze()[sel_idx]

        S0 = float(self.df["Close"].iloc[-1])
        prices = np.zeros((draws, steps), dtype=float)
        last_r = list(self.returns.values[-self.p:]) if self.p > 0 else []

        for i_draw in range(draws):
            price = S0
            r_hist = list(last_r)
            for t in range(steps):
                mu_t = mu0_all[i_draw]
                if self.p > 0 and phi_all is not None:
                    mu_t += np.dot(phi_all[i_draw, :], r_hist[::-1])

                # incorporate learned bias
                mu_t += bias_all[i_draw]

                # effective sigma
                eff_sigma = sigma_all[i_draw] * sigma_scale_all[i_draw]

                # sample return from Student-t
                ret = mu_t + eff_sigma * rng.standard_t(df=nu_all[i_draw])
                
                # Clip extreme returns to help numeric stability
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
