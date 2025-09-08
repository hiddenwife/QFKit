# src/forecast.py
"""
Forecaster: ARIMA (mean) + GARCH (volatility) pipeline

expects a pandas DataFrame with:
 - DatetimeIndex
 - 'Close' column
 - optional 'Log_Returns' column (if missing it's computed as log(close).diff())

Primary API:
 - Forecaster(df)
 - .fit(arima_kwargs=None, garch_kwargs=None)
 - .forecast(steps=30, alpha=0.05) -> DataFrame with mean_forecast, lower_ci, upper_ci
 - .plot_forecast(steps=30, history=100)

Design goals:
 - Safe defaults for use in interactive scripts
 - Handles missing dates by using business-day index for forecasts
 - Uses pmdarima.auto_arima when available (falls back to statsmodels.ARIMA if not) - issues with downloading it - numpy version?
 - Fits a GARCH model on ARIMA residuals using arch if available
 - Produces volatility-adjusted confidence intervals that grow with forecast horizon

"""

# Example: custom ARIMA and GARCH parameters
# fc.fit(
#     arima_kwargs={'order': (2, 0, 2)},  # ARIMA(p,d,q) for statsmodels fallback
#     garch_kwargs={'p': 2, 'q': 2, 'vol': 'Garch', 'dist': 'normal'}
# )


from typing import Optional, Dict, Any, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional imports - allow graceful errors with helpful messages
try:
    from pmdarima import auto_arima  # type: ignore
    _HAS_PMD = True
except Exception:
    _HAS_PMD = False

try:
    from arch import arch_model  # type: ignore
    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA  # fallback if pmdarima not present
from scipy.stats import norm


warnings.filterwarnings("ignore")


class Forecaster:
    """
    ARIMA (mean) + GARCH (volatility) forecaster.

    Args:
        df: DataFrame with DatetimeIndex and 'Close'. If 'Log_Returns' missing it is calculated.
        returns_col: name of the returns column (default 'Log_Returns').
    """
    def __init__(self, df: pd.DataFrame, returns_col: str = "Log_Returns"):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")

        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column.")

        self.df = df.sort_index().copy()
        # ensure business-day frequency when possible
        if self.df.index.freq is None:
            try:
                inferred = pd.infer_freq(self.df.index)
                if inferred is not None:
                    self.df = self.df.asfreq(inferred)
                else:
                    # set to business day to avoid irregular forecasting index issues
                    self.df = self.df.asfreq('B')
            except Exception:
                self.df = self.df.asfreq('B')

        self.returns_col = returns_col
        if returns_col not in self.df.columns:
            self.df[returns_col] = np.log(self.df['Close']).diff()

        self.returns = self.df[returns_col].dropna()

        # models / results
        self.arima_model = None
        self.garch_res = None
        self.is_fitted = False

    def _adf_stationary(self, series: pd.Series, alpha: float = 0.05) -> bool:
        """Return True if series is stationary by ADF test (p <= alpha)."""
        try:
            pvalue = adfuller(series.dropna())[1]
            return pvalue <= alpha
        except Exception:
            return False

    def fit(self,
            arima_kwargs: Optional[Dict[str, Any]] = None,
            garch_kwargs: Optional[Dict[str, Any]] = None):
        """
        Fit ARIMA on returns, then GARCH on ARIMA residuals.

        arima_kwargs: passed to auto_arima or statsmodels ARIMA fallback.
            Examples:
              {'start_p':0, 'start_q':0, 'max_p':5, 'max_q':5, 'seasonal':False, 'trace':False}
        garch_kwargs: passed to arch_model(...).fit()
            Examples:
              {'p':1, 'q':1, 'vol':'Garch', 'dist':'t'}
        """
        arima_kwargs = arima_kwargs or {}
        garch_kwargs = garch_kwargs or {}

        # ---- Fit ARIMA/auto_arima ----
        print("Fitting ARIMA (mean model on returns)...")
        # If pmdarima available, prefer it for stepwise auto-selection
        if _HAS_PMD:
            # let auto_arima decide differencing unless series is clearly stationary
            d = 0 if self._adf_stationary(self.returns) else None
            am_kwargs = dict(
                y=self.returns,
                start_p=arima_kwargs.get('start_p', 0),
                start_q=arima_kwargs.get('start_q', 0),
                max_p=arima_kwargs.get('max_p', 5),
                max_q=arima_kwargs.get('max_q', 5),
                seasonal=arima_kwargs.get('seasonal', False),
                d=d,
                trace=arima_kwargs.get('trace', False),
                error_action='ignore',
                suppress_warnings=True,
                stepwise=arima_kwargs.get('stepwise', True)
            )
            # allow passing other args through
            extra = {k: v for k, v in arima_kwargs.items() if k not in am_kwargs}
            am_kwargs.update(extra)
            self.arima_model = auto_arima(**am_kwargs)
            # pmdarima returns forecasts on returns directly later via .predict
            resid = self.arima_model.resid()
        else:
            # fallback: fit a simple low-order ARIMA on returns using statsmodels
            order = arima_kwargs.get('order', (3, 0, 3))
            sm = SM_ARIMA(self.returns, order=order)
            self.arima_model = sm.fit()
            resid = self.arima_model.resid

        # ---- Fit GARCH on residuals (volatility model) ----
        if not _HAS_ARCH:
            warnings.warn("arch package not available: GARCH volatility modeling skipped. Forecast CIs will use constant vol.")
            self.garch_res = None
        else:
            print("Fitting GARCH on ARIMA residuals...")
            # defaults
            p = garch_kwargs.get('p', 1)
            q = garch_kwargs.get('q', 1)
            vol = garch_kwargs.get('vol', 'Garch')
            dist = garch_kwargs.get('dist', 't')
            am = arch_model(resid.dropna(), vol=vol, p=p, q=q, dist=dist)
            self.garch_res = am.fit(disp='off')
        self.is_fitted = True
        print("Models fitted.")

    def _forecast_arima_returns(self, steps: int, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (mean_returns, lower_returns, upper_returns) for next `steps` periods.
        Values are (not cumulative) log-return forecasts per-step.
        """
        if _HAS_PMD and hasattr(self.arima_model, "predict"):
            # pmdarima auto_arima
            mean, conf = self.arima_model.predict(n_periods=steps, return_conf_int=True, alpha=alpha)
            mean = np.asarray(mean)  # per-step (log) returns
            lower = conf[:, 0]
            upper = conf[:, 1]
            return mean, lower, upper
        else:
            # statsmodels ARIMAResultsWrapper: get_forecast returns cumulative or per-step depending on model.
            # For simplicity we use .get_forecast on the returns series.
            pred = self.arima_model.get_forecast(steps=steps)
            mean = pred.predicted_mean.values
            ci = pred.conf_int(alpha=alpha)
            lower = ci.iloc[:, 0].values
            upper = ci.iloc[:, 1].values
            return mean, lower, upper

    def _forecast_garch_variance(self, steps: int) -> np.ndarray:
        """
        Return per-step forecast standard deviation (volatility) for residuals.
        If GARCH not fitted, return constant sigma = historical std of residuals.
        The returned array is per-step standard deviations (not variances).
        """
        if self.garch_res is None:
            # fallback: use historical std of residuals
            if _HAS_PMD and hasattr(self.arima_model, "resid"):
                resid = np.asarray(self.arima_model.resid())
            else:
                resid = np.asarray(self.arima_model.resid)
            sigma = np.nanstd(resid)
            return np.repeat(sigma, steps)
        else:
            # arch model: use .forecast to get variance forecasts
            fcast = self.garch_res.forecast(horizon=steps, reindex=False)
            # fcast.variance is a DataFrame with cols 1..steps. take last row
            try:
                var_vals = fcast.variance.iloc[-1].values
                # sometimes returns shape (steps,) or (1, steps)
                var_vals = np.asarray(var_vals, dtype=float)
                # standard deviations
                return np.sqrt(var_vals)
            except Exception:
                # fallback to repeating last in-sample vol
                in_sample_sigma = np.nanstd(self.garch_res.std_resid)
                return np.repeat(in_sample_sigma, steps)

    def forecast(self, steps: int = 30, alpha: float = 0.05) -> pd.DataFrame:
        """
        Forecast price for `steps` future periods.

        Returns DataFrame indexed by business days starting next trading day with columns:
          - mean_forecast: point estimate of price
          - lower_ci, upper_ci: volatility-adjusted confidence interval (alpha default 0.05 -> 95% CI)
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .forecast().")

        # get per-step return forecasts and per-step return CI from ARIMA
        mean_r, lower_r, upper_r = self._forecast_arima_returns(steps=steps, alpha=alpha)

        # forecast residual volatility (std per-step)
        sigma_steps = self._forecast_garch_variance(steps=steps)

        # Combine: compute cumulative log-return forecast and CI using volatility
        # cumulative_mean is cumulative sum of per-step mean log returns
        cum_mean = np.cumsum(mean_r)

        # For volatility we assume independent-step vol generated by GARCH vols: cumulative variance = sum(sigma_i^2)
        cum_var = np.cumsum(sigma_steps ** 2)
        cum_std = np.sqrt(cum_var)

        # z for two-sided CI
        z = norm.ppf(1 - alpha / 2.0)

        # last observed price
        S0 = float(self.df['Close'].iloc[-1])

        mean_price = S0 * np.exp(cum_mean)
        lower_price = S0 * np.exp(cum_mean - z * cum_std)
        upper_price = S0 * np.exp(cum_mean + z * cum_std)

        # Build forecast index starting next business day after last real date
        last_date = self.df.index[-1]
        try:
            start = last_date + pd.Timedelta(days=1)
            idx = pd.bdate_range(start=start, periods=steps)
        except Exception:
            idx = pd.RangeIndex(start=0, stop=steps)

        forecast_df = pd.DataFrame({
            'mean_forecast': mean_price,
            'lower_ci': lower_price,
            'upper_ci': upper_price
        }, index=idx)

        return forecast_df

    def plot_forecast(self, steps: int = 30, history: int = 100, alpha: float = 0.05, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot historical close prices (last `history` points) and forecast with CI.
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .plot_forecast().")

        fc = self.forecast(steps=steps, alpha=alpha)
        hist = self.df['Close'].iloc[-history:]

        plt.figure(figsize=figsize)
        plt.plot(hist.index, hist.values, label='Historical', color='black')
        plt.plot(fc.index, fc['mean_forecast'], label='Forecast', color='blue', linestyle='--')
        plt.fill_between(fc.index, fc['lower_ci'], fc['upper_ci'], color='skyblue', alpha=0.4, label=f'{int((1-alpha)*100)}% CI')
        plt.title(f'Price forecast ({steps} steps) — last {history} history shown')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()


def preprocess_and_forecast(df: pd.DataFrame, steps: int = 30, alpha: float = 0.05, returns_col: str = "Log_Returns"):
    """
    Preprocess the data by filtering out returns outside 3 standard deviations,
    then fit the Forecaster model and plot the forecast.

    df: DataFrame with DatetimeIndex and 'Close' column.
    steps: number of steps to forecast.
    alpha: significance level for confidence intervals.
    returns_col: name of the returns column (default 'Log_Returns').
    """
    returns = np.log(df['Close']).diff().dropna()
    # Remove returns outside 3 standard deviations
    filtered_returns = returns[(np.abs(returns - returns.mean()) < 3 * returns.std())]
    filtered_df = df.loc[filtered_returns.index]
    fc = Forecaster(filtered_df, returns_col=returns_col)
    fc.fit()
    fc.plot_forecast(steps=steps, alpha=alpha)
