# src/analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st


class FinancialInstrument:
    def __init__(self, ticker, df):
        self.ticker = ticker
        self.df = df

    def plot_prices(self):
        self.df['Close'].plot(title=f"{self.ticker} Closing Price")
        plt.show()

    def summary_stats(self):
        return self.df['Log_Returns'].describe()

    def annualised_return(self):
        mu = self.df['Log_Returns'].mean() * 252
        sigma = self.df['Log_Returns'].std() * np.sqrt(252)
        return mu, sigma

    def growth_probability(self, horizon_years):
        mu, sigma = self.annualised_return()
        T = horizon_years
        drift = (mu - 0.5 * sigma**2) * T
        vol = sigma * np.sqrt(T)
        prob = 1 - st.norm.cdf(0, loc=drift, scale=vol)
        return prob

    def expected_price_and_ci(self, horizon_years, ci=0.95):
        mu, sigma = self.annualised_return()
        S0 = self.df['Close'].iloc[-1]
        T = horizon_years
        expected = S0 * np.exp(mu * T)
        drift = (mu - 0.5 * sigma**2) * T
        vol = sigma * np.sqrt(T)
        lower = S0 * np.exp(drift + st.norm.ppf((1 - ci) / 2) * vol)
        upper = S0 * np.exp(drift + st.norm.ppf(1 - (1 - ci) / 2) * vol)
        return expected, (lower, upper)

    def value_at_risk(self, horizon_days=252, alpha=0.05):
        mu, sigma = self.annualised_return()
        T = horizon_days / 252
        var = mu * T + sigma * np.sqrt(T) * st.norm.ppf(alpha)
        return var

    def sharpe_ratio(self, risk_free_rate=0.0):
        mu, sigma = self.annualised_return()
        if sigma == 0:
            return 0.0
        return (mu - risk_free_rate) / sigma

    def rolling_volatility(self, window=252):
        return self.df['Log_Returns'].rolling(window).std() * np.sqrt(252)

    def rolling_sharpe(self, window=252, risk_free_rate=0.0):
        rolling_mean = self.df['Log_Returns'].rolling(window).mean() * 252
        rolling_vol = self.df['Log_Returns'].rolling(window).std() * np.sqrt(252)
        return (rolling_mean - risk_free_rate).div(rolling_vol).replace([np.inf, -np.inf], 0).fillna(0)

    def average_yearly_simple_return(self):
        mu_log, _ = self.annualised_return()
        simple_return = np.exp(mu_log) - 1
        return simple_return * 100

    def compute_cagr(self):
        start_price = self.df['Close'].iloc[0]
        end_price = self.df['Close'].iloc[-1]
        num_years = (self.df.index[-1] - self.df.index[0]).days / 365.25
        if num_years == 0 or start_price == 0:
            return 0.0
        cagr = (end_price / start_price) ** (1 / num_years) - 1
        return cagr


class TimeSeriesAnalysis(FinancialInstrument):
    def __init__(self, ticker, df):
        super().__init__(ticker, df)

    def rolling_volatility(self, window=30):
        self.df['Rolling_Vol'] = self.df['Log_Returns'].rolling(window).std()
        return self.df[['Rolling_Vol']]

    def moving_average(self, window=20):
        self.df[f"MA{window}"] = self.df['Close'].rolling(window).mean()
        return self.df[[f"MA{window}"]], window

    def plot_with_ma(self, window=20):
        self.moving_average()
        mu = self.average_yearly_simple_return()
        ax = self.df[['Close', f"MA{window}"]].plot(title=f"{self.ticker} with {window}-day MA")
        ax.plot([], [], ' ', label=f'Average Yearly Return: {mu:.2f}%')
        ax.legend(loc='best')
        plt.show()

    def plot_returns(self):
        returns_pct = self.df['Log_Returns'] * 100
        ax = returns_pct.plot(title=f"{self.ticker} Daily Returns (%)", figsize=(10, 5))
        ax.set_ylabel("Return (%)")
        plt.show()

    def plot_growth(self):
        growth = self.df['Close'] / self.df['Close'].iloc[0]
        cagr = self.compute_cagr()
        ax = growth.plot(title=f"{self.ticker} Relative Growth (Start = 1)", figsize=(10, 5))
        ax.plot([], [], ' ', label=f'CAGR: {cagr*100:.2f}%')
        ax.set_ylabel("Growth (relative to 1)")
        ax.legend()
        plt.show()

    def plot_rolling_sharpe_with_signals(self, window=60):
        if self.df.empty:
            raise ValueError("DataFrame is empty.")
        start_date = self.df.index[-1] - pd.DateOffset(years=1)
        recent_data = self.df.loc[self.df.index >= start_date]

        if len(recent_data) < window:
            raise ValueError("Not enough data in the last year to compute rolling Sharpe.")

        rolling_sharpe_series = recent_data['Log_Returns'].rolling(window).mean() * 252 / \
                                (recent_data['Log_Returns'].rolling(window).std() * np.sqrt(252))
        rolling_sharpe_series = rolling_sharpe_series.dropna()

        signals = pd.DataFrame(index=rolling_sharpe_series.index)
        signals['sharpe'] = rolling_sharpe_series

        signals['trend'] = signals['sharpe'].diff().rolling(window=3).mean()
        signals['dynamic_threshold'] = -0.1 - (0.1 * signals['sharpe'])
        signals['momentum_sell'] = (signals['trend'] < signals['dynamic_threshold']) & (signals['sharpe'] > 1)

        fig, ax = plt.subplots(figsize=(12, 7))
        is_good = signals['sharpe'] > 1.0
        is_bad = signals['sharpe'] < 0.0
        is_neutral = ~is_good & ~is_bad

        ax.plot(signals.index, signals['sharpe'].where(is_good), color='green', linewidth=2, zorder=2)
        ax.plot(signals.index, signals['sharpe'].where(is_bad), color='red', linewidth=2, zorder=2)
        ax.plot(signals.index, signals['sharpe'].where(is_neutral), color='navy',
                label=f'{window}-Day Rolling Sharpe', linewidth=2, zorder=2)

        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Good Threshold (1.0)')
        ax.axhline(y=0.0, color='tomato', linestyle='--', alpha=0.5, label='Negative Threshold (0.0)')

        ax.plot(signals[signals['momentum_sell']].index, signals['sharpe'][signals['momentum_sell']], 'x',
                markersize=10, markeredgewidth=2, color='orange', label='Momentum Sell', zorder=5)

        ax.set_title(f'{self.ticker} | Rolling Sharpe Ratio & Momentum Signals (Last Year)')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

    def plot_moving_average_strategy(self, short_window=50, long_window=200):
        df = self.df.copy()
        df['SMA_short'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
        df['SMA_long'] = df['Close'].rolling(window=long_window, min_periods=1).mean()

        df['Signal'] = 0
        df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = 1
        df.loc[df['SMA_short'] < df['SMA_long'], 'Signal'] = -1

        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], label=f'{self.ticker} Close Price', alpha=0.6, color='lightblue')
        ax.plot(df.index, df['SMA_short'], label=f'{short_window}-day SMA', color='orange')
        ax.plot(df.index, df['SMA_long'], label=f'{long_window}-day SMA', color='green')

        ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=30)
        ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=30)

        ax.set_title(f"{self.ticker} Stock Price with Moving Average Strategy Signals")
        ax.set_ylabel("Stock Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

    @staticmethod
    def plot_all_growth(instruments):
        fig, ax = plt.subplots(figsize=(10, 6))
        for ticker, inst in instruments.items():
            growth = inst.df['Close'] / inst.df['Close'].iloc[0]
            cagr = inst.compute_cagr()
            ax.plot(growth.index, growth.values, label=f"{ticker}, CAGR: {cagr:.2%}", alpha=0.8)
        ax.set_title("Relative Growth Comparison (Start = 1)")
        ax.set_ylabel("Growth (relative to 1)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        return fig

    @staticmethod
    def plot_all_returns(instruments):
        yearly_returns = {}
        for ticker, inst in instruments.items():
            yearly = inst.df['Log_Returns'].resample('YE').sum() * 100
            yearly.index = pd.to_datetime(yearly.index)
            yearly_returns[ticker] = yearly
        yearly_returns_df = pd.DataFrame(yearly_returns)
        fig, ax = plt.subplots(figsize=(10, 6))
        yearly_returns_df.plot(kind='bar', alpha=0.7, ax=ax, width=0.8)
        ax.set_title("Yearly Returns Comparison")
        ax.set_ylabel("Return (%)")
        ax.set_xlabel("Year")
        ax.legend(title="Tickers")
        years = [d.year for d in pd.to_datetime(yearly_returns_df.index)]
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels(years, rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        return fig
