import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st


class FinancialInstrument:
  def __init__(self, ticker, df):
    """
        Base class to hold stock data and basic info.
        df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume', 'Log_Returns']
    """
    self.ticker = ticker
    self.df = df
  
  def plot_prices(self):
    """Plot the closing price"""
    self.df['Close'].plot(title=f"{self.ticker} Closing Price")
    plt.show()

  def summary_stats(self):
    return self.df['Log_Returns'].describe()
  
  def annualised_return(self):
    """Calculate the annualised return for mean and volatility"""
    mu = self.df['Log_Returns'].mean() * 252
    sigma = self.df['Log_Returns'].std() * np.sqrt(252)
    return mu, sigma
  
  def growth_probability(self, horizon_years):
    """
    Estimate probability that price increases over horizon_years
    using lognormal model with drift + volatility.
    """
    mu, sigma = self.annualised_return()

    # horizon
    T = horizon_years

    # Normal approximation: S_T ~ lognormal
    drift = (mu - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)

    # Probability that return > 0
    # P( log(S_T/S_0) > 0 ) = P( Z > -drift/vol )
    prob = 1 - st.norm.cdf(0, loc=drift, scale=vol)
    return prob
  
  def expected_price_and_ci(self, horizon_years, ci=0.95):
    mu, sigma = self.annualised_return()
    S0 = self.df['Close'].iloc[-1]
    T = horizon_years

    # expected price
    expected = S0 * np.exp(mu * T)

    # confidence interval
    drift = (mu - 0.5*sigma**2) * T
    vol = sigma * np.sqrt(T)

    lower = S0 * np.exp(drift + st.norm.ppf((1-ci)/2) * vol)
    upper = S0 * np.exp(drift + st.norm.ppf(1-(1-ci)/2) * vol)

    return expected, (lower, upper)
  
  def value_at_risk(self, horizon_days=252, alpha=0.05):
    """what is the maximum loss we expect with 95% confidence over some horizon?"""
    mu, sigma = self.annualised_return()
    T = horizon_days / 252  # convert to years

    var = mu*T + sigma*np.sqrt(T)*st.norm.ppf(alpha)
    return var  # log return threshold
  
  def sharpe_ratio(self, risk_free_rate=0.0):
    mu, sigma = self.annualised_return()
    return (mu - risk_free_rate) / sigma
  
  def rolling_volatility(self, window=252):
    """Rolling 1-year volatility (default window = 252 trading days)."""
    return self.df['Log_Returns'].rolling(window).std() * np.sqrt(252)

  def rolling_sharpe(self, window=252, risk_free_rate=0.0):
    rolling_mean = self.df['Log_Returns'].rolling(window).mean() * 252
    rolling_vol = self.df['Log_Returns'].rolling(window).std() * np.sqrt(252)
    return (rolling_mean - risk_free_rate) / rolling_vol
  
  def average_yearly_simple_return(self):
      mu_log, _ = self.annualised_return()  # already annualised log-return
      simple_return = np.exp(mu_log) - 1    # convert log return -> simple return
      return simple_return * 100            # in percentage

  def compute_cagr(self):
    # Ensure start and end prices are scalars
    start_price = self.df['Close'].iloc[0] 
    end_price = self.df['Close'].iloc[-1] 

    num_years = (self.df.index[-1] - self.df.index[0]).days / 365.25
    cagr = (end_price / start_price) ** (1 / num_years) - 1

    return cagr



class TimeSeriesAnalysis(FinancialInstrument):
  def __init__(self, ticker, df):
    super().__init__(ticker, df) #inherit constructors from FinancialInstrument

  def rolling_volatility(self, window = 30):
    """Compute rolling volatility"""
    self.df['Rolling_Vol'] = self.df['Log_Returns'].rolling(window).std()
    return self.df[['Rolling_Vol']]

  
  def moving_average(self, window = 20):
    """Compute rolling average"""
    self.df[f"MA{window}"] = self.df['Close'].rolling(window).mean()
    return self.df[[f"MA{window}"]], window
  
  def plot_with_ma(self, window = 20):
    """Plot price with moving average overlay"""
    self.moving_average()
    mu = self.average_yearly_simple_return()
    ax = self.df[['Close', f"MA{window}"]].plot(title=f"{self.ticker} with {window}-day MA")
    # Adding the average return to the legend
    ax.plot([], [], ' ', label=f'Average Yearly Return: {mu:.2f}%')
    ax.legend(loc='best')
    plt.show()

  def plot_returns(self):
    """Plot daily percentage returns"""
    returns_pct = self.df['Log_Returns'] * 100
    ax = returns_pct.plot(title=f"{self.ticker} Daily Returns (%)", figsize=(10,5))
    ax.set_ylabel("Return (%)")
    plt.show()

  def plot_growth(self):
      """Plot relative growth (normalised to 1.0 at start)"""
      growth = self.df['Close'] / self.df['Close'].iloc[0]
      cagr = self.compute_cagr()

      # Plot
      ax = growth.plot(title=f"{self.ticker} Relative Growth (Start = 1)", figsize=(10,5))
      ax.plot([], [], ' ', label=f'CAGR: {cagr*100:.2f}%')
      ax.set_ylabel("Growth (relative to 1)")
      ax.legend()
      plt.show()

  @staticmethod
  def plot_all_growth(instruments):
        """Plot relative growth of multiple tickers on one chart."""
        ax = None
        for ticker, inst in instruments.items():
            growth = inst.df['Close'] / inst.df['Close'].iloc[0]
            cagr = inst.compute_cagr()
            ax = growth.plot(
                ax=ax,
                label=f"{ticker}, CAGR: {cagr:.2%}", 
                alpha=0.7,
                figsize=(10, 6)
            )

        if ax:
            ax.set_title("Relative Growth Comparison (Start = 1)")
            ax.set_ylabel("Growth (relative to 1)")
            ax.legend()
            plt.show()
        else:
            print("❌ No valid data to plot.")


  @staticmethod
  def plot_all_returns(instruments):
    """Plot yearly returns of multiple tickers on one chart."""
    yearly_returns = {}

    # Calculate yearly returns for each instrument
    for ticker, inst in instruments.items():
        # Resample daily log returns to yearly returns
        yearly = inst.df['Log_Returns'].resample('Y').sum() * 100 
        yearly.index = pd.to_datetime(yearly.index)  
        yearly_returns[ticker] = yearly

    yearly_returns_df = pd.DataFrame(yearly_returns)

    # Plotting using matplotlib and set x-axis ticks to show only the year
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_returns_df.plot(kind='bar', alpha=0.7, ax=ax)
    ax.set_title("Yearly Returns Comparison")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Year")
    ax.legend(title="Tickers")

    years = [d.year for d in pd.to_datetime(yearly_returns_df.index)]
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years, rotation=0)

    plt.tight_layout() 
    plt.show()
