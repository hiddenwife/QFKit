from src.data_loader import get_stock_data
from src.analysis import FinancialInstrument, TimeSeriesAnalysis
from src.portfolio import Portfolio
import numpy as np
from src.simulation import Simulation 


#stock_list = ["ACWI", "SPXL", "AAPL", "^GSPC", "^FTSE", "VWCE.DE"]

print("This code will compare funds, ETFs, trackers etc from Yahoo Finance.")
print("It will plot them and calculate growth probabilities and past returns.\n")

# --- Loop until user provides at least one ticker
while True:
    tickers = input(
        'Input Yahoo Finance tickers (multiple allowed, separated by space, e.g. "ACWI AAPL ^GSPC"):\n'
    ).split()

    if tickers:
        break
    else:
        print("❌ No tickers provided. Please try again.\n")

# --- Load data safely
instruments = {}  # Dictionary to store TimeSeriesAnalysis objects
data_dict = {}    # Dictionary to store DataFrames

for ticker in tickers:
    try:
        df = get_stock_data(ticker)
        if df.empty:
            print(f"⚠️ {ticker} returned no data. Skipping.")
            continue
        
        instruments[ticker] = TimeSeriesAnalysis(ticker, df)  # Store ticker as an attribute
        inst = instruments[ticker]
        data_dict[ticker] = df  # Store DataFrame without adding ticker as a column
        print(f"✅ Loaded {ticker}")

        print(f"\nAnalysis for {ticker}:")
        print(f"  Sharpe ratio: {inst.sharpe_ratio():.4f}")
        print(f"  Growth probability (5y): {inst.growth_probability(5):.2%}")
        print(f"  Growth probability (10y): {inst.growth_probability(10):.2%}")
        print(f"  Growth probability (15y): {inst.growth_probability(15):.2%}")
        print(f"  1-year 5% VaR (log return): {inst.value_at_risk():.4f}")
    except Exception as e:
        print(f"❌ {ticker} is not a valid ticker. Skipping. Error: {e}")

while True:
    decision = input("\nDo you want to plot the growth comparison of the tickers? [y/n]\n").strip().lower()
    if decision in ('y', 'yes'):
        TimeSeriesAnalysis.plot_all_growth(instruments)
        break
    elif decision in ('n', 'no'):
        break
    else:
        print("Please answer 'y' or 'n'.")

def prompt_and_plot_portfolio(portfolio: Portfolio):
    """
    Interactive helper to get weights from the user (or use equal weights),
    print key metrics, and optionally plot cumulative returns.
    """
    tickers = list(portfolio.instruments.keys())
    n = len(tickers)

    # get weights
    w_input = input(f"Enter {n} weights for {tickers} separated by space (or press Enter for equal weights):\n").strip()
    if w_input:
        w = np.array([float(x) for x in w_input.split()])
        if len(w) != n:
            raise ValueError("Number of weights must match number of tickers.")
        w = w / w.sum()
    else:
        w = np.ones(n) / n

    # print metrics
    print("\nCorrelation matrix:")
    print(portfolio.correlation_matrix())
    print("\nCovariance matrix:")
    print(portfolio.covariance_matrix())

    vol = portfolio.portfolio_volatility(w)
    print(f"\nPortfolio volatility (annualised if returns are): {vol:.6f}")

    print("\nVariance contributions (pct):")
    contrib = portfolio.variance_contributions(w)
    for t, pct in contrib.items():
        print(f"  {t}: {pct:.2%}")

    # ask to plot
    while True:
        decision = input("\nDo you want to plot the portfolio cumulative return? [y/n]\n").strip().lower()
        if decision in ('y', 'yes'):
            portfolio.plot_cumulative_return(w)
            break
        elif decision in ('n', 'no'):
            break
        else:
            print("Please answer 'y' or 'n'.")


portfolio = Portfolio(instruments)

prompt_and_plot_portfolio(portfolio)

simulations = {t: Simulation(t, inst.df) for t, inst in instruments.items()}


def run_and_print_simulation(sim, ticker_choice):
    res = sim.run_simulation(
        horizon_years=5,
        n_sims=10000,
        steps_per_year=252,
        plot=True,
        n_plot_paths=100,
        seed=42
    )
    print(f"\nSimulation results for {ticker_choice}:")
    print(f"  P(end > today): {res['prob_increase']:.2%}")
    print("  Final price stats:")
    for k, v in res['ST_stats'].items():
        print(f"    {k.capitalize()}: {v:.2f}")
        

while True:
    decision = input("\nDo you want to run and plot a simulation? [y/n]\n").strip().lower()
    if decision in ('y', 'yes'):
        if not simulations:
            print("❌ No valid simulations available.")
            break
        elif len(simulations) == 1:
            ticker_choice = next(iter(simulations.keys()))
            sim = simulations[ticker_choice]
            run_and_print_simulation(sim, ticker_choice)
            break
        else:
            while True:
                ticker_choice = input(f"Which ticker? Available: {', '.join(simulations.keys())}\n").strip().upper()
                if ticker_choice in simulations:
                    sim = simulations[ticker_choice]
                    run_and_print_simulation(sim, ticker_choice)
                    break
                else:
                    print("❌ Invalid ticker. Please choose from the list above.")
            break
    elif decision in ('n', 'no'):
        break
    else:
        print("Please answer 'y' or 'n'.")