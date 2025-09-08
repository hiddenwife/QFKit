from src.data_loader import get_stock_data
from src.analysis import FinancialInstrument, TimeSeriesAnalysis
from src.portfolio import Portfolio
import numpy as np
from src.simulation import Simulation
from src.forecast import Forecaster
from gui import launch_gui


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
    Returns the weights used.
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

    return w

simulations = {t: Simulation(t, inst.df) for t, inst in instruments.items()}

# Only if the user inputs more than one ticker will a portfolio be created
if not len(tickers) in (0, 1):
    portfolio = Portfolio(instruments)
    w = prompt_and_plot_portfolio(portfolio)

    # Calculate portfolio historical price series
    tickers_port = list(portfolio.instruments.keys())

    # Build portfolio price series (weighted sum of normalized prices)
    price_df = np.column_stack([instruments[t].df['Close'].values for t in tickers_port])
    portfolio_prices = np.dot(price_df, w)
    portfolio_dates = instruments[tickers_port[0]].df.index  # assumes all have same dates

    # Create DataFrame for portfolio
    portfolio_df = instruments[tickers_port[0]].df.copy()
    portfolio_df['Close'] = portfolio_prices

    # Add portfolio as a pseudo-ticker
    instruments['PORTFOLIO'] = TimeSeriesAnalysis('PORTFOLIO', portfolio_df)
    simulations['PORTFOLIO'] = Simulation('PORTFOLIO', portfolio_df)


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
                ticker_choice = input(f"Which ticker? Available: {', '.join(simulations.keys())}, 'all', or 'none'):\n").strip().upper()
                if ticker_choice in simulations:
                    sim = simulations[ticker_choice]
                    run_and_print_simulation(sim, ticker_choice)
                    break
                elif ticker_choice == 'ALL':
                    for sim in simulations.values():
                        run_and_print_simulation(sim, ticker_choice)
                    break
                elif ticker_choice == 'NONE':
                    break
                else:
                    print("❌ Invalid ticker. Please choose from the list above.")
            break
    elif decision in ('n', 'no'):
        break
    else:
        print("Please answer 'y' or 'n'.")


def prompt_compare_simulation(simulations):
    """
    Prompt user to choose a ticker to compare simulated vs real paths,
    or run for all tickers.
    """
    if not simulations:
        print("❌ No simulations available.")
        return

    while True:
        choice = input(f"\nCompare simulated vs real price paths for which ticker? (type {', '.join(simulations.keys())}, 'all', or 'none'):\n").strip().upper()
        if choice in ('NONE', 'N'):
            return
        elif choice == 'ALL':
            to_run = list(simulations.keys())
            break
        elif choice in simulations:
            to_run = [choice]
            break
        print(f"❌ Invalid choice. Enter {', '.join(simulations.keys())}, 'all', or 'none'.")

    for t in to_run:
        sim = simulations[t]
        print(f"\nRunning comparison for {t} ...")
        # runs comparison from first real date to last real date and plots
        out = sim.compare_simulation_to_real(n_sims=2000, steps_per_year=252, seed=42, plot=True)
        # print simple numeric comparison at end date
        real_end = out['real'].iloc[-1] if not out['real'].isna().all() else None
        sim_expected_end = out['expected'].iloc[-1]
        sim_median_end = out['median'].iloc[-1]
        print(f"Ticker {t}: Real end = {real_end}, Sim expected end = {sim_expected_end:.2f}, Sim median end = {sim_median_end:.2f}")

# call the prompt at end of main
prompt_compare_simulation(simulations)


def prompt_forecasting(instruments):
    if not instruments:
        print("❌ No instruments loaded.")
        return

    print("\nForecasting menu — available tickers:", ", ".join(instruments.keys()))
    while True:
        choice = input("Select ticker for forecasting (or 'none' to skip):\n").strip()
        if choice.lower() in ('none', ''):
            return
        if choice not in instruments:
            print("❌ Invalid ticker. Try again.")
            continue

        inst = instruments[choice]

        # Ask if user wants to use only recent data
        use_recent = input("Use only recent data (last 250 days)? [y/n]: ").strip().lower() in ('y', 'yes')
        if use_recent:
            df_for_forecast = inst.df.iloc[-250:]
        else:
            df_for_forecast = inst.df

        fc = Forecaster(df_for_forecast)  # uses Log_Returns by default

        # Fit models (safe defaults)
        try:
            fc.fit()
        except Exception as e:
            print("Model fit failed:", e)
            # continue to next ticker or allow retry
            if input("Retry fit? [y/n]: ").strip().lower() in ('y', 'yes'):
                continue
            else:
                if input("Forecast next ticker? [y/n]: ").strip().lower() in ('y', 'yes'):
                    continue
                else:
                    break

        # Produce a numeric preview of the next N forecasts
        steps = 20
        try:
            forecast_df = fc.forecast(steps=steps)
            print(f"\nPrice forecast for next {steps} periods (first 5 rows):")
            print(forecast_df.head().to_string())
        except Exception as e:
            print("Forecast failed:", e)

        # Ask to plot
        if input("Plot price forecast? [y/n]: ").strip().lower() in ('y', 'yes'):
            try:
                fc.plot_forecast(steps=steps, history=100)
            except Exception as e:
                print("Plot failed:", e)

        # repeat or exit
        if input("\nRun forecasting for another ticker? [y/n]: ").strip().lower() not in ('y','yes'):
            break

# call prompt after simulations/prompt_compare_simulation
prompt_forecasting(instruments)

