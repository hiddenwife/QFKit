import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog
import numpy as np
import pandas as pd
import threading
import matplotlib.pyplot as plt

# Import project-specific modules
from src.data_loader import get_stock_data
from src.analysis import TimeSeriesAnalysis
from src.portfolio import Portfolio
from src.simulation import Simulation
from src.forecast import Forecaster

# Helper to run long-running tasks in a separate thread to keep the GUI responsive
def run_in_thread(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        thread.start()
    return wrapper

class TradingAppGUI:
    def __init__(self, master):
        self.master = master
        master.title("Financial Analysis Toolkit")
        master.geometry("800x700")

        # --- Application State ---
        self.instruments = {}  # {ticker: TimeSeriesAnalysis}
        self.simulations = {}  # {ticker: Simulation}
        self.portfolio = None      # The current Portfolio object
        
        # --- FIX: Widget References ---
        # Store references to widgets instead of using nametowidget
        self.plot_portfolio_button = None
        self.stats_portfolio_button = None

        # --- Main GUI Structure ---
        self.nb = ttk.Notebook(master)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        # Build each tab
        self._build_ticker_tab()
        self._build_analysis_tab()
        self._build_portfolio_tab()
        self._build_simulation_tab()
        self._build_compare_tab()
        self._build_forecast_tab()

        # --- Logging Area ---
        log_frame = ttk.LabelFrame(master, text="Log Output")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_print("GUI ready. Enter tickers to begin.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # UI Builder Helper Methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _build_scrollable_checkbox_frame(self, parent, var_dict, show_weights: bool = False, weight_dict: dict | None = None, height: int = 150):
        """
        Create (or recreate) a scrollable frame of checkboxes inside `parent`.
        This version clears `parent` first to avoid duplication on refresh.
        - var_dict: dict to be populated with ticker -> BooleanVar
        - if show_weights is True, also populate weight_dict with ticker -> StringVar
        Returns the container frame (so caller can keep a reference).
        """
        # ensure weight_dict exists
        if weight_dict is None:
            weight_dict = {}

        # Clear any existing children in parent (prevents duplicate UI trees)
        for w in parent.winfo_children():
            w.destroy()
        var_dict.clear()

        # container frame (one per tab, caller stores reference)
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True, padx=6, pady=4)

        # canvas + scrollbar
        canvas = tk.Canvas(container, height=height)
        scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        # Populate checkboxes (and optional weight entries)
        for ticker in sorted(self.instruments.keys()):
            var = tk.BooleanVar(value=False)
            row = ttk.Frame(inner)
            cb = ttk.Checkbutton(row, text=ticker, variable=var)
            cb.pack(side="left", anchor="w", padx=(2, 6))
            var_dict[ticker] = var

            if show_weights:
                wvar = tk.StringVar(value="1.0")
                weight_dict[ticker] = wvar
                ttk.Label(row, text="Weight:").pack(side="left", padx=(6, 2))
                ttk.Entry(row, textvariable=wvar, width=6).pack(side="left")

            row.pack(fill="x", anchor="w", pady=2)

        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return container

    def _build_portfolio_checkbox_frame(self, parent, ticker_vars, weight_vars):
        """Creates the specific scrollable frame for the portfolio tab with weight entries."""
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True, pady=5)
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        for widget in scrollable_frame.winfo_children():
            widget.destroy()
        ticker_vars.clear()
        weight_vars.clear()

        for ticker in sorted(self.instruments.keys()):
            if ticker == "PORTFOLIO": continue
            
            row = ttk.Frame(scrollable_frame)
            var = tk.BooleanVar(value=True)
            ticker_vars[ticker] = var
            ttk.Checkbutton(row, text=f"{ticker:<10}", variable=var).pack(side="left", padx=5)
            
            ttk.Label(row, text="Weight:").pack(side="left")
            weight_var = tk.StringVar(value="1.0")
            weight_vars[ticker] = weight_var
            ttk.Entry(row, textvariable=weight_var, width=8).pack(side="left", padx=5)
            
            row.pack(fill="x", pady=2)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return container

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Core Application Logic & State Management
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def log_print(self, *args):
        text = " ".join(map(str, args)) + "\n"
        self.master.after(0, self._insert_log_text, text)

    def _insert_log_text(self, text):
        self.log.insert("end", text)
        self.log.see("end")
        
    def get_selected_tickers(self, ticker_vars_dict):
        return [ticker for ticker, var in ticker_vars_dict.items() if var.get()]

    def refresh_all_ui_lists(self):
        self.log_print("Refreshing UI lists...")
        self._build_scrollable_checkbox_frame(self.analysis_frame, self.analysis_vars)
        # Clear the portfolio frame before rebuilding
        for widget in self.portfolio_frame.winfo_children():
            widget.destroy()
        self._build_portfolio_checkbox_frame(self.portfolio_frame, self.portfolio_vars, self.portfolio_weight_vars)
        self._build_scrollable_checkbox_frame(self.sim_frame, self.sim_vars)
        self._build_scrollable_checkbox_frame(self.compare_frame, self.compare_vars)
        self._build_scrollable_checkbox_frame(self.forecast_frame, self.forecast_vars)
        self.log_print("UI lists refreshed.")
        
    def clear_loaded_data(self):
        self.instruments.clear()
        self.simulations.clear()
        self.portfolio = None
        self.log_print("\n--- Cleared all loaded data. ---")
        self.refresh_all_ui_lists()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # "Tickers / Load" Tab
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _build_ticker_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text="1. Load Data")

        ttk.Label(tab, text="Enter Tickers (space-separated, e.g., AAPL GOOG ^GSPC):").pack(anchor="w")
        self.ticker_entry = ttk.Entry(tab)
        self.ticker_entry.pack(fill="x", pady=5)
        self.ticker_entry.insert(0, "AAPL MSFT NVDA")

        # Start-date toggle: show/hide the start-date entry when checked
        self.start_date_var = tk.BooleanVar(value=False)

        def _toggle_start_date():
            if self.start_date_var.get():
                self.start_date_frame.pack(fill="x", pady=(6, 0))
            else:
                self.start_date_frame.pack_forget()

        start_chk = ttk.Checkbutton(
            tab,
            text="Set a Start date (yyyy-mm-dd)? (default is 2022-01-01)",
            variable=self.start_date_var,
            command=_toggle_start_date
        )
        start_chk.pack(anchor="w", pady=(10, 0))

        # Frame that holds the actual start-date widgets (created once, shown/hidden)
        self.start_date_frame = ttk.Frame(tab)
        ttk.Label(self.start_date_frame, text="Start Date:").pack(side="left")
        self.start_date_entry = ttk.Entry(self.start_date_frame, width=16)
        self.start_date_entry.pack(side="left", padx=(6,0))
        self.start_date_entry.insert(0, "2022-01-01")

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(
            btn_frame,
            text="Load & analyse",
            command=lambda: self.load_data(self.start_date_entry.get() if self.start_date_var.get() else None)
        ).pack(side="left")
        ttk.Button(btn_frame, text="Clear All Loaded Data", command=self.clear_loaded_data).pack(side="left", padx=10)


    @run_in_thread
    def load_data(self, start=None):
        tickers = self.ticker_entry.get().upper().split()
        if not tickers:
            self.master.after(0, lambda: messagebox.showerror("Input Error", "Please enter at least one ticker."))
            return

        self.log_print(f"\n--- Loading data for: {', '.join(tickers)} ---")
        for ticker in tickers:
            if not ticker: continue
            try:
                self.log_print(f"Fetching {ticker}...")
                df = get_stock_data(ticker, start=start)
                if df is None or df.empty:
                    self.log_print(f"No data returned for {ticker}. Skipping.")
                    continue

                self.instruments[ticker] = TimeSeriesAnalysis(ticker, df)
                self.simulations[ticker] = Simulation(ticker, df)
                
                inst = self.instruments[ticker]
                self.log_print(f"Loaded {ticker} | Sharpe: {inst.sharpe_ratio():.2f} | CAGR: {inst.compute_cagr():.2%}")

            except Exception as e:
                self.log_print(f"❌ Failed to load {ticker}: {e}")
        
        self.master.after(0, self.refresh_all_ui_lists)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # "Analysis / Plots" Tab
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _build_analysis_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text="2. Analysis")
        ttk.Label(tab, text="Select tickers to analyse or plot:").pack(anchor="w")

        # Create the container frame ONCE
        self.analysis_vars = {}
        self.analysis_frame = ttk.Frame(tab)
        self.analysis_frame.pack(fill="both", expand=True, padx=6, pady=4)
        self._populate_analysis_checkboxes()

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="Plot Relative Growth", command=self.plot_selected_growth).pack(side="left")
        ttk.Button(btn_frame, text="Print Key Stats", command=self.print_selected_stats).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Plot Yearly Returns", command=self.plot_selected_returns).pack(side="left", padx=10)

    def _populate_analysis_checkboxes(self):
        # Clear old widgets before repopulating
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()
        self.analysis_vars.clear()

        canvas = tk.Canvas(self.analysis_frame)
        scrollbar = ttk.Scrollbar(self.analysis_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        for ticker in sorted(self.instruments.keys()):
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(scrollable_frame, text=ticker, variable=var)
            cb.pack(anchor="w", padx=10, pady=2)
            self.analysis_vars[ticker] = var

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def refresh_analysis_checkboxes(self):
        self._populate_analysis_checkboxes()

    def plot_selected_growth(self):
        selected = self.get_selected_tickers(self.analysis_vars)
        if not selected:
            messagebox.showinfo("Selection Error", "Please select at least one ticker to plot.")
            return
        
        inst_to_plot = {t: self.instruments[t] for t in selected}
        self.log_print(f"\nPlotting growth for: {', '.join(selected)}")
        TimeSeriesAnalysis.plot_all_growth(inst_to_plot)
        plt.show()

    def plot_selected_returns(self):
        selected = self.get_selected_tickers(self.analysis_vars)
        if not selected:
            messagebox.showinfo("Selection Error", "Please select at least one ticker to plot.")
            return
        
        inst_to_plot = {t: self.instruments[t] for t in selected}
        self.log_print(f"\nPlotting returns for: {', '.join(selected)}")
        TimeSeriesAnalysis.plot_all_returns(inst_to_plot)
        plt.show()


    def print_selected_stats(self):
        selected = self.get_selected_tickers(self.analysis_vars)
        if not selected:
            messagebox.showinfo("Selection Error", "Please select at least one ticker.")
            return

        self.log_print("\n--- Key Statistics ---")
        for t in selected:
            inst = self.instruments[t]
            self.log_print(
                f"{t}: Sharpe={inst.sharpe_ratio():.3f}, "
                f"CAGR={inst.compute_cagr():.2%}, "
                f"Annual Vol={inst.annualised_return()[1]:.2%}, "
                f"5Y Growth Prob={inst.growth_probability(5):.2%}, "
                f"10Y Growth Prob={inst.growth_probability(10):.2%}, "
                f"15Y Growth Prob={inst.growth_probability(15):.2%}, "
            )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # "Portfolio" Tab
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _build_portfolio_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text="3. Portfolio")
        ttk.Label(tab, text="Select assets and specify weights to create a portfolio:").pack(anchor="w")

        self.portfolio_vars = {}
        self.portfolio_weight_vars = {}

        # Only create the frame once
        self.portfolio_frame = ttk.Frame(tab)
        self.portfolio_frame.pack(fill="both", expand=True, padx=6, pady=4)
        self._build_portfolio_checkbox_frame(self.portfolio_frame, self.portfolio_vars, self.portfolio_weight_vars)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="Create/Update Portfolio", command=self.create_portfolio).pack(side="left")
        self.plot_portfolio_button = ttk.Button(btn_frame, text="Plot Portfolio Growth", command=self.plot_portfolio_growth, state="disabled")
        self.plot_portfolio_button.pack(side="left", padx=10)
        self.stats_portfolio_button = ttk.Button(btn_frame, text="Show Matrices & Stats", command=self.show_portfolio_stats, state="disabled")
        self.stats_portfolio_button.pack(side="left")

    def create_portfolio(self):
        selected = self.get_selected_tickers(self.portfolio_vars)
        if len(selected) < 2:
            messagebox.showerror("Portfolio Error", "Please select at least two assets for a portfolio.")
            return
        
        weights = []
        try:
            for ticker in selected:
                w = float(self.portfolio_weight_vars[ticker].get())
                if w < 0: raise ValueError("Weights cannot be negative.")
                weights.append(w)
            weights = np.array(weights)
            weights /= weights.sum()
        except (ValueError, ZeroDivisionError) as e:
            messagebox.showerror("Input Error", f"Invalid weight entered. Please use valid numbers. Details: {e}")
            return

        self.log_print("\n--- Creating Portfolio ---")
        self.log_print(f"Assets: {', '.join(selected)}")
        self.log_print(f"Weights: {', '.join([f'{w:.2%}' for w in weights])}")

        inst_dict = {t: self.instruments[t] for t in selected}
        self.portfolio = Portfolio(inst_dict)

        close_prices = pd.concat({t: inst.df['Close'] for t, inst in inst_dict.items()}, axis=1).dropna()
        portfolio_close = (close_prices * weights).sum(axis=1)

        portfolio_df = pd.DataFrame(index=portfolio_close.index)
        portfolio_df['Close'] = portfolio_close
        portfolio_df['Log_Returns'] = np.log(portfolio_df['Close']).diff()
        portfolio_df.dropna(inplace=True)

        self.instruments['PORTFOLIO'] = TimeSeriesAnalysis('PORTFOLIO', portfolio_df)
        self.simulations['PORTFOLIO'] = Simulation('PORTFOLIO', portfolio_df)
        self.log_print("Portfolio created and added as 'PORTFOLIO' for further analysis.")

        # --- FIX: Use stored references to configure buttons ---
        if self.plot_portfolio_button:
            self.plot_portfolio_button.config(state="normal")
        if self.stats_portfolio_button:
            self.stats_portfolio_button.config(state="normal")
        
        self.refresh_all_ui_lists()
        
    def plot_portfolio_growth(self):
        if not self.portfolio: return
        selected = list(self.portfolio.instruments.keys())
        weights = [float(self.portfolio_weight_vars[t].get()) for t in selected]
        weights = np.array(weights) / np.sum(weights)
        
        self.log_print("\nPlotting portfolio cumulative growth...")
        self.portfolio.plot_cumulative_return(weights)
        plt.show()

    def show_portfolio_stats(self):
        if not self.portfolio: return
        selected = list(self.portfolio.instruments.keys())
        weights = [float(self.portfolio_weight_vars[t].get()) for t in selected]
        weights = np.array(weights) / np.sum(weights)

        self.log_print("\n--- Portfolio Analytics ---")
        self.log_print("Correlation Matrix:\n" + self.portfolio.correlation_matrix().to_string())
        self.log_print("\nVariance Contributions:")
        contrib = self.portfolio.variance_contributions(weights)
        for t, pct in contrib.items():
            self.log_print(f"  {t}: {pct:.2%}")

    # --- FIX: Helper method to display plots on the main thread ---
    def _display_plot(self):
        """This function MUST be called from the main thread."""
        try:
            plt.show()
        except Exception as e:
            self.log_print(f"Error displaying plot: {e}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # "Simulation" Tab
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _build_simulation_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text="4. Simulation")
        ttk.Label(tab, text="Select assets to run Geometric Brownian Motion (GBM) simulation:").pack(anchor="w")
        
        self.sim_vars = {}
        self.sim_frame = self._build_scrollable_checkbox_frame(tab, self.sim_vars)
        
        ttk.Button(tab, text="Run Simulation for Selected", command=self.run_simulations).pack(pady=10)

    @run_in_thread
    def run_simulations(self):
        selected = self.get_selected_tickers(self.sim_vars)
        if not selected:
            self.master.after(0, lambda: messagebox.showinfo("Selection Error", "Please select at least one asset to simulate."))
            return

        self.log_print(f"\n--- Running Simulations for: {', '.join(selected)} ---")
        for ticker in selected:
            sim = self.simulations.get(ticker)
            if not sim:
                self.log_print(f"No simulation object found for {ticker}.")
                continue

            self.log_print(f"Simulating {ticker} (compute only)...")
            # compute results without plotting in background thread
            res = sim.run_simulation(horizon_years=5, n_sims=1000, plot=False, n_plot_paths=50)
            self.log_print(f"  > {ticker} 5Y P(increase): {res['prob_increase']:.2%}")

            # schedule plotting on main thread (re-run plotting step on main thread)
            self.master.after(0, lambda s=sim: (s.run_simulation(horizon_years=5, n_sims=1000, plot=True, n_plot_paths=50), self._display_plot()))


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # "Compare Sim vs Real" Tab
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _build_compare_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text="5. Compare")
        ttk.Label(tab, text="Compare historical performance against a simulation based on historical stats:").pack(anchor="w")

        self.compare_vars = {}
        self.compare_frame = self._build_scrollable_checkbox_frame(tab, self.compare_vars)
        
        ttk.Button(tab, text="Run Comparison for Selected", command=self.run_comparisons).pack(pady=10)

    @run_in_thread
    def run_comparisons(self):
        selected = self.get_selected_tickers(self.compare_vars)
        if not selected:
            self.master.after(0, lambda: messagebox.showinfo("Selection Error", "Please select at least one asset to compare."))
            return

        self.log_print(f"\n--- Running Comparisons for: {', '.join(selected)} ---")
        for ticker in selected:
            sim = self.simulations.get(ticker)
            if not sim:
                self.log_print(f"No simulation object found for {ticker}.")
                continue

            self.log_print(f"Comparing {ticker} (compute only)...")
            # compute comparison without plotting
            sim.compare_simulation_to_real(n_sims=1000, plot=False)
            # schedule the plotting for the main thread
            self.master.after(0, lambda s=sim: (s.compare_simulation_to_real(n_sims=1000, plot=True), self._display_plot()))


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # "Forecast" Tab
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _build_forecast_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text="6. Forecast")
        ttk.Label(tab, text="Select an asset to forecast future price using ARIMA + GARCH models:").pack(anchor="w")
        
        self.forecast_vars = {}
        self.forecast_frame = self._build_scrollable_checkbox_frame(tab, self.forecast_vars)
        
        ttk.Button(tab, text="Run Forecast for Selected", command=self.run_forecasts).pack(pady=10)
    
    # --- FIX: Helper method to run the plotting part of the forecast on the main thread ---
    def _display_forecast(self, forecaster, ticker):
        """Displays the forecast plot. Runs on the main thread."""
        try:
            self.log_print(f"Displaying forecast for {ticker}...")
            forecaster.plot_forecast(steps=30, history=100)
            self._display_plot()
        except Exception as e:
            self.log_print(f"❌ Failed to plot forecast for {ticker}: {e}")

    @run_in_thread
    def run_forecasts(self):
        selected = self.get_selected_tickers(self.forecast_vars)
        if not selected:
            self.master.after(0, lambda: messagebox.showinfo("Selection Error", "Please select at least one asset to forecast."))
            return

        self.log_print(f"\n--- Running Forecasts for: {', '.join(selected)} ---")
        for ticker in selected:
            inst = self.instruments.get(ticker)
            if not inst:
                self.log_print(f"No data found for {ticker}.")
                continue

            self.log_print(f"Fitting forecast model for {ticker}...")
            try:
                # Fit the model in the background thread (no plotting here)
                fc = Forecaster(inst.df.copy())
                fc.fit()
                # Schedule the plotting to run on the main thread (plot only on main thread)
                self.master.after(0, self._display_forecast, fc, ticker)
            except Exception as e:
                self.log_print(f"❌ Forecast model fit for {ticker} failed: {e}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Application Entry Point
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def launch_gui():
    """Initialises and runs the Tkinter GUI application."""
    root = tk.Tk()
    app = TradingAppGUI(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()