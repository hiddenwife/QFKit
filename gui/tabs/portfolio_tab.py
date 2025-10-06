# gui/tabs/portfolio_tab.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame, QHBoxLayout,
    QCheckBox, QLineEdit, QPushButton, QMessageBox, QSizePolicy, QSpacerItem, QDialog
)
from PySide6.QtCore import Slot
from PySide6.QtGui import QDoubleValidator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# Import domain logic
from src.analysis import TimeSeriesAnalysis
from src.portfolio import Portfolio
from src.simulation import SDE_Simulation


class PortfolioTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        self.portfolio = None  # instance of Portfolio (domain class)
        self.portfolio_vars = {}
        self.portfolio_weight_vars = {}

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        root_layout.addWidget(QLabel("Select assets and specify weights to create a portfolio:"))

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QFrame()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setContentsMargins(4, 4, 4, 4)
        self.scroll_layout.setSpacing(6)
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll.setWidget(self.scroll_widget)
        root_layout.addWidget(self.scroll, stretch=1)

        btn_row = QHBoxLayout()
        self.create_btn = QPushButton("Create/Update Portfolio")
        self.create_btn.clicked.connect(self.create_portfolio)
        btn_row.addWidget(self.create_btn)

        self.plot_portfolio_button = QPushButton("Plot Portfolio Growth")
        self.plot_portfolio_button.setEnabled(False)
        self.plot_portfolio_button.clicked.connect(self.plot_portfolio_growth)
        btn_row.addWidget(self.plot_portfolio_button)

        self.stats_portfolio_button = QPushButton("Show Matrices & Stats")
        self.stats_portfolio_button.setEnabled(False)
        self.stats_portfolio_button.clicked.connect(self.show_portfolio_stats)
        btn_row.addWidget(self.stats_portfolio_button)

        btn_row.addItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        root_layout.addLayout(btn_row)

        self.setLayout(root_layout)
        self._build_portfolio_checkbox_frame()

    def log_print(self, msg: str):
        if hasattr(self.main_window, "log_message"):
            self.main_window.log_message(msg)
        else:
            print(msg)

    def _clear_portfolio_list(self):
        for i in reversed(range(self.scroll_layout.count())):
            item = self.scroll_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if w:
                w.setParent(None)
        self.portfolio_vars.clear()
        self.portfolio_weight_vars.clear()

    def _build_portfolio_checkbox_frame(self):
        self._clear_portfolio_list()
        instruments = getattr(self.main_window, "instruments", {}) or {}
        if not instruments:
            lbl = QLabel("No instruments loaded. Load data first.")
            self.scroll_layout.addWidget(lbl)
            return

        def make_on_state_changed(wedit):
            def _on_state(state):
                enabled = bool(state)
                wedit.setEnabled(enabled)
                wedit.setStyleSheet("" if enabled else "color: gray;")
                if enabled and (wedit.text().strip() == "" or wedit.text() == "0"):
                    wedit.setText("1.0")
            return _on_state

        for ticker in sorted(instruments.keys()):
            # --- FIX: Skip the 'PORTFOLIO' ticker ---
            if ticker == 'PORTFOLIO':
                continue

            row = QHBoxLayout()
            cb = QCheckBox(ticker)
            cb.setChecked(False)
            row.addWidget(cb)

            wedit = QLineEdit()
            wedit.setFixedWidth(90)
            wedit.setText("1.0")
            wedit.setEnabled(False)
            wedit.setStyleSheet("color: gray;")
            validator = QDoubleValidator(0.0, 1e12, 8, wedit) 
            validator.setNotation(QDoubleValidator.StandardNotation)
            wedit.setValidator(validator)
            wedit.setPlaceholderText("0.0")
            row.addWidget(wedit)

            cb.stateChanged.connect(make_on_state_changed(wedit))

            self.portfolio_vars[ticker] = cb
            self.portfolio_weight_vars[ticker] = wedit
            container = QFrame()
            container.setLayout(row)
            self.scroll_layout.addWidget(container)

        self.scroll_layout.addStretch(1)

    def _get_selected_tickers(self):
        return [t for t, cb in self.portfolio_vars.items() if cb.isChecked()]

    def _read_weights_for(self, tickers):
        # returns numpy array of normalized weights aligned to tickers order
        try:
            weights = []
            for t in tickers:
                txt = self.portfolio_weight_vars[t].text()
                w = float(txt) if txt.strip() != "" else 0.0
                if w < 0:
                    raise ValueError("Weights cannot be negative.")
                weights.append(w)
            weights = np.array(weights, dtype=float)
            if weights.sum() == 0:
                raise ValueError("Sum of weights must be > 0")
            return weights / weights.sum()
        except ValueError as e:
            raise

    @Slot()
    def create_portfolio(self):
        selected = self._get_selected_tickers()
        if len(selected) < 2:
            QMessageBox.critical(self, "Portfolio Error", "Please select at least two assets for a portfolio.")
            return

        try:
            weights = self._read_weights_for(selected)
        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"Invalid weight entered. Details: {e}")
            return

        self.log_print("\n--- Creating Portfolio ---")
        self.log_print(f"Assets: {', '.join(selected)}")
        self.log_print("Weights: " + ", ".join([f"{w:.2%}" for w in weights]))

        instruments_map = {t: self.main_window.instruments[t] for t in selected}

        # Create domain Portfolio object
        try:
            self.portfolio = Portfolio(instruments_map)
        except Exception as e:
            self.log_print(f"Error constructing Portfolio object: {e}")
            self.portfolio = None

        # Create a time-series object for downstream tools (TimeSeriesAnalysis / SDE_Simulation)
        try:
            ts = Portfolio.from_weighted_close_series('PORTFOLIO', instruments_map, weights)
            if TimeSeriesAnalysis is not None:
                self.main_window.instruments['PORTFOLIO'] = TimeSeriesAnalysis('PORTFOLIO', ts.df)
            else:
                self.main_window.instruments['PORTFOLIO'] = ts
            if SDE_Simulation is not None:
                self.main_window.simulations['PORTFOLIO'] = SDE_Simulation('PORTFOLIO', ts.df)
            else:
                self.main_window.simulations['PORTFOLIO'] = None

            self.log_print("Portfolio created and added as 'PORTFOLIO' for further analysis.")
        except Exception as e:
            QMessageBox.critical(self, "Portfolio Error", f"Failed to construct portfolio time series:\n{e}")
            return

        self.plot_portfolio_button.setEnabled(True)
        self.stats_portfolio_button.setEnabled(True)

        if hasattr(self.main_window, "refresh_all_ui_lists"):
            try:
                self.main_window.refresh_all_ui_lists()
            except Exception:
                pass

    @Slot()
    def plot_portfolio_growth(self):
        if self.portfolio is None and 'PORTFOLIO' not in getattr(self.main_window, "instruments", {}):
            return

        # prefer domain portfolio instrument order if present
        if self.portfolio:
            selected = list(self.portfolio.instruments.keys())
        else:
            selected = self._get_selected_tickers()

        try:
            weights = self._read_weights_for(selected)
        except Exception:
            QMessageBox.critical(self, "Input Error", "Please enter valid numeric weights before plotting.")
            return

        self.log_print("\nPlotting portfolio cumulative growth...")

        try:
            if self.portfolio:
                # domain class produces cumulative series
                cum = self.portfolio.cumulative_value_series(weights)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                cum.plot(ax=ax, title="Portfolio relative growth (Start = 1)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Relative value (start = 1)")
                ax.grid(True)
                self._display_plot_dialog(fig)
            else:
                # fallback to the constructed TimeSeriesAnalysis stored as 'PORTFOLIO'
                ts = self.main_window.instruments.get('PORTFOLIO')
                if ts is None:
                    raise RuntimeError("No portfolio time series found.")
                cum = np.exp(ts.df['Log_Returns'].cumsum())
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(cum.index, cum.values, label='PORTFOLIO')
                ax.set_title("Portfolio Cumulative Growth")
                ax.set_xlabel("Date")
                ax.set_ylabel("Cumulative Return")
                ax.legend()
                self._display_plot_dialog(fig)
        except Exception as e:
            self.log_print(f"Error plotting portfolio: {e}")

    @Slot()
    def show_portfolio_stats(self):
        if not self.portfolio:
            QMessageBox.information(self, "No Portfolio", "No Portfolio object available to show stats.")
            return

        selected = list(self.portfolio.instruments.keys())
        try:
            weights = self._read_weights_for(selected)
        except Exception:
            QMessageBox.critical(self, "Input Error", "Please enter valid numeric weights before viewing stats.")
            return

        self.log_print("\n--- Portfolio Analytics ---")
        try:
            corr = self.portfolio.correlation_matrix()
            self.log_print("Correlation Matrix:\n" + corr.to_string())
            contrib = self.portfolio.variance_contributions(weights)
            self.log_print("\nVariance Contributions:")
            for t, pct in contrib.items():
                self.log_print(f"  {t}: {pct:.2%}")
        except Exception as e:
            self.log_print(f"Error computing portfolio stats: {e}")

    def _display_plot_dialog(self, fig):
        try:
            # Ensure matplotlib is not in blocking interactive mode
            matplotlib.interactive(False)
            dialog = QDialog(self)
            dialog.setWindowTitle("Portfolio Plot")
            dlg_layout = QVBoxLayout(dialog)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, dialog)
            dlg_layout.addWidget(toolbar)
            dlg_layout.addWidget(canvas)
            dialog.resize(900, 600)
            canvas.draw()
            dialog.exec() 
        except Exception as e:
            self.log_print(f"Error displaying plot: {e}")

    def refresh_instruments(self):
        self._build_portfolio_checkbox_frame()