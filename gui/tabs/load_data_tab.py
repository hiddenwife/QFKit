# gui/tabs/load_data_tab.py
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QMessageBox, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Slot, QTimer
import threading
import pandas as pd
from src.data_loader import get_stock_data
from src.analysis import TimeSeriesAnalysis
from src.simulation import Simulation


def run_in_thread(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        thread.start()
    return wrapper


class LoadDataTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.instruments = {}
        self.simulations = {}
        self.portfolio = None

        main = QVBoxLayout()
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(6)

        # Ticker row (label + entry)
        row_ticker = QHBoxLayout()
        row_ticker.setSpacing(8)
        lbl = QLabel("Enter Tickers (space-separated, e.g., AAPL GOOG ^GSPC):")
        lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        row_ticker.addWidget(lbl)

        self.ticker_entry = QLineEdit()
        self.ticker_entry.setText("AAPL MSFT NVDA")
        self.ticker_entry.setPlaceholderText("e.g. AAPL MSFT NVDA")
        self.ticker_entry.setMinimumWidth(320)
        row_ticker.addWidget(self.ticker_entry, stretch=1)
        main.addLayout(row_ticker)

        # Checkbox + inline date input (date input disabled/hidden until checked)
        row_start = QHBoxLayout()
        row_start.setSpacing(8)
        row_end = QHBoxLayout()
        row_end.setSpacing(8)

        self.start_date_chk = QCheckBox("Set a Start date (yyyy-mm-dd)?")
        self.start_date_chk.setChecked(False)
        self.start_date_chk.stateChanged.connect(self._on_start_toggled)
        row_start.addWidget(self.start_date_chk, stretch=0)

        # Inline date entry placed immediately to the right of the checkbox.
        self.start_date_entry = QLineEdit()
        self.start_date_entry.setFixedWidth(140)
        self.start_date_entry.setText("2022-01-01")
        self.start_date_entry.setEnabled(False)
        # visually muted when disabled
        self.start_date_entry.setStyleSheet("color: gray;")
        row_start.addWidget(self.start_date_entry, stretch=0)

        self.end_date_chk = QCheckBox("Set an End date (yyyy-mm-dd)?")
        self.end_date_chk.setChecked(False)
        self.end_date_chk.stateChanged.connect(self._on_end_toggled)
        row_end.addWidget(self.end_date_chk, stretch=0)

        self.end_date_entry = QLineEdit()
        self.end_date_entry.setFixedWidth(140)
        self.end_date_entry.setText("2025-09-23")
        self.end_date_entry.setEnabled(False)

        self.end_date_entry.setStyleSheet("color: gray;")
        row_end.addWidget(self.end_date_entry, stretch=0)

        row_start.addItem(QSpacerItem(20, 0))
        main.addLayout(row_start)

        row_end.addItem(QSpacerItem(20, 0))
        main.addLayout(row_end)

        # Buttons row
        row_buttons = QHBoxLayout()
        row_buttons.setSpacing(8)
        self.load_button = QPushButton("Load & analyse")
        self.clear_button = QPushButton("Clear All Loaded Data")
        row_buttons.addWidget(self.load_button)
        row_buttons.addWidget(self.clear_button)
        row_buttons.addStretch(1)
        main.addLayout(row_buttons)

        main.addStretch(1)
        self.setLayout(main)

        # Connections
        self.load_button.clicked.connect(self._on_load_clicked)
        self.clear_button.clicked.connect(self.clear_loaded_data)

    @Slot()
    def _on_start_toggled(self):
        enabled = self.start_date_chk.isChecked()
        self.start_date_entry.setEnabled(enabled)
        # update visual style
        self.start_date_entry.setStyleSheet("" if enabled else "color: gray;")

    @Slot()
    def _on_end_toggled(self):
        enabled = self.end_date_chk.isChecked()
        self.end_date_entry.setEnabled(enabled)
        # update visual style
        self.end_date_entry.setStyleSheet("" if enabled else "color: gray;")


    def log_message(self, text: str):
        if self.parent and hasattr(self.parent, "log_message"):
            self.parent.log_message(text)

    @Slot()
    def _on_load_clicked(self):
        start = self.start_date_entry.text().strip() if self.start_date_chk.isChecked() else None
        end = self.end_date_entry.text().strip() if self.end_date_chk.isChecked() else None
        self.load_data(start, end)

    @run_in_thread
    def load_data(self, start=None, end=None):
        tickers = [t.strip().upper() for t in self.ticker_entry.text().split() if t.strip()]
        if not tickers:
            def show():
                QMessageBox.warning(self, "Input Error", "Please enter at least one ticker.")
            QTimer.singleShot(0, show)
            return

        if not start:
            start = "2022-01-01"

        if not end:
            end = "2025-09-23"

        self.log_message(f"\n--- Loading data for: {', '.join(tickers)} ---")
        # Write into the main window's shared dicts
        main_instruments = getattr(self.parent, "instruments", None)
        main_simulations = getattr(self.parent, "simulations", None)
        # If main window doesn't have those, keep local ones
        if main_instruments is None:
            main_instruments = self.instruments
        if main_simulations is None:
            main_simulations = self.simulations

        for ticker in tickers:
            try:
                self.log_message(f"Fetching {ticker}...")
                df = self._get_stock_data_fallback(ticker, start=start, end=end)
                if df is None or df.empty:
                    self.log_message(f"No data returned for {ticker}. Skipping.")
                    continue

                inst_obj = TimeSeriesAnalysis(ticker, df)
                sim_obj = Simulation(ticker, df)

                # store on both local tab and main window shared dicts
                self.instruments[ticker] = inst_obj
                self.simulations[ticker] = sim_obj
                try:
                    main_instruments[ticker] = inst_obj
                    main_simulations[ticker] = sim_obj
                except Exception:
                    pass

                sharpe = inst_obj.sharpe_ratio() if hasattr(inst_obj, "sharpe_ratio") else 0.0
                cagr = inst_obj.compute_cagr() if hasattr(inst_obj, "compute_cagr") else None
                if cagr is not None:
                    self.log_message(f"Loaded {ticker} | Sharpe: {sharpe:.2f} | CAGR: {cagr:.2%}")
                else:
                    self.log_message(f"Loaded {ticker} | Sharpe: {sharpe:.2f}")
            except Exception as e:
                self.log_message(f"Failed to load {ticker}: {e}")

        # Notify main window to refresh UI lists on the main thread
        if self.parent and hasattr(self.parent, "refresh_all_ui_lists"):
            QTimer.singleShot(0, self.parent.refresh_all_ui_lists)

    def clear_loaded_data(self):
        self.instruments.clear()
        self.simulations.clear()
        self.portfolio = None
        # Also clear main window shared dicts if present
        if self.parent:
            if hasattr(self.parent, "instruments"):
                try:
                    self.parent.instruments.clear()
                except Exception:
                    pass
            if hasattr(self.parent, "simulations"):
                try:
                    self.parent.simulations.clear()
                except Exception:
                    pass
        self.log_message("\n--- Cleared all loaded data. ---")
        if self.parent and hasattr(self.parent, "refresh_all_ui_lists"):
            self.parent.refresh_all_ui_lists()

    def _get_stock_data_fallback(self, ticker, start="2022-01-01", end="2025-09-23"):
        try:
            return get_stock_data(ticker, start=start, end=end)
        except Exception:
            import numpy as np
            dates = pd.date_range(start, periods=200, freq="B")
            prices = 100 + np.cumsum(np.random.randn(200))
            df = pd.DataFrame({"Close": prices}, index=dates)
            df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
            return df