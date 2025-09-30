from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QMessageBox, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Slot, QTimer, QRunnable, QThreadPool, Signal
import pandas as pd
from src.data_loader import get_stock_data
from src.analysis import TimeSeriesAnalysis
from src.simulation import Simulation
from datetime import date, timedelta


class Worker(QRunnable):
    """Generic worker for running functions in background threads."""
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.fn(*self.args, **self.kwargs)


class LoadDataTab(QWidget):
    log_signal = Signal(str)           # Thread-safe logging
    refresh_signal = Signal()          # Refresh UI safely
    enable_ui_signal = Signal(bool)    # Enable/disable buttons safely
    error_signal = Signal(str, str)    # Show QMessageBox safely

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.instruments = {}
        self.simulations = {}
        self.portfolio = None

        main = QVBoxLayout()
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(6)

        # --- Ticker Row ---
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

        # --- Date Checkboxes + Inputs ---
        row_start = QHBoxLayout()
        row_end = QHBoxLayout()

        self.start_date_chk = QCheckBox("Set a Start date (yyyy-mm-dd)?")
        self.start_date_entry = QLineEdit()
        self.start_date_entry.setFixedWidth(140)
        self.start_date_entry.setText("2022-01-01")
        self.start_date_entry.setEnabled(False)
        self.start_date_entry.setStyleSheet("color: gray;")
        self.start_date_chk.stateChanged.connect(self._on_start_toggled)

        row_start.addWidget(self.start_date_chk)
        row_start.addWidget(self.start_date_entry)

        self.end_date_chk = QCheckBox("Set an End date (yyyy-mm-dd)?")
        self.end_date_entry = QLineEdit()
        self.end_date_entry.setFixedWidth(140)
        formatted_three_days_ago = (date.today() - timedelta(days=3)).isoformat()
        self.end_date_entry.setText(formatted_three_days_ago)
        self.end_date_entry.setEnabled(False)
        self.end_date_entry.setStyleSheet("color: gray;")
        self.end_date_chk.stateChanged.connect(self._on_end_toggled)

        row_end.addWidget(self.end_date_chk)
        row_end.addWidget(self.end_date_entry)

        main.addLayout(row_start)
        main.addLayout(row_end)

        # --- Buttons ---
        row_buttons = QHBoxLayout()
        self.load_button = QPushButton("Load & analyse")
        self.clear_button = QPushButton("Clear All Loaded Data")
        row_buttons.addWidget(self.load_button)
        row_buttons.addWidget(self.clear_button)
        row_buttons.addStretch(1)
        main.addLayout(row_buttons)
        main.addStretch(1)

        self.setLayout(main)

        # --- Signal Connections ---
        self.log_signal.connect(self._append_log)
        self.refresh_signal.connect(self._do_refresh)
        self.enable_ui_signal.connect(self._set_ui_enabled)
        self.error_signal.connect(self._show_error)

        self.load_button.clicked.connect(self._on_load_clicked)
        self.clear_button.clicked.connect(self.clear_loaded_data)

    # ---------------- Slots ----------------
    @Slot()
    def _on_start_toggled(self):
        enabled = self.start_date_chk.isChecked()
        self.start_date_entry.setEnabled(enabled)
        self.start_date_entry.setStyleSheet("" if enabled else "color: gray;")

    @Slot()
    def _on_end_toggled(self):
        enabled = self.end_date_chk.isChecked()
        self.end_date_entry.setEnabled(enabled)
        self.end_date_entry.setStyleSheet("" if enabled else "color: gray;")

    @Slot(str)
    def _append_log(self, text: str):
        if self.parent and hasattr(self.parent, "log_message"):
            self.parent.log_message(text)

    @Slot()
    def _do_refresh(self):
        if self.parent and hasattr(self.parent, "refresh_all_ui_lists"):
            self.parent.refresh_all_ui_lists()

    @Slot(bool)
    def _set_ui_enabled(self, enabled: bool):
        self.load_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.ticker_entry.setEnabled(enabled)

    @Slot(str, str)
    def _show_error(self, title: str, msg: str):
        QMessageBox.warning(self, title, msg)

    # ---------------- Actions ----------------
    def log_message(self, text: str):
        self.log_signal.emit(text)

    @Slot()
    def _on_load_clicked(self):
        start = self.start_date_entry.text().strip() if self.start_date_chk.isChecked() else None
        end = self.end_date_entry.text().strip() if self.end_date_chk.isChecked() else None
        self.load_data(start, end)

    def load_data(self, start=None, end=None):
        tickers = [t.strip().upper() for t in self.ticker_entry.text().split() if t.strip()]
        if not tickers:
            self.error_signal.emit("Input Error", "Please enter at least one ticker.")
            return

        if not start:
            start = "2022-01-01"
        if not end:
            end = "2025-09-23"

        self.log_message(f"\n--- Loading data for: {', '.join(tickers)} ---")
        self.enable_ui_signal.emit(False)  # disable UI while loading

        main_instruments = getattr(self.parent, "instruments", self.instruments)
        main_simulations = getattr(self.parent, "simulations", self.simulations)

        worker = Worker(self._load_data_task, tickers, start, end, main_instruments, main_simulations)
        QThreadPool.globalInstance().start(worker)

    def _load_data_task(self, tickers, start, end, main_instruments, main_simulations):
        for ticker in tickers:
            try:
                self.log_signal.emit(f"Fetching {ticker}...")
                df = self._get_stock_data_fallback(ticker, start=start, end=end)
                if df is None or df.empty:
                    self.log_signal.emit(f"No data returned for {ticker}. Skipping.")
                    continue

                inst_obj = TimeSeriesAnalysis(ticker, df)
                sim_obj = Simulation(ticker, df)

                self.instruments[ticker] = inst_obj
                self.simulations[ticker] = sim_obj
                main_instruments[ticker] = inst_obj
                main_simulations[ticker] = sim_obj

                sharpe = inst_obj.sharpe_ratio() if hasattr(inst_obj, "sharpe_ratio") else 0.0
                cagr = inst_obj.compute_cagr() if hasattr(inst_obj, "compute_cagr") else None
                if cagr is not None:
                    self.log_signal.emit(f"Loaded {ticker} | Sharpe: {sharpe:.2f} | CAGR: {cagr:.2%}")
                else:
                    self.log_signal.emit(f"Loaded {ticker} | Sharpe: {sharpe:.2f}")
            except Exception as e:
                self.log_signal.emit(f"Failed to load {ticker}: {e}")

        self.refresh_signal.emit()
        self.enable_ui_signal.emit(True)  # re-enable UI

    def clear_loaded_data(self):
        self.instruments.clear()
        self.simulations.clear()
        self.portfolio = None

        if self.parent:
            if hasattr(self.parent, "instruments"):
                self.parent.instruments.clear()
            if hasattr(self.parent, "simulations"):
                self.parent.simulations.clear()

        self.log_message("\n--- Cleared all loaded data. ---")
        self.refresh_signal.emit()

    def _get_stock_data_fallback(self, ticker, start="2022-01-01", end="2025-09-23"):
        try:
            return get_stock_data(ticker, start=start, end=end)
        except Exception as e:
            print(f"Unable to load data {ticker}. Problem: {e}")
            return None
