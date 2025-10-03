# gui/tabs/analysis_tab.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame,
    QHBoxLayout, QCheckBox, QMessageBox, QDialog
)
from PySide6.QtCore import Slot, QObject, Signal, QThread
from src.analysis import TimeSeriesAnalysis
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import traceback


class Worker(QObject):
    finished = Signal()
    error = Signal(str)
    result = Signal(object)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        try:
            # NO MATPLOTLIB CALLS HERE.
            output = self.fn(*self.args, **self.kwargs)
            self.result.emit(output)
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()


class AnalysisTab(QWidget):
    def __init__(self, parent=None, load_tab=None):
        super().__init__(parent)
        self.main_window = parent
        self.load_tab = load_tab
        self.analysis_checks = {}
        self.running_threads = [] 

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        root_layout.addWidget(QLabel("Select tickers to analyse or plot:"))

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QFrame()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setContentsMargins(4, 4, 4, 4)
        self.scroll_layout.setSpacing(6)
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll.setWidget(self.scroll_widget)
        root_layout.addWidget(self.scroll, stretch=1)

        # --- First row of simpler analysis buttons ---
        btn_row_simple = QHBoxLayout()
        
        self.plot_growth_btn = QPushButton("Plot Relative Growth")
        self.plot_growth_btn.clicked.connect(self.plot_selected_growth)
        btn_row_simple.addWidget(self.plot_growth_btn)

        self.print_stats_btn = QPushButton("Print Key Stats")
        self.print_stats_btn.clicked.connect(self.print_selected_stats)
        btn_row_simple.addWidget(self.print_stats_btn)

        self.plot_returns_btn = QPushButton("Plot Yearly Returns")
        self.plot_returns_btn.clicked.connect(self.plot_selected_returns)
        btn_row_simple.addWidget(self.plot_returns_btn)
        
        root_layout.addLayout(btn_row_simple)
        
        # --- Second row of strategy/signal buttons ---
        btn_row_strategy = QHBoxLayout()
        
        self.plot_sharpe_btn = QPushButton("Plot Rolling Sharpe Signals")
        self.plot_sharpe_btn.clicked.connect(self.plot_rolling_sharpe)
        btn_row_strategy.addWidget(self.plot_sharpe_btn)

        self.plot_ma_btn = QPushButton("Plot Moving Average Strategy")
        self.plot_ma_btn.clicked.connect(self.plot_moving_average_strategy)
        btn_row_strategy.addWidget(self.plot_ma_btn)
        
        root_layout.addLayout(btn_row_strategy)
        
        self.setLayout(root_layout)

        self._build_analysis_checkbox_frame()

    def _available_tickers(self):
        instruments = getattr(self.main_window, "instruments", {}) or {}
        return sorted(instruments.keys())

    def log_print(self, msg: str):
        if hasattr(self.main_window, "log_message"):
            self.main_window.log_message(msg)
        else:
            print(msg)

    def _clear_analysis_list(self):
        for i in reversed(range(self.scroll_layout.count())):
            item = self.scroll_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if w:
                w.setParent(None)
        self.analysis_checks.clear()

    def _build_analysis_checkbox_frame(self):
        self._clear_analysis_list()
        tickers = self._available_tickers()
        if not tickers:
            lbl = QLabel("No instruments loaded. Load data first.")
            self.scroll_layout.addWidget(lbl)
            return

        for ticker in tickers:
            row = QHBoxLayout()
            cb = QCheckBox(ticker)
            cb.setChecked(False)
            row.addWidget(cb)
            container = QFrame()
            container.setLayout(row)
            self.scroll_layout.addWidget(container)
            self.analysis_checks[ticker] = cb
        self.scroll_layout.addStretch(1)

    def _get_selected_tickers(self):
        return [t for t, cb in self.analysis_checks.items() if cb.isChecked()]

    def _get_instrument_for(self, ticker):
        instruments = getattr(self.main_window, "instruments", {}) or {}
        return instruments.get(ticker)

    def _run_task_in_thread(self, task_function, result_slot, *args):
        """Helper to manage threads, mirroring SimulationTab's logic."""
        thread = QThread()
        worker = Worker(task_function, *args)
        worker.moveToThread(thread)
        worker.result.connect(result_slot)
        worker.error.connect(lambda err: self.log_print(f"Error in worker thread:\n{err}"))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.running_threads.append((thread, worker))
        thread.finished.connect(lambda: self.running_threads.remove((thread, worker)) if (thread, worker) in self.running_threads else None)
        thread.started.connect(worker.run)
        thread.start()

    def _display_plot_dialog(self, fig, title="Plot"):
        """Displays a plot modally using exec(), blocking the calling thread."""
        try:
            # Ensure interactive mode is off for GUI embedding
            matplotlib.interactive(False)
            dialog = QDialog(self)
            dialog.setWindowTitle(title)
            layout = QVBoxLayout(dialog)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, dialog)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            dialog.resize(900, 600)
            canvas.draw()
            dialog.exec()
            matplotlib.pyplot.close(fig)
        except Exception as e:
            self.log_print(f"Error displaying plot: {e}")

    # --- Worker functions (run in separate threads) ---

    def _worker_get_ticker(self, ticker):
        """Worker function to simply pass the ticker back to the main thread."""
        return ticker

    # --- Button Slots (run on main thread) ---

    @Slot()
    def plot_rolling_sharpe(self):
        selected = self._get_selected_tickers()
        if not selected:
            QMessageBox.information(self, "Selection Error", "Please select at least one ticker.")
            return

        for ticker in selected:
            inst = self._get_instrument_for(ticker)
            if not inst:
                self.log_print(f"Skipping {ticker}: instrument not found.")
                continue

            self.log_print(f"\nStarting thread for Rolling Sharpe Ratio plot: {ticker}")
            # Run a worker to immediately pass the ticker back to a main thread slot
            self._run_task_in_thread(self._worker_get_ticker, self._plot_sharpe_from_result, ticker)

    @Slot(object)
    def _plot_sharpe_from_result(self, ticker):
        """Slot to receive result and generate/display plot on main thread."""
        inst = self._get_instrument_for(ticker)
        if not inst:
            self.log_print(f"[{ticker}] Instrument not found for plotting.")
            return

        try:
            # Matplotlib figure generation and display must be done on the main (GUI) thread.
            self.log_print(f"[{ticker}] Generating Rolling Sharpe plot.")
            fig = inst.plot_rolling_sharpe_with_signals()
            self._display_plot_dialog(fig, title=f"Rolling Sharpe Signals: {ticker}")
        except Exception:
            self.log_print(traceback.format_exc())
            QMessageBox.warning(self, "Plotting Error", f"Could not generate plot for {ticker}. Check logs.")


    @Slot()
    def plot_moving_average_strategy(self):
        selected = self._get_selected_tickers()
        if not selected:
            QMessageBox.information(self, "Selection Error", "Please select at least one ticker.")
            return

        for ticker in selected:
            inst = self._get_instrument_for(ticker)
            if not inst:
                self.log_print(f"Skipping {ticker}: instrument not found.")
                continue

            self.log_print(f"\nStarting thread for Moving Average Strategy plot: {ticker}")

            self._run_task_in_thread(self._worker_get_ticker, self._plot_ma_from_result, ticker)

    @Slot(object)
    def _plot_ma_from_result(self, ticker):
        """Slot to receive result and generate/display plot on main thread."""
        inst = self._get_instrument_for(ticker)
        if not inst:
            self.log_print(f"[{ticker}] Instrument not found for plotting.")
            return

        try:
            # Matplotlib figure generation and display must be done on the main (GUI) thread.
            self.log_print(f"[{ticker}] Generating MA Strategy plot.")
            fig = inst.plot_moving_average_strategy()
            self._display_plot_dialog(fig, title=f"MA Strategy: {ticker}")
        except Exception:
            self.log_print(traceback.format_exc())
            QMessageBox.warning(self, "Plotting Error", f"Could not generate plot for {ticker}. Check logs.")

    @Slot()
    def plot_selected_growth(self):
        selected = self._get_selected_tickers()
        if not selected:
            QMessageBox.information(self, "Selection Error", "Please select at least one ticker to plot.")
            return
        inst_to_plot = {t: inst for t in selected if (inst := self._get_instrument_for(t))}
        self.log_print(f"\nPlotting growth for: {', '.join(selected)}")
        fig = TimeSeriesAnalysis.plot_all_growth(inst_to_plot)

        self._display_plot_dialog(fig, title="Relative Growth Comparison")

    @Slot()
    def plot_selected_returns(self):
        selected = self._get_selected_tickers()
        if not selected:
            QMessageBox.information(self, "Selection Error", "Please select at least one ticker to plot.")
            return
        inst_to_plot = {t: inst for t in selected if (inst := self._get_instrument_for(t))}
        self.log_print(f"\nPlotting returns for: {', '.join(selected)}")
        fig = TimeSeriesAnalysis.plot_all_returns(inst_to_plot)

        self._display_plot_dialog(fig, title="Yearly Returns Comparison")

    @Slot()
    def print_selected_stats(self):
        selected = self._get_selected_tickers()
        if not selected:
            QMessageBox.information(self, "Selection Error", "Please select at least one ticker.")
            return
        self.log_print("\n--- Key Statistics ---")
        for t in selected:
            inst = self._get_instrument_for(t)
            if not inst: continue
            try: sharpe = inst.sharpe_ratio()
            except Exception: sharpe = float("nan")
            try: cagr = inst.compute_cagr()
            except Exception: cagr = float("nan")
            try: _, ann_vol = inst.annualised_return()
            except Exception: ann_vol = float("nan")
            try: p5 = inst.growth_probability(5)
            except Exception: p5 = float("nan")
            try: p10 = inst.growth_probability(10)
            except Exception: p10 = float("nan")
            try: p15 = inst.growth_probability(15)
            except Exception: p15 = float("nan")
            self.log_print(
                f"{t}: Sharpe={sharpe:.3f}, "
                f"CAGR={cagr:.2%}, "
                f"Annual Vol={ann_vol:.2%}, "
                f"5Y Growth Prob={p5:.2%}, "
                f"10Y Growth Prob={p10:.2%}, "
                f"15Y Growth Prob={p15:.2%}"
            )

    def refresh_instruments(self):
        self._build_analysis_checkbox_frame()