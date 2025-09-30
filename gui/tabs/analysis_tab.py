# gui/tabs/analysis_tab.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame,
    QHBoxLayout, QCheckBox, QMessageBox, QDialog
)
from PySide6.QtCore import Slot
from src.analysis import TimeSeriesAnalysis
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class AnalysisTab(QWidget):
    def __init__(self, parent=None, load_tab=None):
        super().__init__(parent)
        self.main_window = parent
        self.load_tab = load_tab
        self.analysis_checks = {}

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

        btn_row = QHBoxLayout()
        self.plot_growth_btn = QPushButton("Plot Relative Growth")
        self.plot_growth_btn.clicked.connect(self.plot_selected_growth)
        btn_row.addWidget(self.plot_growth_btn)

        self.print_stats_btn = QPushButton("Print Key Stats")
        self.print_stats_btn.clicked.connect(self.print_selected_stats)
        btn_row.addWidget(self.print_stats_btn)

        self.plot_returns_btn = QPushButton("Plot Yearly Returns")
        self.plot_returns_btn.clicked.connect(self.plot_selected_returns)
        btn_row.addWidget(self.plot_returns_btn)

        root_layout.addLayout(btn_row)
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
            # The 'PORTFOLIO' ticker will be included here automatically
            # as there is no filter. This is the desired behavior.
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

    # --- FIX: Add a non-blocking dialog to display plots ---
    def _display_plot_dialog(self, fig, title="Plot"):
        try:
            # Ensure matplotlib is not in blocking interactive mode
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
            # Use show() for a non-modal dialog, or exec() for a modal one
            dialog.exec() 
        except Exception as e:
            self.log_print(f"Error displaying plot: {e}")

    @Slot()
    def plot_selected_growth(self):
        selected = self._get_selected_tickers()
        if not selected:
            QMessageBox.information(self, "Selection Error", "Please select at least one ticker to plot.")
            return

        inst_to_plot = {}
        for t in selected:
            inst = self._get_instrument_for(t)
            if inst:
                inst_to_plot[t] = inst

        self.log_print(f"\nPlotting growth for: {', '.join(selected)}")
        # --- FIX: Get the figure and display it in a dialog ---
        fig = TimeSeriesAnalysis.plot_all_growth(inst_to_plot)
        self._display_plot_dialog(fig, title="Relative Growth Comparison")


    @Slot()
    def plot_selected_returns(self):
        selected = self._get_selected_tickers()
        if not selected:
            QMessageBox.information(self, "Selection Error", "Please select at least one ticker to plot.")
            return

        inst_to_plot = {}
        for t in selected:
            inst = self._get_instrument_for(t)
            if inst:
                inst_to_plot[t] = inst

        self.log_print(f"\nPlotting returns for: {', '.join(selected)}")
        # --- FIX: Get the figure and display it in a dialog ---
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
            if not inst:
                continue
            
            # Using try-except for each metric to prevent one failure from crashing the loop
            try: sharpe = inst.sharpe_ratio()
            except Exception: sharpe = float("nan")
            try: cagr = inst.compute_cagr()
            except Exception: cagr = float("nan")
            try:
                ann_mu, ann_vol = inst.annualised_return()
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