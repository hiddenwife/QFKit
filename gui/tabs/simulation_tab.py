# gui/tabs/simulation_tab.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame,
    QHBoxLayout, QCheckBox, QMessageBox, QDialog, QSlider
)
from PySide6.QtCore import Slot, QObject, Signal, QThread, Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import traceback

from src.simulation import Simulation


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
            output = self.fn(*self.args, **self.kwargs)
            self.result.emit(output)
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()


class SimulationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.simulation_checks = {}
        self.running_threads = []

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)
        root_layout.addWidget(QLabel("Select tickers to run simulations on:"))

        # Scrollable ticker selection
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QFrame()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll.setWidget(self.scroll_widget)
        root_layout.addWidget(self.scroll, stretch=1)

        # --- parameter controls ---
        controls_row = QVBoxLayout()

        # Volatility window slider
        days_row = QHBoxLayout()
        days_row.addWidget(QLabel("Volatility window (days):"))
        self.days_slider = QSlider(Qt.Horizontal)
        self.days_slider.setMinimum(30)
        self.days_slider.setMaximum(252)
        self.days_slider.setSingleStep(2)
        self.days_slider.setPageStep(2)
        self.days_slider.setTickInterval(10)
        self.days_slider.setTickPosition(QSlider.TicksBelow)
        self.days_slider.setValue(180)
        self.days_value_label = QLabel(str(self.days_slider.value()))
        self.days_slider.valueChanged.connect(lambda v: self._snap_days_slider(v))
        days_row.addWidget(self.days_slider, stretch=1)
        days_row.addWidget(self.days_value_label)
        controls_row.addLayout(days_row)

        # Horizon slider
        years_row = QHBoxLayout()
        years_row.addWidget(QLabel("Horizon (years):"))
        self.years_slider = QSlider(Qt.Horizontal)
        self.years_slider.setMinimum(1)
        self.years_slider.setMaximum(16)
        self.years_slider.setSingleStep(1)
        self.years_slider.setPageStep(1)
        self.years_slider.setTickInterval(1)
        self.years_slider.setTickPosition(QSlider.TicksBelow)
        self.years_slider.setValue(4)  # default = 2.0 years
        self.years_value_label = QLabel("2.0")
        self.years_slider.valueChanged.connect(lambda v: self._on_years_changed(v))
        years_row.addWidget(self.years_slider, stretch=1)
        years_row.addWidget(self.years_value_label)
        controls_row.addLayout(years_row)

        # Checkboxes
        self.cb_timevarying = QCheckBox("Time-varying (mu/sigma)")
        self.cb_jump = QCheckBox("Jump-Diffusion (Merton)")
        controls_row.addWidget(self.cb_timevarying)
        controls_row.addWidget(self.cb_jump)

        root_layout.addLayout(controls_row)

        # Buttons
        btn_row = QHBoxLayout()
        self.run_gbm_btn = QPushButton("Run GBM Simulation")
        self.run_gbm_btn.clicked.connect(self.run_simulation_for_selected)
        btn_row.addWidget(self.run_gbm_btn)
        root_layout.addLayout(btn_row)

        self._build_simulation_controls()

    # -------------------------
    # Utility methods
    # -------------------------
    def log_print(self, msg: str):
        if hasattr(self.main_window, "log_message"):
            self.main_window.log_message(msg)
        else:
            print(msg)

    def _available_tickers(self):
        """Get tickers from instruments (same source as AnalysisTab)."""
        instruments = getattr(self.main_window, "instruments", {}) or {}
        return sorted(instruments.keys())

    def _clear_simulation_list(self):
        for i in reversed(range(self.scroll_layout.count())):
            item = self.scroll_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if w:
                w.setParent(None)
        self.simulation_checks.clear()

    def _build_simulation_controls(self):
        """Build checkboxes from instruments (mirrors AnalysisTab behaviour)."""
        self._clear_simulation_list()
        tickers = self._available_tickers()
        if not tickers:
            self.scroll_layout.addWidget(QLabel("No instruments loaded. Load data first."))
            return

        for ticker in tickers:
            row = QHBoxLayout()
            cb = QCheckBox(ticker)
            cb.setChecked(False)
            row.addWidget(cb)

            container = QFrame()
            container.setLayout(row)
            self.scroll_layout.addWidget(container)
            self.simulation_checks[ticker] = cb

        self.scroll_layout.addStretch(1)

    # -------------------------
    # Worker Thread Handling
    # -------------------------
    def _run_task_in_thread(self, task_function, result_slot, *args):
        thread = QThread()
        worker = Worker(task_function, *args)
        worker.moveToThread(thread)
        worker.result.connect(result_slot)
        worker.error.connect(lambda err: self.log_print(f"Error in worker thread:\n{err}"))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.running_threads.append((thread, worker))
        thread.finished.connect(lambda: self.running_threads.remove((thread, worker)))
        thread.started.connect(worker.run)
        thread.start()

    def _display_plot_dialog(self, fig, title="Simulation Plot", ):
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

    # -------------------------
    # Simulation run (worker fn)
    # -------------------------
    def _run_sim_for_ticker(self, sim_obj, horizon_years, window_days, timevarying, jump):
        # runs in worker thread: perform pure computation and return result dict
        return sim_obj.run_simulation(
            horizon_years=horizon_years,
            n_sims=1000,
            seed=42,
            window_days=window_days if timevarying else None,
            jump=jump
        )

    @Slot(object)
    def _plot_sim_from_result(self, res):
        try:
            if not res:
                return
            ticker = res.get('ticker')
            # get the simulation object we stored before launching worker
            sim_obj = self.main_window.simulations.get(ticker)
            if not sim_obj:
                self.log_print(f"[{ticker}] Simulation result received but simulation object missing.")
                return
            self.log_print(f"[{ticker}] Plotting simulation results.")
            fig = sim_obj.make_sim_figure(res)
            self._display_plot_dialog(fig, title=f"Simulation: {ticker}")
        except Exception:
            self.log_print(traceback.format_exc())

    def run_simulation_for_selected(self):
        selected = self._get_selected_tickers()
        if not selected:
            QMessageBox.information(self, "Selection Error", "Please select at least one ticker.")
            return

        window_days = int(self.days_slider.value())
        horizon_years = float(self.years_slider.value()) / 2.0
        timevarying = self.cb_timevarying.isChecked()
        jump = self.cb_jump.isChecked()

        for t in selected:
            # Create/wrap simulation object ON THE MAIN THREAD (so plotting later can look it up)
            sim_obj = self.main_window.simulations.get(t)
            if sim_obj is None:
                inst = self.main_window.instruments.get(t)
                if inst is None:
                    self.log_print(f"Skipping {t}: instrument object not found in main storage.")
                    continue
                
                try:
                    sim_obj = Simulation(inst.ticker, inst.df.copy())
                except Exception:
                    print("Failed to construct simulation.")

                self.main_window.simulations[t] = sim_obj

            # Now running the heavy computation in worker thread, passing the simulation object
            self.log_print(f"Running simulation for: {t} (horizon={horizon_years}, timevarying={timevarying}, jump={jump})")
            self._run_task_in_thread(self._run_sim_for_ticker, self._plot_sim_from_result,
                                     sim_obj, horizon_years, window_days, timevarying, jump)

    # -------------------------
    # Helpers
    # -------------------------
    def _get_selected_tickers(self):
        return [t for t, cb in self.simulation_checks.items() if cb.isChecked()]

    def refresh_instruments(self):
        self._build_simulation_controls()

    def _snap_days_slider(self, raw_value: int):
        step = 2
        snapped = int(round(raw_value / step) * step)
        snapped = max(30, min(252, snapped))
        if snapped != raw_value:
            self.days_slider.blockSignals(True)
            self.days_slider.setValue(snapped)
            self.days_slider.blockSignals(False)
        self.days_value_label.setText(str(snapped))

    def _on_years_changed(self, slider_val: int):
        years = slider_val / 2.0
        self.years_value_label.setText(f"{years:.1f}")
