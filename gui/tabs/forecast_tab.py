# gui/tabs/forecast_tab.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame,
    QHBoxLayout, QCheckBox, QMessageBox, QDialog, QSlider,
    QPushButton, QComboBox, QSpinBox, QSplitter, QLineEdit
)
from PySide6.QtCore import Slot, Qt, QObject, QThread, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import traceback
import functools
import multiprocessing
import pandas as pd  # Import pandas to handle data slicing

from src.forecast import EpicBayesForecaster  # your Bayesian forecaster class
import plotly.graph_objects as go


# Worker used to run fitting in a background thread
class Worker(QObject):
    finished = Signal()
    error = Signal(str)
    # result now includes a flag for historical forecast
    result = Signal(str, object, bool)  # ticker + forecaster object + is_historical

    def __init__(self, fn, ticker, is_historical):
        super().__init__()
        self.fn = fn
        self.ticker = ticker
        self.is_historical = is_historical

    @Slot()
    def run(self):
        try:
            out = self.fn()
            # Pass the is_historical flag back with the result
            self.result.emit(self.ticker, out, self.is_historical)
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()


class ForecastTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.forecast_checks = {}
        self.status_labels = {}
        self.running_threads = []
        self._setup_ui()
        self._build_forecast_controls()


    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(QLabel("Select tickers for Bayesian Forecast:"))

        splitter = QSplitter(Qt.Vertical)

        # --- Scroll area for ticker selection ---
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QFrame()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll.setWidget(self.scroll_widget)
        splitter.addWidget(self.scroll)

        # --- Parameter controls container ---
        params_container = QFrame()
        params_layout = QVBoxLayout(params_container)

        # --- Forecast horizon ---
        horizon_row = QHBoxLayout()
        horizon_row.addWidget(QLabel("Forecast horizon (days):"))
        self.horizon_slider = QSlider(Qt.Horizontal)
        self.horizon_slider.setMinimum(5)
        self.horizon_slider.setMaximum(252)
        self.horizon_slider.setSingleStep(2)
        self.horizon_slider.setPageStep(2)
        self.horizon_slider.setTickInterval(10)
        self.horizon_slider.setValue(60)
        self.horizon_slider.setTickPosition(QSlider.TicksBelow)
        self.horizon_label = QLabel(str(self.horizon_slider.value()))
        self.horizon_slider.valueChanged.connect(lambda v: self._snap_horizon_slider(v))
        horizon_row.addWidget(self.horizon_slider, stretch=1)
        horizon_row.addWidget(self.horizon_label)
        params_layout.addLayout(horizon_row)

        # --- AR order ---
        ar_row = QHBoxLayout()
        ar_row.addWidget(QLabel("AR Order (p):"))
        self.ar_spin = QSpinBox()
        self.ar_spin.setMinimum(0)
        self.ar_spin.setMaximum(6)
        self.ar_spin.setValue(1)
        ar_row.addWidget(self.ar_spin)
        params_layout.addLayout(ar_row)

        # --- Posterior Draws slider ---
        self.draws_parent = QHBoxLayout()
        self.draws_parent.addWidget(QLabel("Posterior Draws:"))

        self.draws_slider = QSlider(Qt.Horizontal)
        self.draws_slider.setMinimum(100)
        self.draws_slider.setMaximum(10000)
        self.draws_slider.setSingleStep(100)
        self.draws_slider.setPageStep(100)
        self.draws_slider.setTickInterval(100)
        self.draws_slider.setValue(1000)
        self.draws_label = QLabel(str(self.draws_slider.value()))
        self.draws_slider.valueChanged.connect(lambda v: self._snap_draws_slider(v))
        self.draws_parent.addWidget(self.draws_slider, stretch=1)
        self.draws_parent.addWidget(self.draws_label)
        params_layout.addLayout(self.draws_parent)

        # --- Custom Draws input (hidden by default) ---
        self.draws_custom_parent = QHBoxLayout()
        self.draws_custom_parent.addWidget(QLabel("Posterior Draws (custom):"))
        self.draws_input = QLineEdit()
        self.draws_input.setPlaceholderText("Enter integer draws")
        self.draws_input.setValidator(QIntValidator(100, 10_000_000, self))  # min=100
        self.draws_custom_parent.addWidget(self.draws_input, stretch=1)
        params_layout.addLayout(self.draws_custom_parent)
        for i in range(self.draws_custom_parent.count()):
            self.draws_custom_parent.itemAt(i).widget().setVisible(False)

        # --- Inference method ---
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Inference Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Fast (ADVI)", "Full (NUTS)"])
        method_row.addWidget(self.method_combo)
        params_layout.addLayout(method_row)

        try:
            self.cpu_count = multiprocessing.cpu_count()
        except NotImplementedError:
            self.cpu_count = 1

        # --- Cores slider ---
        self.cores_parent = QHBoxLayout()
        self.cores_label_prefix = QLabel(f"Number of cores to use (max {self.cpu_count - 1}):")
        self.cores_slider = QSlider(Qt.Horizontal)
        self.cores_slider.setMinimum(1)
        self.cores_slider.setMaximum(max(1, self.cpu_count - 1))
        self.cores_slider.setSingleStep(1)
        self.cores_slider.setValue(1)
        self.cores_label = QLabel(str(self.cores_slider.value()))
        self.cores_parent.addWidget(self.cores_label_prefix)
        self.cores_parent.addWidget(self.cores_slider, stretch=1)
        self.cores_parent.addWidget(self.cores_label)

        self.cores_container = QFrame()
        self.cores_container.setLayout(self.cores_parent)
        self.cores_container.setVisible(False)
        params_layout.addWidget(self.cores_container)

        # --- Chains slider ---
        self.chains_parent = QHBoxLayout()
        self.chains_parent.addWidget(QLabel("Chains for NUTS:"))
        self.chains_slider = QSlider(Qt.Horizontal)
        self.chains_slider.setMinimum(4)
        self.chains_slider.setMaximum(10)
        self.chains_slider.setSingleStep(1)
        self.chains_slider.setValue(4)
        self.chains_label = QLabel(str(self.chains_slider.value()))
        self.chains_parent.addWidget(self.chains_slider, stretch=1)
        self.chains_parent.addWidget(self.chains_label)
        params_layout.addLayout(self.chains_parent)

        # --- Custom chains input (hidden by default) ---
        self.chains_custom_parent = QHBoxLayout()
        self.chains_custom_parent.addWidget(QLabel("Chains (custom):"))
        self.chains_input = QLineEdit()
        self.chains_input.setPlaceholderText("Enter integer chains")
        self.chains_input.setValidator(QIntValidator(4, 10_000_000, self))  # min=4
        self.chains_custom_parent.addWidget(self.chains_input, stretch=1)
        params_layout.addLayout(self.chains_custom_parent)
        for i in range(self.chains_custom_parent.count()):
            self.chains_custom_parent.itemAt(i).widget().setVisible(False)

        self.more_draws_checkbox = QCheckBox("I want more draws + chains")
        self.more_draws_checkbox.setVisible(False)
        params_layout.addWidget(self.more_draws_checkbox)

        # --- Slider Callbacks ---
        def update_cores_max(chains_value):
            """Set cores slider max = min(chains_value, cpu_count-1)."""
            max_cores = min(chains_value, self.cpu_count - 1)
            self.cores_slider.setMaximum(max_cores)
            self.cores_label_prefix.setText(f"Number of processors to use (max {max_cores}):")
            if self.cores_slider.value() > max_cores:
                self.cores_slider.setValue(max_cores)

        def on_chains_changed(value):
            self.chains_label.setText(str(value))
            update_cores_max(value)

        def on_cores_changed(value):
            self.cores_label.setText(str(value))

        self.chains_slider.valueChanged.connect(on_chains_changed)
        self.cores_slider.valueChanged.connect(on_cores_changed)

        # --- When custom chain input changes, update cores max too ---
        def on_custom_chains_changed():
            text = self.chains_input.text()
            if text.isdigit():
                update_cores_max(int(text))

        self.chains_input.textChanged.connect(lambda _: on_custom_chains_changed())

        on_chains_changed(self.chains_slider.value())

        # --- Toggle helpers for default/custom draws & chains ---
        def _show_default_draws_widgets():
            self.draws_slider.setVisible(True)
            self.draws_label.setVisible(True)
            for i in range(self.draws_custom_parent.count()):
                self.draws_custom_parent.itemAt(i).widget().setVisible(False)
            self.draws_parent.itemAt(0).widget().setVisible(True)

        def _show_custom_draws_widgets():
            self.draws_slider.setVisible(False)
            self.draws_label.setVisible(False)
            for i in range(self.draws_custom_parent.count()):
                self.draws_custom_parent.itemAt(i).widget().setVisible(True)
            self.draws_parent.itemAt(0).widget().setVisible(False)

        def _show_default_chains_widgets():
            self.chains_slider.setVisible(True)
            self.chains_label.setVisible(True)
            for i in range(self.chains_custom_parent.count()):
                self.chains_custom_parent.itemAt(i).widget().setVisible(False)
            self.chains_parent.itemAt(0).widget().setVisible(True)

        def _show_custom_chains_widgets():
            self.chains_slider.setVisible(False)
            self.chains_label.setVisible(False)
            for i in range(self.chains_custom_parent.count()):
                self.chains_custom_parent.itemAt(i).widget().setVisible(True)
            self.chains_parent.itemAt(0).widget().setVisible(False)

        def chains_row_enable(enable: bool):
            for i in range(self.chains_parent.count()):
                self.chains_parent.itemAt(i).widget().setVisible(enable)
            for i in range(self.chains_custom_parent.count()):
                self.chains_custom_parent.itemAt(i).widget().setVisible(enable and self.more_draws_checkbox.isChecked())

        self._show_default_draws_widgets = _show_default_draws_widgets
        self._show_custom_draws_widgets = _show_custom_draws_widgets
        self._show_default_chains_widgets = _show_default_chains_widgets
        self._show_custom_chains_widgets = _show_custom_chains_widgets
        self.chains_row_enable = chains_row_enable

        def on_method_changed(txt):
            is_nuts = "NUTS" in txt.upper()
            self.more_draws_checkbox.setVisible(is_nuts)
            self.cores_container.setVisible(is_nuts)
            if is_nuts:
                self.chains_row_enable(True)
                _show_default_chains_widgets()
            else:
                self.more_draws_checkbox.setChecked(False)
                self.chains_row_enable(False)
                for i in range(self.chains_parent.count()):
                    self.chains_parent.itemAt(i).widget().setVisible(False)
                for i in range(self.chains_custom_parent.count()):
                    self.chains_custom_parent.itemAt(i).widget().setVisible(False)

        self.method_combo.currentTextChanged.connect(on_method_changed)

        def on_more_draws_toggled(checked):
            if checked:
                _show_custom_draws_widgets()
                _show_custom_chains_widgets()
            else:
                _show_default_draws_widgets()
                _show_default_chains_widgets()

        self.more_draws_checkbox.toggled.connect(on_more_draws_toggled)

        _show_default_draws_widgets()
        _show_default_chains_widgets()

        # --- Plot type ---
        plot_row = QHBoxLayout()
        plot_row.addWidget(QLabel("Plot Type:"))
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Interactive (Plotly)", "Static (Matplotlib)"])
        plot_row.addWidget(self.plot_type)
        params_layout.addLayout(plot_row)

        # --- Run buttons ---
        btn_row = QHBoxLayout()
        self.run_forecast_btn = QPushButton("Run Forecast")
        self.run_forecast_btn.clicked.connect(self.run_forecast_for_selected)
        btn_row.addWidget(self.run_forecast_btn)

        # NEW: Button for historical forecast comparison
        self.run_historical_btn = QPushButton("Compare Forecast to History")
        self.run_historical_btn.clicked.connect(self.run_historical_forecast)
        btn_row.addWidget(self.run_historical_btn)

        self.global_status = QLabel("")
        btn_row.addWidget(self.global_status, stretch=1)
        params_layout.addLayout(btn_row)

        params_container.setLayout(params_layout)
        splitter.addWidget(params_container)

        on_method_changed(self.method_combo.currentText())
        root.addWidget(splitter, stretch=1)

    # ---------------- Utility Helpers ----------------
    def log_print(self, msg: str):
        if hasattr(self.main_window, "log_message"):
            self.main_window.log_message(msg)
        else:
            print(msg)

    def _available_tickers(self):
        instruments = getattr(self.main_window, "instruments", {}) or {}
        return sorted(instruments.keys())

    def _clear_forecast_list(self):
        for i in reversed(range(self.scroll_layout.count())):
            item = self.scroll_layout.itemAt(i)
            w = item.widget()
            if w:
                w.setParent(None)
        self.forecast_checks.clear()
        self.status_labels.clear()

    def _build_forecast_controls(self):
        self._clear_forecast_list()
        tickers = self._available_tickers()
        if not tickers:
            self.scroll_layout.addWidget(QLabel("No instruments loaded. Load data first."))
            return

        for ticker in tickers:
            row = QHBoxLayout()
            cb = QCheckBox(ticker)
            row.addWidget(cb)
            status = QLabel("")
            status.setFixedWidth(180)
            row.addWidget(status)

            container = QFrame()
            container.setLayout(row)
            self.scroll_layout.addWidget(container)
            self.forecast_checks[ticker] = cb
            self.status_labels[ticker] = status

        self.scroll_layout.addStretch(1)

    def refresh_instruments(self):
        self._build_forecast_controls()

    def _get_selected_tickers(self):
        return [t for t, cb in self.forecast_checks.items() if cb.isChecked()]

    def _snap_draws_slider(self, raw_value: int):
        step = 100
        snapped = int(round(raw_value / step) * step)
        snapped = max(100, min(10000, snapped))
        if snapped != raw_value:
            self.draws_slider.blockSignals(True)
            self.draws_slider.setValue(snapped)
            self.draws_slider.blockSignals(False)
        self.draws_label.setText(str(snapped))

    def _snap_horizon_slider(self, raw_value: int):
        step = 2
        snapped = int(round(raw_value / step) * step)
        snapped = max(5, min(252, snapped))
        if snapped != raw_value:
            self.horizon_slider.blockSignals(True)
            self.horizon_slider.setValue(snapped)
            self.horizon_slider.blockSignals(False)
        self.horizon_label.setText(str(snapped))

    # ---------------- Thread helper ----------------
    def _run_task_in_thread(self, fn, ticker, result_slot, is_historical):
        thread = QThread()
        # Pass is_historical to the Worker
        worker = Worker(fn, ticker, is_historical)
        worker.moveToThread(thread)

        worker.result.connect(result_slot)
        worker.error.connect(lambda err: self._on_worker_error(ticker, err))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self.running_threads.append((thread, worker))
        thread.finished.connect(lambda: self.running_threads.remove((thread, worker)))

        thread.started.connect(worker.run)
        thread.start()

    def _on_worker_error(self, ticker, tb):
        self.log_print(f"[{ticker}] Worker ERROR:\n{tb}")
        lbl = self.status_labels.get(ticker)
        if lbl:
            lbl.setText("Error")
        self._maybe_enable_run_button()

    # ---------------- Main entry ----------------
    def _get_forecast_params(self):
        """Helper to get all forecast parameters from the UI."""
        steps = int(self.horizon_slider.value())
        p = int(self.ar_spin.value())
        plot_type = self.plot_type.currentText()
        method = "advi" if "ADVI" in self.method_combo.currentText() else "nuts"

        # If user wants custom draws/chains, read from QLineEdit instead of sliders
        if self.more_draws_checkbox.isChecked():
            try:
                draws = max(100, int(self.draws_input.text()))
            except ValueError:
                draws = 100

            try:
                chains = max(4, int(self.chains_input.text()))
            except ValueError:
                chains = 4
        else:
            draws = int(self.draws_slider.value())
            chains = int(self.chains_slider.value())

        cores = int(self.cores_slider.value())
        return steps, p, plot_type, method, draws, chains, cores

    def _setup_run(self, selected_tickers, is_historical: bool = False):
        """Disables buttons and sets global status."""
        if not selected_tickers:
            QMessageBox.information(self, "Selection Error", "Please select at least one ticker.")
            return False

        self.run_forecast_btn.setEnabled(False)
        self.run_historical_btn.setEnabled(False)
        status = "Running forecasts..."
        if is_historical:
            status = "Running historical forecasts..."
        
        steps, p, plot_type, method, draws, chains, cores = self._get_forecast_params()
        
        statement = (f"\nParameters: \n AR Order: {p} \n Plot Type: {plot_type} \n "
                     f"Method: {method} \n Posterior Draws: {draws} \n")

        if "nuts" in method:
            status += " (this may take a while)"
            statement += f" Chains: {chains} \n Processors: {cores}\n"
        
        self.global_status.setText(status)
        return steps, p, plot_type, method, draws, chains, cores, statement

    def run_forecast_for_selected(self):
        """Runs a forecast for the selected tickers starting from 'now'."""
        selected = self._get_selected_tickers()
        params = self._setup_run(selected, is_historical=False)
        if not params: return

        steps, p, _, method, draws, chains, cores, statement = params
        
        for ticker in selected:
            inst = self.main_window.instruments.get(ticker)
            if inst is None:
                self.log_print(f"Skipping {ticker}: no data.")
                continue

            lbl = self.status_labels.get(ticker)
            if lbl:
                lbl.setText("Fitting...")

            # Forecast: Use all available data
            data_to_fit = inst.df

            def task_fit(df=data_to_fit):
                fc = EpicBayesForecaster(df)
                self.log_print(statement)
                self.log_print("Don't worry about pytensor errors!")
                fc.fit(p=p, draws=draws, method=method, tune=max(100, draws // 2),
                       chains=chains, cores=cores, random_seed=42)
                return fc

            bound_fn = functools.partial(task_fit)
            # is_historical = False
            self._run_task_in_thread(bound_fn, ticker, self._on_worker_result, False)

    def run_historical_forecast(self):
        """Runs a historical forecast, predicting the last 'steps' days."""
        selected = self._get_selected_tickers()
        params = self._setup_run(selected, is_historical=True)
        if not params: return

        steps, p, _, method, draws, chains, cores, statement = params

        for ticker in selected:
            inst = self.main_window.instruments.get(ticker)
            if inst is None or inst.df is None or len(inst.df) <= steps:
                msg = f"Skipping {ticker}: not enough data for historical forecast of {steps} days."
                self.log_print(msg)
                QMessageBox.warning(self, "Data Error", msg)
                continue

            lbl = self.status_labels.get(ticker)
            if lbl:
                lbl.setText("Fitting (Historical)...")

            data_to_fit = inst.df.iloc[:-steps]

            full_data = inst.df

            def task_fit(df=data_to_fit, full_df=full_data):
                fc = EpicBayesForecaster(df)
                self.log_print(statement)
                self.log_print("Don't worry about pytensor errors!")
                fc.fit(p=p, draws=draws, method=method, tune=max(100, draws // 2),
                       chains=chains, cores=cores, random_seed=42)
                fc._full_df = full_df 
                return fc

            bound_fn = functools.partial(task_fit)
            # is_historical = True
            self._run_task_in_thread(bound_fn, ticker, self._on_worker_result, True)


    @Slot(str, object, bool)
    def _on_worker_result(self, ticker, forecaster, is_historical):

        try:
            lbl = self.status_labels.get(ticker)
            if lbl:
                lbl.setText("Plotting...")
                self.global_status.setText("Opening plot... Done!")

            steps = int(self.horizon_slider.value())

            if is_historical:
                
                full_df = getattr(forecaster, "_full_df", None)

                if full_df is None:
                    # fallback to normal plotting if full data wasn't attached
                    self.log_print(f"[{ticker}] Warning: no full_df attached to forecaster; plotting forecast only.")
                    if self.plot_type.currentText().startswith("Interactive"):
                        fig = forecaster.plot_forecast_interactive(steps=steps)
                        self._show_plotly_in_dialog(fig, title=f"Historical Forecast: {ticker}")
                    else:
                        fig = forecaster.plot_forecast_matplotlib(steps=steps)
                        self._show_matplotlib_in_dialog(fig, title=f"Historical Forecast: {ticker}")
                else:
                    price_col = None
                    for c in ("Close"):
                        if c in full_df.columns:
                            price_col = c
                            break
                    if price_col is None:
                        for c in full_df.columns:
                            if pd.api.types.is_numeric_dtype(full_df[c]):
                                price_col = c
                                break
                    if price_col is None:
                        price_col = full_df.columns[0]

                    data_used_for_fit = getattr(forecaster, "df", None)
                    if data_used_for_fit is not None and len(data_used_for_fit) > 0:
                        start_date = data_used_for_fit.index[-1]
                    else:
                        start_date = full_df.index[-steps] if len(full_df) > steps else full_df.index[0]

                    if self.plot_type.currentText().startswith("Interactive"):
                        try:
                            fc_fig = forecaster.plot_forecast_interactive(steps=steps)
                        except TypeError:
                            fc_fig = forecaster.plot_forecast_interactive(steps)

                        try:
                            mask = full_df.index >= start_date
                            if hasattr(fc_fig, "data"):
                                combined = fc_fig
                            else:
                                combined = go.Figure()
                            combined.add_trace(
                                go.Scatter(
                                    x=full_df.index[mask],
                                    y=full_df.loc[mask, price_col],
                                    mode="lines",
                                    name="Actual",
                                    line=dict(width=2, dash="dash")
                                )
                            )
                            title = f"Historical Forecast: {ticker} (Predicting from {start_date.strftime('%Y-%m-%d')})"
                            self._show_plotly_in_dialog(combined, title=title)
                        except Exception:
                            self.log_print(f"[{ticker}] failed to overlay actual (interactive):\n{traceback.format_exc()}")
                            self._show_plotly_in_dialog(fc_fig, title=f"Historical Forecast: {ticker}")
                    else:
                        # Matplotlib path
                        try:
                            fc_fig = forecaster.plot_forecast_matplotlib(steps=steps)
                        except TypeError:
                            fc_fig = forecaster.plot_forecast_matplotlib(steps)

                        try:
                            fig = fc_fig
                            ax = None
                            if hasattr(fig, "axes") and fig.axes:
                                ax = fig.axes[0]
                            else:
                                try:
                                    ax = fig.gca()
                                except Exception:
                                    ax = None
                            if ax is None:
                                ax = fig.add_subplot(111)

                            mask = full_df.index >= start_date
                            ax.plot(full_df.index[mask], full_df.loc[mask, price_col], linestyle="--", label="Actual")
                            ax.legend()
                            title = f"Historical Forecast: {ticker} (Predicting from {start_date.strftime('%Y-%m-%d')})"
                            self._show_matplotlib_in_dialog(fig, title=title)
                        except Exception:
                            self.log_print(f"[{ticker}] failed to overlay actual (matplotlib):\n{traceback.format_exc()}")
                            self._show_matplotlib_in_dialog(fc_fig, title=f"Historical Forecast: {ticker}")
            else:
                # Live forecast
                if self.plot_type.currentText().startswith("Interactive"):
                    fig = forecaster.plot_forecast_interactive(steps=steps)
                    self._show_plotly_in_dialog(fig, title=f"Forecast: {ticker}")
                else:
                    fig = forecaster.plot_forecast_matplotlib(steps=steps)
                    self._show_matplotlib_in_dialog(fig, title=f"Forecast: {ticker}")

            if lbl:
                lbl.setText("Done")
            self.log_print(f"[{ticker}] Forecast complete (Historical: {is_historical}).")
        except Exception:
            self.log_print(f"[{ticker}] Plotting error:\n{traceback.format_exc()}")
            lbl = self.status_labels.get(ticker)
            if lbl:
                lbl.setText("Plot error")
        finally:
            self._maybe_enable_run_button()

    def _maybe_enable_run_button(self):
        """Enables both run buttons if no threads are running."""
        if not self.running_threads:
            self.run_forecast_btn.setEnabled(True)
            self.run_historical_btn.setEnabled(True)
            self.global_status.setText("")

    # ---------------- Plot helpers ----------------
    def _show_plotly_in_dialog(self, fig, title="Forecast"):
        try:
            if hasattr(fig, 'to_html'):
                html = fig.to_html(include_plotlyjs="cdn")
                dialog = QDialog(self)
                dialog.setWindowTitle(title)
                layout = QVBoxLayout(dialog)
                view = QWebEngineView()
                view.setHtml(html)
                layout.addWidget(view)
                dialog.resize(900, 650)
                dialog.exec()
            else:
                 self.log_print(f"Plotly display failed: Figure object not recognized.")

        except Exception:
            self.log_print(f"Plotly display failed:\n{traceback.format_exc()}")

    def _show_matplotlib_in_dialog(self, fig, title="Forecast"):
        try:
            if fig and hasattr(fig, 'canvas'):
                dialog = QDialog(self)
                dialog.setWindowTitle(title)
                layout = QVBoxLayout(dialog)
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
                dialog.resize(900, 650)
                canvas.draw()
                dialog.exec()
            else:
                self.log_print(f"Matplotlib display failed: Figure object not recognized.")

        except Exception:
            self.log_print(f"Matplotlib display failed:\n{traceback.format_exc()}")