# gui/tabs/forecast_tab.py
import os
import sys
import tempfile
import subprocess
import pickle
import traceback
import time

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame,
    QHBoxLayout, QCheckBox, QMessageBox, QDialog, QSlider,
    QPushButton, QComboBox, QSpinBox, QSplitter, QLineEdit,
    QStackedLayout
)
from PySide6.QtGui import QIntValidator
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

import multiprocessing as mp


class ForecastWorkerThread(QThread):
    finished = Signal(dict)
    error = Signal(str)
    status_update = Signal(str)

    def __init__(self, cmd, env, timeout=1200):
        super().__init__()
        self.cmd = cmd
        self.env = env
        self.timeout = timeout
        self.process = None

    def run(self):
        try:
            self.status_update.emit("Starting forecast process...")
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW
            
            self.process = subprocess.Popen(
                self.cmd, env=self.env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, creationflags=creationflags
            )
            stdout, stderr = self.process.communicate(timeout=self.timeout)
            ret = self.process.returncode
            output_file = self.cmd[self.cmd.index("--output-file") + 1]

            if os.path.exists(output_file):
                try:
                    with open(output_file, 'rb') as f:
                        results = pickle.load(f)
                    if results.get("status") == "error" and stderr:
                        results["traceback"] = results.get("traceback", "") + f"\n\nStderr:\n{stderr}"
                    self.finished.emit(results)
                except Exception as e:
                    self.error.emit(f"Failed to read output file: {e}\nStderr: {stderr}")
            else:
                self.error.emit(f"Process exited with code {ret} without creating output.\nStderr:\n{stderr}")
        except subprocess.TimeoutExpired:
            self.error.emit(f"Forecast timed out after {self.timeout} seconds.")
            self.stop()
        except Exception as e:
            self.error.emit(f"Worker thread error: {e}\n{traceback.format_exc()}")

    def stop(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            except Exception:
                pass


class ForecastTab(QWidget):
    _update_status_label = Signal(str, str)
    _plot_figure_ready = Signal(object, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.forecast_checks = {}
        self.status_labels = {}
        self.worker_thread = None
        self.active_ticker = None
        self.temp_files = []
        self._setup_ui()
        self._build_forecast_controls()
        self._update_status_label.connect(self._on_update_status_label)
        self._plot_figure_ready.connect(self._display_plot_dialog)

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(QLabel("Select tickers for Bayesian Forecast:"))
        splitter = QSplitter(Qt.Vertical)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QFrame()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll.setWidget(self.scroll_widget)
        splitter.addWidget(self.scroll)
        self.params_container = QFrame()
        params_layout = QVBoxLayout(self.params_container)

        horizon_row = QHBoxLayout()
        horizon_row.addWidget(QLabel("Forecast horizon (days):"))
        self.horizon_slider = QSlider(Qt.Horizontal)
        self.horizon_slider.setRange(5, 252)
        self.horizon_slider.setValue(60)
        self.horizon_label = QLabel(str(self.horizon_slider.value()))
        self.horizon_slider.valueChanged.connect(lambda v: self.horizon_label.setText(str(v)))
        horizon_row.addWidget(self.horizon_slider, stretch=1)
        horizon_row.addWidget(self.horizon_label)
        params_layout.addLayout(horizon_row)

        ar_row = QHBoxLayout()
        ar_row.addWidget(QLabel("AR Order (p):"))
        self.ar_spin = QSpinBox()
        self.ar_spin.setRange(0, 6)
        self.ar_spin.setValue(1)
        ar_row.addWidget(self.ar_spin)
        params_layout.addLayout(ar_row)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Inference Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Fast (ADVI)", "Full (NUTS)"])
        method_row.addWidget(self.method_combo)
        params_layout.addLayout(method_row)
        
        self.advi_container = QWidget()
        advi_row = QHBoxLayout(self.advi_container)
        advi_row.setContentsMargins(0, 0, 0, 0)
        advi_row.addWidget(QLabel("ADVI Iterations:"))
        self.advi_spin = QSpinBox()
        self.advi_spin.setRange(10000, 200000)
        self.advi_spin.setValue(30000)
        self.advi_spin.setSingleStep(5000)
        advi_row.addWidget(self.advi_spin)
        params_layout.addWidget(self.advi_container)

        self.draws_stack = QStackedLayout()
        draws_default_container = QWidget()
        draws_default_layout = QHBoxLayout(draws_default_container)
        draws_default_layout.setContentsMargins(0, 0, 0, 0)
        draws_default_layout.addWidget(QLabel("Posterior Draws:"))
        self.draws_slider = QSlider(Qt.Horizontal)
        self.draws_slider.setRange(1000, 50000)
        self.draws_slider.setValue(10000)
        self.draws_label = QLabel(str(self.draws_slider.value()))
        self.draws_slider.valueChanged.connect(lambda v: self.draws_label.setText(str(v)))
        draws_default_layout.addWidget(self.draws_slider, stretch=1)
        draws_default_layout.addWidget(self.draws_label)

        draws_custom_container = QWidget()
        draws_custom_layout = QHBoxLayout(draws_custom_container)
        draws_custom_layout.setContentsMargins(0, 0, 0, 0)
        draws_custom_layout.addWidget(QLabel("Posterior Draws (custom):"))
        self.draws_input = QLineEdit("10000")
        self.draws_input.setValidator(QIntValidator(500, 300000))
        draws_custom_layout.addWidget(self.draws_input, stretch=1)
        self.draws_stack.addWidget(draws_default_container)
        self.draws_stack.addWidget(draws_custom_container)
        params_layout.addLayout(self.draws_stack)

        self.nuts_options_container = QFrame()
        nuts_layout = QVBoxLayout(self.nuts_options_container)
        nuts_layout.setContentsMargins(0, 0, 0, 0)

        try:
            self.cpu_count = mp.cpu_count()
        except Exception:
            self.cpu_count = 1
        
        self.chains_stack = QStackedLayout()
        chains_default_container = QWidget()
        chains_default_layout = QHBoxLayout(chains_default_container)
        chains_default_layout.setContentsMargins(0, 0, 0, 0)
        chains_default_layout.addWidget(QLabel("Chains for NUTS:"))
        self.chains_slider = QSlider(Qt.Horizontal)
        self.chains_slider.setRange(2, max(2, self.cpu_count))
        self.chains_slider.setValue(min(4, self.cpu_count))
        self.chains_label = QLabel(str(self.chains_slider.value()))
        self.chains_slider.valueChanged.connect(lambda v: self.chains_label.setText(str(v)))
        chains_default_layout.addWidget(self.chains_slider, stretch=1)
        chains_default_layout.addWidget(self.chains_label)
        chains_custom_container = QWidget()
        chains_custom_layout = QHBoxLayout(chains_custom_container)
        chains_custom_layout.setContentsMargins(0, 0, 0, 0)
        chains_custom_layout.addWidget(QLabel("Chains (custom):"))
        self.chains_input = QLineEdit("8")
        self.chains_input.setValidator(QIntValidator(2, 64))
        chains_custom_layout.addWidget(self.chains_input, stretch=1)
        self.chains_stack.addWidget(chains_default_container)
        self.chains_stack.addWidget(chains_custom_container)

        nuts_layout.addLayout(self.chains_stack)
        cores_row = QHBoxLayout()
        self.cores_label_prefix = QLabel("Cores to use:")
        cores_row.addWidget(self.cores_label_prefix)
        self.cores_slider = QSlider(Qt.Horizontal)
        self.cores_slider.setRange(1, self.cpu_count)
        self.cores_slider.setValue(min(4, self.cpu_count))
        self.cores_label = QLabel(str(self.cores_slider.value()))
        self.cores_slider.valueChanged.connect(lambda v: self.cores_label.setText(str(v)))
        cores_row.addWidget(self.cores_slider, stretch=1)
        cores_row.addWidget(self.cores_label)
        nuts_layout.addLayout(cores_row)
        self.more_draws_checkbox = QCheckBox("Use custom draws/chains")
        nuts_layout.addWidget(self.more_draws_checkbox)
        params_layout.addWidget(self.nuts_options_container)

        plot_row = QHBoxLayout()
        plot_row.addWidget(QLabel("Plot Type:"))
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Interactive (Plotly)", "Static (Matplotlib)"])
        plot_row.addWidget(self.plot_type)
        params_layout.addLayout(plot_row)

        btn_row = QHBoxLayout()
        self.run_forecast_btn = QPushButton("Run Forecast")
        self.run_historical_btn = QPushButton("Compare Forecast to History")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        btn_row.addWidget(self.run_forecast_btn)
        btn_row.addWidget(self.run_historical_btn)
        btn_row.addWidget(self.cancel_btn)
        self.global_status = QLabel("")
        btn_row.addWidget(self.global_status, stretch=1)
        params_layout.addLayout(btn_row)
        self.run_forecast_btn.clicked.connect(lambda: self.run_forecast_for_selected(is_historical=False))
        self.run_historical_btn.clicked.connect(lambda: self.run_forecast_for_selected(is_historical=True))
        self.cancel_btn.clicked.connect(self._cancel_forecast)
        splitter.addWidget(self.params_container)
        root.addWidget(splitter, stretch=1)

        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        self.more_draws_checkbox.toggled.connect(self._on_more_draws_toggled)
        self.chains_slider.valueChanged.connect(self.update_cores_max)
        self.chains_input.textChanged.connect(self.update_cores_max)
        QTimer.singleShot(0, lambda: self._on_method_changed(self.method_combo.currentText()))
        QTimer.singleShot(0, lambda: self._on_more_draws_toggled(self.more_draws_checkbox.isChecked()))
        QTimer.singleShot(0, self.update_cores_max)

    @Slot(str, str)
    def _on_update_status_label(self, ticker, text):
        if lbl := self.status_labels.get(ticker): lbl.setText(text)

    def _on_method_changed(self, text):
        is_nuts = "NUTS" in text.upper()
        self.nuts_options_container.setVisible(is_nuts)
        self.advi_container.setVisible(not is_nuts)
        self.update_cores_max()

    def _on_more_draws_toggled(self, checked):
        self.draws_stack.setCurrentIndex(1 if checked else 0)
        self.chains_stack.setCurrentIndex(1 if checked else 0)
        self.update_cores_max()

    def update_cores_max(self):
        try:
            chains_visible = self.more_draws_checkbox.isChecked() and self.nuts_options_container.isVisible()
            chains = int(self.chains_input.text()) if chains_visible else self.chains_slider.value()
        except (ValueError, AttributeError):
            chains = self.chains_slider.minimum()
        max_cores = min(chains, getattr(self, "cpu_count", 1))
        self.cores_slider.setRange(1, max_cores)
        self.cores_label_prefix.setText(f"Cores to use (max {max_cores}):")

    def _get_forecast_params(self):
        method = "nuts" if "NUTS" in self.method_combo.currentText() else "advi"
        params = {
            "steps": self.horizon_slider.value(),
            "p": self.ar_spin.value(),
            "plot_type": "plotly" if "Plotly" in self.plot_type.currentText() else "mpl",
            "method": method,
            "advi_iter": self.advi_spin.value() if method == "advi" else 20000
        }
        if self.more_draws_checkbox.isChecked() and method == "nuts":
            params.update({"draws": int(self.draws_input.text()), "chains": int(self.chains_input.text())})
        else:
            params.update({"draws": self.draws_slider.value(), "chains": self.chains_slider.value()})
        params["cores"] = self.cores_slider.value()
        return params

    def log_print(self, msg):
        getattr(self.main_window, "log_message", print)(msg)

    def _get_selected_tickers(self):
        return [t for t, cb in self.forecast_checks.items() if cb.isChecked()]

    def _build_forecast_controls(self):
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if widget := item.widget():
                widget.deleteLater()
        self.forecast_checks.clear()
        self.status_labels.clear()

        tickers = sorted((getattr(self.main_window, "instruments", {}) or {}).keys())
        if not tickers:
            self.scroll_layout.addWidget(QLabel("No instruments loaded."))
            return

        for ticker in tickers:
            row_container = QFrame()
            row_layout = QHBoxLayout(row_container)
            row_layout.setContentsMargins(2, 2, 2, 2)
            
            cb = QCheckBox(ticker)
            status = QLabel("")
            
            row_layout.addWidget(cb)
            row_layout.addStretch(1)
            row_layout.addWidget(status)
            
            self.scroll_layout.addWidget(row_container)
            
            self.forecast_checks[ticker] = cb
            self.status_labels[ticker] = status
            
        self.scroll_layout.addStretch(1)

    def refresh_instruments(self):
        self._build_forecast_controls()

    def run_forecast_for_selected(self, is_historical):
        selected = self._get_selected_tickers()
        if not selected:
            QMessageBox.information(self, "Selection Error", "Please select at least one ticker.")
            return

        self.active_ticker = selected[0]
        inst = getattr(self.main_window, "instruments", {}).get(self.active_ticker)
        if inst is None or getattr(inst, "df", None) is None:
            QMessageBox.warning(self, "Data Error", f"Instrument {self.active_ticker} has no data.")
            return

        params = self._get_forecast_params()
        if is_historical and len(inst.df) <= params["steps"]:
            QMessageBox.information(self, "Not enough data", "Not enough data for historical comparison.")
            return

        returns = np.log(inst.df["Close"]).diff().dropna()
        sigma_prior = max(0.005, np.std(returns.tail(60))) if len(returns) >= 30 else 0.05
        
        log_msg = (f"[{self.active_ticker}] Starting {'Historical' if is_historical else 'Live'} Forecast...\n"
                   f"  - Method: {params['method'].upper()}, Horizon: {params['steps']} days, AR Order: {params['p']}\n"
                   f"  - Posterior Draws: {params['draws']}\n"
                   f"  - Sigma Prior STD: {sigma_prior:.4f} (from last 60 days)")
        if params['method'] == 'advi':
            log_msg += f"\n  - ADVI Iterations: {params['advi_iter']}"
        else:
            log_msg += f"\n  - Chains: {params['chains']}, Cores: {params['cores']}"
        self.log_print(log_msg)

        input_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl").name
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl").name
        self.temp_files.extend([input_file, output_file])

        with open(input_file, "wb") as f:
            pickle.dump(inst.df, f, protocol=pickle.HIGHEST_PROTOCOL)

        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "forecast_worker.py"),
            "--input-file", input_file, "--output-file", output_file,
            "--steps", str(params["steps"]), "--p", str(params["p"]), "--method", params["method"],
            "--draws", str(params["draws"]), "--chains", str(params["chains"]),
            "--cores", str(params["cores"]), "--advi-iter", str(params["advi_iter"]),
            "--sigma-prior", str(sigma_prior),
        ]
        if is_historical:
            cmd.append("--is-historical")

        env = os.environ.copy()
        env.update({"MPLCONFIGDIR": tempfile.mkdtemp()})
        self._set_ui_running(True)
        self.global_status.setText(f"Running forecast for {self.active_ticker}...")
        self.status_labels[self.active_ticker].setText("Fitting...")

        self.worker_thread = ForecastWorkerThread(cmd, env)
        self.worker_thread.finished.connect(self._on_forecast_finished)
        self.worker_thread.error.connect(self._on_forecast_error)
        self.worker_thread.status_update.connect(self.log_print)
        self.worker_thread.start()

    def _cancel_forecast(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.log_print("Cancelling forecast...")
            self.worker_thread.stop()
            self.worker_thread.wait(5000)
            self._cleanup_after_forecast("Forecast cancelled.")

    # ==========================================================
    # FIX: Disable controls individually to keep Cancel button active
    # ==========================================================
    def _set_ui_running(self, is_running):
        """Enable/disable UI controls based on forecast status."""
        # List of all controls to disable, exclusing the cancel button
        controls_to_toggle = [
            self.scroll,
            self.horizon_slider,
            self.ar_spin,
            self.method_combo,
            self.advi_container,
            self.draws_stack,
            self.nuts_options_container,
            self.plot_type,
            self.run_forecast_btn,
            self.run_historical_btn
        ]
        
        for control in controls_to_toggle:
            control.setEnabled(not is_running)
        
        # Handle the cancel button separately
        self.cancel_btn.setEnabled(is_running)


    @Slot(dict)
    def _on_forecast_finished(self, results):
        if results.get("status") == "success":
            self.status_labels[self.active_ticker].setText("Plotting...")
            plot_type = self._get_forecast_params()['plot_type']
            is_historical = "--is-historical" in self.worker_thread.cmd
            forecast_df = results["forecast_df"]
            full_df = results["full_df"]
            
            history_len = 200
            history_df = full_df['Close']
            
            actuals_df = None
            if is_historical:
                common_index = forecast_df.index.intersection(full_df.index)
                actuals_df = full_df.loc[common_index, 'Close']
                if not common_index.empty:
                    history_df = full_df.loc[full_df.index < common_index.min(), 'Close']

            history_df = history_df.tail(history_len)

            title = f"{'Historical ' if is_historical else ''}Forecast: {self.active_ticker}"

            fig = self._create_plot_figure(plot_type, history_df, forecast_df, actuals_df)
            self._plot_figure_ready.emit(fig, title, plot_type)
            self.status_labels[self.active_ticker].setText("Done")
            self._cleanup_after_forecast("Done.")
        else:
            error_msg = f"[{self.active_ticker}] Subprocess ERROR: {results.get('traceback')}"
            self.log_print(error_msg)
            self.status_labels[self.active_ticker].setText("Error")
            self._cleanup_after_forecast("Error.")

    @Slot(str)
    def _on_forecast_error(self, error_msg):
        self.log_print(f"Forecast error: {error_msg}")
        if self.active_ticker and self.active_ticker in self.status_labels:
            self.status_labels[self.active_ticker].setText("Error")
        self._cleanup_after_forecast("Error.")

    def _cleanup_after_forecast(self, status_msg):
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None
        self._cleanup_temp_files()
        self._set_ui_running(False)
        self.global_status.setText(status_msg)
        self.active_ticker = None

    def _cleanup_temp_files(self):
        for f in self.temp_files:
            if os.path.exists(f):
                try: os.remove(f)
                except Exception: pass
        self.temp_files.clear()

    def _create_plot_figure(self, plot_type, history_df, forecast_df, actuals_df=None):
        method = self._get_forecast_params()['method']
        method_label = "ADVI" if method == "advi" else "NUTS" 
        if plot_type == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history_df.index, y=history_df.values, mode="lines", name="History", line=dict(color="black")))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["median"], mode="lines", name="Forecast Median", line=dict(color="blue", dash="dash")))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["upper_95"], mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["lower_95"], mode="lines", fill="tonexty", name="95% CI", line=dict(width=0), fillcolor="rgba(135,206,250,0.3)"))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["upper_80"], mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["lower_80"], mode="lines", fill="tonexty", name="80% CI", line=dict(width=0), fillcolor="rgba(30,144,255,0.25)"))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["upper_50"], mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["lower_50"], mode="lines", fill="tonexty", name="50% CI", line=dict(width=0), fillcolor="rgba(65,105,225,0.5)"))
            if actuals_df is not None and not actuals_df.empty:
                fig.add_trace(go.Scatter(x=actuals_df.index, y=actuals_df.values, mode="lines", name="Actual", line=dict(width=2, dash="dot", color="red")))
            fig.update_layout(
                template="plotly_white",
                hovermode="x unified",
                title=f"Forecast for {self.active_ticker} with {method_label}",
                xaxis_title="Date", 
                yaxis_title="Price"  
            )
            return fig
        else:
            fig = Figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            ax.plot(history_df.index, history_df.values, label="History", color="black")
            ax.plot(forecast_df.index, forecast_df["median"], label="Median Forecast", linestyle="--", color="blue")
            ax.fill_between(forecast_df.index, forecast_df["lower_95"], forecast_df["upper_95"], color="skyblue", alpha=0.3, label="95% CI")
            ax.fill_between(forecast_df.index, forecast_df["lower_80"], forecast_df["upper_80"], color="dodgerblue", alpha=0.3, label="80% CI")
            ax.fill_between(forecast_df.index, forecast_df["lower_50"], forecast_df["upper_50"], color="steelblue", alpha=0.3, label="50% CI")
            if actuals_df is not None and not actuals_df.empty:
                ax.plot(actuals_df.index, actuals_df.values, label="Actual", linestyle=":", color="red")

            ax.set_title(f"Forecast for {self.active_ticker} with {method_label}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.tight_layout()
            return fig

    @Slot(object, str, str)
    def _display_plot_dialog(self, fig, title, plot_type):
        if plot_type == 'plotly':
            try:
                import webbrowser
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
                    fig.write_html(f, include_plotlyjs="cdn")
                webbrowser.open(f"file://{os.path.realpath(f.name)}")
            except Exception as e:
                self.log_print(f"Failed to show Plotly plot: {e}")
        else:
            try:
                dialog = QDialog(self)
                dialog.setWindowTitle(title)
                layout = QVBoxLayout(dialog)
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, dialog)
                layout.addWidget(toolbar)
                layout.addWidget(canvas)
                dialog.resize(900, 600)
                dialog.exec()
            except Exception as e:
                self.log_print(f"Matplotlib display failed: {e}")