from PySide6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QTextEdit, QSplitter, QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor
from .tabs import (LoadDataTab, PortfolioTab, AnalysisTab,
                   SimulationTab, ForecastTab, CompareTab)
from PySide6.QtGui import QFont

def apply_theme(app: QApplication):
    """Apply a simple, easy-to-tweak palette + stylesheet to the app."""
    
    QApplication.setFont(QFont("Arial", 10))  # Use a common font
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#f3f4f6"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.Text, QColor("#2d374d"))
    pal.setColor(QPalette.Button, QColor("#2563eb"))
    pal.setColor(QPalette.ButtonText, QColor("#ffffff"))
    app.setPalette(pal)

    app.setStyleSheet("""
    QTabWidget::pane { background: #ffffff; border: 1px solid #e5e7eb; }
    QTabBar::tab { background: #eaeef6; color: #0f172a; padding: 6px 10px; border: 1px solid #e5e7eb; border-bottom: none; }
    QTabBar::tab:selected { background: #ffffff; font-weight: 600; }
    QTextEdit#logPanel { background: #0f172a; color: #f8fafc; border: 1px solid #374151; }
    QPushButton { background: #2563eb; color: white; border-radius: 4px; padding: 6px 10px; }
    QPushButton:hover { background: #1e40af; }
    QLineEdit, QComboBox { background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; padding: 4px; }
    QLabel { color: #0f172a; }
    """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QFKit")
        self.resize(700, 700)
        self.setMinimumWidth(600)

        # Shared app state used by tabs
        self.instruments = {}   # ticker -> TimeSeriesAnalysis-like object
        self.simulations = {}   # name -> SDE_Simulation-like object

        # Central layout: splitter with tabs above and log below
        splitter = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

        self.log_panel = QTextEdit()
        self.log_panel.setObjectName("logPanel")
        self.log_panel.setReadOnly(True)
        splitter.addWidget(self.log_panel)
        splitter.setSizes([240, 560])

        self.setCentralWidget(splitter)

        # Create tabs and keep references
        self.init_tabs()

    def init_tabs(self):
        self.load_tab = LoadDataTab(self)
        self.portfolio_tab = PortfolioTab(self)
        self.analysis_tab = AnalysisTab(self, load_tab=self.load_tab)
        self.simulation_tab = SimulationTab(self)
        self.forecast_tab = ForecastTab(self)
        self.compare_tab = CompareTab(self)

        # Add tabs
        self.tabs.addTab(self.load_tab, "Load Data")
        self.tabs.addTab(self.portfolio_tab, "Portfolio")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.simulation_tab, "Simulation")
        self.tabs.addTab(self.forecast_tab, "Forecast")

        self.tabs.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index: int):
        widget = self.tabs.widget(index)
        # Refresh tabs with dynamic content when they become visible
        if widget in (self.portfolio_tab, self.analysis_tab, self.simulation_tab, self.forecast_tab):
            try:
                widget.refresh_instruments()
            except Exception:
                pass

    def add_tab(self, name, widget: QWidget):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        container = QWidget()
        container.setLayout(layout)
        self.tabs.addTab(container, name)

    def log_message(self, text: str):
        self.log_panel.append(text)

    def refresh_all_ui_lists(self):
        """
        Called by other tabs after they modify the central data store.
        This function's only job is to tell all relevant tabs to update their view.
        """
        try:
            self.portfolio_tab.refresh_instruments()
        except Exception as e:
            print(f"Error refreshing portfolio tab: {e}")
            pass
        try:
            self.analysis_tab.refresh_instruments()
        except Exception as e:
            print(f"Error refreshing analysis tab: {e}")
            pass
        try:
            self.simulation_tab.refresh_instruments()
        except Exception as e:
            print(f"Error refreshing simulation tab: {e}")
            pass
        try:
            self.forecast_tab.refresh_instruments() 
        except Exception as e:
            print(f"Error refreshing forecast tab: {e}")
