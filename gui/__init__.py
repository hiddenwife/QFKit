# gui/__init__.py
from .main_window import MainWindow, apply_theme
from PySide6.QtWidgets import QApplication
import sys


def launch_gui():
    """Launch the PySide6 GUI application."""
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
