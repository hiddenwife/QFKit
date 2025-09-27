# gui/tabs/compare_tab.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class CompareTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Compare Tools Coming Soon"))
        self.setLayout(layout)
