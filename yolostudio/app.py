"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow
from .ui.theme import BG, STYLESHEET, TEXT


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Studio")
    app.setOrganizationName("YOLOStudio")
    app.setStyle("Fusion")

    # Fusion plus a base palette keeps native dialogs -- file picker, colour
    # picker -- consistent with the stylesheet instead of flashing white.
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor("#141418"))
    palette.setColor(QPalette.AlternateBase, QColor(BG))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor("#232329"))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor("#2f5aa8"))
    palette.setColor(QPalette.HighlightedText, QColor(TEXT))
    palette.setColor(QPalette.ToolTipBase, QColor("#232329"))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
