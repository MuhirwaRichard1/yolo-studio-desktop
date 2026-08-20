"""A dark theme. Annotation is long-session work against arbitrary photos, and
a neutral dark chrome keeps the eye on the image instead of the UI.
"""

from __future__ import annotations

BG = "#1b1b21"
BG_ALT = "#232329"
BG_DEEP = "#141418"
LINE = "#33333c"
TEXT = "#e4e4ea"
TEXT_DIM = "#9a9aa8"
ACCENT = "#4f8cff"
ACCENT_DIM = "#2f5aa8"
GOOD = "#43c07a"
WARN = "#e0a33a"
BAD = "#e2564c"

STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-size: 13px;
}}
QMainWindow::separator {{
    background: {LINE};
    width: 1px;
    height: 1px;
}}
QToolBar {{
    background: {BG_ALT};
    border: none;
    border-bottom: 1px solid {LINE};
    padding: 4px 6px;
    spacing: 4px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 5px 9px;
}}
QToolButton:hover {{ background: {LINE}; }}
QToolButton:checked {{
    background: {ACCENT_DIM};
    border-color: {ACCENT};
}}
QStatusBar {{
    background: {BG_ALT};
    border-top: 1px solid {LINE};
    color: {TEXT_DIM};
}}
QStatusBar::item {{ border: none; }}
QDockWidget {{
    titlebar-close-icon: none;
    font-weight: 600;
}}
QDockWidget::title {{
    background: {BG_ALT};
    padding: 7px 10px;
    border-bottom: 1px solid {LINE};
}}
QTabWidget::pane {{
    border: none;
    border-top: 1px solid {LINE};
}}
QTabBar::tab {{
    background: transparent;
    padding: 7px 14px;
    border-bottom: 2px solid transparent;
    color: {TEXT_DIM};
}}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom-color: {ACCENT};
}}
QTabBar::tab:hover {{ color: {TEXT}; }}

QPushButton {{
    background: {BG_ALT};
    border: 1px solid {LINE};
    border-radius: 5px;
    padding: 6px 12px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {LINE}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {LINE}; }}
QPushButton[accent="true"] {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: #08101f;
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{ background: #679dff; }}
QPushButton[accent="true"]:disabled {{ background: {ACCENT_DIM}; color: {TEXT_DIM}; }}
QPushButton[danger="true"] {{ border-color: {BAD}; color: {BAD}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {BG_DEEP};
    border: 1px solid {LINE};
    border-radius: 5px;
    padding: 5px 7px;
    selection-background-color: {ACCENT_DIM};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {BG_ALT};
    border: 1px solid {LINE};
    selection-background-color: {ACCENT_DIM};
    outline: none;
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 14px; }}

QListWidget, QTreeWidget, QTableWidget {{
    background: {BG_DEEP};
    border: 1px solid {LINE};
    border-radius: 5px;
    outline: none;
    alternate-background-color: {BG};
}}
QListWidget::item, QTreeWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {ACCENT_DIM};
    color: {TEXT};
}}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: {LINE}; }}
QHeaderView::section {{
    background: {BG_ALT};
    border: none;
    border-bottom: 1px solid {LINE};
    padding: 5px;
    color: {TEXT_DIM};
}}

QGroupBox {{
    border: 1px solid {LINE};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: {TEXT_DIM};
}}

QProgressBar {{
    background: {BG_DEEP};
    border: 1px solid {LINE};
    border-radius: 5px;
    text-align: center;
    height: 18px;
    color: {TEXT};
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0; }}
QScrollBar::handle {{ background: #45454f; border-radius: 5px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: #565663; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {LINE}; }}
QToolTip {{
    background: {BG_ALT};
    color: {TEXT};
    border: 1px solid {LINE};
    padding: 4px 6px;
}}
QMenu {{
    background: {BG_ALT};
    border: 1px solid {LINE};
    padding: 4px;
}}
QMenu::item {{ padding: 5px 22px 5px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT_DIM}; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px 6px; }}
QMenuBar {{ background: {BG_ALT}; border-bottom: 1px solid {LINE}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {LINE}; border-radius: 4px; }}

QLabel[hint="true"] {{ color: {TEXT_DIM}; }}
QLabel[heading="true"] {{ font-size: 15px; font-weight: 600; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; }}
QCheckBox::indicator:unchecked {{
    border: 1px solid {LINE};
    border-radius: 3px;
    background: {BG_DEEP};
}}
QCheckBox::indicator:checked {{
    border: 1px solid {ACCENT};
    border-radius: 3px;
    background: {ACCENT};
}}
"""
