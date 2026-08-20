"""Image browser: search, filter by annotation state, and navigate.

Annotation state is cached in a dict rather than re-stat'ing every label file on
each keystroke, which keeps filtering responsive on large projects.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout,
                               QWidget)

from ..core.project import ImageEntry, Project
from .theme import BAD, GOOD, TEXT_DIM

FILTER_ALL = "All images"
FILTER_DONE = "Annotated"
FILTER_TODO = "Not annotated"
FILTER_MISSING = "Missing file"


def _dot(color: str) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(2, 2, 8, 8)
    painter.end()
    return QIcon(pixmap)


class ImageListPanel(QWidget):

    currentChanged = Signal(object)      # ImageEntry or None
    removeRequested = Signal(list)       # List[ImageEntry]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._project: Optional[Project] = None
        self._entries: List[ImageEntry] = []
        self._annotated: Dict[str, bool] = {}
        self._counts: Dict[str, int] = {}
        self._icons = {"done": _dot(GOOD), "todo": _dot(TEXT_DIM), "missing": _dot(BAD)}
        self._loading = False

        self.search = QLineEdit(placeholderText="Filter by filename…")
        self.search.setClearButtonEnabled(True)
        self.filter = QComboBox()
        self.filter.addItems([FILTER_ALL, FILTER_DONE, FILTER_TODO, FILTER_MISSING])

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setUniformItemSizes(True)
        self.list.setAlternatingRowColors(True)

        self.summary = QLabel("No project open")
        self.summary.setProperty("hint", True)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.search, 1)
        top.addWidget(self.filter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.summary)

        self.search.textChanged.connect(self.refresh)
        self.filter.currentIndexChanged.connect(self.refresh)
        self.list.currentItemChanged.connect(self._on_current)
        self.list.keyPressEvent = self._list_key_press  # type: ignore[method-assign]

    # ------------------------------------------------------------------ data

    def set_project(self, project: Optional[Project]) -> None:
        self._project = project
        self._annotated.clear()
        self._counts.clear()
        if project is not None:
            for entry in project.images:
                shapes = project.shapes_for(entry)
                self._annotated[entry.iid] = bool(shapes)
                self._counts[entry.iid] = len(shapes)
        self.refresh()

    def mark(self, entry: ImageEntry, count: int) -> None:
        """Update one row's state after a save, without a full rebuild."""
        self._annotated[entry.iid] = count > 0
        self._counts[entry.iid] = count
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.data(Qt.UserRole) is entry or (
                    isinstance(item.data(Qt.UserRole), ImageEntry)
                    and item.data(Qt.UserRole).iid == entry.iid):
                self._decorate(item, entry)
                break
        self._update_summary()

    def refresh_states(self) -> None:
        """Re-read every label file (after auto-labelling or a class delete)."""
        if self._project is None:
            return
        for entry in self._project.images:
            shapes = self._project.shapes_for(entry)
            self._annotated[entry.iid] = bool(shapes)
            self._counts[entry.iid] = len(shapes)
        self.refresh(keep_current=True)

    # --------------------------------------------------------------- display

    def _decorate(self, item: QListWidgetItem, entry: ImageEntry) -> None:
        count = self._counts.get(entry.iid, 0)
        if not entry.exists():
            item.setIcon(self._icons["missing"])
            item.setToolTip(f"{entry.path}\nFile is missing")
        elif self._annotated.get(entry.iid):
            item.setIcon(self._icons["done"])
            item.setToolTip(f"{entry.path}\n{count} annotation(s)")
        else:
            item.setIcon(self._icons["todo"])
            item.setToolTip(entry.path)
        suffix = f"   {count}" if count else ""
        item.setText(f"{entry.name}{suffix}")

    def refresh(self, keep_current: bool = False) -> None:
        current = self.current_entry() if keep_current else None
        self._loading = True
        self.list.clear()
        self._entries = []
        if self._project is None:
            self._loading = False
            self.summary.setText("No project open")
            return

        needle = self.search.text().strip().lower()
        mode = self.filter.currentText()
        for entry in self._project.images:
            if needle and needle not in entry.name.lower():
                continue
            annotated = self._annotated.get(entry.iid, False)
            exists = entry.exists()
            if mode == FILTER_DONE and not annotated:
                continue
            if mode == FILTER_TODO and (annotated or not exists):
                continue
            if mode == FILTER_MISSING and exists:
                continue
            item = QListWidgetItem()
            item.setData(Qt.UserRole, entry)
            self._decorate(item, entry)
            self.list.addItem(item)
            self._entries.append(entry)

        self._loading = False
        self._update_summary()
        if current is not None:
            self.select_entry(current)
        elif self.list.count():
            self.list.setCurrentRow(0)
        else:
            self.currentChanged.emit(None)

    def _update_summary(self) -> None:
        if self._project is None:
            return
        total = len(self._project.images)
        done = sum(1 for v in self._annotated.values() if v)
        shown = self.list.count()
        pct = (done / total * 100) if total else 0
        extra = f" · showing {shown}" if shown != total else ""
        self.summary.setText(f"{done}/{total} annotated ({pct:.0f}%){extra}")

    # ------------------------------------------------------------ navigation

    def current_entry(self) -> Optional[ImageEntry]:
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def selected_entries(self) -> List[ImageEntry]:
        return [i.data(Qt.UserRole) for i in self.list.selectedItems()]

    def select_entry(self, entry: ImageEntry) -> bool:
        for row in range(self.list.count()):
            candidate = self.list.item(row).data(Qt.UserRole)
            if candidate is not None and candidate.iid == entry.iid:
                self.list.setCurrentRow(row)
                return True
        return False

    def step(self, delta: int) -> None:
        if not self.list.count():
            return
        row = max(0, min(self.list.count() - 1, self.list.currentRow() + delta))
        self.list.setCurrentRow(row)

    def next_unannotated(self) -> bool:
        """Jump to the next image with no labels, wrapping around."""
        total = self.list.count()
        if not total:
            return False
        start = self.list.currentRow()
        for offset in range(1, total + 1):
            row = (start + offset) % total
            entry = self.list.item(row).data(Qt.UserRole)
            if entry is not None and not self._annotated.get(entry.iid, False):
                self.list.setCurrentRow(row)
                return True
        return False

    def _on_current(self, item: Optional[QListWidgetItem], _prev) -> None:
        if self._loading:
            return
        self.currentChanged.emit(item.data(Qt.UserRole) if item else None)

    def _list_key_press(self, event) -> None:
        if event.key() == Qt.Key_Delete:
            entries = self.selected_entries()
            if entries:
                self.removeRequested.emit(entries)
            event.accept()
            return
        QListWidget.keyPressEvent(self.list, event)
