"""Class list: the active label for new shapes, plus class CRUD.

Rows 1-9 are bound to the number keys, which is how most annotation actually
gets done -- draw, press 2, draw, press 1.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QColorDialog, QHBoxLayout, QInputDialog, QLabel,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from ..core.project import Project


def _swatch(color: str) -> QIcon:
    pixmap = QPixmap(14, 14)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawRoundedRect(1, 1, 12, 12, 3, 3)
    painter.end()
    return QIcon(pixmap)


class ClassPanel(QWidget):

    activeClassChanged = Signal(int)
    classesChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._project: Optional[Project] = None
        self._counts: List[int] = []

        self.list = QListWidget()
        self.list.setUniformItemSizes(True)

        self.btn_add = QPushButton("Add")
        self.btn_rename = QPushButton("Rename")
        self.btn_color = QPushButton("Colour")
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setProperty("danger", True)

        hint = QLabel("Press 1-9 to switch class · Enter renames")
        hint.setProperty("hint", True)
        hint.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(4)
        for button in (self.btn_add, self.btn_rename, self.btn_color, self.btn_delete):
            buttons.addWidget(button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.list, 1)
        layout.addLayout(buttons)
        layout.addWidget(hint)

        self.btn_add.clicked.connect(self.add_class)
        self.btn_rename.clicked.connect(self.rename_current)
        self.btn_color.clicked.connect(self.recolor_current)
        self.btn_delete.clicked.connect(self.delete_current)
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.list.itemDoubleClicked.connect(lambda _: self.rename_current())

    # ------------------------------------------------------------------ data

    def set_project(self, project: Optional[Project]) -> None:
        self._project = project
        self.refresh()

    def set_counts(self, counts: List[int]) -> None:
        self._counts = list(counts)
        self.refresh(keep_current=True)

    def refresh(self, keep_current: bool = False) -> None:
        row = self.list.currentRow() if keep_current else 0
        self.list.blockSignals(True)
        self.list.clear()
        if self._project is not None:
            for index, name in enumerate(self._project.classes):
                item = QListWidgetItem(_swatch(self._project.color_for(index)), "")
                count = self._counts[index] if index < len(self._counts) else 0
                key = f"{index + 1}  " if index < 9 else "    "
                item.setText(f"{key}{name}" + (f"   ({count})" if count else ""))
                item.setData(Qt.UserRole, index)
                self.list.addItem(item)
        self.list.blockSignals(False)
        if self.list.count():
            self.list.setCurrentRow(max(0, min(row, self.list.count() - 1)))

    @property
    def active_class(self) -> int:
        return max(0, self.list.currentRow())

    def select_class(self, index: int) -> None:
        if 0 <= index < self.list.count():
            self.list.setCurrentRow(index)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.activeClassChanged.emit(row)

    # --------------------------------------------------------------- actions

    def add_class(self) -> None:
        if self._project is None:
            return
        name, ok = QInputDialog.getText(self, "Add class", "Class name:")
        if not ok or not name.strip():
            return
        try:
            index = self._project.add_class(name)
        except ValueError as exc:
            QMessageBox.warning(self, "Add class", str(exc))
            return
        self.refresh()
        self.select_class(index)
        self.classesChanged.emit()

    def rename_current(self) -> None:
        if self._project is None or self.list.currentRow() < 0:
            return
        index = self.list.currentRow()
        current = self._project.classes[index]
        name, ok = QInputDialog.getText(self, "Rename class", "Class name:", text=current)
        if not ok or name.strip() == current:
            return
        try:
            self._project.rename_class(index, name)
        except ValueError as exc:
            QMessageBox.warning(self, "Rename class", str(exc))
            return
        self.refresh(keep_current=True)
        self.classesChanged.emit()

    def recolor_current(self) -> None:
        if self._project is None or self.list.currentRow() < 0:
            return
        index = self.list.currentRow()
        chosen = QColorDialog.getColor(QColor(self._project.color_for(index)), self,
                                       "Class colour")
        if not chosen.isValid():
            return
        self._project.set_color(index, chosen.name())
        self.refresh(keep_current=True)
        self.classesChanged.emit()

    def delete_current(self) -> None:
        if self._project is None or self.list.currentRow() < 0:
            return
        index = self.list.currentRow()
        name = self._project.classes[index]
        count = self._counts[index] if index < len(self._counts) else 0
        detail = (f"\n\n{count} existing annotation(s) of this class will be deleted, "
                  "and the remaining class ids will be renumbered.") if count else ""
        answer = QMessageBox.question(
            self, "Delete class",
            f"Delete the class {name!r}?{detail}\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        self._project.delete_class(index)
        self.refresh()
        self.classesChanged.emit()
