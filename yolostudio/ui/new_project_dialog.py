"""Dialog for creating a project directory."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)


class NewProjectDialog(QDialog):

    def __init__(self, default_dir: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.setMinimumWidth(520)

        self.name = QLineEdit("my-dataset")
        self.location = QLineEdit(str(default_dir))
        browse = QPushButton("Browse…")
        self.task = QComboBox()
        self.task.addItem("Detect  —  bounding boxes", "detect")
        self.task.addItem("Segment  —  polygon masks", "segment")
        self.classes = QPlainTextEdit()
        self.classes.setPlaceholderText("One class name per line, e.g.\nperson\nhelmet\nvest")
        self.classes.setFixedHeight(110)

        hint = QLabel("The task sets the default export format. You can annotate boxes and "
                      "polygons in the same project either way, and change this later.")
        hint.setProperty("hint", True)
        hint.setWordWrap(True)

        path_row = QHBoxLayout()
        path_row.addWidget(self.location, 1)
        path_row.addWidget(browse)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.addRow("Project name", self.name)
        form.addRow("Location", path_row)
        form.addRow("Task", self.task)
        form.addRow("Classes", self.classes)
        form.addRow("", hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Create")
        buttons.button(QDialogButtonBox.Ok).setProperty("accent", True)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        browse.clicked.connect(self._browse)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Where should the project live?",
                                                self.location.text())
        if path:
            self.location.setText(path)

    def _accept(self) -> None:
        if not self.name.text().strip():
            QMessageBox.warning(self, "New project", "Give the project a name.")
            return
        parent = Path(self.location.text().strip() or ".")
        if not parent.is_dir():
            QMessageBox.warning(self, "New project", f"Not a folder:\n{parent}")
            return
        target = self.project_path()
        if target.exists() and any(target.iterdir()):
            answer = QMessageBox.question(
                self, "New project",
                f"{target} already exists and is not empty.\n\nUse it anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        self.accept()

    # ---------------------------------------------------------------- result

    def project_path(self) -> Path:
        return Path(self.location.text().strip()) / self.name.text().strip()

    def project_name(self) -> str:
        return self.name.text().strip()

    def project_task(self) -> str:
        return self.task.currentData()

    def class_names(self) -> List[str]:
        seen: List[str] = []
        for line in self.classes.toPlainText().splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.append(line)
        return seen
