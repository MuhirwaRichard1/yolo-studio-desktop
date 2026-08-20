"""Export a trained checkpoint to a deployment format."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from ..core import models as model_catalog
from ..core.project import Project
from ..core.runner import JobRunner
from .theme import BAD, GOOD, TEXT_DIM

FORMATS = [
    ("ONNX", "onnx", "Portable. Runs under onnxruntime on CPU or GPU."),
    ("TensorRT engine", "engine", "Fastest on your RTX card. Build takes several minutes "
                                  "and the file only works on this GPU + driver."),
    ("TorchScript", "torchscript", "Self-contained PyTorch graph, no Python needed."),
    ("OpenVINO", "openvino", "For Intel CPUs and iGPUs."),
]


class ExportModelDialog(QDialog):

    def __init__(self, project: Project, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Export trained model")
        self.setMinimumWidth(620)
        self._project = project
        self._runner = JobRunner(self)

        self.checkpoint = QComboBox()
        for path in model_catalog.find_checkpoints(project.runs_dir):
            self.checkpoint.addItem(model_catalog.describe_checkpoint(path), str(path))
        browse = QPushButton("Browse…")

        self.format = QComboBox()
        for label, key, _ in FORMATS:
            self.format.addItem(label, key)
        self.note = QLabel()
        self.note.setProperty("hint", True)
        self.note.setWordWrap(True)

        self.imgsz = QSpinBox(minimum=64, maximum=2048, value=640, singleStep=32)
        self.half = QCheckBox("FP16 (half precision)")
        self.half.setChecked(True)
        self.dynamic = QCheckBox("Dynamic input shape")
        self.simplify = QCheckBox("Simplify ONNX graph")
        self.simplify.setChecked(True)

        self.log = QPlainTextEdit(readOnly=True)
        self.log.setFixedHeight(150)
        self.log.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        self.status = QLabel("")
        self.status.setProperty("hint", True)

        row = QHBoxLayout()
        row.addWidget(self.checkpoint, 1)
        row.addWidget(browse)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.addRow("Checkpoint", row)
        form.addRow("Format", self.format)
        form.addRow("", self.note)
        form.addRow("Image size", self.imgsz)
        form.addRow("", self.half)
        form.addRow("", self.dynamic)
        form.addRow("", self.simplify)

        self.buttons = QDialogButtonBox()
        self.btn_run = self.buttons.addButton("Export", QDialogButtonBox.AcceptRole)
        self.btn_run.setProperty("accent", True)
        self.btn_close = self.buttons.addButton("Close", QDialogButtonBox.RejectRole)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.log)
        layout.addWidget(self.status)
        layout.addWidget(self.buttons)

        browse.clicked.connect(self._browse)
        self.format.currentIndexChanged.connect(self._update_note)
        self.btn_run.clicked.connect(self._run)
        self.btn_close.clicked.connect(self.reject)
        self._runner.log.connect(self.log.appendPlainText)
        self._runner.event.connect(self._on_event)
        self._runner.failed.connect(lambda msg, hint: self._set_status(hint or msg, BAD))
        self._runner.finished.connect(self._on_finished)
        self._update_note()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a .pt checkpoint",
                                              str(self._project.runs_dir),
                                              "PyTorch weights (*.pt)")
        if path:
            self.checkpoint.insertItem(0, path, path)
            self.checkpoint.setCurrentIndex(0)

    def _update_note(self) -> None:
        key = self.format.currentData()
        for _, candidate, blurb in FORMATS:
            if candidate == key:
                self.note.setText(blurb)
        self.simplify.setEnabled(key == "onnx")
        self.dynamic.setEnabled(key in ("onnx", "engine"))

    def _run(self) -> None:
        model = self.checkpoint.currentData()
        if not model or not Path(model).exists():
            QMessageBox.warning(self, "Export", "Choose a checkpoint to export.")
            return
        if self._runner.busy:
            return
        args = {
            "format": self.format.currentData(),
            "imgsz": self.imgsz.value(),
            "half": self.half.isChecked(),
        }
        if self.dynamic.isEnabled() and self.dynamic.isChecked():
            args["dynamic"] = True
            args["half"] = False  # ultralytics rejects dynamic+half together
        if self.simplify.isEnabled():
            args["simplify"] = self.simplify.isChecked()

        self.log.clear()
        self._set_status("Exporting… TensorRT builds can take several minutes.", TEXT_DIM)
        self.btn_run.setEnabled(False)
        self._runner.start({"command": "export", "model": model, "args": args},
                           workdir=self._project.root)

    def _on_event(self, message: dict) -> None:
        if message.get("event") == "result":
            summary = message.get("summary") or {}
            self._set_status(f"Wrote {summary.get('exported', '')}", GOOD)

    def _on_finished(self, ok: bool) -> None:
        self.btn_run.setEnabled(True)
        if not ok and not self.status.text():
            self._set_status("Export failed — see the log.", BAD)

    def _set_status(self, text: str, color: str) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color};")

    def reject(self) -> None:
        if self._runner.busy:
            self._runner.stop()
        super().reject()
