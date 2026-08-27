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
from ..core import npu
from ..core.project import Project
from ..core.runner import JobRunner
from .theme import BAD, GOOD, TEXT_DIM

FORMATS = [
    ("ONNX", "onnx", "Portable. Runs under onnxruntime on CPU or GPU."),
    ("TensorRT engine", "engine", "Fastest on your RTX card. Build takes several minutes "
                                  "and the file only works on this GPU + driver."),
    ("TorchScript", "torchscript", "Self-contained PyTorch graph, no Python needed."),
    ("OpenVINO", "openvino", "For Intel CPUs and iGPUs."),
    ("RKNN (Radxa / Rockchip)", "rknn",
     "For the NPU on Radxa boards. Converted inside WSL — rknn-toolkit2 has no "
     "Windows build. The first run downloads about 1 GB of toolchain."),
    ("D-Robotics .bin (RDK)", "horizon",
     "For the BPU on RDK boards. Runs D-Robotics' OpenExplorer toolchain in "
     "Docker inside WSL; the image must already be loaded there."),
]

# Formats handled by the WSL bridge rather than by ultralytics on Windows.
NPU_FORMATS = {"rknn", "horizon"}


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

        # ---- SBC NPU controls, shown only for the WSL-backed formats --------
        self.distro = QComboBox()
        for distro in npu.usable_distros():
            label = f"{distro.name} (default)" if distro.default else distro.name
            self.distro.addItem(label, distro.name)
        if not self.distro.count():
            self.distro.addItem("no WSL 2 distribution found", "")

        self.chip = QComboBox()
        for chip in npu.RKNN_CHIPS:
            suffix = "  — INT8 only" if chip in npu.RKNN_INT8_ONLY else ""
            self.chip.addItem(f"{chip}{suffix}", chip)

        self.board = QComboBox()
        for board, march in npu.HORIZON_MARCH.items():
            self.board.addItem(f"{board}  ({march})", board)

        self.quantize = QComboBox()
        self.quantize.addItem("INT8 — smallest and fastest on NPU", 8)
        self.quantize.addItem("FP16 — larger, no calibration needed", 16)

        self.calib = QSpinBox(minimum=8, maximum=1000, value=50, singleStep=10)

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
        form.addRow("WSL distribution", self.distro)
        form.addRow("Target chip", self.chip)
        form.addRow("Board", self.board)
        form.addRow("Precision", self.quantize)
        form.addRow("Calibration images", self.calib)
        self._form = form

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
        self.quantize.currentIndexChanged.connect(self._update_note)
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

        npu_target = key in NPU_FORMATS
        # The NPU toolchains pick their own precision and always emit a static
        # graph, so the desktop-format switches would be misleading here.
        for widget in (self.half, self.dynamic, self.simplify):
            self._form.setRowVisible(widget, not npu_target)
        self._form.setRowVisible(self.distro, npu_target)
        self._form.setRowVisible(self.chip, key == "rknn")
        self._form.setRowVisible(self.board, key == "horizon")
        # D-Robotics quantizes unconditionally, so precision is not a choice.
        self._form.setRowVisible(self.quantize, key == "rknn")
        # Only the D-Robotics path writes its own calibration set. RKNN hands
        # the dataset to ultralytics, which calibrates over the whole val split
        # and ignores any count we pass, so offering the spinbox there would be
        # a control that quietly does nothing.
        self._form.setRowVisible(self.calib, key == "horizon")

        if npu_target and not self.distro.currentData():
            self.note.setText(self.note.text() + "\n\nNo WSL 2 distribution was found. "
                                                 "Install one with:  wsl --install -d Ubuntu")
        self.adjustSize()

    def _needs_calibration(self, key: str) -> bool:
        if key == "horizon":
            return True                       # OpenExplorer always calibrates
        return key == "rknn" and self.quantize.currentData() == 8

    def _dataset_yaml(self) -> str:
        """The exported dataset descriptor used for INT8 calibration."""
        candidate = self._project.datasets_dir / "data.yaml"
        return str(candidate) if candidate.exists() else ""

    def _run(self) -> None:
        model = self.checkpoint.currentData()
        if not model or not Path(model).exists():
            QMessageBox.warning(self, "Export", "Choose a checkpoint to export.")
            return
        if self._runner.busy:
            return

        key = self.format.currentData()
        if key in NPU_FORMATS:
            self._run_npu(model, key)
            return

        args = {
            "format": key,
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

    def _run_npu(self, model: str, key: str) -> None:
        """Start a WSL-hosted conversion for one of the SBC NPU targets."""
        distro = self.distro.currentData()
        if not distro:
            QMessageBox.warning(
                self, "Export",
                "No WSL 2 distribution is installed, and these toolchains only "
                "exist for Linux.\n\nInstall one with:\n    wsl --install -d Ubuntu")
            return

        data_yaml = ""
        if self._needs_calibration(key):
            data_yaml = self._dataset_yaml()
            if not data_yaml:
                QMessageBox.warning(
                    self, "Export",
                    "Quantizing needs calibration images, and this project has no "
                    "exported dataset yet.\n\nRun Tools ▸ Export dataset first, or "
                    "pick the FP16 build.")
                return

        args = {
            "target": key,
            "distro": distro,
            "imgsz": self.imgsz.value(),
            "data": data_yaml,
            "calibration_images": self.calib.value(),
        }
        if key == "rknn":
            args["chip"] = self.chip.currentData()
            args["quantize"] = self.quantize.currentData()
        else:
            args["board"] = self.board.currentData()
            args["quantize"] = 8

        self.log.clear()
        self._set_status(
            "Converting in WSL… the first run installs the toolchain and can "
            "take 10 minutes.", TEXT_DIM)
        self.btn_run.setEnabled(False)
        self._runner.start({"command": "export_npu", "model": model, "args": args},
                           workdir=self._project.root)

    # What each stage of a WSL conversion is doing, so a ten-minute first run
    # does not look like a hang.
    PHASES = {
        "provision": "Building the Linux toolchain inside WSL (first run only)…",
        "check": "Checking the toolchain container…",
        "onnx": "Exporting ONNX as the conversion input…",
        "calibration": "Preparing calibration images…",
        "convert": "Converting for the NPU…",
    }

    def _on_event(self, message: dict) -> None:
        event = message.get("event")
        if event == "phase":
            text = self.PHASES.get(message.get("name", ""))
            if text:
                self._set_status(text, TEXT_DIM)
        elif event == "result":
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
