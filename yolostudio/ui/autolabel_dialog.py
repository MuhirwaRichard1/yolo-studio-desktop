"""Model-assisted pre-labelling.

Runs a checkpoint over chosen images and writes YOLO label files the annotator
then corrects. The class-mapping table is the important part: a COCO-pretrained
model emits its own 80 class ids, which almost never line up with a project's
classes, so every model class must be explicitly mapped or skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox,
                               QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QProgressBar, QPushButton, QRadioButton,
                               QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from ..core import models as model_catalog
from ..core.project import ImageEntry, Project
from ..core.runner import JobRunner
from .theme import BAD, GOOD, TEXT_DIM

SKIP = -1


class AutoLabelDialog(QDialog):

    labelsWritten = Signal()

    def __init__(self, project: Project, selected: List[ImageEntry],
                 device_items: List[tuple], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Auto-label with a model")
        self.setMinimumSize(760, 640)
        self._project = project
        self._selected = list(selected)
        self._model_names: Dict[int, str] = {}
        self._model_task = ""
        self._runner = JobRunner(self)
        self._names_runner = JobRunner(self)

        self._build(device_items)
        self._wire()
        self._update_source_visibility()
        self._update_scope_note()

    # ----------------------------------------------------------------- build

    def _build(self, device_items: List[tuple]) -> None:
        # ---- model ------------------------------------------------------
        self.source = QComboBox()
        self.source.addItem("One of my trained runs", "run")
        self.source.addItem("Custom .pt file…", "file")
        self.source.addItem("Pretrained COCO model", "pretrained")

        self.run_pick = QComboBox()
        for path in model_catalog.find_checkpoints(self._project.runs_dir):
            self.run_pick.addItem(model_catalog.describe_checkpoint(path), str(path))
        if not self.run_pick.count():
            self.run_pick.addItem("No finished runs yet", "")

        self.pretrained = QComboBox()
        for info in model_catalog.CATALOG:
            self.pretrained.addItem(f"{info.label} {info.task}   ·   {info.blurb}", info.key)

        self.file_label = QLabel("—")
        self.file_label.setProperty("hint", True)
        self.file_label.setWordWrap(True)
        self.btn_browse = QPushButton("Browse…")
        self.btn_load = QPushButton("Load classes")

        model_box = QGroupBox("Model")
        model_form = QFormLayout(model_box)
        model_form.setLabelAlignment(Qt.AlignRight)
        model_form.addRow("Source", self.source)
        self._lbl_run = QLabel("Run")
        self._lbl_pre = QLabel("Checkpoint")
        self._lbl_file = QLabel("File")
        model_form.addRow(self._lbl_run, self.run_pick)
        model_form.addRow(self._lbl_pre, self.pretrained)
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(self.btn_browse)
        model_form.addRow(self._lbl_file, file_row)
        model_form.addRow("", self.btn_load)
        self._model_form = model_form

        # ---- scope ------------------------------------------------------
        self.scope_all = QRadioButton("All images in the project")
        self.scope_todo = QRadioButton("Only images with no labels yet")
        self.scope_sel = QRadioButton(f"Selected in the image list ({len(self._selected)})")
        self.scope_todo.setChecked(True)
        self.scope_sel.setEnabled(bool(self._selected))
        group = QButtonGroup(self)
        for button in (self.scope_all, self.scope_todo, self.scope_sel):
            group.addButton(button)
        self.overwrite = QCheckBox("Overwrite labels that already exist")
        self.scope_note = QLabel("")
        self.scope_note.setProperty("hint", True)

        scope_box = QGroupBox("Apply to")
        scope_layout = QVBoxLayout(scope_box)
        for widget in (self.scope_all, self.scope_todo, self.scope_sel,
                       self.overwrite, self.scope_note):
            scope_layout.addWidget(widget)

        # ---- inference --------------------------------------------------
        self.conf = QDoubleSpinBox(minimum=0.01, maximum=0.99, decimals=2, value=0.25)
        self.conf.setSingleStep(0.05)
        self.conf.setToolTip("Lower catches more objects but writes more false positives "
                             "for you to delete.")
        self.iou = QDoubleSpinBox(minimum=0.1, maximum=0.95, decimals=2, value=0.70)
        self.iou.setSingleStep(0.05)
        self.imgsz = QComboBox()
        for size in (320, 416, 512, 640, 768, 896, 1024, 1280):
            self.imgsz.addItem(str(size), size)
        self.imgsz.setCurrentText("640")
        self.max_det = QSpinBox(minimum=1, maximum=2000, value=300)
        self.device = QComboBox()
        for text, data in device_items:
            self.device.addItem(text, data)
        self.shape = QComboBox()
        self.shape.addItem("Bounding boxes", "box")
        self.shape.addItem("Polygons (segmentation models only)", "polygon")

        infer_box = QGroupBox("Inference")
        infer = QFormLayout(infer_box)
        infer.setLabelAlignment(Qt.AlignRight)
        infer.addRow("Confidence", self.conf)
        infer.addRow("NMS IoU", self.iou)
        infer.addRow("Image size", self.imgsz)
        infer.addRow("Max detections", self.max_det)
        infer.addRow("Device", self.device)
        infer.addRow("Write as", self.shape)

        # ---- mapping ----------------------------------------------------
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Model class", "Becomes project class"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)

        self.btn_automatch = QPushButton("Match by name")
        self.btn_skip_all = QPushButton("Skip all")
        self.map_note = QLabel("Load a model to see its classes.")
        self.map_note.setProperty("hint", True)

        map_box = QGroupBox("Class mapping")
        map_layout = QVBoxLayout(map_box)
        map_buttons = QHBoxLayout()
        map_buttons.addWidget(self.btn_automatch)
        map_buttons.addWidget(self.btn_skip_all)
        map_buttons.addStretch(1)
        map_layout.addWidget(self.table, 1)
        map_layout.addLayout(map_buttons)
        map_layout.addWidget(self.map_note)

        # ---- run --------------------------------------------------------
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFormat("idle")
        self.status = QLabel("")
        self.status.setProperty("hint", True)

        self.buttons = QDialogButtonBox()
        self.btn_run = self.buttons.addButton("Run", QDialogButtonBox.AcceptRole)
        self.btn_run.setProperty("accent", True)
        self.btn_cancel = self.buttons.addButton("Stop", QDialogButtonBox.DestructiveRole)
        self.btn_cancel.setEnabled(False)
        self.btn_close = self.buttons.addButton("Close", QDialogButtonBox.RejectRole)

        left = QVBoxLayout()
        left.addWidget(model_box)
        left.addWidget(scope_box)
        left.addWidget(infer_box)
        left.addStretch(1)

        columns = QHBoxLayout()
        columns.addLayout(left, 0)
        columns.addWidget(map_box, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(columns, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.buttons)

    def _wire(self) -> None:
        self.source.currentIndexChanged.connect(self._update_source_visibility)
        self.btn_browse.clicked.connect(self._browse)
        self.btn_load.clicked.connect(self._load_names)
        self.btn_automatch.clicked.connect(self._automatch)
        self.btn_skip_all.clicked.connect(lambda: self._set_all(SKIP))
        for button in (self.scope_all, self.scope_todo, self.scope_sel):
            button.toggled.connect(self._update_scope_note)
        self.overwrite.toggled.connect(self._update_scope_note)

        self.btn_run.clicked.connect(self._run)
        self.btn_cancel.clicked.connect(self._runner.stop)
        self.btn_close.clicked.connect(self.reject)

        self._names_runner.event.connect(self._on_names_event)
        self._names_runner.failed.connect(
            lambda msg, hint: self._set_status(f"Could not read model classes: {msg}", BAD))
        self._runner.event.connect(self._on_event)
        self._runner.failed.connect(lambda msg, hint: self._set_status(hint or msg, BAD))
        self._runner.finished.connect(self._on_finished)

    # ---------------------------------------------------------------- model

    def _update_source_visibility(self) -> None:
        mode = self.source.currentData()
        for widget in (self.run_pick, self._lbl_run):
            widget.setVisible(mode == "run")
        for widget in (self.pretrained, self._lbl_pre):
            widget.setVisible(mode == "pretrained")
        for widget in (self.file_label, self.btn_browse, self._lbl_file):
            widget.setVisible(mode == "file")

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a .pt checkpoint",
                                              str(self._project.runs_dir),
                                              "PyTorch weights (*.pt)")
        if path:
            self.file_label.setText(path)

    def _model(self) -> str:
        mode = self.source.currentData()
        if mode == "run":
            return self.run_pick.currentData() or ""
        if mode == "pretrained":
            return self.pretrained.currentData() or ""
        text = self.file_label.text().strip()
        return "" if text == "—" else text

    def _load_names(self) -> None:
        model = self._model()
        if not model:
            QMessageBox.warning(self, "Auto-label", "Choose a model first.")
            return
        if self._names_runner.busy:
            return
        self._set_status("Loading model classes…", TEXT_DIM)
        self.btn_load.setEnabled(False)
        self._names_runner.start({"command": "names", "model": model},
                                 workdir=self._project.root)

    def _on_names_event(self, message: dict) -> None:
        if message.get("event") == "names":
            self._model_names = {int(k): v for k, v in (message.get("names") or {}).items()}
            self._model_task = message.get("task", "")
            self._populate_table()
            self._automatch()
            if self._model_task == "segment":
                self.shape.setCurrentIndex(1)
            else:
                self.shape.setCurrentIndex(0)
            self._set_status(
                f"Loaded {len(self._model_names)} classes from a {self._model_task or 'YOLO'} model.",
                GOOD)
        if message.get("event") in ("done", "error"):
            self.btn_load.setEnabled(True)

    def _populate_table(self) -> None:
        self.table.setRowCount(0)
        for index in sorted(self._model_names):
            row = self.table.rowCount()
            self.table.insertRow(row)
            item = QTableWidgetItem(f"{index}  {self._model_names[index]}")
            item.setData(Qt.UserRole, index)
            self.table.setItem(row, 0, item)
            combo = QComboBox()
            combo.addItem("— skip —", SKIP)
            for cls, name in enumerate(self._project.classes):
                combo.addItem(f"{cls}  {name}", cls)
            self.table.setCellWidget(row, 1, combo)
        self.table.resizeRowsToContents()

    def _set_all(self, value: int) -> None:
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 1)
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)

    def _automatch(self) -> None:
        lookup = {name.strip().lower(): i for i, name in enumerate(self._project.classes)}
        matched = 0
        for row in range(self.table.rowCount()):
            model_index = self.table.item(row, 0).data(Qt.UserRole)
            name = self._model_names.get(model_index, "").strip().lower()
            target = lookup.get(name, SKIP)
            combo = self.table.cellWidget(row, 1)
            index = combo.findData(target)
            combo.setCurrentIndex(max(0, index))
            if target != SKIP:
                matched += 1
        if self.table.rowCount():
            self.map_note.setText(
                f"{matched} of {self.table.rowCount()} model classes matched by name. "
                "Unmatched classes are skipped — set them manually if you want them.")

    def _class_map(self) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for row in range(self.table.rowCount()):
            model_index = self.table.item(row, 0).data(Qt.UserRole)
            target = self.table.cellWidget(row, 1).currentData()
            if target is not None and target != SKIP:
                mapping[int(model_index)] = int(target)
        return mapping

    # ---------------------------------------------------------------- scope

    def _targets(self) -> List[ImageEntry]:
        if self.scope_sel.isChecked():
            pool = self._selected
        elif self.scope_todo.isChecked():
            pool = [e for e in self._project.images if not self._project.is_annotated(e)]
        else:
            pool = list(self._project.images)
        pool = [e for e in pool if e.exists()]
        if not self.overwrite.isChecked():
            pool = [e for e in pool if not self._project.is_annotated(e)]
        return pool

    def _update_scope_note(self) -> None:
        count = len(self._targets())
        self.scope_note.setText(f"{count} image(s) will be processed.")

    # ------------------------------------------------------------------ run

    def _run(self) -> None:
        if self._runner.busy:
            return
        model = self._model()
        if not model:
            QMessageBox.warning(self, "Auto-label", "Choose a model first.")
            return
        if self.source.currentData() != "pretrained" and not Path(model).exists():
            QMessageBox.warning(self, "Auto-label", f"Checkpoint not found:\n{model}")
            return
        mapping = self._class_map()
        if not mapping:
            QMessageBox.warning(
                self, "Auto-label",
                "No class mapping is set, so nothing would be written.\n\n"
                "Press 'Load classes', then map at least one model class to a project class.")
            return
        targets = self._targets()
        if not targets:
            QMessageBox.information(self, "Auto-label", "No images match the chosen scope.")
            return
        if self.shape.currentData() == "polygon" and self._model_task not in ("segment", ""):
            QMessageBox.warning(
                self, "Auto-label",
                "This model is a detector, so it cannot produce polygons.\n"
                "Switch 'Write as' to bounding boxes, or pick a -seg model.")
            return

        config = {
            "command": "predict",
            "model": model,
            "conf": self.conf.value(),
            "iou": self.iou.value(),
            "imgsz": int(self.imgsz.currentData()),
            "device": self.device.currentData() or "0",
            "max_det": self.max_det.value(),
            "shape": self.shape.currentData(),
            "class_map": {str(k): v for k, v in mapping.items()},
            "items": [{"image": e.path, "label": str(self._project.label_path(e))}
                      for e in targets],
        }
        self.progress.setValue(0)
        self.progress.setFormat("0 / %d" % len(targets))
        self._set_status("Running…", TEXT_DIM)
        if self._runner.start(config, workdir=self._project.root):
            self.btn_run.setEnabled(False)
            self.btn_cancel.setEnabled(True)

    def _on_event(self, message: dict) -> None:
        kind = message.get("event")
        if kind == "status":
            self._set_status(message.get("msg", ""), TEXT_DIM)
        elif kind == "download":
            done, total = int(message.get("done", 0)), int(message.get("total", 0))
            name = message.get("name", "weights")
            if total:
                self.progress.setValue(int(done / total * 100))
                self.progress.setFormat(f"downloading {name}  {done / total:.0%}")
        elif kind == "progress":
            done, total = int(message.get("done", 0)), max(1, int(message.get("total", 1)))
            self.progress.setValue(int(done / total * 100))
            self.progress.setFormat(f"{done} / {total}")
            found = message.get("found")
            name = message.get("name", "")
            if message.get("error"):
                self._set_status(f"{name}: {message['error']}", BAD)
            elif found is not None:
                self._set_status(f"{name} — {found} object(s)", TEXT_DIM)
        elif kind == "result":
            summary = message.get("summary") or {}
            self._set_status(" · ".join(f"{k}: {v}" for k, v in summary.items()), GOOD)

    def _on_finished(self, ok: bool) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setFormat("done" if ok else "stopped")
        self.labelsWritten.emit()
        self._update_scope_note()

    def _set_status(self, text: str, color: str) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color};")

    # ------------------------------------------------------------ lifecycle

    def reject(self) -> None:
        if self._runner.busy:
            answer = QMessageBox.question(
                self, "Auto-label", "A job is still running. Stop it and close?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
            self._runner.stop()
        super().reject()
