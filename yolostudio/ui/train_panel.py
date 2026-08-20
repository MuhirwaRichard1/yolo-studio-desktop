"""Training tab: pick a base model, set hyperparameters, watch it learn.

Pressing Start does three things in order: export a fresh dataset from the
project, write a worker config, and launch :mod:`yolostudio.worker` as a
subprocess. Progress arrives as protocol events and is mirrored into the chart,
the progress bar and the log.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                               QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..core import dataset as dataset_mod
from ..core import models as model_catalog
from ..core.project import Project
from ..core.runner import JobRunner
from .chart import MetricsChart
from .theme import BAD, GOOD, TEXT_DIM, WARN

LOSS_COLORS = {
    "train/box_loss": "#4f8cff",
    "train/seg_loss": "#42d4f4",
    "train/cls_loss": "#f58231",
    "train/dfl_loss": "#9a6324",
    "val/box_loss": "#8ab4ff",
    "val/cls_loss": "#ffb87a",
}
METRIC_COLORS = {
    "metrics/mAP50-95(B)": "#43c07a",
    "metrics/mAP50(B)": "#bfef45",
    "metrics/precision(B)": "#f032e6",
    "metrics/recall(B)": "#e6194b",
    "metrics/mAP50-95(M)": "#3cb44b",
}
SHORT = {
    "train/box_loss": "box", "train/seg_loss": "seg", "train/cls_loss": "cls",
    "train/dfl_loss": "dfl", "val/box_loss": "val box", "val/cls_loss": "val cls",
    "metrics/mAP50-95(B)": "mAP50-95", "metrics/mAP50(B)": "mAP50",
    "metrics/precision(B)": "precision", "metrics/recall(B)": "recall",
    "metrics/mAP50-95(M)": "mask mAP",
}
MAX_LOG_BLOCKS = 4000


class TrainPanel(QWidget):

    statusMessage = Signal(str)
    runFinished = Signal(str)        # path to best.pt, "" when the run failed

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._project: Optional[Project] = None
        self._sysinfo: Dict[str, Any] = {}
        self._vram_gb = 16.0
        self._runner = JobRunner(self)
        self._epoch_times: List[float] = []
        self._run_started = 0.0
        self._last_epoch_at = 0.0
        self._best_path = ""
        self._save_dir = ""

        self._build()
        self._wire()
        self._refresh_models()

    # ----------------------------------------------------------------- build

    def _build(self) -> None:
        # ---- model -----------------------------------------------------
        self.task = QComboBox()
        self.task.addItem("Detect  (bounding boxes)", "detect")
        self.task.addItem("Segment  (polygon masks)", "segment")

        self.source = QComboBox()
        self.source.addItem("Pretrained COCO checkpoint", "pretrained")
        self.source.addItem("Continue from one of my runs", "run")
        self.source.addItem("Custom .pt file…", "file")

        self.family = QComboBox()
        self.scale = QComboBox()
        self.model_note = QLabel()
        self.model_note.setProperty("hint", True)
        self.model_note.setWordWrap(True)

        self.run_pick = QComboBox()
        self.custom_path = QLabel("—")
        self.custom_path.setProperty("hint", True)
        self.custom_path.setWordWrap(True)
        self.btn_browse = QPushButton("Browse…")

        model_box = QGroupBox("Base model")
        model_form = QFormLayout(model_box)
        model_form.setLabelAlignment(Qt.AlignRight)
        model_form.addRow("Task", self.task)
        model_form.addRow("Start from", self.source)
        # Row labels are kept as real widgets so they can be hidden alongside
        # their field. QFormLayout.labelForField cannot find the label for a row
        # whose field is a layout, which the File row is.
        self._lbl_family = QLabel("Family")
        self._lbl_scale = QLabel("Size")
        self._lbl_run = QLabel("Run")
        self._lbl_file = QLabel("File")
        model_form.addRow(self._lbl_family, self.family)
        model_form.addRow(self._lbl_scale, self.scale)
        model_form.addRow("", self.model_note)
        model_form.addRow(self._lbl_run, self.run_pick)
        browse_row = QHBoxLayout()
        browse_row.addWidget(self.custom_path, 1)
        browse_row.addWidget(self.btn_browse)
        model_form.addRow(self._lbl_file, browse_row)
        self._model_form = model_form

        # ---- hyperparameters -------------------------------------------
        self.epochs = QSpinBox(minimum=1, maximum=10000, value=100)
        self.imgsz = QComboBox()
        for size in (320, 416, 512, 640, 768, 896, 1024, 1280):
            self.imgsz.addItem(str(size), size)
        self.imgsz.setCurrentText("640")
        self.batch = QSpinBox(minimum=-1, maximum=256, value=16)
        self.batch.setToolTip("-1 lets ultralytics pick a batch size that fills ~60% of VRAM.")
        self.btn_suggest = QPushButton("Suggest")
        self.device = QComboBox()
        self.device.setEditable(True)
        # Sensible defaults until the probe subprocess reports the real devices,
        # so a run started in the first second still gets a valid device string.
        self.device.addItem("0", "0")
        self.device.addItem("cpu", "cpu")
        self.workers = QSpinBox(minimum=0, maximum=32, value=8)
        self.patience = QSpinBox(minimum=0, maximum=1000, value=50)
        self.patience.setToolTip("Stop early after this many epochs with no improvement. 0 disables.")
        self.optimizer = QComboBox()
        self.optimizer.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RMSProp"])
        self.lr0 = QDoubleSpinBox(minimum=0.00001, maximum=1.0, decimals=5, value=0.01)
        self.lr0.setSingleStep(0.001)
        self.seed = QSpinBox(minimum=0, maximum=999999, value=0)
        self.freeze = QSpinBox(minimum=0, maximum=24, value=0)
        self.freeze.setToolTip("Freeze the first N layers. 10 is a common choice for small datasets.")
        self.amp = QCheckBox("Mixed precision (AMP)")
        self.amp.setChecked(True)
        self.cache = QCheckBox("Cache images in RAM")
        self.cos_lr = QCheckBox("Cosine LR schedule")

        hyper_box = QGroupBox("Training")
        hyper = QFormLayout(hyper_box)
        hyper.setLabelAlignment(Qt.AlignRight)
        hyper.addRow("Epochs", self.epochs)
        hyper.addRow("Image size", self.imgsz)
        batch_row = QHBoxLayout()
        batch_row.addWidget(self.batch, 1)
        batch_row.addWidget(self.btn_suggest)
        hyper.addRow("Batch", batch_row)
        hyper.addRow("Device", self.device)
        hyper.addRow("Dataloader workers", self.workers)
        hyper.addRow("Patience", self.patience)
        hyper.addRow("Optimizer", self.optimizer)
        hyper.addRow("Initial LR", self.lr0)
        hyper.addRow("Freeze layers", self.freeze)
        hyper.addRow("Seed", self.seed)
        hyper.addRow("", self.amp)
        hyper.addRow("", self.cache)
        hyper.addRow("", self.cos_lr)

        # ---- dataset ----------------------------------------------------
        self.val_split = QDoubleSpinBox(minimum=0.0, maximum=0.9, decimals=2, value=0.20)
        self.val_split.setSingleStep(0.05)
        self.test_split = QDoubleSpinBox(minimum=0.0, maximum=0.5, decimals=2, value=0.0)
        self.test_split.setSingleStep(0.05)
        self.include_bg = QCheckBox("Include unannotated images as background")
        self.reexport = QCheckBox("Re-export dataset before training")
        self.reexport.setChecked(True)
        self.dataset_note = QLabel("—")
        self.dataset_note.setProperty("hint", True)
        self.dataset_note.setWordWrap(True)

        data_box = QGroupBox("Dataset")
        data = QFormLayout(data_box)
        data.setLabelAlignment(Qt.AlignRight)
        data.addRow("Validation split", self.val_split)
        data.addRow("Test split", self.test_split)
        data.addRow("", self.include_bg)
        data.addRow("", self.reexport)
        data.addRow("", self.dataset_note)

        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(10)
        settings_layout.addWidget(model_box)
        settings_layout.addWidget(hyper_box)
        settings_layout.addWidget(data_box)
        settings_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(settings)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(460)

        # ---- run controls ------------------------------------------------
        self.btn_start = QPushButton("Start training")
        self.btn_start.setProperty("accent", True)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setProperty("danger", True)
        self.btn_stop.setEnabled(False)
        self.btn_folder = QPushButton("Open run folder")
        self.btn_folder.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("idle")
        self.headline = QLabel("Ready")
        self.headline.setProperty("heading", True)
        self.sub = QLabel("")
        self.sub.setProperty("hint", True)

        controls = QHBoxLayout()
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_folder)
        controls.addStretch(1)

        self.chart = MetricsChart()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(MAX_LOG_BLOCKS)
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log.setStyleSheet("font-family: Consolas, 'Cascadia Mono', monospace; font-size: 12px;")

        tabs = QTabWidget()
        tabs.addTab(self.chart, "Curves")
        tabs.addTab(self.log, "Log")

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)
        right_layout.addWidget(self.headline)
        right_layout.addWidget(self.sub)
        right_layout.addWidget(self.progress)
        right_layout.addLayout(controls)
        right_layout.addWidget(tabs, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

    def _wire(self) -> None:
        self.task.currentIndexChanged.connect(self._refresh_models)
        self.source.currentIndexChanged.connect(self._update_source_visibility)
        self.family.currentIndexChanged.connect(self._refresh_scales)
        self.scale.currentIndexChanged.connect(self._update_model_note)
        self.imgsz.currentIndexChanged.connect(self._update_model_note)
        self.btn_suggest.clicked.connect(self._apply_suggested_batch)
        self.btn_browse.clicked.connect(self._browse_weights)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self._runner.stop)
        self.btn_folder.clicked.connect(self._open_run_folder)

        self._runner.event.connect(self._on_event)
        self._runner.log.connect(self._append_log)
        self._runner.finished.connect(self._on_finished)
        self._runner.failed.connect(self._on_failed)

    # ------------------------------------------------------------------ data

    def set_project(self, project: Optional[Project]) -> None:
        self._project = project
        if project is None:
            self.dataset_note.setText("—")
            return
        cfg = project.train
        self.task.setCurrentIndex(0 if project.task != "segment" else 1)
        self.epochs.setValue(cfg.epochs)
        self.imgsz.setCurrentText(str(cfg.imgsz))
        self.batch.setValue(cfg.batch)
        self.workers.setValue(cfg.workers)
        self.patience.setValue(cfg.patience)
        self.optimizer.setCurrentText(cfg.optimizer)
        self.lr0.setValue(cfg.lr0)
        self.seed.setValue(cfg.seed)
        self.freeze.setValue(cfg.freeze)
        self.amp.setChecked(cfg.amp)
        self.cache.setChecked(cfg.cache)
        self.cos_lr.setChecked(cfg.cos_lr)
        self.val_split.setValue(cfg.val_split)
        self.test_split.setValue(cfg.test_split)
        self._refresh_models()
        self._select_model_key(cfg.model)
        self._refresh_runs()
        self.refresh_stats()

    def set_system_info(self, info: Dict[str, Any]) -> None:
        self._sysinfo = info or {}
        devices = self._sysinfo.get("devices") or []
        self.device.clear()
        if devices:
            for dev in devices:
                self.device.addItem(f"{dev['index']}  —  {dev['name']} ({dev['vram_gb']} GB)",
                                    str(dev["index"]))
            self._vram_gb = float(devices[0].get("vram_gb") or 16.0)
        self.device.addItem("cpu", "cpu")
        if not devices:
            self.device.setCurrentIndex(self.device.count() - 1)
            self.sub.setText("No CUDA device detected — training will run on CPU and be very slow.")
            self.sub.setStyleSheet(f"color: {WARN};")
        self._update_model_note()

    def refresh_stats(self) -> None:
        if self._project is None:
            return
        stats = self._project.stats()
        bits = [f"{stats['annotated']} annotated of {stats['images']} images",
                f"{stats['instances']} instances",
                f"{len(self._project.classes)} classes"]
        if stats["missing_files"]:
            bits.append(f"{stats['missing_files']} missing files")
        self.dataset_note.setText(" · ".join(bits))

    # ------------------------------------------------------------ model list

    def _current_task(self) -> str:
        return self.task.currentData()

    def _refresh_models(self) -> None:
        task = self._current_task()
        self.family.blockSignals(True)
        self.family.clear()
        for fam in model_catalog.families_for_task(task):
            self.family.addItem(fam, fam)
        self.family.blockSignals(False)
        self._refresh_scales()
        self._update_source_visibility()

    def _refresh_scales(self) -> None:
        task = self._current_task()
        fam = self.family.currentData()
        self.scale.blockSignals(True)
        self.scale.clear()
        for info in model_catalog.for_task(task):
            if info.family == fam:
                self.scale.addItem(f"{info.label}   ·   {info.blurb}", info.key)
        self.scale.blockSignals(False)
        if self.scale.count():
            self.scale.setCurrentIndex(min(1, self.scale.count() - 1))  # 's' by default
        self._update_model_note()

    def _select_model_key(self, key: str) -> None:
        info = model_catalog.get(key)
        if info is None:
            return
        self.task.setCurrentIndex(0 if info.task == "detect" else 1)
        self._refresh_models()
        index = self.family.findData(info.family)
        if index >= 0:
            self.family.setCurrentIndex(index)
        self._refresh_scales()
        index = self.scale.findData(key)
        if index >= 0:
            self.scale.setCurrentIndex(index)

    def _update_source_visibility(self) -> None:
        mode = self.source.currentData()
        pretrained = mode == "pretrained"
        for widget in (self.family, self.scale, self.model_note,
                       self._lbl_family, self._lbl_scale):
            widget.setVisible(pretrained)
        for widget in (self.run_pick, self._lbl_run):
            widget.setVisible(mode == "run")
        for widget in (self.custom_path, self.btn_browse, self._lbl_file):
            widget.setVisible(mode == "file")
        if mode == "run":
            self._refresh_runs()

    def _refresh_runs(self) -> None:
        self.run_pick.clear()
        if self._project is None:
            return
        for path in model_catalog.find_checkpoints(self._project.runs_dir):
            self.run_pick.addItem(model_catalog.describe_checkpoint(path), str(path))
        if not self.run_pick.count():
            self.run_pick.addItem("No finished runs yet", "")

    def _update_model_note(self) -> None:
        key = self.scale.currentData()
        info = model_catalog.get(key) if key else None
        if info is None:
            self.model_note.setText("")
            return
        imgsz = int(self.imgsz.currentData() or 640)
        suggested = model_catalog.suggest_batch(key, imgsz, self._vram_gb)
        note = model_catalog.FAMILY_NOTES.get(info.family, "")
        self.model_note.setText(
            f"{note}\nSuggested batch at {imgsz}px on {self._vram_gb:.0f} GB: {suggested}")

    def _apply_suggested_batch(self) -> None:
        key = self._resolve_catalog_key()
        imgsz = int(self.imgsz.currentData() or 640)
        self.batch.setValue(model_catalog.suggest_batch(key, imgsz, self._vram_gb))

    def _resolve_catalog_key(self) -> str:
        if self.source.currentData() == "pretrained":
            return self.scale.currentData() or "yolo11s.pt"
        return "yolo11s.pt"

    def _browse_weights(self) -> None:
        start = str(self._project.runs_dir) if self._project else ""
        path, _ = QFileDialog.getOpenFileName(self, "Choose a .pt checkpoint", start,
                                              "PyTorch weights (*.pt)")
        if path:
            self.custom_path.setText(path)

    def _resolve_model(self) -> str:
        mode = self.source.currentData()
        if mode == "pretrained":
            return self.scale.currentData() or ""
        if mode == "run":
            return self.run_pick.currentData() or ""
        text = self.custom_path.text().strip()
        return "" if text == "—" else text

    # ----------------------------------------------------------------- start

    def _persist_config(self, model: str) -> None:
        if self._project is None:
            return
        cfg = self._project.train
        cfg.model = model
        cfg.epochs = self.epochs.value()
        cfg.imgsz = int(self.imgsz.currentData())
        cfg.batch = self.batch.value()
        cfg.device = self.device.currentData() or self.device.currentText()
        cfg.workers = self.workers.value()
        cfg.patience = self.patience.value()
        cfg.optimizer = self.optimizer.currentText()
        cfg.lr0 = self.lr0.value()
        cfg.seed = self.seed.value()
        cfg.freeze = self.freeze.value()
        cfg.amp = self.amp.isChecked()
        cfg.cache = self.cache.isChecked()
        cfg.cos_lr = self.cos_lr.isChecked()
        cfg.val_split = self.val_split.value()
        cfg.test_split = self.test_split.value()
        self._project.task = self._current_task()
        self._project.save()

    def start(self) -> None:
        if self._project is None:
            QMessageBox.information(self, "Train", "Open a project first.")
            return
        if self._runner.busy:
            return

        model = self._resolve_model()
        if not model:
            QMessageBox.warning(self, "Train", "Choose a base model to finetune.")
            return
        if self.source.currentData() != "pretrained" and not Path(model).exists():
            QMessageBox.warning(self, "Train", f"Checkpoint not found:\n{model}")
            return

        task = self._current_task()
        stats = self._project.stats()
        if not self._project.classes:
            QMessageBox.warning(self, "Train", "Add at least one class first.")
            return
        if stats["annotated"] < 2:
            QMessageBox.warning(self, "Train",
                                "Annotate at least a couple of images before training.")
            return
        if task == "segment" and stats["polygons"] == 0:
            answer = QMessageBox.question(
                self, "Train",
                "This project has no polygon annotations. Boxes will be converted to "
                "rectangular masks, which usually trains a poor segmenter.\n\nContinue anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return

        self._persist_config(model)

        # Export the dataset the run will read.
        try:
            if self.reexport.isChecked() or not (
                    self._project.datasets_dir / "train_set" / "data.yaml").exists():
                self.statusMessage.emit("Exporting dataset…")
                result = dataset_mod.export(
                    self._project, task=task, name="train_set",
                    val_split=self.val_split.value(), test_split=self.test_split.value(),
                    seed=self.seed.value(),
                    include_unannotated=self.include_bg.isChecked())
                self._append_log(
                    f"Dataset: train={result.counts['train']} val={result.counts['val']} "
                    f"test={result.counts['test']} · {result.instances} instances · "
                    f"{result.linked} linked, {result.copied} copied\n{result.yaml_path}\n\n")
                data_yaml = result.yaml_path
            else:
                data_yaml = self._project.datasets_dir / "train_set" / "data.yaml"
        except ValueError as exc:
            QMessageBox.warning(self, "Dataset", str(exc))
            return

        # 'task' is deliberately not passed: it is inherited from the checkpoint.
        # Overriding it conflicts with the loaded model and raises before the
        # first epoch. Detect training reads polygon labels fine either way.
        args: Dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": self.epochs.value(),
            "imgsz": int(self.imgsz.currentData()),
            "batch": self.batch.value(),
            "device": self.device.currentData() or self.device.currentText(),
            "workers": self.workers.value(),
            "patience": self.patience.value(),
            "optimizer": self.optimizer.currentText(),
            "lr0": self.lr0.value(),
            "seed": self.seed.value(),
            "amp": self.amp.isChecked(),
            "cache": self.cache.isChecked(),
            "cos_lr": self.cos_lr.isChecked(),
            "project": str(self._project.runs_dir),
            "name": f"{task}_{time.strftime('%Y%m%d-%H%M%S')}",
            "exist_ok": True,
            "plots": True,
            "val": True,
        }
        if self.freeze.value() > 0:
            args["freeze"] = self.freeze.value()

        self._prepare_chart(task)
        self.log.clear()
        self._epoch_times.clear()
        self._run_started = self._last_epoch_at = time.monotonic()
        self._best_path = ""
        self._save_dir = ""
        self.progress.setValue(0)
        self.progress.setFormat("starting…")
        self.headline.setText("Starting…")
        self.sub.setStyleSheet("")
        self.sub.setText(f"{Path(model).name} → {args['name']}")
        self._append_log(f"$ yolo {task} train model={model} "
                         + " ".join(f"{k}={v}" for k, v in args.items()) + "\n\n")

        started = self._runner.start({"command": "train", "model": model, "args": args},
                                     workdir=self._project.root)
        if started:
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_folder.setEnabled(False)
            self.statusMessage.emit("Training started")

    def _prepare_chart(self, task: str) -> None:
        self.chart.reset_series()
        keys = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
        if task == "segment":
            keys.insert(1, "train/seg_loss")
        for key in keys:
            self.chart.add_series(key, SHORT.get(key, key), LOSS_COLORS.get(key, "#888"), "left")
        metric_keys = ["metrics/mAP50-95(B)", "metrics/mAP50(B)"]
        if task == "segment":
            metric_keys.append("metrics/mAP50-95(M)")
        for key in metric_keys:
            self.chart.add_series(key, SHORT.get(key, key), METRIC_COLORS.get(key, "#43c07a"),
                                  "right")
        self.chart.clear()

    # ---------------------------------------------------------------- events

    def _on_event(self, message: Dict[str, Any]) -> None:
        kind = message.get("event")
        if kind == "status":
            note = message.get("msg", "")
            self.sub.setText(note)
            self._append_log(note + "\n")
        elif kind == "download":
            done = int(message.get("done", 0))
            total = int(message.get("total", 0))
            name = message.get("name", "weights")
            if total:
                self.progress.setValue(int(done / total * 100))
                self.progress.setFormat(f"downloading {name}  {done / total:.0%}")
            else:
                self.progress.setFormat(f"downloading {name}  {done / 1e6:.1f} MB")
            self.headline.setText("Fetching base model")
        elif kind == "train_start":
            self._save_dir = message.get("save_dir", "")
            self.headline.setText("Training")
        elif kind == "epoch":
            self._on_epoch(message)
        elif kind == "train_end":
            self._best_path = message.get("best") or ""
            self._save_dir = message.get("save_dir") or self._save_dir
        elif kind == "result":
            summary = message.get("summary") or {}
            if summary:
                text = " · ".join(f"{k} {v}" for k, v in summary.items() if v is not None)
                self.sub.setText(text)
                self._append_log("\n" + text + "\n")

    def _on_epoch(self, message: Dict[str, Any]) -> None:
        epoch = int(message.get("epoch", 0))
        total = max(1, int(message.get("epochs", 1)))
        metrics = message.get("metrics") or {}
        self.chart.append(epoch, metrics)

        now = time.monotonic()
        self._epoch_times.append(now - self._last_epoch_at)
        self._last_epoch_at = now
        recent = self._epoch_times[-8:]
        per_epoch = sum(recent) / len(recent)
        remaining = max(0, total - epoch) * per_epoch

        self.progress.setValue(int(epoch / total * 100))
        self.progress.setFormat(f"epoch {epoch}/{total}  —  {_hms(remaining)} left")
        self.headline.setText(f"Epoch {epoch} of {total}")

        parts = []
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/mAP50-95(M)"):
            if key in metrics:
                parts.append(f"{SHORT[key]} {metrics[key]:.3f}")
        vram = message.get("vram_gb")
        if vram:
            parts.append(f"{vram} GB VRAM")
        parts.append(f"{per_epoch:.1f}s/epoch")
        self.sub.setText(" · ".join(parts))

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)

    def _on_failed(self, message: str, hint: str) -> None:
        self.headline.setText("Training failed")
        self.sub.setStyleSheet(f"color: {BAD};")
        self.sub.setText(hint or message)
        self._append_log(f"\nERROR: {message}\n" + (hint + "\n" if hint else ""))

    def _on_finished(self, ok: bool) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_folder.setEnabled(bool(self._save_dir))
        elapsed = _hms(time.monotonic() - self._run_started)
        if ok:
            self.progress.setValue(100)
            self.progress.setFormat(f"done in {elapsed}")
            self.headline.setText("Training complete")
            self.sub.setStyleSheet(f"color: {GOOD};")
            if self._best_path:
                self._append_log(f"\nBest weights: {self._best_path}\n")
            self.statusMessage.emit(f"Training finished in {elapsed}")
            self.runFinished.emit(self._best_path)
        else:
            self.progress.setFormat("stopped")
            if self.headline.text() != "Training failed":
                self.headline.setText("Stopped")
            self.statusMessage.emit("Training stopped")
            self.runFinished.emit("")
        self._refresh_runs()

    def _open_run_folder(self) -> None:
        target = self._save_dir or (str(self._project.runs_dir) if self._project else "")
        if target:
            QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    # ------------------------------------------------------------- lifecycle

    @property
    def busy(self) -> bool:
        return self._runner.busy

    def stop(self) -> None:
        self._runner.stop()


def _hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
