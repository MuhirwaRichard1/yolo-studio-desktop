"""Main window: docks, toolbar, shortcuts, and the annotate/train tabs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QActionGroup, QColor, QDesktopServices, QIcon,
                           QKeySequence, QPainter, QPixmap)
from PySide6.QtWidgets import (QAbstractItemView, QDockWidget, QFileDialog, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QTabWidget, QToolBar, QVBoxLayout, QWidget)

from ..core.project import PROJECT_FILE, Project
from ..core.runner import JobRunner
from .autolabel_dialog import AutoLabelDialog
from .canvas import MODE_BOX, MODE_POLYGON, MODE_SELECT, AnnotationCanvas
from .class_panel import ClassPanel
from .export_model_dialog import ExportModelDialog
from .image_list import ImageListPanel
from .new_project_dialog import NewProjectDialog
from .train_panel import TrainPanel
from .theme import TEXT_DIM

ORG = "YOLOStudio"
APP = "YOLOStudio"
MAX_RECENT = 8

SHORTCUT_HELP = """\
<h3>Keyboard</h3>
<table cellspacing="6">
<tr><td><b>V</b> / <b>W</b> / <b>E</b></td><td>Select / draw box / draw polygon</td></tr>
<tr><td><b>1</b> – <b>9</b></td><td>Set active class (also relabels the selection)</td></tr>
<tr><td><b>A</b> / <b>D</b></td><td>Previous / next image</td></tr>
<tr><td><b>Tab</b></td><td>Jump to next unannotated image</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save labels for this image</td></tr>
<tr><td><b>Del</b></td><td>Delete selected shapes</td></tr>
<tr><td><b>Ctrl+A</b></td><td>Select all shapes</td></tr>
<tr><td><b>Ctrl+Z</b> / <b>Ctrl+Y</b></td><td>Undo / redo</td></tr>
<tr><td><b>F</b> / <b>Ctrl+0</b></td><td>Fit to window / actual size</td></tr>
<tr><td><b>Esc</b></td><td>Cancel the shape being drawn</td></tr>
</table>
<h3>Mouse</h3>
<table cellspacing="6">
<tr><td><b>Wheel</b></td><td>Zoom at the cursor</td></tr>
<tr><td><b>Middle-drag</b> / <b>Space-drag</b></td><td>Pan</td></tr>
<tr><td><b>Double-click</b></td><td>Close a polygon, or add a vertex to one</td></tr>
<tr><td><b>Alt+click</b> a vertex</td><td>Remove that vertex</td></tr>
</table>
"""


def _color_icon(color: str) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawRoundedRect(1, 1, 10, 10, 2, 2)
    painter.end()
    return QIcon(pixmap)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Studio")
        self.resize(1500, 940)

        self._settings = QSettings(ORG, APP)
        self._project: Optional[Project] = None
        self._entry = None
        self._dirty = False
        self._sysinfo: Dict[str, Any] = {}
        self._probe = JobRunner(self)

        self._build_ui()
        self._build_actions()
        self._wire()
        self._restore_state()
        self._update_enabled()

        QTimer.singleShot(50, self._run_probe)
        QTimer.singleShot(120, self._reopen_last)

    # ----------------------------------------------------------------- build

    def _build_ui(self) -> None:
        self.canvas = AnnotationCanvas()
        self.train_panel = TrainPanel()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.canvas, "Annotate")
        self.tabs.addTab(self.train_panel, "Train")
        self.setCentralWidget(self.tabs)

        self.images = ImageListPanel()
        dock_images = QDockWidget("Images", self)
        dock_images.setObjectName("dock_images")
        dock_images.setWidget(self.images)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_images)

        self.classes = ClassPanel()
        dock_classes = QDockWidget("Classes", self)
        dock_classes.setObjectName("dock_classes")
        dock_classes.setWidget(self.classes)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_classes)

        self.shape_list = QListWidget()
        self.shape_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.shape_list.setUniformItemSizes(True)
        shapes_holder = QWidget()
        shapes_layout = QVBoxLayout(shapes_holder)
        shapes_layout.setContentsMargins(8, 8, 8, 8)
        shapes_layout.addWidget(self.shape_list)
        dock_shapes = QDockWidget("Annotations on this image", self)
        dock_shapes.setObjectName("dock_shapes")
        dock_shapes.setWidget(shapes_holder)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_shapes)
        self._docks = [dock_images, dock_classes, dock_shapes]

        self.status_image = QLabel("No image")
        self.status_mode = QLabel("")
        self.status_zoom = QLabel("")
        self.status_gpu = QLabel("checking GPU…")
        self.status_gpu.setStyleSheet(f"color: {TEXT_DIM};")
        bar = self.statusBar()
        bar.addWidget(self.status_image, 1)
        for widget in (self.status_mode, self.status_zoom, self.status_gpu):
            bar.addPermanentWidget(widget)

    def _build_actions(self) -> None:
        # ---- file --------------------------------------------------------
        self.act_new = QAction("&New project…", self, shortcut=QKeySequence.New)
        self.act_open = QAction("&Open project…", self, shortcut=QKeySequence.Open)
        self.act_save = QAction("&Save labels", self, shortcut=QKeySequence.Save)
        self.act_import_files = QAction("Import &images…", self)
        self.act_import_folder = QAction("Import &folder…", self, shortcut="Ctrl+Shift+O")
        self.act_reveal = QAction("Open project folder", self)
        self.act_quit = QAction("E&xit", self, shortcut="Ctrl+Q")

        # ---- edit --------------------------------------------------------
        self.act_undo = QAction("Undo", self, shortcut=QKeySequence.Undo)
        self.act_redo = QAction("Redo", self, shortcut="Ctrl+Y")
        self.act_delete = QAction("Delete selected", self, shortcut="Del")
        self.act_select_all = QAction("Select all shapes", self, shortcut=QKeySequence.SelectAll)
        self.act_prev = QAction("Previous image", self, shortcut="A")
        self.act_next = QAction("Next image", self, shortcut="D")
        self.act_next_todo = QAction("Next unannotated", self, shortcut="Tab")

        # ---- modes -------------------------------------------------------
        self.act_select = QAction("Select", self, checkable=True, shortcut="V")
        self.act_box = QAction("Box", self, checkable=True, shortcut="W")
        self.act_polygon = QAction("Polygon", self, checkable=True, shortcut="E")
        self.act_select.setChecked(True)
        modes = QActionGroup(self)
        for action in (self.act_select, self.act_box, self.act_polygon):
            modes.addAction(action)
        modes.setExclusive(True)

        # ---- view --------------------------------------------------------
        self.act_fit = QAction("Fit to window", self, shortcut="F")
        self.act_actual = QAction("Actual size", self, shortcut="Ctrl+0")

        # ---- tools -------------------------------------------------------
        self.act_autolabel = QAction("&Auto-label with a model…", self, shortcut="Ctrl+L")
        self.act_export_data = QAction("Export &dataset…", self, shortcut="Ctrl+E")
        self.act_export_model = QAction("Export trained &model…", self)
        self.act_goto_train = QAction("Go to &Training", self, shortcut="Ctrl+T")

        self.act_shortcuts = QAction("Keyboard && mouse", self, shortcut="F1")
        self.act_about = QAction("About", self)

        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(self.act_new)
        file_menu.addAction(self.act_open)
        self.recent_menu = file_menu.addMenu("Open &recent")
        file_menu.addSeparator()
        file_menu.addAction(self.act_import_files)
        file_menu.addAction(self.act_import_folder)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_reveal)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        edit_menu = menu.addMenu("&Edit")
        for action in (self.act_undo, self.act_redo, None, self.act_delete,
                       self.act_select_all, None, self.act_prev, self.act_next,
                       self.act_next_todo):
            edit_menu.addSeparator() if action is None else edit_menu.addAction(action)

        view_menu = menu.addMenu("&View")
        view_menu.addAction(self.act_select)
        view_menu.addAction(self.act_box)
        view_menu.addAction(self.act_polygon)
        view_menu.addSeparator()
        view_menu.addAction(self.act_fit)
        view_menu.addAction(self.act_actual)
        view_menu.addSeparator()
        for dock in self._docks:
            view_menu.addAction(dock.toggleViewAction())

        tools_menu = menu.addMenu("&Tools")
        tools_menu.addAction(self.act_autolabel)
        tools_menu.addSeparator()
        tools_menu.addAction(self.act_export_data)
        tools_menu.addAction(self.act_export_model)
        tools_menu.addSeparator()
        tools_menu.addAction(self.act_goto_train)

        help_menu = menu.addMenu("&Help")
        help_menu.addAction(self.act_shortcuts)
        help_menu.addAction(self.act_about)

        toolbar = QToolBar("Main")
        toolbar.setObjectName("toolbar_main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(self.act_select)
        toolbar.addAction(self.act_box)
        toolbar.addAction(self.act_polygon)
        toolbar.addSeparator()
        toolbar.addAction(self.act_prev)
        toolbar.addAction(self.act_next)
        toolbar.addAction(self.act_next_todo)
        toolbar.addSeparator()
        toolbar.addAction(self.act_fit)
        toolbar.addAction(self.act_actual)
        toolbar.addSeparator()
        toolbar.addAction(self.act_save)
        toolbar.addAction(self.act_autolabel)
        toolbar.addAction(self.act_export_data)
        toolbar.addAction(self.act_goto_train)

        # Single-letter and bare-key shortcuts are scoped to the canvas. As
        # window-wide shortcuts they would swallow ordinary typing: "a" in the
        # image filter box would jump to the previous image, and Ctrl+A would
        # select shapes instead of the text under the cursor.
        for action in (self.act_prev, self.act_next, self.act_next_todo,
                       self.act_select, self.act_box, self.act_polygon,
                       self.act_fit, self.act_actual, self.act_delete,
                       self.act_select_all, self.act_undo, self.act_redo):
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            self.canvas.addAction(action)

        # Class hotkeys, same reasoning: typing "3" in a filter box must not
        # silently relabel a shape.
        for index in range(9):
            action = QAction(self, shortcut=str(index + 1))
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.triggered.connect(lambda _=False, i=index: self._pick_class(i))
            self.canvas.addAction(action)

        self._rebuild_recent()

    def _wire(self) -> None:
        self.act_new.triggered.connect(self.new_project)
        self.act_open.triggered.connect(self.open_project_dialog)
        self.act_save.triggered.connect(lambda: self._save_current(force=True))
        self.act_import_files.triggered.connect(self.import_images)
        self.act_import_folder.triggered.connect(self.import_folder)
        self.act_reveal.triggered.connect(self._reveal_project)
        self.act_quit.triggered.connect(self.close)

        self.act_undo.triggered.connect(self.canvas.undo)
        self.act_redo.triggered.connect(self.canvas.redo)
        self.act_delete.triggered.connect(self.canvas.delete_selected)
        self.act_select_all.triggered.connect(self.canvas.select_all)
        self.act_prev.triggered.connect(lambda: self.images.step(-1))
        self.act_next.triggered.connect(lambda: self.images.step(1))
        self.act_next_todo.triggered.connect(self._next_unannotated)

        self.act_select.triggered.connect(lambda: self._set_mode(MODE_SELECT))
        self.act_box.triggered.connect(lambda: self._set_mode(MODE_BOX))
        self.act_polygon.triggered.connect(lambda: self._set_mode(MODE_POLYGON))
        self.act_fit.triggered.connect(self.canvas.fit_to_window)
        self.act_actual.triggered.connect(self.canvas.zoom_actual)

        self.act_autolabel.triggered.connect(self.auto_label)
        self.act_export_data.triggered.connect(self.export_dataset)
        self.act_export_model.triggered.connect(self.export_model)
        self.act_goto_train.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        self.act_shortcuts.triggered.connect(self._show_shortcuts)
        self.act_about.triggered.connect(self._show_about)

        self.images.currentChanged.connect(self._on_image_selected)
        self.images.removeRequested.connect(self._remove_images)
        self.classes.activeClassChanged.connect(self._on_active_class)
        self.classes.classesChanged.connect(self._on_classes_changed)

        self.canvas.shapesChanged.connect(self._on_shapes_changed)
        self.canvas.selectionChanged.connect(self._sync_shape_selection)
        self.canvas.statusMessage.connect(self._flash)
        self.canvas.zoomChanged.connect(
            lambda z: self.status_zoom.setText(f"{z * 100:.0f}%"))
        self.canvas.classNeeded.connect(self._prompt_first_class)
        self.shape_list.itemSelectionChanged.connect(self._select_from_list)

        self.train_panel.statusMessage.connect(self._flash)
        self.train_panel.runFinished.connect(self._on_run_finished)
        self._probe.event.connect(self._on_probe)

    # ------------------------------------------------------------- lifecycle

    def _run_probe(self) -> None:
        self._probe.start({"command": "probe"})

    def _on_probe(self, message: dict) -> None:
        if message.get("event") != "probe":
            return
        info = message.get("info") or {}
        self._sysinfo = info
        devices = info.get("devices") or []
        if info.get("torch_error"):
            self.status_gpu.setText("PyTorch not installed")
        elif devices:
            first = devices[0]
            self.status_gpu.setText(
                f"{first['name']} · {first['vram_gb']} GB · torch {info.get('torch', '?')}")
        elif info.get("cuda_available") is False:
            self.status_gpu.setText(f"CPU only · torch {info.get('torch', '?')}")
        self.train_panel.set_system_info(info)

    def device_items(self) -> List[tuple]:
        items = [(f"{d['index']}  —  {d['name']}", str(d["index"]))
                 for d in (self._sysinfo.get("devices") or [])]
        items.append(("cpu", "cpu"))
        return items

    def _restore_state(self) -> None:
        geometry = self._settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = self._settings.value("windowState")
        if state:
            self.restoreState(state)

    def _reopen_last(self) -> None:
        recent = self._recent()
        if recent and Path(recent[0], PROJECT_FILE).exists():
            self.open_project(Path(recent[0]))

    def closeEvent(self, event) -> None:
        if self.train_panel.busy:
            answer = QMessageBox.question(
                self, "Quit", "A training run is still going. Stop it and quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.train_panel.stop()
        self._save_current()
        if self._project is not None:
            self._project.save()
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
        super().closeEvent(event)

    # --------------------------------------------------------------- project

    def _recent(self) -> List[str]:
        return list(self._settings.value("recent", []) or [])

    def _push_recent(self, path: Path) -> None:
        entries = [p for p in self._recent() if Path(p) != path]
        entries.insert(0, str(path))
        self._settings.setValue("recent", entries[:MAX_RECENT])
        self._rebuild_recent()

    def _rebuild_recent(self) -> None:
        self.recent_menu.clear()
        entries = [p for p in self._recent() if Path(p, PROJECT_FILE).exists()]
        if not entries:
            action = self.recent_menu.addAction("Nothing yet")
            action.setEnabled(False)
            return
        for path in entries:
            action = self.recent_menu.addAction(path)
            action.triggered.connect(lambda _=False, p=path: self.open_project(Path(p)))

    def new_project(self) -> None:
        dialog = NewProjectDialog(Path.home(), self)
        if dialog.exec() != NewProjectDialog.Accepted:
            return
        try:
            project = Project.create(dialog.project_path(), dialog.project_name(),
                                     dialog.project_task(), dialog.class_names())
        except OSError as exc:
            QMessageBox.critical(self, "New project", f"Could not create the project:\n{exc}")
            return
        self._adopt(project)
        self._flash(f"Created {project.root}")
        if not project.images:
            self.import_folder()

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", str(Path.home()),
                                              f"YOLO Studio project ({PROJECT_FILE})")
        if path:
            self.open_project(Path(path).parent)

    def open_project(self, root: Path) -> None:
        try:
            project = Project.open(root)
        except Exception as exc:
            QMessageBox.critical(self, "Open project", f"Could not open {root}:\n{exc}")
            return
        self._adopt(project)
        self._flash(f"Opened {project.name}")

    def _adopt(self, project: Project) -> None:
        self._save_current()
        if self._project is not None:
            self._project.save()
        self._project = project
        self._entry = None
        self._dirty = False
        self.setWindowTitle(f"YOLO Studio — {project.name}")
        self.canvas.clear_all()
        self.canvas.set_classes(project.classes, project.colors)
        self.classes.set_project(project)
        self.images.set_project(project)
        self.train_panel.set_project(project)
        self._refresh_counts()
        self._push_recent(project.root)
        self._update_enabled()

    def _reveal_project(self) -> None:
        if self._project is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._project.root)))

    def _update_enabled(self) -> None:
        have = self._project is not None
        for action in (self.act_save, self.act_import_files, self.act_import_folder,
                       self.act_autolabel, self.act_export_data, self.act_export_model,
                       self.act_reveal):
            action.setEnabled(have)

    # ---------------------------------------------------------------- images

    def import_images(self) -> None:
        if self._project is None:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import images", str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if paths:
            self._after_import(self._project.add_images(Path(p) for p in paths))

    def import_folder(self) -> None:
        if self._project is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Import a folder of images",
                                                  str(Path.home()))
        if folder:
            self._after_import(self._project.add_folder(Path(folder)))

    def _after_import(self, added: list) -> None:
        if self._project is None:
            return
        self._project.save()
        self.images.set_project(self._project)
        self.train_panel.refresh_stats()
        self._flash(f"Imported {len(added)} image(s)" if added
                    else "No new images (already in the project, or unsupported types)")

    def _remove_images(self, entries: list) -> None:
        if self._project is None or not entries:
            return
        answer = QMessageBox.question(
            self, "Remove images",
            f"Remove {len(entries)} image(s) from the project?\n\n"
            "Their label files are deleted. The image files on disk are left alone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        if self._entry is not None and any(e.iid == self._entry.iid for e in entries):
            self._entry = None
            self._dirty = False
            self.canvas.clear_all()
        self._project.remove_images(entries)
        self._project.save()
        self.images.set_project(self._project)
        self._refresh_counts()

    def _on_image_selected(self, entry) -> None:
        self._save_current()
        self._entry = entry
        if entry is None or self._project is None:
            self.canvas.clear_all()
            self.status_image.setText("No image")
            self._refresh_shape_list()
            return
        if not entry.exists():
            self.canvas.clear_all()
            self.status_image.setText(f"{entry.name} — file is missing")
            self._refresh_shape_list()
            return
        if self.canvas.load_image(entry.file):
            self.canvas.set_classes(self._project.classes, self._project.colors)
            self.canvas.set_shapes(self._project.shapes_for(entry))
            self.canvas.set_active_class(self.classes.active_class)
            width, height = self.canvas.image_size
            self.status_image.setText(f"{entry.name}   {width}×{height}")
        self._dirty = False
        self._refresh_shape_list()

    def _next_unannotated(self) -> None:
        if not self.images.next_unannotated():
            self._flash("Every image in this view is annotated")

    # ---------------------------------------------------------------- labels

    def _save_current(self, force: bool = False) -> None:
        if self._project is None or self._entry is None:
            return
        if not (self._dirty or force):
            return
        if not self.canvas.has_image:
            return
        shapes = self.canvas.get_shapes()
        self._project.write_shapes(self._entry, shapes)
        self.images.mark(self._entry, len(shapes))
        self._dirty = False
        self._refresh_counts()
        if force:
            self._flash(f"Saved {len(shapes)} annotation(s)")

    def _on_shapes_changed(self) -> None:
        self._dirty = True
        self._refresh_shape_list()
        if self._entry is not None:
            self.images.mark(self._entry, self.canvas.count())

    def _refresh_counts(self) -> None:
        if self._project is None:
            return
        stats = self._project.stats()
        self.classes.set_counts(stats["per_class"])
        self.train_panel.refresh_stats()

    # ----------------------------------------------------------------- shape

    def _refresh_shape_list(self) -> None:
        self.shape_list.blockSignals(True)
        self.shape_list.clear()
        if self._project is not None:
            for index, shape in enumerate(self.canvas.get_shapes()):
                kind = "box" if shape.is_box else f"polygon ({len(shape.points())})"
                item = QListWidgetItem(_color_icon(self._project.color_for(shape.cls)),
                                       f"{index + 1}.  {self._project.class_name(shape.cls)}   {kind}")
                item.setData(Qt.UserRole, index)
                self.shape_list.addItem(item)
        self.shape_list.blockSignals(False)

    def _select_from_list(self) -> None:
        rows = {i.data(Qt.UserRole) for i in self.shape_list.selectedItems()}
        if not rows:
            return
        items = sorted(self.canvas._shape_items(), key=lambda i: (i.cls,))
        for index, item in enumerate(items):
            item.setSelected(index in rows)

    def _sync_shape_selection(self) -> None:
        selected = {id(i) for i in self.canvas.selected_items()}
        items = sorted(self.canvas._shape_items(), key=lambda i: (i.cls,))
        self.shape_list.blockSignals(True)
        for row in range(self.shape_list.count()):
            index = self.shape_list.item(row).data(Qt.UserRole)
            if 0 <= index < len(items):
                self.shape_list.item(row).setSelected(id(items[index]) in selected)
        self.shape_list.blockSignals(False)

    # --------------------------------------------------------------- classes

    def _pick_class(self, index: int) -> None:
        if self._project is None or index >= len(self._project.classes):
            return
        self.classes.select_class(index)
        changed = self.canvas.assign_class_to_selection(index)
        if changed:
            self._flash(f"Relabelled {changed} shape(s) as {self._project.classes[index]}")

    def _on_active_class(self, index: int) -> None:
        self.canvas.set_active_class(index)

    def _on_classes_changed(self) -> None:
        if self._project is None:
            return
        self._project.save()
        self.canvas.set_classes(self._project.classes, self._project.colors)
        if self._entry is not None:
            # A class delete rewrites label files underneath us; reload from disk.
            self.canvas.set_shapes(self._project.shapes_for(self._entry))
            self._dirty = False
        self.images.refresh_states()
        self._refresh_counts()
        self._refresh_shape_list()

    def _prompt_first_class(self) -> None:
        self._flash("Add a class before drawing")
        self.classes.add_class()
        self.canvas.set_classes(self._project.classes if self._project else [],
                                self._project.colors if self._project else [])

    # ----------------------------------------------------------------- tools

    def auto_label(self) -> None:
        if self._project is None:
            return
        self._save_current()
        dialog = AutoLabelDialog(self._project, self.images.selected_entries(),
                                 self.device_items(), self)
        dialog.labelsWritten.connect(self._after_autolabel)
        dialog.exec()

    def _after_autolabel(self) -> None:
        self.images.refresh_states()
        self._refresh_counts()
        if self._entry is not None and self._project is not None:
            self.canvas.set_shapes(self._project.shapes_for(self._entry))
            self._dirty = False
            self._refresh_shape_list()

    def export_dataset(self) -> None:
        if self._project is None:
            return
        self._save_current()
        self.tabs.setCurrentIndex(1)
        self._flash("Set the split in the Dataset box, then press Start training — "
                    "the dataset is exported automatically.")

    def export_model(self) -> None:
        if self._project is None:
            return
        ExportModelDialog(self._project, self).exec()

    def _on_run_finished(self, best: str) -> None:
        if best:
            self._flash(f"Best weights: {best}")

    # ------------------------------------------------------------------ misc

    def _set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        self.tabs.setCurrentIndex(0)
        self.canvas.setFocus()
        self.status_mode.setText({MODE_SELECT: "select",
                                  MODE_BOX: "drawing boxes",
                                  MODE_POLYGON: "drawing polygons"}[mode])

    def _flash(self, text: str) -> None:
        self.statusBar().showMessage(text, 4000)

    def _show_shortcuts(self) -> None:
        QMessageBox.information(self, "Keyboard & mouse", SHORTCUT_HELP)

    def _show_about(self) -> None:
        info = self._sysinfo
        devices = ", ".join(f"{d['name']} ({d['vram_gb']} GB)"
                            for d in (info.get("devices") or [])) or "none detected"
        QMessageBox.information(
            self, "About YOLO Studio",
            f"<h3>YOLO Studio</h3>"
            f"<p>Annotate images, then finetune a YOLO model on your own GPU.</p>"
            f"<p><b>Python</b> {info.get('python', '?')}<br>"
            f"<b>PyTorch</b> {info.get('torch', 'not installed')} "
            f"(CUDA {info.get('cuda_build', '—')})<br>"
            f"<b>Ultralytics</b> {info.get('ultralytics', 'not installed')}<br>"
            f"<b>GPU</b> {devices}</p>")
