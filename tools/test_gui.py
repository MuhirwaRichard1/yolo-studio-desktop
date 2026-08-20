"""Headless construction and interaction test for the Qt layer.

Builds the real MainWindow on the offscreen platform, drives a small
annotation session through the canvas API, and checks the shapes come back out
correctly. Catches Qt API mistakes that a syntax check cannot::

    .\\.venv\\Scripts\\python.exe tools\\test_gui.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF, QRectF, Qt                       # noqa: E402
from PySide6.QtGui import QColor, QImage                             # noqa: E402
from PySide6.QtWidgets import QApplication                           # noqa: E402

from yolostudio.core.annotations import BOX, POLYGON, Shape          # noqa: E402
from yolostudio.core.project import Project                          # noqa: E402
from yolostudio.ui.canvas import MODE_BOX, MODE_POLYGON, MODE_SELECT  # noqa: E402
from yolostudio.ui.main_window import MainWindow                     # noqa: E402
from yolostudio.ui.shapes import BoxItem, PolygonItem                # noqa: E402

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) < tol


def make_project(tmp: Path) -> Project:
    """A project with three real 640x480 images on disk."""
    src = tmp / "images"
    src.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        image = QImage(640, 480, QImage.Format_RGB32)
        image.fill(QColor(40 + index * 30, 60, 90))
        image.save(str(src / f"shot_{index}.png"))
    project = Project.create(tmp / "proj", "gui", "detect", ["person", "helmet"])
    project.add_images(sorted(src.glob("*.png")))
    project.save()
    return project


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="yolostudio-gui-"))
    app = QApplication.instance() or QApplication([])
    try:
        print("Window construction")
        window = MainWindow()
        check("MainWindow builds", window is not None)
        check("two central tabs", window.tabs.count() == 2, window.tabs.tabText(1))
        check("three docks", len(window._docks) == 3)
        check("train panel model catalogue populated", window.train_panel.scale.count() > 0,
              f"{window.train_panel.scale.count()} entries")

        print("\nProject load")
        project = make_project(tmp)
        window._adopt(project)
        check("images listed", window.images.list.count() == 3,
              str(window.images.list.count()))
        check("classes listed", window.classes.list.count() == 2)
        check("first image selected", window.images.current_entry() is not None)
        check("canvas has the image", window.canvas.has_image)
        check("image size read", window.canvas.image_size == (640, 480),
              str(window.canvas.image_size))

        print("\nDrawing")
        canvas = window.canvas
        canvas.set_mode(MODE_SELECT)
        canvas.set_active_class(0)
        # Add shapes through the same path the mouse handlers use.
        box = BoxItem(QRectF(64, 48, 128, 96), 0, QColor("#e6194b"), "person")
        canvas._scene.addItem(box)
        poly = PolygonItem([QPointF(320, 240), QPointF(480, 240), QPointF(400, 400)],
                           1, QColor("#3cb44b"), "helmet")
        canvas._scene.addItem(poly)
        canvas.shapesChanged.emit()
        check("canvas counts both shapes", canvas.count() == 2, str(canvas.count()))

        shapes = canvas.get_shapes()
        check("two shapes exported", len(shapes) == 2)
        by_kind = {s.kind: s for s in shapes}
        check("box normalised cx", approx(by_kind[BOX].coords[0], (64 + 64) / 640),
              f"{by_kind[BOX].coords[0]:.4f}")
        check("box normalised w", approx(by_kind[BOX].coords[2], 128 / 640),
              f"{by_kind[BOX].coords[2]:.4f}")
        check("polygon has 3 points", len(by_kind[POLYGON].points()) == 3)
        check("polygon class preserved", by_kind[POLYGON].cls == 1)
        check("all coordinates in range",
              all(0.0 <= v <= 1.0 for s in shapes for v in s.coords))

        print("\nSave and reload")
        window._save_current(force=True)
        entry = window.images.current_entry()
        on_disk = project.shapes_for(entry)
        check("labels written to disk", len(on_disk) == 2, str(len(on_disk)))
        check("mixed kinds survive the file",
              {s.kind for s in on_disk} == {BOX, POLYGON})
        text = project.label_path(entry).read_text(encoding="utf-8")
        check("file is plain YOLO text", all(line.split()[0].isdigit()
                                             for line in text.splitlines() if line))

        print("\nNavigation")
        window.images.step(1)
        check("moved to second image", window.images.list.currentRow() == 1)
        check("second image starts empty", window.canvas.count() == 0,
              str(window.canvas.count()))
        window.images.step(-1)
        check("returned to first image", window.images.list.currentRow() == 0)
        check("annotations reloaded", window.canvas.count() == 2,
              str(window.canvas.count()))
        moved = window.images.next_unannotated()
        check("next_unannotated finds one", moved and window.images.list.currentRow() != 0)

        print("\nUndo / redo")
        window.images.list.setCurrentRow(0)
        before = window.canvas.count()
        window.canvas.select_all()
        removed = window.canvas.delete_selected()
        check("delete removed both", removed == 2 and window.canvas.count() == 0)
        window.canvas.undo()
        check("undo restores shapes", window.canvas.count() == before,
              str(window.canvas.count()))
        window.canvas.redo()
        check("redo removes them again", window.canvas.count() == 0)
        window.canvas.undo()
        check("undo again restores", window.canvas.count() == before)

        print("\nZoom")
        window.canvas.fit_to_window()
        fitted = window.canvas.zoom_factor()
        window.canvas.scale_by(2.0)
        check("zoom doubles", approx(window.canvas.zoom_factor(), fitted * 2, 1e-2),
              f"{fitted:.3f} -> {window.canvas.zoom_factor():.3f}")
        window.canvas.zoom_actual()
        check("actual size is 1.0", approx(window.canvas.zoom_factor(), 1.0))

        print("\nModes")
        for mode in (MODE_BOX, MODE_POLYGON, MODE_SELECT):
            window.canvas.set_mode(mode)
            check(f"mode {mode}", window.canvas.mode == mode)

        print("\nClass edits propagate")
        window.canvas.select_all()
        changed = window.canvas.assign_class_to_selection(1)
        check("relabelled selection", changed == 2, str(changed))
        check("shapes now class 1",
              all(s.cls == 1 for s in window.canvas.get_shapes()))
        window._save_current(force=True)
        stats = project.stats()
        check("project stats see the change", stats["per_class"] == [0, 2],
              str(stats["per_class"]))

        project.delete_class(0)
        window._on_classes_changed()
        check("class removal renumbers on canvas",
              all(s.cls == 0 for s in window.canvas.get_shapes()),
              str([s.cls for s in window.canvas.get_shapes()]))
        check("class panel updated", window.classes.list.count() == 1)

        print("\nDialogs construct")
        from yolostudio.ui.autolabel_dialog import AutoLabelDialog
        from yolostudio.ui.export_model_dialog import ExportModelDialog
        from yolostudio.ui.new_project_dialog import NewProjectDialog

        dialog = AutoLabelDialog(project, [], [("cpu", "cpu")])
        check("auto-label dialog builds", dialog.table.columnCount() == 2)
        dialog.deleteLater()
        export_dialog = ExportModelDialog(project)
        check("export dialog builds", export_dialog.format.count() == 4)
        export_dialog.deleteLater()
        new_dialog = NewProjectDialog(tmp)
        new_dialog.classes.setPlainText("a\nb\na\n\n")
        check("new-project dialog dedupes classes",
              new_dialog.class_names() == ["a", "b"], str(new_dialog.class_names()))
        new_dialog.deleteLater()

        print("\nTrain panel")
        panel = window.train_panel
        panel.set_system_info({"torch": "2.6.0", "cuda_available": True,
                               "devices": [{"index": 0, "name": "RTX 4090 Laptop",
                                            "vram_gb": 16.0, "capability": "8.9"}]})
        check("device list has the GPU", panel.device.count() >= 2,
              str(panel.device.count()))
        panel.task.setCurrentIndex(1)   # segment
        seg_keys = [panel.scale.itemData(i) for i in range(panel.scale.count())]
        check("segment models are -seg", all("-seg" in k for k in seg_keys),
              str(seg_keys[:2]))
        panel.task.setCurrentIndex(0)
        det_keys = [panel.scale.itemData(i) for i in range(panel.scale.count())]
        check("detect models are not -seg", all("-seg" not in k for k in det_keys),
              str(det_keys[:2]))
        panel._apply_suggested_batch()
        check("suggested batch is sane", 1 <= panel.batch.value() <= 128,
              str(panel.batch.value()))
        panel._prepare_chart("detect")
        panel.chart.append(1, {"train/box_loss": 2.4, "metrics/mAP50-95(B)": 0.10})
        panel.chart.append(2, {"train/box_loss": 1.8, "metrics/mAP50-95(B)": 0.31})
        check("chart accepted points", panel.chart.has_data())
        panel.chart.grab()   # force a real paint pass
        check("chart paints without error", True)

        window.canvas.grab()
        check("canvas paints without error", True)
        window.close()

        print("\n" + "=" * 52)
        if FAILURES:
            print(f"  {len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        else:
            print("  all GUI tests passed")
        print("=" * 52)
        return 1 if FAILURES else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
