"""The annotation canvas: image display, drawing tools, zoom/pan and undo.

Scene coordinates are image pixels, one-to-one, so converting to and from
normalised YOLO coordinates is a plain divide by width/height.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImageReader, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
                               QWidget)

from ..core.annotations import BOX, POLYGON, Shape
from .shapes import BaseShapeItem, BoxItem, PolygonItem

MODE_SELECT = "select"
MODE_BOX = "box"
MODE_POLYGON = "polygon"

MIN_BOX_PX = 3.0
MAX_ZOOM = 40.0
MIN_ZOOM = 0.02
UNDO_DEPTH = 80


class _Scene(QGraphicsScene):
    """Scene that lets items report user edits back to the canvas."""

    edited = Signal()

    def notify_edited(self) -> None:
        self.edited.emit()


class AnnotationCanvas(QGraphicsView):

    shapesChanged = Signal()
    selectionChanged = Signal()
    statusMessage = Signal(str)
    zoomChanged = Signal(float)
    # Emitted when the user finishes a shape while no class exists yet.
    classNeeded = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._scene = _Scene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor("#17171c"))

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._image_size: Tuple[int, int] = (0, 0)
        self._mode = MODE_SELECT
        self._active_class = 0
        self._classes: List[str] = []
        self._colors: List[str] = []

        self._drawing_box: Optional[BoxItem] = None
        self._box_origin: Optional[QPointF] = None
        self._poly_points: List[QPointF] = []
        self._poly_preview: Optional[PolygonItem] = None
        self._cursor_scene: Optional[QPointF] = None

        self._panning = False
        self._pan_anchor = QPointF()
        self._space_held = False

        self._undo: List[list] = []
        self._redo: List[list] = []
        # State captured when a drag starts, committed to the undo stack only if
        # the drag actually changed something.
        self._pending: Optional[list] = None

        self._scene.selectionChanged.connect(self.selectionChanged.emit)
        self._scene.edited.connect(self._on_item_edited)
        # Undo/redo are bound by MainWindow so there is exactly one registration
        # per key sequence; two would make Qt report an ambiguous shortcut and
        # fire neither.

    # ------------------------------------------------------------ image load

    def load_image(self, path: Path) -> bool:
        self.clear_all()
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)          # honour EXIF orientation
        image = reader.read()
        if image.isNull():
            self._image_size = (0, 0)
            self.statusMessage.emit(f"Could not read {Path(path).name}: {reader.errorString()}")
            return False
        pixmap = QPixmap.fromImage(image)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(-100)
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._image_size = (pixmap.width(), pixmap.height())
        self._scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.fit_to_window()
        return True

    def clear_all(self) -> None:
        self._cancel_drawing()
        self._scene.clear()
        self._pixmap_item = None
        self._undo.clear()
        self._redo.clear()
        self._pending = None

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    @property
    def has_image(self) -> bool:
        return self._pixmap_item is not None

    # --------------------------------------------------------------- classes

    def set_classes(self, names: Sequence[str], colors: Sequence[str]) -> None:
        self._classes = list(names)
        self._colors = list(colors)
        for item in self._shape_items():
            item.set_class(item.cls, self._color(item.cls), self._name(item.cls))

    def set_active_class(self, index: int) -> None:
        self._active_class = max(0, index)

    @property
    def active_class(self) -> int:
        return self._active_class

    def _color(self, cls: int) -> QColor:
        if 0 <= cls < len(self._colors):
            return QColor(self._colors[cls])
        return QColor("#888888")

    def _name(self, cls: int) -> str:
        if 0 <= cls < len(self._classes):
            return self._classes[cls]
        return f"class_{cls}"

    # ------------------------------------------------------------------ mode

    def set_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._cancel_drawing()
        self._mode = mode
        interactive = mode == MODE_SELECT
        for item in self._shape_items():
            item.setEnabled(interactive)
        self.viewport().setCursor(Qt.ArrowCursor if interactive else Qt.CrossCursor)
        self.viewport().update()

    @property
    def mode(self) -> str:
        return self._mode

    # ---------------------------------------------------------------- shapes

    def _shape_items(self) -> List[BaseShapeItem]:
        return [i for i in self._scene.items() if isinstance(i, BaseShapeItem)]

    def set_shapes(self, shapes: Sequence[Shape]) -> None:
        for item in self._shape_items():
            self._scene.removeItem(item)
        width, height = self._image_size
        if width == 0 or height == 0:
            return
        for shape in shapes:
            self._add_item_from_shape(shape, width, height)
        self._sync_view_scale()
        self._undo.clear()
        self._redo.clear()
        self._pending = None

    def _add_item_from_shape(self, shape: Shape, width: int, height: int) -> BaseShapeItem:
        color, name = self._color(shape.cls), self._name(shape.cls)
        if shape.is_box:
            x1, y1, x2, y2 = shape.xyxy()
            item: BaseShapeItem = BoxItem(
                QRectF(QPointF(x1 * width, y1 * height), QPointF(x2 * width, y2 * height)),
                shape.cls, color, name)
        else:
            pts = [QPointF(x * width, y * height) for x, y in shape.points()]
            item = PolygonItem(pts, shape.cls, color, name)
        item.setEnabled(self._mode == MODE_SELECT)
        self._scene.addItem(item)
        return item

    def get_shapes(self) -> List[Shape]:
        width, height = self._image_size
        if width == 0 or height == 0:
            return []
        out: List[Shape] = []
        for item in sorted(self._shape_items(), key=lambda i: (i.cls,)):
            if isinstance(item, BoxItem):
                rect = item.rect()
                cx = (rect.left() + rect.right()) / 2 / width
                cy = (rect.top() + rect.bottom()) / 2 / height
                out.append(Shape(item.cls, BOX,
                                 [cx, cy, rect.width() / width, rect.height() / height]).clamped())
            elif isinstance(item, PolygonItem):
                flat: List[float] = []
                for point in item.points():
                    flat.extend((point.x() / width, point.y() / height))
                out.append(Shape(item.cls, POLYGON, flat).clamped())
        return [s for s in out if s.area() > 0]

    def selected_items(self) -> List[BaseShapeItem]:
        return [i for i in self._scene.selectedItems() if isinstance(i, BaseShapeItem)]

    def delete_selected(self) -> int:
        items = self.selected_items()
        if not items:
            return 0
        self._snapshot()
        for item in items:
            self._scene.removeItem(item)
        self.shapesChanged.emit()
        return len(items)

    def select_all(self) -> None:
        for item in self._shape_items():
            item.setSelected(True)

    def assign_class_to_selection(self, cls: int) -> int:
        items = self.selected_items()
        if not items:
            return 0
        self._snapshot()
        for item in items:
            item.set_class(cls, self._color(cls), self._name(cls))
        self.shapesChanged.emit()
        return len(items)

    def count(self) -> int:
        return len(self._shape_items())

    # ------------------------------------------------------------------ undo

    def _snapshot(self) -> None:
        self._push_undo(self._current_state())

    def _push_undo(self, state: list) -> None:
        self._undo.append(state)
        del self._undo[:-UNDO_DEPTH]
        self._redo.clear()

    def _current_state(self) -> list:
        return [(i.cls, isinstance(i, BoxItem), [(p.x(), p.y()) for p in i.points()])
                for i in self._shape_items()]

    def _restore(self, state: list) -> None:
        for item in self._shape_items():
            self._scene.removeItem(item)
        for cls, is_box, pts in state:
            color, name = self._color(cls), self._name(cls)
            if is_box:
                item: BaseShapeItem = BoxItem(
                    QRectF(QPointF(*pts[0]), QPointF(*pts[1])), cls, color, name)
            else:
                item = PolygonItem([QPointF(x, y) for x, y in pts], cls, color, name)
            item.setEnabled(self._mode == MODE_SELECT)
            self._scene.addItem(item)
        self._sync_view_scale()
        self.shapesChanged.emit()

    def undo(self) -> None:
        if not self._undo:
            self.statusMessage.emit("Nothing to undo")
            return
        self._redo.append(self._current_state())
        self._restore(self._undo.pop())
        self.statusMessage.emit("Undo")

    def redo(self) -> None:
        if not self._redo:
            self.statusMessage.emit("Nothing to redo")
            return
        self._undo.append(self._current_state())
        self._restore(self._redo.pop())
        self.statusMessage.emit("Redo")

    def _on_item_edited(self) -> None:
        # An item finished a drag that changed its geometry. Commit the pre-drag
        # state captured on mouse-press.
        if self._pending is not None:
            self._push_undo(self._pending)
            self._pending = None
        self.shapesChanged.emit()

    # ------------------------------------------------------------------ zoom

    def zoom_factor(self) -> float:
        return self.transform().m11()

    def _sync_view_scale(self) -> None:
        scale = self.zoom_factor()
        for item in self._shape_items():
            item.set_view_scale(scale)
        if self._poly_preview is not None:
            self._poly_preview.set_view_scale(scale)
        self.zoomChanged.emit(scale)

    def scale_by(self, factor: float, anchor_view: Optional[QPointF] = None) -> None:
        current = self.zoom_factor()
        target = max(MIN_ZOOM, min(MAX_ZOOM, current * factor))
        factor = target / current
        if abs(factor - 1.0) < 1e-9:
            return
        if anchor_view is None:
            anchor_view = QPointF(self.viewport().rect().center())
        before = self.mapToScene(anchor_view.toPoint())
        self.scale(factor, factor)
        after = self.mapToScene(anchor_view.toPoint())
        delta = after - before
        self.translate(delta.x(), delta.y())
        self._sync_view_scale()

    def fit_to_window(self) -> None:
        if not self.has_image:
            return
        self.resetTransform()
        rect = self._scene.sceneRect()
        view = self.viewport().rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        scale = min(view.width() / rect.width(), view.height() / rect.height()) * 0.98
        scale = max(MIN_ZOOM, min(MAX_ZOOM, scale))
        self.scale(scale, scale)
        self.centerOn(rect.center())
        self._sync_view_scale()

    def zoom_actual(self) -> None:
        if not self.has_image:
            return
        center = self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        self.centerOn(center)
        self._sync_view_scale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_view_scale()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - event.angleDelta().y())
            event.accept()
            return
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.scale_by(1.18 ** steps, QPointF(event.position()))
        event.accept()

    # -------------------------------------------------------------- drawing

    def _cancel_drawing(self) -> None:
        if self._drawing_box is not None:
            self._scene.removeItem(self._drawing_box)
            self._drawing_box = None
        self._box_origin = None
        if self._poly_preview is not None:
            self._scene.removeItem(self._poly_preview)
            self._poly_preview = None
        self._poly_points = []
        self.viewport().update()

    def _ensure_class(self) -> bool:
        if self._classes:
            return True
        self.classNeeded.emit()
        return bool(self._classes)

    def _finish_polygon(self) -> None:
        if len(self._poly_points) < 3:
            self.statusMessage.emit("A polygon needs at least 3 points")
            return
        pts = list(self._poly_points)
        self._cancel_drawing()
        self._snapshot()
        cls = self._active_class
        item = PolygonItem(pts, cls, self._color(cls), self._name(cls))
        item.setEnabled(self._mode == MODE_SELECT)
        item.set_view_scale(self.zoom_factor())
        self._scene.addItem(item)
        self.shapesChanged.emit()
        self.statusMessage.emit(f"Added polygon ({len(pts)} points)")

    # ---------------------------------------------------------- mouse events

    def mousePressEvent(self, event):
        if not self.has_image:
            return
        pos = QPointF(event.position())
        scene_pos = self.mapToScene(pos.toPoint())

        if event.button() == Qt.MiddleButton or (self._space_held and event.button() == Qt.LeftButton):
            self._panning = True
            self._pan_anchor = pos
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if self._mode == MODE_BOX and event.button() == Qt.LeftButton:
            if not self._ensure_class():
                return
            self._box_origin = scene_pos
            cls = self._active_class
            self._drawing_box = BoxItem(QRectF(scene_pos, scene_pos), cls,
                                        self._color(cls), self._name(cls))
            self._drawing_box.set_view_scale(self.zoom_factor())
            self._drawing_box.setEnabled(False)
            self._scene.addItem(self._drawing_box)
            event.accept()
            return

        if self._mode == MODE_POLYGON and event.button() == Qt.LeftButton:
            if not self._ensure_class():
                return
            self._poly_points.append(scene_pos)
            if self._poly_preview is None:
                cls = self._active_class
                self._poly_preview = PolygonItem(list(self._poly_points), cls,
                                                 self._color(cls), self._name(cls))
                self._poly_preview.setEnabled(False)
                self._poly_preview.set_view_scale(self.zoom_factor())
                self._scene.addItem(self._poly_preview)
            else:
                self._poly_preview.prepareGeometryChange()
                self._poly_preview.set_points(list(self._poly_points))
                self._poly_preview.update()
            self.statusMessage.emit(
                f"Polygon: {len(self._poly_points)} points — double-click or Enter to close")
            event.accept()
            return

        if self._mode == MODE_SELECT and event.button() == Qt.LeftButton:
            item = self.itemAt(pos.toPoint())
            if isinstance(item, PolygonItem) and event.modifiers() & Qt.AltModifier:
                before = self._current_state()
                if item.remove_vertex_at(scene_pos):
                    self._push_undo(before)
                    self.shapesChanged.emit()
                    self.statusMessage.emit("Vertex removed")
                else:
                    self.statusMessage.emit(
                        "Alt-click a vertex to remove it (a polygon keeps at least 3)")
                event.accept()
                return
            if isinstance(item, BaseShapeItem):
                self._pending = self._current_state()   # committed only if it moves
            super().mousePressEvent(event)
            if not isinstance(item, BaseShapeItem):
                self._scene.clearSelection()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = QPointF(event.position())
        self._cursor_scene = self.mapToScene(pos.toPoint())

        if self._panning:
            delta = pos - self._pan_anchor
            self._pan_anchor = pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return

        if self._drawing_box is not None and self._box_origin is not None:
            self._drawing_box.set_rect(QRectF(self._box_origin, self._cursor_scene).normalized())
            event.accept()
            return

        if self._mode in (MODE_BOX, MODE_POLYGON):
            self.viewport().update()   # redraw crosshair

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self.viewport().setCursor(
                Qt.ArrowCursor if self._mode == MODE_SELECT else Qt.CrossCursor)
            event.accept()
            return

        if self._drawing_box is not None and self._box_origin is not None:
            rect = self._drawing_box.rect()
            self._scene.removeItem(self._drawing_box)
            self._drawing_box = None
            self._box_origin = None
            if rect.width() >= MIN_BOX_PX and rect.height() >= MIN_BOX_PX:
                self._snapshot()
                cls = self._active_class
                item = BoxItem(rect, cls, self._color(cls), self._name(cls))
                item.setEnabled(self._mode == MODE_SELECT)
                item.set_view_scale(self.zoom_factor())
                self._scene.addItem(item)
                self.shapesChanged.emit()
                self.statusMessage.emit(
                    f"Added box {int(rect.width())}x{int(rect.height())} px")
            else:
                self.statusMessage.emit("Box too small — drag further")
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._mode == MODE_POLYGON and self._poly_points:
            self._finish_polygon()
            event.accept()
            return
        if self._mode == MODE_SELECT:
            scene_pos = self.mapToScene(QPointF(event.position()).toPoint())
            item = self.itemAt(QPointF(event.position()).toPoint())
            if isinstance(item, PolygonItem):
                before = self._current_state()
                if item.insert_vertex_near(scene_pos):
                    self._push_undo(before)
                    self.shapesChanged.emit()
                    self.statusMessage.emit("Vertex added")
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    # ------------------------------------------------------------ key events

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        if key == Qt.Key_Escape:
            if self._poly_points or self._drawing_box is not None:
                self._cancel_drawing()
                self.statusMessage.emit("Cancelled")
            else:
                self._scene.clearSelection()
            event.accept()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter) and self._mode == MODE_POLYGON:
            self._finish_polygon()
            event.accept()
            return
        if key == Qt.Key_Backspace and self._poly_points:
            self._poly_points.pop()
            if self._poly_preview is not None:
                if self._poly_points:
                    self._poly_preview.prepareGeometryChange()
                    self._poly_preview.set_points(list(self._poly_points))
                    self._poly_preview.update()
                else:
                    self._scene.removeItem(self._poly_preview)
                    self._poly_preview = None
            event.accept()
            return
        if key == Qt.Key_Delete:
            if self.delete_selected():
                self.statusMessage.emit("Deleted")
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            if not self._panning:
                self.viewport().setCursor(
                    Qt.ArrowCursor if self._mode == MODE_SELECT else Qt.CrossCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    # ------------------------------------------------------------- overlays

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        """Crosshair guides while drawing, plus the rubber line to the cursor."""
        if self._mode == MODE_SELECT or self._cursor_scene is None or not self.has_image:
            return
        painter.save()
        scale = max(1e-6, self.zoom_factor())
        pen = QPen(QColor(255, 255, 255, 90), 1.0 / scale)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        x, y = self._cursor_scene.x(), self._cursor_scene.y()
        painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        if self._mode == MODE_POLYGON and self._poly_points:
            live = QPen(self._color(self._active_class), 1.5 / scale)
            live.setStyle(Qt.DashLine)
            painter.setPen(live)
            painter.drawLine(self._poly_points[-1], self._cursor_scene)
            painter.drawLine(self._cursor_scene, self._poly_points[0])
        painter.restore()
