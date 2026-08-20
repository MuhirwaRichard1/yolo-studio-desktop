"""Graphics items for boxes and polygons.

Both items keep their geometry in **scene coordinates, which equal image
pixels**, and always sit at ``pos() == (0, 0)``. Dragging translates the stored
points rather than moving the item, so a polygon vertex and a box corner behave
identically and nothing has to be un-transformed when converting to normalised
YOLO coordinates.

Handles, outlines and text are drawn at a constant *screen* size by dividing by
``view_scale``, so annotating stays comfortable at 8x zoom.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QPainter, QPainterPath,
                           QPen, QPolygonF)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

HANDLE_PX = 5.0        # half-size of a handle square, in screen pixels
GRAB_PX = 8.0          # click tolerance around a handle
OUTLINE_PX = 2.0
LABEL_PT = 11.0
FILL_ALPHA = 45
FILL_ALPHA_SEL = 80

# Box handle order: corners and edge midpoints, clockwise from top-left.
TL, TC, TR, RC, BR, BC, BL, LC = range(8)


class BaseShapeItem(QGraphicsObject):
    """Shared painting, hit-testing and drag plumbing."""

    def __init__(self, cls: int, color: QColor, name: str):
        super().__init__()
        self.cls = cls
        self.color = QColor(color)
        self.name = name
        self.view_scale = 1.0
        self._hover_handle = -1
        self._drag_handle = -1
        self._drag_origin: Optional[QPointF] = None
        self._drag_points: List[QPointF] = []
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsFocusable)
        self.setAcceptHoverEvents(True)

    # -------------------------------------------------------------- geometry

    def points(self) -> List[QPointF]:
        raise NotImplementedError

    def set_points(self, pts: List[QPointF]) -> None:
        raise NotImplementedError

    def handle_positions(self) -> List[QPointF]:
        raise NotImplementedError

    def resize_to(self, handle: int, pos: QPointF) -> None:
        raise NotImplementedError

    # ---------------------------------------------------------------- pixels

    def _px(self, value: float) -> float:
        """Convert a screen-pixel length into scene units at the current zoom."""
        return value / max(1e-6, self.view_scale)

    def _label_extent(self) -> Tuple[float, float]:
        """Scene-unit (width, height) the class chip occupies above the shape.

        The chip is drawn in screen pixels, so a long class name at low zoom can
        reach far past the geometry. boundingRect has to cover it or Qt leaves
        trails when the item scrolls.
        """
        if not self.name:
            return (0.0, 0.0)
        return (self._px(9.0 * len(self.name) + 14.0), self._px(LABEL_PT * 2.2))

    def set_view_scale(self, scale: float) -> None:
        if abs(scale - self.view_scale) > 1e-9:
            self.prepareGeometryChange()
            self.view_scale = scale
            self.update()

    def set_class(self, cls: int, color: QColor, name: str) -> None:
        self.cls = cls
        self.color = QColor(color)
        self.name = name
        self.update()

    # ------------------------------------------------------------- hit tests

    def handle_at(self, scene_pos: QPointF) -> int:
        tol = self._px(GRAB_PX)
        for index, point in enumerate(self.handle_positions()):
            if abs(point.x() - scene_pos.x()) <= tol and abs(point.y() - scene_pos.y()) <= tol:
                return index
        return -1

    # ---------------------------------------------------------------- events

    def hoverMoveEvent(self, event):
        handle = self.handle_at(event.scenePos()) if self.isSelected() else -1
        if handle != self._hover_handle:
            self._hover_handle = handle
            self.update()
        self.setCursor(self._cursor_for(handle))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover_handle = -1
        self.update()
        super().hoverLeaveEvent(event)

    def _cursor_for(self, handle: int) -> Qt.CursorShape:
        return Qt.SizeAllCursor if handle < 0 else Qt.CrossCursor

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        self.setSelected(True)
        self._drag_handle = self.handle_at(event.scenePos())
        self._drag_origin = event.scenePos()
        self._drag_points = list(self.points())
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_origin is None:
            return
        if self._drag_handle >= 0:
            self.prepareGeometryChange()
            self.resize_to(self._drag_handle, event.scenePos())
        else:
            delta = event.scenePos() - self._drag_origin
            self.prepareGeometryChange()
            self.set_points([p + delta for p in self._drag_points])
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag_origin is not None:
            self._drag_origin = None
            self._drag_handle = -1
            # Only report an edit if the geometry actually moved. A plain click
            # to select must not land on the undo stack.
            moved = [(p.x(), p.y()) for p in self.points()] != \
                    [(p.x(), p.y()) for p in self._drag_points]
            scene = self.scene()
            if moved and scene is not None and hasattr(scene, "notify_edited"):
                scene.notify_edited()
        event.accept()

    # --------------------------------------------------------------- drawing

    def _paint_label(self, painter: QPainter, anchor: QPointF) -> None:
        if not self.name:
            return
        font = QFont()
        font.setPointSizeF(LABEL_PT)
        painter.save()
        painter.translate(anchor)
        painter.scale(self._px(1.0), self._px(1.0))   # draw in screen pixels
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        width = metrics.horizontalAdvance(self.name) + 8
        height = metrics.height() + 2
        rect = QRectF(0, -height, width, height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.color))
        painter.drawRect(rect)
        text_color = Qt.black if self.color.lightnessF() > 0.6 else Qt.white
        painter.setPen(QPen(text_color))
        painter.drawText(rect.adjusted(4, 0, -4, -1), Qt.AlignVCenter | Qt.AlignLeft, self.name)
        painter.restore()

    def _paint_handles(self, painter: QPainter) -> None:
        if not self.isSelected():
            return
        size = self._px(HANDLE_PX)
        painter.setPen(QPen(QColor("#101014"), self._px(1.0)))
        for index, point in enumerate(self.handle_positions()):
            painter.setBrush(QBrush(QColor("#ffffff") if index != self._hover_handle
                                    else self.color.lighter(150)))
            painter.drawRect(QRectF(point.x() - size, point.y() - size, size * 2, size * 2))

    def _pens(self) -> Tuple[QPen, QBrush]:
        pen = QPen(self.color, self._px(OUTLINE_PX))
        pen.setCosmetic(False)
        pen.setJoinStyle(Qt.MiterJoin)
        fill = QColor(self.color)
        fill.setAlpha(FILL_ALPHA_SEL if self.isSelected() else FILL_ALPHA)
        if self.isSelected():
            pen.setWidthF(self._px(OUTLINE_PX + 1.0))
        return pen, QBrush(fill)


class BoxItem(BaseShapeItem):
    """An axis-aligned bounding box with eight resize handles."""

    def __init__(self, rect: QRectF, cls: int, color: QColor, name: str):
        super().__init__(cls, color, name)
        self._rect = QRectF(rect).normalized()

    # -------------------------------------------------------------- geometry

    def rect(self) -> QRectF:
        return QRectF(self._rect)

    def points(self) -> List[QPointF]:
        return [self._rect.topLeft(), self._rect.bottomRight()]

    def set_points(self, pts: List[QPointF]) -> None:
        self._rect = QRectF(pts[0], pts[1]).normalized()

    def set_rect(self, rect: QRectF) -> None:
        self.prepareGeometryChange()
        self._rect = QRectF(rect).normalized()
        self.update()

    def handle_positions(self) -> List[QPointF]:
        r = self._rect
        cx, cy = r.center().x(), r.center().y()
        return [QPointF(r.left(), r.top()), QPointF(cx, r.top()),
                QPointF(r.right(), r.top()), QPointF(r.right(), cy),
                QPointF(r.right(), r.bottom()), QPointF(cx, r.bottom()),
                QPointF(r.left(), r.bottom()), QPointF(r.left(), cy)]

    def resize_to(self, handle: int, pos: QPointF) -> None:
        r = QRectF(self._rect)
        if handle in (TL, LC, BL):
            r.setLeft(pos.x())
        if handle in (TR, RC, BR):
            r.setRight(pos.x())
        if handle in (TL, TC, TR):
            r.setTop(pos.y())
        if handle in (BL, BC, BR):
            r.setBottom(pos.y())
        self._rect = r.normalized()

    def _cursor_for(self, handle: int) -> Qt.CursorShape:
        return {
            TL: Qt.SizeFDiagCursor, BR: Qt.SizeFDiagCursor,
            TR: Qt.SizeBDiagCursor, BL: Qt.SizeBDiagCursor,
            TC: Qt.SizeVerCursor, BC: Qt.SizeVerCursor,
            LC: Qt.SizeHorCursor, RC: Qt.SizeHorCursor,
        }.get(handle, Qt.SizeAllCursor)

    # --------------------------------------------------------------- drawing

    def boundingRect(self) -> QRectF:
        pad = self._px(GRAB_PX + OUTLINE_PX)
        label_w, label_h = self._label_extent()
        return self._rect.adjusted(-pad, -pad - label_h, pad + label_w, pad)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self._rect)
        stroker_pad = self._px(GRAB_PX)
        outer = QPainterPath()
        outer.addRect(self._rect.adjusted(-stroker_pad, -stroker_pad,
                                          stroker_pad, stroker_pad))
        return outer if self.isSelected() else path

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen, brush = self._pens()
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRect(self._rect)
        self._paint_label(painter, self._rect.topLeft())
        self._paint_handles(painter)


class PolygonItem(BaseShapeItem):
    """A closed polygon with one handle per vertex."""

    def __init__(self, points: List[QPointF], cls: int, color: QColor, name: str):
        super().__init__(cls, color, name)
        self._pts = [QPointF(p) for p in points]

    # -------------------------------------------------------------- geometry

    def points(self) -> List[QPointF]:
        return list(self._pts)

    def set_points(self, pts: List[QPointF]) -> None:
        self._pts = [QPointF(p) for p in pts]

    def handle_positions(self) -> List[QPointF]:
        return list(self._pts)

    def resize_to(self, handle: int, pos: QPointF) -> None:
        if 0 <= handle < len(self._pts):
            self._pts[handle] = QPointF(pos)

    def insert_vertex_near(self, scene_pos: QPointF) -> bool:
        """Split the edge closest to ``scene_pos`` and add a vertex there."""
        if len(self._pts) < 2:
            return False
        best_index, best_dist = -1, float("inf")
        for i in range(len(self._pts)):
            a = self._pts[i]
            b = self._pts[(i + 1) % len(self._pts)]
            dist = _point_segment_distance(scene_pos, a, b)
            if dist < best_dist:
                best_index, best_dist = i, dist
        if best_index < 0 or best_dist > self._px(GRAB_PX * 2):
            return False
        self.prepareGeometryChange()
        self._pts.insert(best_index + 1, QPointF(scene_pos))
        self.update()
        return True

    def remove_vertex_at(self, scene_pos: QPointF) -> bool:
        """Delete a vertex, refusing to drop below a valid triangle."""
        handle = self.handle_at(scene_pos)
        if handle < 0 or len(self._pts) <= 3:
            return False
        self.prepareGeometryChange()
        del self._pts[handle]
        self._hover_handle = -1
        self.update()
        return True

    # --------------------------------------------------------------- drawing

    def _polygon(self) -> QPolygonF:
        return QPolygonF(self._pts)

    def boundingRect(self) -> QRectF:
        rect = self._polygon().boundingRect()
        pad = self._px(GRAB_PX + OUTLINE_PX)
        label_w, label_h = self._label_extent()
        return rect.adjusted(-pad, -pad - label_h, pad + label_w, pad)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addPolygon(self._polygon())
        path.closeSubpath()
        return path

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen, brush = self._pens()
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawPolygon(self._polygon())
        rect = self._polygon().boundingRect()
        self._paint_label(painter, rect.topLeft())
        self._paint_handles(painter)


def _point_segment_distance(p: QPointF, a: QPointF, b: QPointF) -> float:
    line = QLineF(a, b)
    length = line.length()
    if length < 1e-9:
        return QLineF(p, a).length()
    t = ((p.x() - a.x()) * (b.x() - a.x()) + (p.y() - a.y()) * (b.y() - a.y())) / (length ** 2)
    t = max(0.0, min(1.0, t))
    proj = QPointF(a.x() + t * (b.x() - a.x()), a.y() + t * (b.y() - a.y()))
    return QLineF(p, proj).length()
