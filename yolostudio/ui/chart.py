"""A small multi-series line chart, drawn with QPainter.

Purpose-built for training curves so the app needs no plotting dependency.
Series are grouped into two independently scaled bands -- losses on the left
axis, metrics (0-1) on the right -- because plotting a box loss of 3.2 and an
mAP of 0.41 on one axis makes both unreadable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from .theme import BG_DEEP, LINE, TEXT, TEXT_DIM

PAD_L, PAD_R, PAD_T, PAD_B = 46, 46, 26, 26
LEGEND_H = 20


class Series:
    def __init__(self, key: str, label: str, color: str, axis: str = "left"):
        self.key = key
        self.label = label
        self.color = color
        self.axis = axis            # "left" (losses) or "right" (0-1 metrics)
        self.points: List[Tuple[float, float]] = []
        self.visible = True


class MetricsChart(QWidget):
    """Append-only chart: feed it ``(x, {key: value})`` per epoch."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._series: Dict[str, Series] = {}
        self._order: List[str] = []
        self._xmax = 1.0
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ------------------------------------------------------------------ data

    def add_series(self, key: str, label: str, color: str, axis: str = "left") -> None:
        if key not in self._series:
            self._series[key] = Series(key, label, color, axis)
            self._order.append(key)

    def reset_series(self) -> None:
        """Drop every series, so a new run can define its own set."""
        self._series.clear()
        self._order.clear()
        self._xmax = 1.0
        self.update()

    def clear(self) -> None:
        for series in self._series.values():
            series.points.clear()
        self._xmax = 1.0
        self.update()

    def append(self, x: float, values: Dict[str, float]) -> None:
        for key, value in values.items():
            series = self._series.get(key)
            if series is None or value is None:
                continue
            try:
                series.points.append((float(x), float(value)))
            except (TypeError, ValueError):
                continue
        self._xmax = max(self._xmax, float(x))
        self.update()

    def has_data(self) -> bool:
        return any(s.points for s in self._series.values())

    # --------------------------------------------------------------- scaling

    def _range(self, axis: str) -> Tuple[float, float]:
        values = [v for s in self._series.values() if s.axis == axis and s.visible
                  for _, v in s.points]
        if not values:
            return (0.0, 1.0)
        if axis == "right":
            return (0.0, 1.0)          # metrics are already normalised
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            hi = lo + 1.0
        pad = (hi - lo) * 0.08
        return (max(0.0, lo - pad), hi + pad)

    # --------------------------------------------------------------- drawing

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(BG_DEEP))

        plot = QRectF(PAD_L, PAD_T,
                      max(10.0, self.width() - PAD_L - PAD_R),
                      max(10.0, self.height() - PAD_T - PAD_B - LEGEND_H))

        if not self.has_data():
            painter.setPen(QPen(QColor(TEXT_DIM)))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Training curves appear here once the first epoch completes")
            return

        left_lo, left_hi = self._range("left")
        font = QFont()
        font.setPointSizeF(8.5)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        # Grid and axis labels.
        painter.setPen(QPen(QColor(LINE), 1))
        for i in range(5):
            t = i / 4.0
            y = plot.bottom() - t * plot.height()
            painter.setPen(QPen(QColor(LINE), 1, Qt.DotLine))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QPen(QColor(TEXT_DIM)))
            left_val = left_lo + t * (left_hi - left_lo)
            painter.drawText(QRectF(0, y - 8, PAD_L - 6, 16),
                             Qt.AlignRight | Qt.AlignVCenter, f"{left_val:.2f}")
            painter.drawText(QRectF(plot.right() + 6, y - 8, PAD_R - 8, 16),
                             Qt.AlignLeft | Qt.AlignVCenter, f"{t:.2f}")

        painter.setPen(QPen(QColor(LINE), 1))
        painter.drawRect(plot)

        xmax = max(1.0, self._xmax)
        for label, t in (("1", 0.0), (f"{int(xmax)}", 1.0)):
            x = plot.left() + t * plot.width()
            painter.setPen(QPen(QColor(TEXT_DIM)))
            painter.drawText(QRectF(x - 24, plot.bottom() + 3, 48, 16),
                             Qt.AlignCenter, label)
        painter.drawText(QRectF(plot.left(), plot.bottom() + 3, plot.width(), 16),
                         Qt.AlignCenter, "epoch")

        # Series.
        for key in self._order:
            series = self._series[key]
            if not series.visible or len(series.points) < 1:
                continue
            lo, hi = (0.0, 1.0) if series.axis == "right" else (left_lo, left_hi)
            span = max(1e-9, hi - lo)
            polygon = QPolygonF()
            for x, y in series.points:
                px = plot.left() + (x / xmax) * plot.width() if xmax > 0 else plot.left()
                py = plot.bottom() - ((y - lo) / span) * plot.height()
                polygon.append(QPointF(px, max(plot.top(), min(plot.bottom(), py))))
            pen = QPen(QColor(series.color), 1.8)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            if len(polygon) == 1:
                painter.drawEllipse(polygon[0], 2.2, 2.2)
            else:
                painter.drawPolyline(polygon)

        # Legend, with the latest value of each series.
        x = plot.left()
        y = self.height() - LEGEND_H + 2
        for key in self._order:
            series = self._series[key]
            if not series.points:
                continue
            latest = series.points[-1][1]
            text = f"{series.label} {latest:.3f}"
            width = metrics.horizontalAdvance(text) + 22
            if x + width > plot.right():
                break
            painter.setPen(QPen(QColor(series.color), 2.4))
            painter.drawLine(QPointF(x, y + 8), QPointF(x + 12, y + 8))
            painter.setPen(QPen(QColor(TEXT if series.visible else TEXT_DIM)))
            painter.drawText(QRectF(x + 16, y, width, 16),
                             Qt.AlignLeft | Qt.AlignVCenter, text)
            x += width
