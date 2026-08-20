"""Shape model and YOLO label-file I/O.

A project stores one ``.txt`` per image in native YOLO format. Both box and
polygon shapes live in the same file and are told apart by token count:

    ``cls cx cy w h``                       -> 5 tokens  -> box
    ``cls x1 y1 x2 y2 x3 y3 ...``           -> 7+ odd    -> polygon

All coordinates are normalised to ``[0, 1]``. That means the same label file
trains a detector (polygons collapse to their bounding box) or a segmenter
(boxes expand to a 4-point rectangle) without re-annotating anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

BOX = "box"
POLYGON = "polygon"

# Below this normalised area a shape is treated as an accidental click.
MIN_AREA = 1e-6


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


@dataclass
class Shape:
    """A single annotation on a single image.

    ``coords`` holds normalised values: ``[cx, cy, w, h]`` for a box, or a flat
    ``[x1, y1, x2, y2, ...]`` ring for a polygon.
    """

    cls: int
    kind: str
    coords: List[float] = field(default_factory=list)

    # ---------------------------------------------------------------- helpers

    @property
    def is_box(self) -> bool:
        return self.kind == BOX

    @property
    def is_polygon(self) -> bool:
        return self.kind == POLYGON

    def points(self) -> List[Tuple[float, float]]:
        """Normalised vertices. A box yields its four corners, clockwise."""
        if self.is_box:
            x1, y1, x2, y2 = self.xyxy()
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return [(self.coords[i], self.coords[i + 1]) for i in range(0, len(self.coords) - 1, 2)]

    def xyxy(self) -> Tuple[float, float, float, float]:
        """Normalised ``(x1, y1, x2, y2)`` bounding box of the shape."""
        if self.is_box:
            cx, cy, w, h = self.coords
            return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        xs = self.coords[0::2]
        ys = self.coords[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    def xywh(self) -> Tuple[float, float, float, float]:
        """Normalised ``(cx, cy, w, h)`` of the shape's bounding box."""
        x1, y1, x2, y2 = self.xyxy()
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    def area(self) -> float:
        _, _, w, h = self.xywh()
        return max(0.0, w) * max(0.0, h)

    def clamped(self) -> "Shape":
        return replace(self, coords=[_clamp01(v) for v in self.coords])

    # ------------------------------------------------------------ conversions

    def as_box(self) -> "Shape":
        """This shape as a detect-task box (identity for boxes)."""
        if self.is_box:
            return self
        return Shape(self.cls, BOX, list(self.xywh()))

    def as_polygon(self) -> "Shape":
        """This shape as a segment-task polygon (a box becomes a rectangle)."""
        if self.is_polygon:
            return self
        flat: List[float] = []
        for x, y in self.points():
            flat.extend((x, y))
        return Shape(self.cls, POLYGON, flat)

    # ---------------------------------------------------------- serialisation

    def to_line(self) -> str:
        vals = " ".join(f"{v:.6f}" for v in self.coords)
        return f"{self.cls} {vals}"

    @staticmethod
    def from_line(line: str) -> "Shape | None":
        parts = line.split()
        if len(parts) < 5:
            return None
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            return None
        if len(vals) == 4:
            return Shape(cls, BOX, vals)
        if len(vals) >= 6 and len(vals) % 2 == 0:
            return Shape(cls, POLYGON, vals)
        return None


# --------------------------------------------------------------------- file IO


def load_labels(path: Path) -> List[Shape]:
    """Read a YOLO label file. Missing or unreadable files yield ``[]``."""
    if not path.exists():
        return []
    shapes: List[Shape] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        shape = Shape.from_line(line)
        if shape is not None and shape.area() >= MIN_AREA:
            shapes.append(shape)
    return shapes


def save_labels(path: Path, shapes: Sequence[Shape]) -> None:
    """Write shapes to a YOLO label file, removing the file when empty.

    An empty file is meaningful to YOLO (an explicit background image), but an
    absent one is how we mark "not annotated yet", so we delete instead.
    """
    keep = [s for s in shapes if s.area() >= MIN_AREA]
    if not keep:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(s.clamped().to_line() for s in keep)
    path.write_text(body + "\n", encoding="utf-8")


def convert_for_task(shapes: Iterable[Shape], task: str) -> List[Shape]:
    """Coerce mixed shapes into the single form a given YOLO task expects."""
    if task == "segment":
        return [s.as_polygon() for s in shapes]
    return [s.as_box() for s in shapes]
