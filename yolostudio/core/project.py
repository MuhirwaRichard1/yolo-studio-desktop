"""Project model: the on-disk format and the in-memory API around it.

Layout of a project directory::

    my-project/
        project.json      classes, image registry, training defaults
        labels/           <image-id>.txt, native YOLO format
        datasets/         generated train/val/test splits + data.yaml
        runs/             ultralytics training output

Images are referenced where they live rather than copied, so importing a
100k-image folder is instant. Splits are materialised at export time using
hardlinks when the filesystem allows it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .annotations import Shape, load_labels, save_labels

PROJECT_FILE = "project.json"
FORMAT_VERSION = 1

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Distinguishable at small sizes and on both light and dark image content.
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffe119",
]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "image"


@dataclass
class ImageEntry:
    """One image in the project. ``iid`` is the stable label-file basename."""

    iid: str
    path: str  # absolute, POSIX-style separators

    @property
    def file(self) -> Path:
        return Path(self.path)

    @property
    def name(self) -> str:
        return Path(self.path).name

    def exists(self) -> bool:
        return Path(self.path).is_file()


@dataclass
class TrainConfig:
    """Defaults for the training panel, persisted per project."""

    model: str = "yolo11s.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "0"
    workers: int = 8
    patience: int = 50
    optimizer: str = "auto"
    lr0: float = 0.01
    seed: int = 0
    cache: bool = False
    amp: bool = True
    cos_lr: bool = False
    freeze: int = 0
    val_split: float = 0.2
    test_split: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(data: dict) -> "TrainConfig":
        cfg = TrainConfig()
        for key, value in (data or {}).items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


class Project:
    """An open annotation project."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.name: str = self.root.name
        self.task: str = "detect"
        self.classes: List[str] = []
        self.colors: List[str] = []
        self.images: List[ImageEntry] = []
        self.train: TrainConfig = TrainConfig()
        self._by_path: Dict[str, ImageEntry] = {}
        self._dirty = False

    # ------------------------------------------------------------------ paths

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def json_path(self) -> Path:
        return self.root / PROJECT_FILE

    def label_path(self, entry: ImageEntry) -> Path:
        return self.labels_dir / f"{entry.iid}.txt"

    # ------------------------------------------------------- create/open/save

    @staticmethod
    def create(root: Path, name: str, task: str = "detect",
               classes: Optional[Sequence[str]] = None) -> "Project":
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        proj = Project(root)
        proj.name = name or root.name
        proj.task = task
        for label in classes or []:
            proj.add_class(label)
        for sub in (proj.labels_dir, proj.datasets_dir, proj.runs_dir):
            sub.mkdir(parents=True, exist_ok=True)
        proj.save()
        return proj

    @staticmethod
    def open(path: Path) -> "Project":
        path = Path(path)
        root = path.parent if path.name == PROJECT_FILE else path
        data = json.loads((root / PROJECT_FILE).read_text(encoding="utf-8"))
        proj = Project(root)
        proj.name = data.get("name", root.name)
        proj.task = data.get("task", "detect")
        proj.classes = list(data.get("classes", []))
        proj.colors = list(data.get("colors", []))
        proj._sync_colors()
        proj.train = TrainConfig.from_dict(data.get("train", {}))
        for item in data.get("images", []):
            entry = ImageEntry(item["iid"], item["path"])
            proj.images.append(entry)
            proj._by_path[os.path.normcase(entry.path)] = entry
        for sub in (proj.labels_dir, proj.datasets_dir, proj.runs_dir):
            sub.mkdir(parents=True, exist_ok=True)
        proj._dirty = False
        return proj

    def save(self) -> None:
        payload = {
            "version": FORMAT_VERSION,
            "name": self.name,
            "task": self.task,
            "classes": self.classes,
            "colors": self.colors,
            "train": self.train.to_dict(),
            "images": [{"iid": e.iid, "path": e.path} for e in self.images],
        }
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.json_path)
        self._dirty = False

    @property
    def dirty(self) -> bool:
        return self._dirty

    def touch(self) -> None:
        self._dirty = True

    # ---------------------------------------------------------------- classes

    def _sync_colors(self) -> None:
        while len(self.colors) < len(self.classes):
            self.colors.append(PALETTE[len(self.colors) % len(PALETTE)])
        del self.colors[len(self.classes):]

    def color_for(self, cls: int) -> str:
        if 0 <= cls < len(self.colors):
            return self.colors[cls]
        return PALETTE[cls % len(PALETTE)] if cls >= 0 else "#888888"

    def class_name(self, cls: int) -> str:
        if 0 <= cls < len(self.classes):
            return self.classes[cls]
        return f"class_{cls}"

    def add_class(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("Class name cannot be empty.")
        if name in self.classes:
            return self.classes.index(name)
        self.classes.append(name)
        self._sync_colors()
        self.touch()
        return len(self.classes) - 1

    def rename_class(self, index: int, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Class name cannot be empty.")
        if name in self.classes and self.classes.index(name) != index:
            raise ValueError(f"A class named {name!r} already exists.")
        self.classes[index] = name
        self.touch()

    def set_color(self, index: int, color: str) -> None:
        self._sync_colors()
        self.colors[index] = color
        self.touch()

    def delete_class(self, index: int) -> int:
        """Remove a class and rewrite every label file that referenced it.

        Shapes of the deleted class are dropped; higher class ids shift down so
        the id space stays contiguous, which is what YOLO requires.
        Returns the number of shapes removed.
        """
        del self.classes[index]
        self._sync_colors()
        removed = 0
        for entry in self.images:
            lpath = self.label_path(entry)
            shapes = load_labels(lpath)
            if not shapes:
                continue
            kept: List[Shape] = []
            changed = False
            for shape in shapes:
                if shape.cls == index:
                    removed += 1
                    changed = True
                    continue
                if shape.cls > index:
                    shape.cls -= 1
                    changed = True
                kept.append(shape)
            if changed:
                save_labels(lpath, kept)
        self.touch()
        return removed

    # ----------------------------------------------------------------- images

    def _make_iid(self, path: Path) -> str:
        digest = hashlib.sha1(os.path.normcase(str(path)).encode("utf-8")).hexdigest()[:8]
        return f"{_slug(path.stem)}_{digest}"

    def add_images(self, paths: Iterable[Path]) -> List[ImageEntry]:
        """Register images, skipping duplicates and non-image files."""
        added: List[ImageEntry] = []
        for raw in paths:
            path = Path(raw).resolve()
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                continue
            key = os.path.normcase(path.as_posix())
            if key in self._by_path:
                continue
            entry = ImageEntry(self._make_iid(path), path.as_posix())
            self.images.append(entry)
            self._by_path[key] = entry
            added.append(entry)
        if added:
            self.touch()
        return added

    def add_folder(self, folder: Path, recursive: bool = True) -> List[ImageEntry]:
        folder = Path(folder)
        it = folder.rglob("*") if recursive else folder.glob("*")
        return self.add_images(sorted(p for p in it if p.suffix.lower() in IMAGE_SUFFIXES))

    def remove_images(self, entries: Sequence[ImageEntry], delete_labels: bool = True) -> None:
        targets = {e.iid for e in entries}
        for entry in entries:
            if delete_labels:
                lpath = self.label_path(entry)
                if lpath.exists():
                    try:
                        lpath.unlink()
                    except OSError:
                        pass
            self._by_path.pop(os.path.normcase(entry.path), None)
        self.images = [e for e in self.images if e.iid not in targets]
        self.touch()

    # ------------------------------------------------------------- label data

    def shapes_for(self, entry: ImageEntry) -> List[Shape]:
        return load_labels(self.label_path(entry))

    def write_shapes(self, entry: ImageEntry, shapes: Sequence[Shape]) -> None:
        save_labels(self.label_path(entry), shapes)

    def is_annotated(self, entry: ImageEntry) -> bool:
        return self.label_path(entry).exists()

    def stats(self) -> dict:
        """Counts used by the dashboard and the pre-train sanity check."""
        per_class = [0] * max(1, len(self.classes))
        annotated = 0
        boxes = polygons = 0
        missing = 0
        for entry in self.images:
            if not entry.exists():
                missing += 1
            shapes = self.shapes_for(entry)
            if shapes:
                annotated += 1
            for shape in shapes:
                if 0 <= shape.cls < len(per_class):
                    per_class[shape.cls] += 1
                if shape.is_box:
                    boxes += 1
                else:
                    polygons += 1
        return {
            "images": len(self.images),
            "annotated": annotated,
            "unannotated": len(self.images) - annotated,
            "missing_files": missing,
            "boxes": boxes,
            "polygons": polygons,
            "instances": boxes + polygons,
            "per_class": per_class,
        }
