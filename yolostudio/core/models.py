"""The catalogue of finetunable YOLO checkpoints, plus VRAM sizing hints.

Ultralytics downloads any name in ``ALL`` on first use and caches it, so the
catalogue is just metadata -- nothing here touches the network.

``suggest_batch`` exists because the single most common way a first finetune
fails is a CUDA OOM three minutes in. The numbers are conservative starting
points for AMP training measured against the listed VRAM, not hard limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

DETECT = "detect"
SEGMENT = "segment"


@dataclass(frozen=True)
class ModelInfo:
    key: str            # weight filename ultralytics resolves, e.g. "yolo11s.pt"
    family: str         # "YOLO11"
    scale: str          # "n" | "s" | "m" | "l" | "x"
    task: str
    params_m: float     # millions of parameters
    coco: str           # reported COCO mAP50-95, for relative comparison
    # Rough per-GPU batch ceiling at imgsz=640 with AMP, 16 GB VRAM.
    batch16: int

    @property
    def label(self) -> str:
        return f"{self.family}{self.scale}"

    @property
    def blurb(self) -> str:
        return f"{self.params_m:.1f}M params · COCO mAP {self.coco}"


def _family(family: str, task: str, prefix: str, rows) -> List[ModelInfo]:
    suffix = "-seg" if task == SEGMENT else ""
    return [
        ModelInfo(f"{prefix}{scale}{suffix}.pt", family, scale, task, params, coco, batch)
        for scale, params, coco, batch in rows
    ]


# (scale, params_m, coco_map, batch_at_16gb)
_V8_DET = [("n", 3.2, "37.3", 64), ("s", 11.2, "44.9", 48), ("m", 25.9, "50.2", 28),
           ("l", 43.7, "52.9", 16), ("x", 68.2, "53.9", 10)]
_V8_SEG = [("n", 3.4, "30.5", 40), ("s", 11.8, "36.8", 28), ("m", 27.3, "40.8", 16),
           ("l", 46.0, "42.6", 10), ("x", 71.8, "43.4", 6)]
_11_DET = [("n", 2.6, "39.5", 64), ("s", 9.4, "47.0", 48), ("m", 20.1, "51.5", 28),
           ("l", 25.3, "53.4", 20), ("x", 56.9, "54.7", 12)]
_11_SEG = [("n", 2.9, "32.0", 40), ("s", 10.1, "37.8", 28), ("m", 22.4, "41.5", 16),
           ("l", 27.6, "42.9", 12), ("x", 62.1, "43.8", 8)]
_12_DET = [("n", 2.6, "40.6", 56), ("s", 9.3, "48.0", 40), ("m", 20.2, "52.5", 24),
           ("l", 26.4, "53.7", 16), ("x", 59.1, "55.2", 10)]

CATALOG: List[ModelInfo] = (
    _family("YOLO11", DETECT, "yolo11", _11_DET)
    + _family("YOLO11", SEGMENT, "yolo11", _11_SEG)
    + _family("YOLOv8", DETECT, "yolov8", _V8_DET)
    + _family("YOLOv8", SEGMENT, "yolov8", _V8_SEG)
    + _family("YOLO12", DETECT, "yolo12", _12_DET)
)

BY_KEY: Dict[str, ModelInfo] = {m.key: m for m in CATALOG}

# Families in the order the picker should show them.
FAMILY_ORDER = ["YOLO11", "YOLOv8", "YOLO12"]

FAMILY_NOTES = {
    "YOLO11": "Current default. Best accuracy-per-parameter; detect and segment.",
    "YOLOv8": "Most battle-tested. Widest ecosystem and export support.",
    "YOLO12": "Attention-centric, detect only here. Slightly slower per epoch.",
}


def for_task(task: str) -> List[ModelInfo]:
    return [m for m in CATALOG if m.task == task]


def families_for_task(task: str) -> List[str]:
    seen = [f for f in FAMILY_ORDER if any(m.family == f and m.task == task for m in CATALOG)]
    return seen


def get(key: str) -> Optional[ModelInfo]:
    return BY_KEY.get(key)


def suggest_batch(key: str, imgsz: int = 640, vram_gb: float = 16.0) -> int:
    """Scale the tabulated batch by image size and available VRAM.

    Activation memory grows roughly with pixel count, so halving batch for a
    doubled edge length is the right first approximation.
    """
    info = BY_KEY.get(key)
    base = info.batch16 if info else 16
    scaled = base * (640.0 / max(320, imgsz)) ** 2 * (vram_gb / 16.0)
    return max(1, min(128, int(scaled)))


def find_checkpoints(runs_dir: Path) -> List[Path]:
    """Every ``best.pt``/``last.pt`` under a project's runs directory, newest first."""
    if not runs_dir.exists():
        return []
    found = [p for p in runs_dir.rglob("weights/*.pt") if p.stem in ("best", "last")]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def describe_checkpoint(path: Path) -> str:
    """``run-name / best.pt`` -- enough to tell two runs apart in a combo box."""
    parts = path.parts
    run = parts[-3] if len(parts) >= 3 else path.parent.name
    return f"{run} / {path.name}"
