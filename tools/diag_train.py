"""Minimal training run with ultralytics output left on the console.

Used to localise a stall: the smoke test captures worker stderr and only prints
it on failure, which is useless when the run hangs instead of failing.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from smoke_test import build_project, synthesise            # noqa: E402
from yolostudio.core import dataset as dataset_mod          # noqa: E402


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="yolostudio-diag-"))
    print(f"workspace: {workdir}", flush=True)

    truth = synthesise(workdir / "images", 16)
    project = build_project(workdir / "proj", truth)
    export = dataset_mod.export(project, task="detect", name="ts",
                                val_split=0.25, seed=0)
    print(f"data.yaml: {export.yaml_path}", flush=True)

    print("importing ultralytics...", flush=True)
    from ultralytics import YOLO

    weights = Path.home() / ".yolostudio" / "weights" / "yolo11n.pt"
    print(f"loading {weights} (exists={weights.exists()})", flush=True)
    model = YOLO(str(weights))

    print("calling train()...", flush=True)
    model.train(data=str(export.yaml_path), epochs=2, imgsz=416, batch=8,
                device=0, workers=0, project=str(project.runs_dir), name="diag",
                exist_ok=True, plots=False, val=True)
    print("train() returned", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
