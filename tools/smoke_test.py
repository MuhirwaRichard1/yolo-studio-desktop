"""End-to-end check: synthesise a dataset, export a split, finetune, predict.

Run it after bootstrap to confirm the whole pipeline works on this machine
before pointing the GUI at real data::

    .\\.venv\\Scripts\\python.exe tools\\smoke_test.py
    .\\.venv\\Scripts\\python.exe tools\\smoke_test.py --epochs 5 --device cpu

It writes to a temporary project directory and prints a PASS/FAIL summary.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolostudio.core import dataset as dataset_mod          # noqa: E402
from yolostudio.core.annotations import BOX, POLYGON, Shape  # noqa: E402
from yolostudio.core.project import Project                  # noqa: E402

IMG_W, IMG_H = 416, 416
CLASSES = ["square", "circle"]


def synthesise(folder: Path, count: int, seed: int = 7) -> list:
    """Draw a coloured square or circle on noise; return per-image truth boxes."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    truth = []
    for index in range(count):
        base = Image.new("RGB", (IMG_W, IMG_H),
                         (rng.randint(20, 70), rng.randint(20, 70), rng.randint(20, 70)))
        draw = ImageDraw.Draw(base)
        for _ in range(40):  # texture, so the model has something to ignore
            x, y = rng.randint(0, IMG_W), rng.randint(0, IMG_H)
            draw.point((x, y), fill=(rng.randint(0, 255),) * 3)

        shapes = []
        for cls in range(2):
            if rng.random() < 0.25:
                continue
            size = rng.randint(60, 130)
            x1 = rng.randint(4, IMG_W - size - 4)
            y1 = rng.randint(4, IMG_H - size - 4)
            x2, y2 = x1 + size, y1 + size
            color = (230, 80, 60) if cls == 0 else (70, 170, 240)
            if cls == 0:
                draw.rectangle([x1, y1, x2, y2], fill=color)
            else:
                draw.ellipse([x1, y1, x2, y2], fill=color)
            shapes.append((cls, x1, y1, x2, y2))

        if not shapes:                       # never emit a blank image
            draw.rectangle([100, 100, 200, 200], fill=(230, 80, 60))
            shapes.append((0, 100, 100, 200, 200))

        path = folder / f"synthetic_{index:03d}.png"
        base.save(path)
        truth.append((path, shapes))
    return truth


def build_project(root: Path, truth: list) -> Project:
    project = Project.create(root, "smoke", task="detect", classes=CLASSES)
    project.add_images(path for path, _ in truth)
    for path, shapes in truth:
        entry = next(e for e in project.images if Path(e.path) == path)
        out = []
        for cls, x1, y1, x2, y2 in shapes:
            cx, cy = (x1 + x2) / 2 / IMG_W, (y1 + y2) / 2 / IMG_H
            w, h = (x2 - x1) / IMG_W, (y2 - y1) / IMG_H
            out.append(Shape(cls, BOX, [cx, cy, w, h]))
            # One polygon per image too, to exercise mixed-shape storage.
            if cls == 1:
                out.append(Shape(cls, POLYGON, [
                    x1 / IMG_W, y1 / IMG_H, x2 / IMG_W, y1 / IMG_H,
                    x2 / IMG_W, y2 / IMG_H, x1 / IMG_W, y2 / IMG_H]))
        project.write_shapes(entry, out)
    project.save()
    return project


def run_worker(config: dict, cwd: Path) -> tuple:
    """Run the real worker the GUI uses; return (ok, events).

    stderr goes to a file rather than a second pipe. Draining only stdout while
    ultralytics fills the stderr pipe buffer deadlocks: the worker blocks on
    write, so it emits no further stdout, and both processes wait on each other
    forever. (The GUI is immune -- QProcess drains both channels via signals.)
    """
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(config, handle, default=str)
    handle.close()

    # mkstemp hands back an open descriptor; close it before reopening by name,
    # or Windows keeps a handle and refuses to unlink the file afterwards.
    log_fd, log_name = tempfile.mkstemp(prefix="yolostudio-worker-", suffix=".log")
    os.close(log_fd)
    log_path = Path(log_name)
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "yolostudio.worker", handle.name],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=log_file,
        text=True, encoding="utf-8", errors="replace")

    events = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(message)
        kind = message.get("event")
        if kind == "epoch":
            metrics = message.get("metrics", {})
            print(f"    epoch {message['epoch']}/{message['epochs']}  "
                  f"mAP50-95={metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        elif kind == "progress" and message.get("done", 0) % 5 == 0:
            print(f"    {message['done']}/{message['total']}")
        elif kind == "error":
            print(f"    ERROR: {message.get('msg')}")
            print(f"    HINT : {message.get('hint')}")
    proc.wait()
    log_file.close()
    stderr = log_path.read_text(encoding="utf-8", errors="replace")
    Path(handle.name).unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
    ok = proc.returncode == 0 and not any(e.get("event") == "error" for e in events)
    if not ok:
        print("    --- worker stderr (tail) ---")
        print("\n".join(stderr.splitlines()[-25:]))
    return ok, events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--keep", action="store_true", help="do not delete the temp project")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="yolostudio-smoke-"))
    results = {}
    try:
        print(f"Workspace: {workdir}\n")

        print("[1/6] probe")
        ok, events = run_worker({"command": "probe"}, workdir)
        info = next((e["info"] for e in events if e.get("event") == "probe"), {})
        print(f"    torch={info.get('torch')} cuda={info.get('cuda_available')} "
              f"ultralytics={info.get('ultralytics')}")
        for device in info.get("devices", []):
            print(f"    gpu {device['index']}: {device['name']} {device['vram_gb']} GB")
        results["probe"] = ok
        if not info.get("cuda_available") and args.device != "cpu":
            print("    CUDA unavailable - falling back to --device cpu")
            args.device = "cpu"

        print("\n[2/6] synthesise images + project")
        truth = synthesise(workdir / "images", args.images)
        project = build_project(workdir / "project", truth)
        stats = project.stats()
        print(f"    {stats['images']} images, {stats['annotated']} annotated, "
              f"{stats['boxes']} boxes, {stats['polygons']} polygons")
        results["project"] = stats["annotated"] == args.images and stats["boxes"] > 0

        print("\n[3/6] round-trip labels through disk")
        entry = project.images[0]
        reloaded = project.shapes_for(entry)
        results["roundtrip"] = bool(reloaded) and all(
            0.0 <= v <= 1.0 for s in reloaded for v in s.coords)
        print(f"    {len(reloaded)} shapes reloaded, coordinates in range: "
              f"{results['roundtrip']}")

        print("\n[4/6] export dataset")
        export = dataset_mod.export(project, task="detect", name="train_set",
                                    val_split=0.25, test_split=0.0, seed=1)
        # ASCII only: the Windows console codepage mangles punctuation like U+00B7.
        print(f"    {export.counts} | {export.instances} instances | "
              f"{export.linked} linked / {export.copied} copied")
        print(f"    {export.yaml_path}")
        results["export"] = export.counts["train"] > 0 and export.counts["val"] > 0

        print(f"\n[5/6] finetune {args.model} for {args.epochs} epochs on device={args.device}")
        ok, events = run_worker({
            "command": "train",
            "model": args.model,
            "args": {
                "data": str(export.yaml_path),
                "epochs": args.epochs, "imgsz": 416, "batch": 8,
                "device": args.device, "workers": 0, "seed": 0,
                "project": str(project.runs_dir), "name": "smoke",
                "exist_ok": True, "plots": False, "val": True, "verbose": True,
            },
        }, workdir)
        best = next((e.get("best") for e in events if e.get("event") == "train_end"), None)
        print(f"    best weights: {best}")
        results["train"] = ok and bool(best) and Path(best).exists()

        print("\n[6/6] auto-label with the model just trained")
        if results["train"]:
            items = [{"image": e.path, "label": str(workdir / "pred" / f"{e.iid}.txt")}
                     for e in project.images[:6]]
            ok, events = run_worker({
                "command": "predict", "model": best, "items": items,
                "class_map": {"0": 0, "1": 1}, "conf": 0.15, "iou": 0.7,
                "imgsz": 416, "device": args.device, "shape": "box",
            }, workdir)
            summary = next((e.get("summary") for e in events
                            if e.get("event") == "result"), {})
            print(f"    {summary}")
            results["predict"] = ok and summary.get("instances written", 0) > 0
        else:
            results["predict"] = False
            print("    skipped (training did not produce weights)")

        print("\n" + "=" * 52)
        for name, passed in results.items():
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        print("=" * 52)
        return 0 if all(results.values()) else 1
    finally:
        if args.keep:
            print(f"\nKept workspace: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
