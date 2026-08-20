"""Unit tests for the project/annotation/dataset layer.

Depends only on the standard library, so it runs without torch or PySide6::

    .\\.venv\\Scripts\\python.exe tools\\test_core.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolostudio.core import dataset as dataset_mod          # noqa: E402
from yolostudio.core.annotations import (BOX, POLYGON, Shape, convert_for_task,  # noqa: E402
                                         load_labels, save_labels)
from yolostudio.core.models import suggest_batch            # noqa: E402
from yolostudio.core.project import Project                 # noqa: E402

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) < tol


def make_images(folder: Path, count: int) -> list:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        path = folder / f"img_{i:02d}.png"
        # A 1x1 PNG is enough: nothing here decodes pixels.
        path.write_bytes(bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000a49444154789c6300010000050001-0d0a2db40000000049454e44ae"
            "426082".replace("-", "")))
        paths.append(path)
    return paths


def test_shape_geometry() -> None:
    print("\nShape geometry")
    box = Shape(0, BOX, [0.5, 0.5, 0.4, 0.2])
    x1, y1, x2, y2 = box.xyxy()
    check("box xyxy", approx(x1, 0.3) and approx(y1, 0.4) and approx(x2, 0.7) and approx(y2, 0.6))
    check("box area", approx(box.area(), 0.08), f"{box.area():.4f}")

    poly = Shape(1, POLYGON, [0.1, 0.1, 0.5, 0.2, 0.3, 0.6])
    check("polygon point count", len(poly.points()) == 3)
    check("polygon bbox", approx(poly.xyxy()[0], 0.1) and approx(poly.xyxy()[2], 0.5))

    as_box = poly.as_box()
    check("polygon -> box keeps class", as_box.cls == 1 and as_box.is_box)
    check("polygon -> box matches bbox", approx(as_box.coords[2], 0.4), f"w={as_box.coords[2]:.3f}")

    as_poly = box.as_polygon()
    check("box -> polygon has 4 corners", len(as_poly.points()) == 4)
    check("box -> polygon round-trips", approx(as_poly.as_box().coords[2], 0.4))

    clamped = Shape(0, POLYGON, [-0.5, 0.2, 1.4, 0.3, 0.5, 0.9]).clamped()
    check("clamping to 0..1", all(0.0 <= v <= 1.0 for v in clamped.coords))


def test_label_io(tmp: Path) -> None:
    print("\nLabel file I/O")
    path = tmp / "labels" / "a.txt"
    shapes = [Shape(0, BOX, [0.5, 0.5, 0.2, 0.2]),
              Shape(2, POLYGON, [0.1, 0.1, 0.4, 0.1, 0.4, 0.5, 0.1, 0.5])]
    save_labels(path, shapes)
    text = path.read_text(encoding="utf-8")
    check("box line has 5 tokens", len(text.splitlines()[0].split()) == 5)
    check("polygon line has 9 tokens", len(text.splitlines()[1].split()) == 9)

    back = load_labels(path)
    check("shape count round-trips", len(back) == 2)
    check("kinds round-trip", back[0].is_box and back[1].is_polygon)
    check("class ids round-trip", back[0].cls == 0 and back[1].cls == 2)
    check("coordinates round-trip", approx(back[0].coords[2], 0.2))

    save_labels(path, [])
    check("empty save deletes the file", not path.exists())

    save_labels(path, [Shape(0, BOX, [0.5, 0.5, 0.0000001, 0.0000001])])
    check("degenerate shapes are dropped", not path.exists())

    bad = tmp / "labels" / "bad.txt"
    bad.write_text("0 0.5 0.5 0.2\n1 0.1 0.1 0.4 0.1 0.4\nnot a label\n\n"
                   "0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    parsed = load_labels(bad)
    check("malformed lines are skipped", len(parsed) == 1, f"parsed {len(parsed)}")


def test_convert() -> None:
    print("\nTask conversion")
    mixed = [Shape(0, BOX, [0.5, 0.5, 0.2, 0.2]),
             Shape(1, POLYGON, [0.1, 0.1, 0.4, 0.1, 0.4, 0.5])]
    detect = convert_for_task(mixed, "detect")
    check("detect export is all boxes", all(s.is_box for s in detect))
    segment = convert_for_task(mixed, "segment")
    check("segment export is all polygons", all(s.is_polygon for s in segment))
    check("segment export keeps classes", [s.cls for s in segment] == [0, 1])


def test_project(tmp: Path) -> None:
    print("\nProject")
    images = make_images(tmp / "src", 10)
    project = Project.create(tmp / "proj", "test", "detect", ["cat", "dog"])
    added = project.add_images(images)
    check("images registered", len(added) == 10)
    check("duplicate import is a no-op", len(project.add_images(images)) == 0)
    check("ids are unique", len({e.iid for e in project.images}) == 10)

    for index, entry in enumerate(project.images):
        cls = index % 2
        project.write_shapes(entry, [Shape(cls, BOX, [0.5, 0.5, 0.3, 0.3])])
    stats = project.stats()
    check("annotated count", stats["annotated"] == 10, str(stats["annotated"]))
    check("per-class counts", stats["per_class"] == [5, 5], str(stats["per_class"]))

    project.save()
    reopened = Project.open(tmp / "proj")
    check("project reopens", len(reopened.images) == 10 and reopened.classes == ["cat", "dog"])
    check("colours persist", len(reopened.colors) == 2)

    # Deleting class 0 must drop its shapes and renumber class 1 down to 0.
    removed = reopened.delete_class(0)
    check("delete_class removed shapes", removed == 5, str(removed))
    after = reopened.stats()
    check("remaining shapes renumbered", after["per_class"] == [5], str(after["per_class"]))
    check("classes list updated", reopened.classes == ["dog"])


def test_split() -> None:
    print("\nDataset split")

    class Fake:
        def __init__(self, i):
            self.iid = str(i)

    items = [Fake(i) for i in range(100)]
    train, val, test = dataset_mod.split_images(items, 0.2, 0.1, seed=3)
    check("split sizes", (len(train), len(val), len(test)) == (70, 20, 10),
          f"{len(train)}/{len(val)}/{len(test)}")
    check("no overlap", len({i.iid for i in train + val + test}) == 100)

    again = dataset_mod.split_images(items, 0.2, 0.1, seed=3)
    check("split is deterministic", [i.iid for i in again[0]] == [i.iid for i in train])
    different = dataset_mod.split_images(items, 0.2, 0.1, seed=4)
    check("different seed shuffles", [i.iid for i in different[0]] != [i.iid for i in train])

    tiny_train, tiny_val, _ = dataset_mod.split_images([Fake(0), Fake(1)], 0.0, 0.0, seed=0)
    check("val is never empty with 2+ images", len(tiny_val) == 1 and len(tiny_train) == 1)


def test_export(tmp: Path) -> None:
    print("\nDataset export")
    images = make_images(tmp / "src2", 12)
    project = Project.create(tmp / "proj2", "export", "detect", ["a", "b"])
    project.add_images(images)
    for index, entry in enumerate(project.images):
        shapes = [Shape(0, BOX, [0.5, 0.5, 0.3, 0.3])]
        if index % 3 == 0:
            shapes.append(Shape(1, POLYGON, [0.1, 0.1, 0.4, 0.1, 0.4, 0.5, 0.1, 0.5]))
        project.write_shapes(entry, shapes)

    result = dataset_mod.export(project, task="detect", name="ds", val_split=0.25, seed=0)
    check("data.yaml written", result.yaml_path.exists())
    yaml_text = result.yaml_path.read_text(encoding="utf-8")
    check("yaml lists both classes", '0: "a"' in yaml_text and '1: "b"' in yaml_text)
    check("yaml has train and val", "train: images/train" in yaml_text
          and "val: images/val" in yaml_text)
    check("no test key when test_split is 0", "test:" not in yaml_text)

    train_labels = sorted((result.root / "labels" / "train").glob("*.txt"))
    train_images = sorted((result.root / "images" / "train").glob("*"))
    check("one label per image", len(train_labels) == len(train_images),
          f"{len(train_labels)} vs {len(train_images)}")
    check("stems match", {p.stem for p in train_labels} == {p.stem for p in train_images})
    check("detect export has only 5-token lines",
          all(len(line.split()) == 5
              for p in train_labels for line in p.read_text().splitlines() if line))

    seg = dataset_mod.export(project, task="segment", name="ds_seg", val_split=0.25, seed=0)
    seg_labels = sorted((seg.root / "labels" / "train").glob("*.txt"))
    check("segment export has only polygon lines",
          all(len(line.split()) >= 9
              for p in seg_labels for line in p.read_text().splitlines() if line))

    # Re-exporting must replace, not accumulate.
    again = dataset_mod.export(project, task="detect", name="ds", val_split=0.25, seed=0)
    check("re-export is idempotent",
          len(list((again.root / "images" / "train").glob("*"))) == len(train_images))


def test_batch_hint() -> None:
    print("\nBatch suggestion")
    big = suggest_batch("yolo11n.pt", 640, 16)
    small = suggest_batch("yolo11x.pt", 640, 16)
    check("smaller model allows a larger batch", big > small, f"{big} vs {small}")
    check("larger images shrink the batch",
          suggest_batch("yolo11s.pt", 1280, 16) < suggest_batch("yolo11s.pt", 640, 16))
    check("more VRAM grows the batch",
          suggest_batch("yolo11s.pt", 640, 24) > suggest_batch("yolo11s.pt", 640, 16))
    check("never returns zero", suggest_batch("yolo11x.pt", 1280, 4) >= 1)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="yolostudio-core-"))
    try:
        test_shape_geometry()
        test_label_io(tmp)
        test_convert()
        test_project(tmp)
        test_split()
        test_export(tmp)
        test_batch_hint()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 52)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: " + ", ".join(FAILURES))
    else:
        print("  all core tests passed")
    print("=" * 52)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
