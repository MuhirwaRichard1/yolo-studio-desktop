"""Out-of-process ultralytics runner.

Launched as ``python -m yolostudio.worker <config.json>``. The config's
``command`` selects the job: ``probe``, ``train``, ``val``, ``predict`` or
``export``.

Two streams come back to the GUI:

* **stdout** -- one JSON object per line, the structured event protocol below.
* **stderr** -- ultralytics' own console output, shown verbatim in the log pane.

Keeping them apart is why ``sys.stdout`` is redirected to stderr *before*
ultralytics is imported: its logger binds to whatever ``sys.stdout`` is at
import time, and a stray progress bar in the middle of the JSON stream would
break the parser.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

# Real stdout is reserved for the event protocol. Do this first: anything that
# imports ultralytics afterwards inherits the redirected stream.
_EVENTS = sys.stdout
sys.stdout = sys.stderr


def emit(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    try:
        _EVENTS.write(json.dumps(payload, default=str) + "\n")
        _EVENTS.flush()
    except (ValueError, OSError):
        pass  # GUI closed the pipe; the job is being cancelled.


def _float(value: Any) -> Any:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


# Ultralytics runs its own asset downloads for things we never asked for -- most
# notably the AMP check, which fetches a small model to compare fp16 and fp32
# output before epoch 1. On a flaky connection that hangs a run before any epoch
# starts, with the GPU sitting idle and nothing in the log to explain it.
#
# ``attempt_download_asset`` looks in ``SETTINGS["weights_dir"]`` before hitting
# the network, so pointing that at our cache and pre-fetching the probe model
# turns the whole thing into a local file read.
FALLBACK_AMP_MODEL = "yolo11n.pt"


def amp_probe_model() -> str:
    """The checkpoint ultralytics' AMP check will try to load.

    The name is hardcoded inside ultralytics and changes between releases --
    8.3 used ``yolo11n.pt``, 8.4 uses ``yolo26n.pt`` -- so read it out of the
    installed source instead of guessing and silently pre-fetching the wrong
    file.
    """
    try:
        import inspect
        import re

        from ultralytics.utils import checks

        source = inspect.getsource(checks.check_amp)
        match = re.search(r'YOLO\(\s*["\']([\w.\-]+\.pt)["\']', source)
        if match:
            return match.group(1)
    except Exception:
        pass
    return FALLBACK_AMP_MODEL


def align_weights_dir() -> None:
    from yolostudio.core import weights

    try:
        from ultralytics import settings as ul_settings

        target = str(weights.weights_dir())
        if str(ul_settings.get("weights_dir", "")) != target:
            ul_settings.update({"weights_dir": target})
        # Analytics upload is another network call that can stall a run on a
        # poor connection, and this tool is meant to work entirely offline.
        if ul_settings.get("sync", False):
            ul_settings.update({"sync": False})
        emit("status", msg=f"Weights cache: {target}")
    except Exception as exc:
        emit("status", msg=f"Could not set ultralytics weights_dir ({exc})")


def resolve_model(cfg: Dict[str, Any]) -> str:
    """Turn a catalogue name into a cached local checkpoint path.

    Downloading here rather than letting ultralytics do it means one clear
    progress stream and one actionable error, instead of a run that dies several
    minutes in with 'Retry limit reached'.
    """
    from yolostudio.core import weights

    model = str(cfg.get("model", ""))
    if not weights.is_bare_name(model):
        return model

    last = [0.0]

    def progress(done: int, total: int) -> None:
        now = time.time()
        if now - last[0] < 0.4 and done != total:
            return
        last[0] = now
        emit("download", name=model, done=done, total=total)

    emit("status", msg=f"Resolving {model}")
    path = weights.resolve(model, progress, lambda note: emit("status", msg=note))
    emit("status", msg=f"Using {path}")
    return path


# ------------------------------------------------------------------- commands


def cmd_probe(_: Dict[str, Any]) -> None:
    """Report the runtime the GUI is about to train on."""
    info: Dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_build"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        devices = []
        for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "vram_gb": round(props.total_memory / (1024 ** 3), 1),
                "capability": f"{props.major}.{props.minor}",
            })
        info["devices"] = devices
    except Exception as exc:  # torch missing or broken install
        info["torch_error"] = str(exc)
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception as exc:
        info["ultralytics_error"] = str(exc)
    emit("probe", info=info)


def cmd_train(cfg: Dict[str, Any]) -> None:
    from ultralytics import YOLO

    args = dict(cfg["args"])
    model = YOLO(resolve_model(cfg))
    total_epochs = int(args.get("epochs", 100))

    # Pre-fetch the model ultralytics uses for its AMP check, so that check
    # cannot stall the run on a slow network.
    if args.get("amp", True):
        probe = amp_probe_model()
        try:
            from yolostudio.core import weights

            if weights.cached_path(probe) is None:
                emit("status", msg=f"Fetching the AMP check model ({probe})…")
                weights.ensure(probe)
        except Exception as exc:
            emit("status", msg=f"{probe} unavailable ({exc}); training without AMP")
            args["amp"] = False

    def on_train_start(trainer):
        emit("train_start",
             epochs=total_epochs,
             save_dir=str(getattr(trainer, "save_dir", "")))

    def on_fit_epoch_end(trainer):
        """Fires after validation, so val metrics for this epoch are present."""
        metrics: Dict[str, Any] = {}
        for key, value in (getattr(trainer, "metrics", None) or {}).items():
            num = _float(value)
            if num is not None:
                metrics[key] = num
        try:
            losses = trainer.label_loss_items(trainer.tloss, prefix="train")
            for key, value in (losses or {}).items():
                num = _float(value)
                if num is not None:
                    metrics[key] = num
        except Exception:
            pass
        lr = None
        try:
            lr = _float(next(iter(trainer.optimizer.param_groups))["lr"])
        except Exception:
            pass
        mem = None
        try:
            import torch

            if torch.cuda.is_available():
                mem = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2)
        except Exception:
            pass
        emit("epoch",
             epoch=int(getattr(trainer, "epoch", 0)) + 1,
             epochs=total_epochs,
             metrics=metrics,
             lr=lr,
             vram_gb=mem)

    def on_train_end(trainer):
        best = getattr(trainer, "best", None)
        last = getattr(trainer, "last", None)
        emit("train_end",
             best=str(best) if best else None,
             last=str(last) if last else None,
             save_dir=str(getattr(trainer, "save_dir", "")))

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    results = model.train(**args)

    summary = {}
    box = getattr(getattr(results, "box", None), "map", None)
    if box is not None:
        summary["mAP50-95"] = _float(box)
        summary["mAP50"] = _float(getattr(results.box, "map50", None))
    seg = getattr(results, "seg", None)
    if seg is not None:
        summary["mask mAP50-95"] = _float(getattr(seg, "map", None))
    emit("result", summary=summary, save_dir=str(getattr(results, "save_dir", "")))


def cmd_val(cfg: Dict[str, Any]) -> None:
    from ultralytics import YOLO

    model = YOLO(resolve_model(cfg))
    results = model.val(**cfg.get("args", {}))
    summary = {
        "mAP50-95": _float(getattr(getattr(results, "box", None), "map", None)),
        "mAP50": _float(getattr(getattr(results, "box", None), "map50", None)),
        "precision": _float(getattr(getattr(results, "box", None), "mp", None)),
        "recall": _float(getattr(getattr(results, "box", None), "mr", None)),
    }
    seg = getattr(results, "seg", None)
    if seg is not None:
        summary["mask mAP50-95"] = _float(getattr(seg, "map", None))
    emit("result", summary={k: v for k, v in summary.items() if v is not None},
         save_dir=str(getattr(results, "save_dir", "")))


def cmd_predict(cfg: Dict[str, Any]) -> None:
    """Pre-label images, writing YOLO ``.txt`` files the annotator can correct."""
    from ultralytics import YOLO

    model = YOLO(resolve_model(cfg))
    items = cfg["items"]                     # [{"image": ..., "label": ...}, ...]
    # Model class index -> project class index. Anything unmapped is dropped.
    class_map = {int(k): int(v) for k, v in cfg.get("class_map", {}).items()}
    conf = float(cfg.get("conf", 0.25))
    iou = float(cfg.get("iou", 0.7))
    imgsz = int(cfg.get("imgsz", 640))
    device = cfg.get("device", "0")
    want_polygons = cfg.get("shape", "box") == "polygon"
    max_det = int(cfg.get("max_det", 300))

    total = len(items)
    written = 0
    instances = 0

    for index, item in enumerate(items, start=1):
        image_path = item["image"]
        try:
            preds = model.predict(source=image_path, conf=conf, iou=iou, imgsz=imgsz,
                                  device=device, max_det=max_det, verbose=False)
        except Exception as exc:
            emit("progress", done=index, total=total, name=Path(image_path).name,
                 error=str(exc))
            continue

        lines = []
        for result in preds:
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)
            if boxes is None:
                continue
            polygons = None
            if want_polygons and masks is not None:
                polygons = masks.xyn  # already normalised, one array per instance

            for i in range(len(boxes)):
                model_cls = int(boxes.cls[i].item())
                if model_cls not in class_map:
                    continue
                target = class_map[model_cls]
                if polygons is not None and i < len(polygons) and len(polygons[i]) >= 3:
                    flat = [f"{float(v):.6f}" for pt in polygons[i] for v in pt]
                    lines.append(f"{target} " + " ".join(flat))
                else:
                    cx, cy, w, h = boxes.xywhn[i].tolist()
                    lines.append(f"{target} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_path = Path(item["label"])
        if lines:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            written += 1
            instances += len(lines)
        elif cfg.get("write_empty", False) and label_path.exists():
            label_path.unlink()

        emit("progress", done=index, total=total, name=Path(image_path).name,
             found=len(lines))

    emit("result", summary={"images labelled": written,
                            "instances written": instances,
                            "images seen": total})


def cmd_export(cfg: Dict[str, Any]) -> None:
    from ultralytics import YOLO

    model = YOLO(resolve_model(cfg))
    out = model.export(**cfg.get("args", {}))
    emit("result", summary={"exported": str(out)})


def cmd_names(cfg: Dict[str, Any]) -> None:
    """Report a checkpoint's class names and task, for the class-mapping UI."""
    from ultralytics import YOLO

    model = YOLO(resolve_model(cfg))
    names = getattr(model, "names", None) or {}
    emit("names",
         names={int(k): str(v) for k, v in dict(names).items()},
         task=str(getattr(model, "task", "") or ""))


COMMANDS = {
    "probe": cmd_probe,
    "names": cmd_names,
    "train": cmd_train,
    "val": cmd_val,
    "predict": cmd_predict,
    "export": cmd_export,
}


def main() -> int:
    if len(sys.argv) < 2:
        emit("error", msg="worker: missing config path")
        return 2
    try:
        cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    except Exception as exc:
        emit("error", msg=f"worker: unreadable config ({exc})")
        return 2

    command = cfg.get("command", "")
    handler = COMMANDS.get(command)
    if handler is None:
        emit("error", msg=f"worker: unknown command {command!r}")
        return 2

    # Keep ultralytics from phoning home or opening a settings wizard.
    os.environ.setdefault("YOLO_VERBOSE", "true")
    os.environ.setdefault("ULTRALYTICS_OFFLINE_SYNC", "1")

    if command != "probe":
        align_weights_dir()

    try:
        handler(cfg)
        emit("done", ok=True)
        return 0
    except KeyboardInterrupt:
        emit("error", msg="Cancelled.")
        return 130
    except Exception as exc:
        message = str(exc)
        hint = ""
        low = message.lower()
        if "out of memory" in low:
            hint = ("CUDA ran out of memory. Lower 'batch', reduce 'imgsz', or pick a "
                    "smaller model scale, then start again.")
        elif "no labels found" in low or "no images found" in low:
            hint = ("The dataset directory looks empty. Re-export the dataset from the "
                    "Dataset tab and check the split counts.")
        emit("error", msg=message, hint=hint, trace=traceback.format_exc())
        emit("done", ok=False)
        return 1


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # Windows dataloader workers re-import this file.
    sys.exit(main())
