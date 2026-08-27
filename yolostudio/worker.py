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
#
# A frozen windowed build can start with sys.stdout/sys.stderr set to None, and
# losing the protocol channel would make the job look like a silent hang. Fall
# back to the raw file descriptors, then to devnull, so emit() never raises.
def _stream(existing, fd: int):
    if existing is not None:
        return existing
    try:
        return os.fdopen(fd, "w", encoding="utf-8", buffering=1, closefd=False)
    except (OSError, ValueError):
        return open(os.devnull, "w", encoding="utf-8")


_EVENTS = _stream(sys.stdout, 1)
sys.stderr = _stream(sys.stderr, 2)
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

    # Where 'Self' comes from. typing_extensions aliases typing.Self on 3.11+
    # and defines its own below that, and a frozen build that mixes the two
    # produces "Plain typing.Self is not valid as type argument" deep inside
    # torch rather than anything resembling an import error.
    try:
        import typing

        import typing_extensions

        info["typing_file"] = getattr(typing, "__file__", "?")
        info["typing_has_self"] = hasattr(typing, "Self")
        info["typing_extensions_file"] = getattr(typing_extensions, "__file__", "?")
        info["self_repr"] = repr(getattr(typing_extensions, "Self", None))
        try:
            from typing import Union

            class _Probe:
                pass

            Union[_Probe, typing_extensions.Self]
            info["self_usable"] = True
        except Exception as exc:
            info["self_usable"] = False
            info["self_error"] = str(exc)
    except Exception as exc:
        info["typing_error"] = str(exc)

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
            # Include the traceback: a per-image failure is reported as one
            # short string in the UI, which is not enough to diagnose anything
            # that only reproduces in a packaged build.
            emit("progress", done=index, total=total, name=Path(image_path).name,
                 error=str(exc), trace=traceback.format_exc())
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


# Third-party modules each export format needs, and the pip name to install if
# it is missing. ultralytics would normally pip-install these on demand, but a
# frozen build has no pip (see the guard in main()), so the requirement has to
# be checked up front and reported as something the user can act on.
EXPORT_REQUIREMENTS = {
    "onnx": [("onnx", "onnx"), ("onnxruntime", "onnxruntime")],
    "engine": [("tensorrt", "tensorrt")],
    "openvino": [("openvino", "openvino")],
}


def missing_export_modules(fmt: str, simplify: bool = False) -> list:
    """Return the pip names of modules `fmt` needs that cannot be imported."""
    import importlib.util

    required = list(EXPORT_REQUIREMENTS.get(fmt, []))
    if fmt == "onnx" and simplify:
        required.append(("onnxslim", "onnxslim"))
    return [pip_name for module, pip_name in required
            if importlib.util.find_spec(module) is None]


def cmd_export(cfg: Dict[str, Any]) -> None:
    from ultralytics import YOLO

    args = dict(cfg.get("args", {}))
    fmt = str(args.get("format", ""))

    missing = missing_export_modules(fmt, bool(args.get("simplify")))
    if missing:
        joined = ", ".join(missing)
        if getattr(sys, "frozen", False):
            raise RuntimeError(
                f"This build cannot export to {fmt.upper()}: {joined} "
                f"{'is' if len(missing) == 1 else 'are'} not bundled in the "
                f"installed application. Run YOLO Studio from source, or "
                f"rebuild with {joined} added to packaging/yolostudio.spec.")
        raise RuntimeError(
            f"{fmt.upper()} export needs {joined}. Install with: pip install {' '.join(missing)}")

    model = YOLO(resolve_model(cfg))
    out = model.export(**args)
    emit("result", summary={"exported": str(out)})


def _calibration_images(data_yaml: str, limit: int) -> list:
    """Image paths from a YOLO data.yaml's train split, for INT8 calibration."""
    import yaml

    spec = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    root = Path(spec.get("path") or Path(data_yaml).parent)
    train = spec.get("train") or "images/train"
    folder = root / train if not Path(train).is_absolute() else Path(train)
    if not folder.is_dir():
        raise RuntimeError(f"Calibration images not found at {folder}. Export the dataset first.")

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in suffixes)
    if not images:
        raise RuntimeError(f"No images under {folder} to calibrate with.")
    return images[:limit]


def _write_horizon_calibration(images: list, dest: Path, imgsz: int) -> int:
    """Letterbox images to imgsz and write them as raw float32 NCHW.

    hb_mapper reads a directory of headerless binaries, one per sample, in the
    dtype named by `cal_data_type` -- not .npy, which would put a header in
    front of the tensor.
    """
    import cv2
    import numpy as np

    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, path in enumerate(images):
        img = cv2.imread(str(path))
        if img is None:
            continue
        height, width = img.shape[:2]
        scale = min(imgsz / height, imgsz / width)
        resized = cv2.resize(img, (int(round(width * scale)), int(round(height * scale))))
        # Pad to square with 114 grey, the value ultralytics letterboxes with,
        # so calibration sees the same borders inference will.
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1)[None].astype("float32")
        tensor.tofile(str(dest / f"cal_{index:04d}.bin"))
        written += 1
    if not written:
        raise RuntimeError("None of the calibration images could be read.")
    return written


def _wsl_dataset_yaml(data_yaml: str) -> Path:
    """Rewrite a dataset descriptor so its paths resolve inside WSL.

    dataset.export() records `path:` as a Windows location (``C:/proj/...``).
    Read from inside the distro that is not absolute, so ultralytics resolves it
    against its own working directory, finds no images, and the calibration pass
    fails with an empty-dataset error that says nothing about drive letters.
    """
    import yaml
    from yolostudio.core import npu

    source = Path(data_yaml)
    spec = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    root = spec.get("path")
    if root:
        spec["path"] = npu.to_wsl_path(str(root))
    translated = source.with_name(f"{source.stem}.wsl.yaml")
    translated.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
                          encoding="utf-8")
    return translated


def cmd_export_npu(cfg: Dict[str, Any]) -> None:
    """Export for an SBC NPU by driving a Linux toolchain inside WSL."""
    from yolostudio.core import npu

    args = dict(cfg.get("args", {}))
    target = str(args.get("target", ""))
    imgsz = int(args.get("imgsz", 640))
    quantize = int(args.get("quantize", 8))
    data_yaml = args.get("data") or ""

    distro = args.get("distro") or npu.default_distro()
    if not distro:
        raise RuntimeError(
            "No WSL 2 distribution found. Install one with:  wsl --install -d Ubuntu")

    model_path = Path(resolve_model(cfg)).resolve()
    log = lambda line: print(line, flush=True)  # noqa: E731 - stdout is the log pane

    if target == "rknn":
        emit("phase", name="provision")
        print(f"[yolostudio] preparing conversion environment in {distro}")
        code = npu.run_script(distro, npu.provision_rknn_script(), log)
        if code != 0:
            raise RuntimeError(
                f"Could not build the RKNN toolchain inside {distro} (exit {code}). "
                "See the log above for the failing step.")

        emit("phase", name="convert")
        calibration = _wsl_dataset_yaml(data_yaml) if data_yaml and quantize == 8 else None
        script = npu.rknn_export_script(
            model_wsl=npu.to_wsl_path(model_path),
            chip=str(args.get("chip", "rk3588")),
            imgsz=imgsz,
            quantize=quantize,
            data_yaml_wsl=npu.to_wsl_path(calibration) if calibration else None,
        )
        code = npu.run_script(distro, script, log)
        if code != 0:
            raise RuntimeError(f"RKNN conversion failed (exit {code}).")

        produced = model_path.parent / f"{model_path.stem}_rknn_model"
        emit("result", summary={"exported": str(produced)})
        return

    if target == "horizon":
        board = str(args.get("board", "RDK X5"))
        march = npu.HORIZON_MARCH.get(board)
        image = npu.HORIZON_IMAGE.get(board)
        if not march:
            raise RuntimeError(f"Unknown D-Robotics board: {board}")

        emit("phase", name="check")
        status: Dict[str, str] = {}

        def collect(line: str) -> None:
            """Split the probe's key=value lines out of its ordinary output."""
            if "=" in line and not line.startswith(("[", " ")):
                key, _, value = line.partition("=")
                status[key.strip()] = value.strip()
            else:
                log(line)

        code = npu.run_script(distro, npu.horizon_check_script(image), collect)
        if code != 0:
            raise RuntimeError(f"Could not query the toolchain in {distro} (exit {code}).")
        if status.get("docker") == "missing":
            raise RuntimeError(
                f"Docker is not installed inside {distro}, and the D-Robotics "
                f"toolchain ships only as a container.\n"
                f"Install it with:  wsl -d {distro} -- curl -fsSL https://get.docker.com | sh")
        if status.get("docker") == "nodaemon":
            raise RuntimeError(
                f"Docker is installed in {distro} but the daemon is not running.\n"
                f"Start it with:  wsl -d {distro} -- sudo service docker start")
        if status.get("image") != "present":
            raise RuntimeError(
                f"The OpenExplorer image '{image}' is not present in {distro}.\n"
                f"It is not publicly pullable -- download the {board} toolchain from "
                f"developer.d-robotics.cc, then load it with:  docker load -i <archive>.tar")

        # Everything hb_mapper touches has to sit under one bind-mounted folder.
        workdir = model_path.parent / f"{model_path.stem}_horizon_{march}"
        workdir.mkdir(parents=True, exist_ok=True)

        emit("phase", name="onnx")
        print(f"[yolostudio] exporting ONNX for {board} ({march})")
        missing = missing_export_modules("onnx", simplify=True)
        if missing:
            raise RuntimeError(f"ONNX export needs {', '.join(missing)}.")
        from ultralytics import YOLO

        # OpenExplorer's ONNX parser tops out at opset 11, and the BPU needs a
        # static shape with detection left to the host, so NMS stays out.
        onnx_path = YOLO(str(model_path)).export(
            format="onnx", imgsz=imgsz, opset=11, simplify=True,
            dynamic=False, half=False, nms=False)
        onnx_file = Path(onnx_path)
        target_onnx = workdir / onnx_file.name
        if onnx_file.resolve() != target_onnx.resolve():
            target_onnx.write_bytes(onnx_file.read_bytes())

        emit("phase", name="calibration")
        if not data_yaml:
            raise RuntimeError(
                "The D-Robotics toolchain always quantizes, so it needs calibration "
                "images. Export the dataset first.")
        images = _calibration_images(data_yaml, int(args.get("calibration_images", 50)))
        count = _write_horizon_calibration(images, workdir / "calibration_data", imgsz)
        print(f"[yolostudio] wrote {count} calibration samples")

        config_name = "hb_mapper_config.yaml"
        (workdir / config_name).write_text(
            npu.horizon_config(target_onnx.name, march, imgsz, model_path.stem),
            encoding="utf-8")

        emit("phase", name="convert")
        code = npu.run_script(
            distro,
            npu.horizon_export_script(npu.to_wsl_path(workdir), image, config_name),
            log)
        if code != 0:
            raise RuntimeError(f"hb_mapper failed (exit {code}). See the log above.")

        produced = sorted((workdir / "output").glob("*.bin"))
        emit("result", summary={"exported": str(produced[0]) if produced else str(workdir)})
        return

    raise RuntimeError(f"Unknown NPU target: {target!r}")


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
    "export_npu": cmd_export_npu,
}


def main() -> int:
    if len(sys.argv) < 2:
        emit("error", msg="worker: missing config path")
        return 2

    # ultralytics installs missing export dependencies with
    # subprocess.run([sys.executable, "-m", "pip", "install", ...]). In a frozen
    # build sys.executable is *this* executable, so that call re-launches the
    # worker with "-m" as its config path. Both processes then sit there
    # forever, which looks like an export that simply never finishes.
    # YOLO_AUTOINSTALL=0 below stops it happening; this is the backstop for any
    # other library that tries the same trick.
    if sys.argv[1] == "-m":
        emit("error",
             msg="worker: refusing to run as 'python -m'. A dependency tried to "
                 "install packages by re-invoking this executable, which a frozen "
                 "build cannot do.")
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

    # No pip inside a frozen build, so auto-install can only misfire. Turning it
    # off makes check_requirements return False instead of shelling out; the
    # preflight in cmd_export turns that into a message naming what is missing.
    if getattr(sys, "frozen", False):
        os.environ["YOLO_AUTOINSTALL"] = "0"

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
