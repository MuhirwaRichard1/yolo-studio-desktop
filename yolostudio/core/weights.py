"""Local cache for pretrained YOLO checkpoints.

Ultralytics downloads missing weights itself, but its fetch is a single attempt
per transport and gives up with a bare "Retry limit reached" that says nothing
useful. On a flaky link that turns into a failed training run several minutes
in. This module fetches once, verifies, caches under ``~/.yolostudio/weights``
and hands ultralytics a local path, so a checkpoint is downloaded at most once
per machine and a bad connection produces a clear message.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, List, Optional

# The 'latest' redirect keeps working as ultralytics publishes new assets;
# the pinned tags are fallbacks for names retired from the latest release.
LATEST_URL = "https://github.com/ultralytics/assets/releases/latest/download/{name}"
TAGGED_URL = "https://github.com/ultralytics/assets/releases/download/{tag}/{name}"
FALLBACK_TAGS = ["v8.4.0", "v8.3.0", "v8.2.0"]

CHUNK = 1 << 18          # 256 KiB
MIN_VALID_BYTES = 1 << 18
ATTEMPTS = 4

Progress = Callable[[int, int], None]     # downloaded, total (0 when unknown)


def weights_dir() -> Path:
    root = Path(os.environ.get("YOLOSTUDIO_HOME", Path.home() / ".yolostudio")) / "weights"
    root.mkdir(parents=True, exist_ok=True)
    return root


def is_bare_name(model: str) -> bool:
    """True for catalogue names like ``yolo11s.pt``, false for real paths."""
    if not model or model.endswith(("/", "\\")):
        return False
    if "/" in model or "\\" in model:
        return False
    return model.endswith(".pt") and not Path(model).exists()


def cached_path(name: str) -> Optional[Path]:
    dest = weights_dir() / name
    if dest.is_file() and dest.stat().st_size >= MIN_VALID_BYTES:
        return dest
    return None


def _urls_for(name: str) -> List[str]:
    return [LATEST_URL.format(name=name)] + [
        TAGGED_URL.format(tag=tag, name=name) for tag in FALLBACK_TAGS
    ]


def ensure(name: str, on_progress: Optional[Progress] = None,
           on_note: Optional[Callable[[str], None]] = None) -> Path:
    """Return a local path for ``name``, downloading it if necessary.

    Raises ``RuntimeError`` with an actionable message if every attempt fails.
    """
    hit = cached_path(name)
    if hit is not None:
        return hit

    import requests  # a dependency of ultralytics; imported lazily

    dest = weights_dir() / name
    tmp = dest.with_suffix(dest.suffix + ".part")
    errors: List[str] = []

    for attempt in range(1, ATTEMPTS + 1):
        for url in _urls_for(name):
            try:
                if on_note:
                    on_note(f"Downloading {name} (attempt {attempt})…")
                with requests.get(url, stream=True, timeout=60,
                                  headers={"User-Agent": "yolo-studio"}) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("Content-Length") or 0)
                    done = 0
                    with open(tmp, "wb") as handle:
                        for chunk in response.iter_content(CHUNK):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            done += len(chunk)
                            if on_progress:
                                on_progress(done, total)
                if tmp.stat().st_size < MIN_VALID_BYTES:
                    raise OSError(f"file too small ({tmp.stat().st_size} bytes)")
                if total and tmp.stat().st_size != total:
                    raise OSError(f"truncated: {tmp.stat().st_size} of {total} bytes")
                os.replace(tmp, dest)
                if on_note:
                    on_note(f"Cached {name} at {dest}")
                return dest
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
        if attempt < ATTEMPTS:
            time.sleep(min(8.0, 1.5 * attempt))

    raise RuntimeError(
        f"Could not download {name} after {ATTEMPTS} attempts.\n"
        f"Last error: {errors[-1] if errors else 'unknown'}\n\n"
        f"Download it manually from\n  {LATEST_URL.format(name=name)}\n"
        f"and save it to\n  {weights_dir()}\n"
        f"then start again — the app reuses the cached copy.")


def resolve(model: str, on_progress: Optional[Progress] = None,
            on_note: Optional[Callable[[str], None]] = None) -> str:
    """Map a catalogue name to a cached local path; pass paths through."""
    if is_bare_name(model):
        return str(ensure(model, on_progress, on_note))
    return model
