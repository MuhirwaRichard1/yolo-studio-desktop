"""Convert trained models for SBC NPUs, using WSL to host the Linux toolchains.

Neither vendor toolchain runs on Windows:

* **Rockchip** (Radxa boards, ``.rknn``) needs ``rknn-toolkit2``, which publishes
  no Windows wheels at all -- ``pip install rknn-toolkit2`` reports
  "from versions: none".
* **D-Robotics** (RDK boards, ``.bin``) needs Horizon's OpenExplorer toolchain,
  which is distributed only as a Linux Docker image.

Both are driven here through ``wsl.exe``. Commands are written to a shell script
and executed with ``bash <script>`` rather than passed as a command line: the
Windows -> WSL -> bash quoting chain mangles anything with spaces or quotes in
it, and a script file sidesteps the problem entirely.

The RKNN conversion deliberately runs *ultralytics itself* inside WSL rather
than converting a Windows-exported ONNX. For INT8 the exporter wraps the torch
model in ``_NormalizeCoords`` before tracing, so the ONNX graph differs from a
normal one; feeding a plain ONNX to ``rknn-toolkit2`` produces a model whose
class scores collapse under the per-tensor INT8 scale.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Rockchip targets rknn-toolkit2 accepts. rv1103/rv1106 are INT8-only, which
# onnx2rknn enforces itself, so they are annotated rather than filtered.
RKNN_CHIPS = ["rk3588", "rk3576", "rk3568", "rk3566", "rk3562", "rv1103", "rv1106"]
RKNN_INT8_ONLY = {"rv1103", "rv1106"}

# D-Robotics BPU architectures, as hb_mapper's `march` key spells them.
HORIZON_MARCH = {
    "RDK X5": "nash-e",
    "RDK X3": "bernoulli2",
}

# OpenExplorer image per board. These are not fetchable without a D-Robotics
# account, so the code checks for them and explains rather than pulling.
HORIZON_IMAGE = {
    "RDK X5": "openexplorer/ai_toolchain_ubuntu_20_x5_cpu",
    "RDK X3": "openexplorer/ai_toolchain_ubuntu_20_xj3_cpu",
}

# Where the provisioned toolchain venv lives inside the distro. On the Linux
# filesystem, not /mnt/c: the env is tens of thousands of small files and the
# 9p mount makes both install and import roughly an order of magnitude slower.
WSL_ENV_DIR = "$HOME/.yolostudio/rknn-env"

Logger = Callable[[str], None]


class WSLError(RuntimeError):
    """Raised with a message intended to be shown to the user as-is."""


@dataclass
class Distro:
    name: str
    version: int
    default: bool
    state: str = ""


# ------------------------------------------------------------------ discovery

def wsl_exe() -> Optional[str]:
    """Path to wsl.exe, or None when this is not a Windows host with WSL."""
    if os.name != "nt":
        return None
    candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "wsl.exe"
    return str(candidate) if candidate.exists() else None


def _run(args: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, timeout=timeout,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def list_distros() -> List[Distro]:
    """Installed WSL distributions, newest API first.

    ``wsl.exe --list --verbose`` writes UTF-16 with no BOM, which is why the
    output is decoded explicitly instead of trusting the console encoding.
    """
    exe = wsl_exe()
    if not exe:
        return []
    try:
        proc = _run([exe, "--list", "--verbose"])
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    text = proc.stdout.decode("utf-16-le", errors="ignore")
    if "NAME" not in text:  # some builds emit UTF-8 instead
        text = proc.stdout.decode("utf-8", errors="ignore")

    found: List[Distro] = []
    for line in text.splitlines()[1:]:
        row = line.strip()
        if not row:
            continue
        default = row.startswith("*")
        parts = row.lstrip("*").split()
        if len(parts) < 3:
            continue
        name, state, version = parts[0], parts[1], parts[-1]
        if not version.isdigit():
            continue
        found.append(Distro(name=name, version=int(version), default=default, state=state))
    return found


def usable_distros() -> List[Distro]:
    """WSL 2 distributions only -- WSL 1 cannot run the vendor toolchains."""
    return [d for d in list_distros() if d.version == 2]


def default_distro() -> Optional[str]:
    usable = usable_distros()
    if not usable:
        return None
    for d in usable:
        if d.default:
            return d.name
    return usable[0].name


# ----------------------------------------------------------------- path bridge

def to_wsl_path(path: Path | str) -> str:
    """Translate ``C:\\Users\\x`` to ``/mnt/c/Users/x``.

    wslpath(1) would be authoritative but costs a distro round-trip per call,
    and this covers every path the app can produce: a local drive letter or an
    already-POSIX path.
    """
    text = str(Path(path)).replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", text)
    if match:
        return f"/mnt/{match.group(1).lower()}/{match.group(2)}"
    if text.startswith("//"):
        raise WSLError(
            f"UNC network paths cannot be reached from WSL: {path}\n"
            "Move the project onto a local drive and try again.")
    return text


# ------------------------------------------------------------------- execution

def run_script(distro: str, script: str, log: Logger,
               workdir_wsl: Optional[str] = None, timeout: Optional[int] = None) -> int:
    """Run a bash script inside `distro`, streaming its output to `log`.

    The script is written to a temp file on the Windows side and executed by
    path, so nothing has to survive Windows -> WSL -> bash quoting.
    """
    exe = wsl_exe()
    if not exe:
        raise WSLError("WSL is not available on this machine.")

    import tempfile

    body = script if not workdir_wsl else f'cd "{workdir_wsl}" || exit 1\n{script}'
    handle = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False,
                                         encoding="utf-8", newline="\n")
    try:
        handle.write(body)
        handle.close()
        args = [exe, "-d", distro, "--exec", "bash", to_wsl_path(handle.name)]
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        assert proc.stdout is not None
        for line in proc.stdout:
            log(line.rstrip("\n"))
        return proc.wait(timeout=timeout)
    finally:
        try:
            os.unlink(handle.name)
        except OSError:
            pass


def probe(distro: str) -> Dict[str, str]:
    """Report what the distro already has, for the UI's readiness panel."""
    lines: List[str] = []
    script = r"""
. /etc/os-release 2>/dev/null
echo "distro=$PRETTY_NAME"
echo "arch=$(uname -m)"
export PATH="$HOME/.local/bin:$PATH"
echo "uv=$(command -v uv || echo no)"
echo "docker=$(command -v docker || echo no)"
if sudo -n true 2>/dev/null; then echo "sudo=passwordless"; else echo "sudo=password"; fi
if [ -x "$HOME/.yolostudio/rknn-env/bin/python" ]; then
    echo "rknn_env=ready"
else
    echo "rknn_env=missing"
fi
"""
    try:
        run_script(distro, script, lines.append, timeout=120)
    except (WSLError, subprocess.SubprocessError, OSError) as exc:
        return {"error": str(exc)}

    info: Dict[str, str] = {}
    for line in lines:
        if "=" in line:
            key, _, value = line.partition("=")
            info[key.strip()] = value.strip()
    return info


# ---------------------------------------------------------------- provisioning

def provision_rknn_script() -> str:
    """Create the WSL side conversion environment if it is not already there.

    torch comes from PyTorch's CPU index. rknn-toolkit2 pulls torch 2.4 and
    would otherwise drag in the whole CUDA wheel set -- about 3 GB of GPU
    libraries that a graph converter never executes.
    """
    return f"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ENVDIR="{WSL_ENV_DIR}"
PY="$ENVDIR/bin/python"

# Importability alone is not enough to call the environment good. rknn-toolkit2
# 2.3.2 is built against numpy 1.x: under numpy 2 it imports cleanly and then
# dies deep in graph optimisation, because converting a one-element array to a
# Python scalar became a TypeError. That surfaces as
# "only 0-dimensional arrays can be converted to Python scalars" from
# _p_fuse_mul_into_sdpa, with nothing pointing at numpy. Check the version that
# actually has to hold, not just that the imports resolve.
if [ -x "$PY" ] && "$PY" -c "
import numpy, rknn.api, ultralytics
assert numpy.__version__.split('.')[0] == '1', 'numpy ' + numpy.__version__
" 2>/dev/null; then
    echo "[yolostudio] conversion environment already present"
    exit 0
fi

# Present but not importable: a previous run was interrupted, or resolved to an
# inconsistent set. Rebuilding is far cheaper to reason about than repairing,
# and the wheels are still in uv's cache so it costs little.
if [ -e "$ENVDIR" ]; then
    echo "[yolostudio] existing environment is incomplete, rebuilding it"
    rm -rf "$ENVDIR"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[yolostudio] installing uv (no root required)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[yolostudio] creating Python 3.11 environment at $ENVDIR"
mkdir -p "$(dirname "$ENVDIR")"
uv venv --python 3.11 "$ENVDIR"

# One resolution pass for everything. Installing in two steps lets the second
# pass upgrade numpy past 2.0, which leaves rknn-toolkit2's compiled extensions
# reading a 1.x .so through a 2.x package layout -- it fails later and far from
# the cause, as "cannot import name '_center' from numpy.core._multiarray_umath".
#
# The +cpu build only exists on PyTorch's index, so naming it explicitly keeps
# the ~3 GB of CUDA wheels out without having to trust index precedence.
echo "[yolostudio] installing the conversion toolchain (this takes a few minutes)"
uv pip install --python "$PY" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    "torch==2.4.0+cpu" \
    "rknn-toolkit2>=2.3.2" \
    "numpy<2" \
    "onnx>=1.16.1,<1.19.0" \
    "onnxslim>=0.1.82" \
    "setuptools<82" \
    ultralytics

# Import everything the conversion touches now, so a broken environment is
# reported here instead of after the next model has already been traced.
"$PY" - <<'CHECK'
import numpy, scipy, ultralytics
from rknn.api import RKNN
print(f"[yolostudio] environment ready: numpy {{numpy.__version__}}, "
      f"scipy {{scipy.__version__}}, ultralytics {{ultralytics.__version__}}")
CHECK
"""


def rknn_export_script(model_wsl: str, chip: str, imgsz: int, quantize: int,
                       data_yaml_wsl: Optional[str]) -> str:
    """Run the RKNN export inside the provisioned environment.

    ``quantize`` is 8 for INT8 or 16 for a float build, matching ultralytics.
    INT8 needs a calibration set, which is the project's exported data.yaml.
    """
    if quantize == 8 and not data_yaml_wsl:
        raise WSLError(
            "INT8 quantization needs calibration images. Export the dataset "
            "first, or choose the FP16 build instead.")
    if chip in RKNN_INT8_ONLY and quantize != 8:
        raise WSLError(f"{chip} has no floating-point support; choose INT8 for this target.")

    data_arg = f", data=r'{data_yaml_wsl}'" if quantize == 8 else ""
    return f"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Never let ultralytics install anything here. Given the chance it resolves a
# missing extra -- onnxslim, typically -- and drags numpy up to 2.x on the way,
# which breaks rknn-toolkit2's compiled extensions in the middle of the very run
# doing the conversion. Everything it needs is installed during provisioning, so
# a missing package now is a bug to report rather than something to paper over.
export YOLO_AUTOINSTALL=0
PY="{WSL_ENV_DIR}/bin/python"

"$PY" - <<'PYEOF'
from ultralytics import YOLO

model = YOLO(r'{model_wsl}')
out = model.export(format='rknn', name='{chip}', imgsz={imgsz},
                   quantize={quantize}{data_arg})
print(f'[yolostudio] RKNN_OUTPUT={{out}}')
PYEOF
"""


# ------------------------------------------------------------------- D-Robotics

def horizon_config(onnx_name: str, march: str, imgsz: int, prefix: str) -> str:
    """hb_mapper YAML for a YOLO detector.

    input_type_rt is left as ``featuremap`` so the .bin takes plain float input.
    The nv12 path is faster on-device but requires the caller to feed camera
    format, which is a deployment decision rather than an export one.
    """
    return f"""# Generated by YOLO Studio for {march}
model_parameters:
  onnx_model: '{onnx_name}'
  march: '{march}'
  working_dir: './output'
  output_model_file_prefix: '{prefix}'
  layer_out_dump: False

input_parameters:
  input_name: 'images'
  input_type_rt: 'featuremap'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  input_shape: '1x3x{imgsz}x{imgsz}'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451

calibration_parameters:
  cal_data_dir: './calibration_data'
  cal_data_type: 'float32'
  calibration_type: 'default'

compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O3'
  debug: False
"""


def horizon_check_script(image: str) -> str:
    """Report whether Docker and the OpenExplorer image are usable."""
    return f"""
if ! command -v docker >/dev/null 2>&1; then
    echo "docker=missing"; exit 0
fi
if ! docker info >/dev/null 2>&1; then
    echo "docker=nodaemon"; exit 0
fi
echo "docker=ok"
if docker image inspect "{image}" >/dev/null 2>&1; then
    echo "image=present"
else
    echo "image=missing"
fi
"""


def horizon_export_script(workdir_wsl: str, image: str, config_name: str) -> str:
    """Invoke hb_mapper inside the OpenExplorer container.

    The working directory is bind-mounted rather than copied: the ONNX and the
    calibration set together run to hundreds of megabytes.
    """
    return f"""
set -euo pipefail
docker run --rm \
    -v "{workdir_wsl}":/work -w /work \
    "{image}" \
    hb_mapper makertbin --config "{config_name}" --model-type onnx
echo "[yolostudio] hb_mapper finished"
"""
