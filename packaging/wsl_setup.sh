#!/usr/bin/env bash
# Prepare the Linux build environment inside WSL (or any Ubuntu 22.04 box).
#
# Builds happen on a native ext4 path rather than /mnt/c: PyInstaller writes
# tens of thousands of small files and doing that over the 9p mount is roughly
# an order of magnitude slower.
set -euo pipefail

SRC="${1:-/mnt/c/Users/USER/yolo-studio}"
BUILD_HOME="${BUILD_HOME:-$HOME/yolo-studio-build}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "==> source:     $SRC"
echo "==> build home: $BUILD_HOME"

mkdir -p "$BUILD_HOME"
# Copy only what the app needs; never drag .venv or runs/ across the mount.
rsync -a --delete \
    --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
    --exclude 'runs' --exclude 'datasets' --exclude 'dist' --exclude 'build' \
    "$SRC/yolostudio" "$SRC/requirements.txt" "$SRC/README.md" \
    "$BUILD_HOME/" 2>/dev/null || {
        echo "rsync unavailable, falling back to cp"
        rm -rf "$BUILD_HOME/yolostudio"
        cp -r "$SRC/yolostudio" "$BUILD_HOME/"
        cp "$SRC/requirements.txt" "$SRC/README.md" "$BUILD_HOME/"
    }

cd "$BUILD_HOME"

# A venv is only usable if it has pip too. A previous half-finished run can
# leave bin/python in place without it, and testing for the interpreter alone
# would silently skip recreation and fail later at the first pip call.
if [ -x ".venv/bin/python" ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    echo "==> existing venv has no pip; recreating"
    rm -rf .venv
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "==> creating venv"
    # Ubuntu splits ensurepip into the python3-venv package, which needs root to
    # install. virtualenv ships its own pip seed, so it works from a plain user
    # account -- which matters on a build box you do not own.
    if python3 -m venv .venv 2>/dev/null && [ -x ".venv/bin/pip" ]; then
        echo "    using stdlib venv"
    else
        echo "    stdlib venv unavailable (no ensurepip); using virtualenv"
        rm -rf .venv
        python3 -m pip install --user --quiet virtualenv
        python3 -m virtualenv .venv
    fi
fi
PY=".venv/bin/python"

echo "==> upgrading pip"
"$PY" -m pip install --upgrade pip setuptools wheel --quiet

echo "==> installing PyTorch (CUDA) — multi-GB download"
"$PY" -m pip install torch torchvision --index-url "$CUDA_INDEX"

echo "==> installing app dependencies"
"$PY" -m pip install -r requirements.txt

echo "==> installing PyInstaller"
"$PY" -m pip install pyinstaller

echo "==> verifying"
"$PY" - <<'PYEOF'
import torch, sys
print("torch      ", torch.__version__)
print("cuda build ", torch.version.cuda)
print("available  ", torch.cuda.is_available())
import PySide6, ultralytics
print("PySide6    ", PySide6.__version__)
print("ultralytics", ultralytics.__version__)
PYEOF

echo "==> venv size"
du -sh "$BUILD_HOME/.venv"
echo "==> done"
