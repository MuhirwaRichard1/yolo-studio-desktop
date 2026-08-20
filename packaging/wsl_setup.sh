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
RECREATE="${RECREATE:-0}"

# Build on Python 3.11, not the 3.10 that ships with Ubuntu 22.04.
#
# On 3.10, typing_extensions picks its `Self` with `if hasattr(typing, "Self")`.
# In a PyInstaller bundle that check comes out True even though 3.10's stdlib
# has no `Self`, so typing_extensions aliases a `typing.Self` that `Union`
# then rejects. The failure surfaces far from the cause, as
# "Plain typing.Self is not valid as type argument" inside torch._dynamo when
# torchvision is imported during inference. On 3.11 `Self` is native and the
# whole branch disappears.
#
# uv installs a standalone CPython without root:
#   python3 -m pip install --user uv && uv python install 3.11
pick_python() {
    if [ -n "${PYBIN:-}" ]; then echo "$PYBIN"; return; fi
    for c in "$HOME/.local/bin/python3.11" \
             "$HOME/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11" \
             python3.11 python3.12 python3; do
        if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then echo "$c"; return; fi
    done
    echo python3
}
PYBIN="$(pick_python)"

echo "==> source:     $SRC"
echo "==> build home: $BUILD_HOME"
echo "==> python:     $PYBIN ($("$PYBIN" --version 2>&1))"

case "$("$PYBIN" --version 2>&1)" in
    *3.10*) echo "    WARNING: building on 3.10 reintroduces the typing.Self bug." ;;
esac

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

if [ "$RECREATE" = "1" ] && [ -d ".venv" ]; then
    echo "==> RECREATE=1, removing existing venv"
    rm -rf .venv
fi

# A venv is only usable if it has pip too. A previous half-finished run can
# leave bin/python in place without it, and testing for the interpreter alone
# would silently skip recreation and fail later at the first pip call.
if [ -x ".venv/bin/python" ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    echo "==> existing venv has no pip; recreating"
    rm -rf .venv
fi

# Rebuild if the venv was made with a different interpreter than the one chosen.
if [ -x ".venv/bin/python" ]; then
    have="$(.venv/bin/python -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "?")"
    want="$("$PYBIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "?")"
    if [ "$have" != "$want" ]; then
        echo "==> venv is Python $have but $want was selected; recreating"
        rm -rf .venv
    fi
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "==> creating venv"
    # Ubuntu splits ensurepip into the python3-venv package, which needs root to
    # install. virtualenv ships its own pip seed, so it works from a plain user
    # account -- which matters on a build box you do not own. A uv-installed
    # CPython has ensurepip, so this fallback normally goes unused.
    if "$PYBIN" -m venv .venv 2>/dev/null && [ -x ".venv/bin/pip" ]; then
        echo "    using stdlib venv"
    else
        echo "    stdlib venv unavailable (no ensurepip); using virtualenv"
        rm -rf .venv
        "$PYBIN" -m pip install --user --quiet virtualenv || \
            python3 -m pip install --user --quiet virtualenv
        "$PYBIN" -m virtualenv .venv 2>/dev/null || python3 -m virtualenv -p "$PYBIN" .venv
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
