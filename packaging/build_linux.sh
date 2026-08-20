#!/usr/bin/env bash
# Build the Linux AppImage.
#
# Run inside WSL or any Ubuntu 22.04 box, after packaging/wsl_setup.sh has
# prepared the build venv. Building on 22.04 targets glibc 2.35, which covers
# Ubuntu 22.04+, Debian 12+ and Fedora 36+.
#
#   bash packaging/build_linux.sh [/mnt/c/Users/USER/yolo-studio]
set -euo pipefail

SRC="${1:-/mnt/c/Users/USER/yolo-studio}"
BUILD_HOME="${BUILD_HOME:-$HOME/yolo-studio-build}"
VERSION="${VERSION:-1.0.0}"
PY="$BUILD_HOME/.venv/bin/python"
APPDIR="$BUILD_HOME/AppDir"
TOOL="$BUILD_HOME/appimagetool-x86_64.AppImage"

[ -x "$PY" ] || { echo "No build venv at $PY — run packaging/wsl_setup.sh first." >&2; exit 1; }

echo "==> syncing sources"
rm -rf "$BUILD_HOME/yolostudio" "$BUILD_HOME/packaging"
cp -r "$SRC/yolostudio" "$BUILD_HOME/"
cp -r "$SRC/packaging" "$BUILD_HOME/"
cp "$SRC/requirements.txt" "$SRC/README.md" "$BUILD_HOME/" 2>/dev/null || true

cd "$BUILD_HOME"

echo "==> generating icons"
"$PY" packaging/make_icon.py

echo "==> running PyInstaller (this takes a while)"
rm -rf build dist
"$PY" -m PyInstaller packaging/yolostudio.spec --noconfirm --distpath dist --workpath build

[ -x "dist/YOLOStudio/YOLOStudio" ] || { echo "PyInstaller produced no GUI binary" >&2; exit 1; }
echo "==> bundle size: $(du -sh dist/YOLOStudio | cut -f1)"

echo "==> assembling AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/512x512/apps"
cp -r dist/YOLOStudio/. "$APPDIR/usr/bin/"

install -m 755 packaging/AppRun "$APPDIR/AppRun"
# Line endings matter: the file is copied from a Windows checkout and a stray
# CR makes the kernel fail to find the interpreter.
sed -i 's/\r$//' "$APPDIR/AppRun"

cp packaging/yolostudio.desktop "$APPDIR/yolostudio.desktop"
sed -i 's/\r$//' "$APPDIR/yolostudio.desktop"
cp packaging/yolostudio.desktop "$APPDIR/usr/share/applications/"

cp packaging/icons/yolostudio.png "$APPDIR/yolostudio.png"
cp packaging/icons/yolostudio.png \
   "$APPDIR/usr/share/icons/hicolor/512x512/apps/yolostudio.png"
# appimagetool also looks for .DirIcon.
cp packaging/icons/yolostudio.png "$APPDIR/.DirIcon"

echo "==> fetching appimagetool"
if [ ! -x "$TOOL" ]; then
    URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
    # GitHub asset downloads are unreliable on some networks; retry properly
    # rather than failing the whole build on one reset connection.
    "$PY" - "$URL" "$TOOL" <<'PYEOF'
import sys, time
import requests

url, dest = sys.argv[1], sys.argv[2]
for attempt in range(1, 7):
    try:
        with requests.get(url, stream=True, timeout=120,
                          headers={"User-Agent": "yolo-studio-build"}) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 18):
                    f.write(chunk)
        import os
        if total and os.path.getsize(dest) != total:
            raise OSError("truncated download")
        print(f"    downloaded {os.path.getsize(dest)} bytes")
        break
    except Exception as exc:
        print(f"    attempt {attempt} failed: {exc}")
        time.sleep(2 * attempt)
else:
    sys.exit("could not download appimagetool")
PYEOF
    chmod +x "$TOOL"
fi

echo "==> building AppImage"
mkdir -p "$BUILD_HOME/out"
OUTPUT="$BUILD_HOME/out/YOLOStudio-${VERSION}-x86_64.AppImage"
rm -f "$OUTPUT"

# --appimage-extract-and-run avoids needing FUSE on the build machine, which
# WSL does not reliably provide. ARCH must be set explicitly for the same reason.
ARCH=x86_64 "$TOOL" --appimage-extract-and-run \
    --no-appstream "$APPDIR" "$OUTPUT"

chmod +x "$OUTPUT"
echo
echo "==> built: $OUTPUT"
ls -lh "$OUTPUT" | awk '{print "    size: " $5}'
