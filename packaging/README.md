# Packaging

Builds a self-contained desktop application for Windows and Linux. Everything
ships inside the bundle — Python, PySide6, PyTorch with CUDA, ultralytics — so
the installed app needs no Python on the target machine and no internet except
to fetch pretrained model weights.

That completeness is why the packages are large: **PyTorch with CUDA is about
4.5 GB unpacked on its own.**

| | Bundle | Distributable |
| --- | --- | --- |
| Windows | `dist\YOLOStudio\` | `dist\installer\YOLOStudio-<ver>-windows-x64-setup.exe` |
| Linux | `dist/YOLOStudio/` | `out/YOLOStudio-<ver>-x86_64.AppImage` |

---

## Windows

```powershell
.\bootstrap.ps1                    # once: creates .venv with CUDA torch
.\packaging\build_windows.ps1      # bundle + installer
```

Options: `-SkipInstaller` (bundle only), `-SkipBundle` (recompile the installer
from an existing bundle), `-Version 1.1.0`.

The installer needs [Inno Setup 6](https://jrsoftware.org/isinfo.php):

```powershell
winget install --id JRSoftware.InnoSetup
```

It installs per-user by default, so no admin prompt appears; the user can opt
into an all-users install from the wizard.

## Linux

Built on Ubuntu 22.04 to target glibc 2.35, which covers Ubuntu 22.04+,
Debian 12+ and Fedora 36+. Under Windows, WSL works:

```bash
wsl -d Ubuntu-22.04
bash /mnt/c/Users/USER/yolo-studio/packaging/wsl_setup.sh   # once
bash /mnt/c/Users/USER/yolo-studio/packaging/build_linux.sh
```

`wsl_setup.sh` copies the source to a native ext4 path and builds a venv there.
PyInstaller writes tens of thousands of small files, and doing that over the
`/mnt/c` 9p mount is roughly an order of magnitude slower.

Neither script needs root. Ubuntu splits `ensurepip` into the `python3-venv`
package, so the setup script falls back to `virtualenv`, which carries its own
pip seed.

---

## Design notes

**Two executables, one payload.** `YOLOStudio` is a windowed binary;
`yolostudio-worker` is a console binary. They share a single PyInstaller
`Analysis` and `COLLECT`, so the gigabytes are stored once and only the small
bootstrap is duplicated.

The split is not cosmetic. Training and inference run in a subprocess that
speaks JSON on stdout, and a *windowed* process on Windows can start with
`sys.stdout` set to `None` — which would silently destroy the protocol. The
worker therefore has to be a console build. It is launched with
`CREATE_NO_WINDOW` so no console flashes on screen, and `worker.py` additionally
falls back to raw file descriptors if the streams are missing.

**One-directory, not one-file.** A onefile build would unpack several GB to a
temporary directory on every launch, costing minutes of startup and double the
disk.

**`sys.executable` changes meaning when frozen.** From source it is a Python
interpreter and the worker is a module (`-m yolostudio.worker`). In a bundle it
is the application itself, and there is no interpreter CLI. `core/runner.py`
resolves this in `worker_program()` / `worker_arguments()`.

**No UPX.** It corrupts some CUDA DLLs and saves little on already-compressed
payloads.

**Excluded Qt modules.** WebEngine, Quick/QML, 3D, Charts, Multimedia and
friends are dropped in the spec. The app uses only QtCore, QtGui and QtWidgets,
and WebEngine alone is several hundred MB.

---

## Size and hosting

GitHub Releases rejects any **single file over 2 GB**. If a build exceeds that,
split it and let users rejoin:

```bash
# publisher
split -b 1900M YOLOStudio-1.0.0-x86_64.AppImage YOLOStudio-1.0.0-x86_64.AppImage.part-

# user
cat YOLOStudio-1.0.0-x86_64.AppImage.part-* > YOLOStudio-1.0.0-x86_64.AppImage
chmod +x YOLOStudio-1.0.0-x86_64.AppImage
```

Always publish a checksum next to the artifacts:

```bash
sha256sum YOLOStudio-* > SHA256SUMS
```

---

## Troubleshooting

**`could not load the Qt platform plugin "xcb"`** — the host is missing X11
client libraries. On a minimal server install:
`sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libgl1`.

**AppImage will not start, mentions FUSE** — install `libfuse2`, or run it
extracted: `./YOLOStudio-*.AppImage --appimage-extract-and-run`.

**Windows SmartScreen warning** — the binaries are unsigned. Users click
*More info → Run anyway*. Removing the warning requires a code-signing
certificate; there is no free option.

**Training says CPU only in the packaged app** — the bundle carries the CUDA
runtime but not the driver. The host needs an NVIDIA driver new enough for
CUDA 12.4 (R550+).
