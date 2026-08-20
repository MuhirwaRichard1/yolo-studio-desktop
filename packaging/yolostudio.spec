# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for YOLO Studio, shared by the Windows and Linux builds.

Produces a one-directory bundle. Onefile is deliberately avoided: the payload
is several gigabytes, and a onefile build would unpack all of it to a temp
directory on every launch, costing minutes of startup and double the disk.

Run from the repository root::

    python -m PyInstaller packaging/yolostudio.spec --noconfirm
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
ICONS = ROOT / "packaging" / "icons"

block_cipher = None

# --------------------------------------------------------------------- data

datas = []
# Ultralytics reads model/task YAML out of its package directory at runtime and
# ships the sample images used by its checks, so the whole data tree must come
# along or the first train() call fails on a missing config.
datas += collect_data_files("ultralytics")
datas += [(str(ICONS / "yolostudio.png"), "yolostudio/resources")]
if (ICONS / "yolostudio.ico").exists():
    datas += [(str(ICONS / "yolostudio.ico"), "yolostudio/resources")]

# ------------------------------------------------------------ hidden imports

hiddenimports = []
# Ultralytics resolves trainers, validators and model heads by string name, so
# static analysis cannot see them.
hiddenimports += collect_submodules("ultralytics")
hiddenimports += [
    "yolostudio.worker",
    "yolostudio.app",
    # Python 3.10 has no typing.Self, so torch and ultralytics take it from
    # typing_extensions. PyInstaller does not always pull the whole module in,
    # and a partial copy shows up as "Plain typing.Self is not valid as type
    # argument" at inference time rather than as an import error.
    "typing_extensions",
]

# ----------------------------------------------------------------- excludes

# Qt modules the app never touches. PySide6-Addons alone is several hundred MB
# and WebEngine is the bulk of it.
QT_EXCLUDES = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebView", "PySide6.QtWebChannel", "PySide6.QtWebSockets",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic", "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtGraphs",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets",
    "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning",
    "PySide6.QtLocation", "PySide6.QtSerialPort", "PySide6.QtSerialBus",
    "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtHelp",
    "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtSensors",
    "PySide6.QtSpatialAudio", "PySide6.QtTextToSpeech", "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
]

# Only exclude what is genuinely unreachable. Trimming stdlib modules to save
# space backfires here: torch imports unittest internally (via torch.testing
# and the dynamo stack), and excluding it produces a bundle that builds and
# launches but cannot import torch at all -- the failure surfaces as
# "No module named 'unittest'" on the first job, not at build time.
excludes = QT_EXCLUDES + [
    "tkinter",
    "IPython", "jupyter", "notebook",
]

# ------------------------------------------------------------------ analysis

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

icon_file = None
if sys.platform == "win32" and (ICONS / "yolostudio.ico").exists():
    icon_file = str(ICONS / "yolostudio.ico")

# Two executables share one Analysis and one COLLECT, so the multi-gigabyte
# payload is stored once and only the small bootstrap is duplicated.
#
# They differ only in subsystem. The GUI must be windowed or Windows shows a
# console behind it; the worker must be a console build, because a windowed
# build can leave sys.stdout as None and the worker's whole protocol is JSON on
# stdout. Both binaries run the same entry point, which dispatches on argv.

exe_gui = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="YOLOStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX corrupts some CUDA DLLs and saves little here
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

exe_worker = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="yolostudio-worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe_gui,
    exe_worker,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="YOLOStudio",
)
