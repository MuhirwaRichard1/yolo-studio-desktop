# YOLO Studio

A desktop app for annotating images and finetuning YOLO models on your own GPU.
Everything runs locally — no accounts, no uploads, no cloud training.

- **Annotate** bounding boxes and polygons on the same canvas
- **Pre-label** with any YOLO checkpoint, then correct instead of drawing from scratch
- **Finetune** YOLO11 / YOLOv8 / YOLO12 with live loss and mAP curves
- **Export** the trained model to ONNX, TensorRT, TorchScript or OpenVINO

---

## Install

Prebuilt packages contain everything — Python, Qt, PyTorch with CUDA,
ultralytics — so nothing needs installing first. They are correspondingly
large, because PyTorch with CUDA is ~4.5 GB on its own.

**Windows 10/11 (x64)** — download the `-setup.exe` and run it. It installs
per-user, so there is no admin prompt. Windows SmartScreen will warn that the
publisher is unknown, because the binary is unsigned: choose *More info → Run
anyway*.

**Linux (x86_64)** — download the `.AppImage`, then:

```bash
chmod +x YOLOStudio-*.AppImage
./YOLOStudio-*.AppImage
```

Built against glibc 2.35, so it runs on Ubuntu 22.04+, Debian 12+, Fedora 36+
and similar. If it complains about FUSE, either install `libfuse2` or run it
with `--appimage-extract-and-run`.

**Both** need roughly 8 GB of free disk, and an NVIDIA driver supporting
CUDA 12.4 (R550 or newer) for GPU training. Without a suitable driver the app
still runs and trains, but on the CPU.

Verify a download before running it:

```bash
sha256sum -c SHA256SUMS
```

---

## Running from source

```powershell
cd C:\Users\USER\yolo-studio
.\bootstrap.ps1      # creates .venv, installs CUDA PyTorch + ultralytics + PySide6
.\run.ps1            # launches the app
```

To build the installers yourself, see [`packaging/README.md`](packaging/README.md).

`bootstrap.ps1` is safe to re-run; add `-Recreate` to rebuild the environment from
scratch. `run.ps1 -Console` keeps the terminal attached so you can see tracebacks.

Verify GPU access at any time:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The **Help → About** dialog shows the same information from inside the app, and
the status bar bottom-right names the GPU it will train on.

### Tests

```powershell
.\.venv\Scripts\python.exe tools\test_core.py    # annotations, project, splits, export
.\.venv\Scripts\python.exe tools\test_gui.py     # builds the real window offscreen
.\.venv\Scripts\python.exe tools\smoke_test.py   # synthesises data and finetunes on GPU
```

`smoke_test.py` is the one to run if something looks wrong: it exercises the
same worker subprocess the GUI uses, so a failure there localises the problem to
the environment rather than the interface. `--device cpu` and `--epochs N` are
accepted.

---

## Workflow

**1. Create a project.** `File → New project`. Name it, pick detect or segment,
and type your class names one per line. The task only sets the default export
format — you can draw boxes and polygons in either kind of project and decide at
training time.

**2. Import images.** `File → Import folder` registers a folder recursively.
Images are referenced where they live and never copied, so importing a large
folder is instant and costs no disk.

**3. Annotate.** Press `W` for boxes, `E` for polygons, `V` to go back to
selecting. Number keys pick the active class. `D` moves to the next image, `Tab`
jumps to the next image with no labels yet. Labels save automatically when you
change images.

**4. Pre-label the rest.** Once ~50–100 images are done, train a quick model,
then `Tools → Auto-label` with that checkpoint over the remaining images. Correct
what it got wrong. This is usually several times faster than annotating by hand,
and it gets faster with each round.

**5. Train.** The **Train** tab. Pick a base model, set epochs and batch, press
Start. The dataset is exported and split automatically; curves and the raw
ultralytics log update as it runs.

**6. Export.** `Tools → Export trained model` for a deployment format. TensorRT
is fastest on your RTX card, but the `.engine` file only works on that same GPU
and driver version.

---

## Keyboard

These are scoped to the canvas, so typing in a filter box does what you expect.

| Key | Action |
| --- | --- |
| `V` / `W` / `E` | Select / draw box / draw polygon |
| `1`–`9` | Set active class (also relabels the current selection) |
| `A` / `D` | Previous / next image |
| `Tab` | Next unannotated image |
| `Ctrl+S` | Save labels for this image |
| `Del` | Delete selected shapes |
| `Ctrl+A` | Select all shapes |
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo |
| `F` / `Ctrl+0` | Fit to window / actual size |
| `Esc` | Cancel the shape being drawn |

Mouse: wheel zooms at the cursor, middle-drag or space-drag pans, double-click
closes a polygon or adds a vertex to an existing one, `Alt`+click a vertex
removes it.

---

## Choosing a model

| | Best for | Notes |
| --- | --- | --- |
| **YOLO11** | Default choice | Best accuracy per parameter; detect and segment |
| **YOLOv8** | Maximum compatibility | Widest ecosystem and export support |
| **YOLO12** | Squeezing out accuracy | Detect only here; slower per epoch |

Scales run `n → s → m → l → x`, trading speed for accuracy. On a 16 GB card,
**`s` or `m` at 640px** is the sweet spot for most finetunes. Start with `s`: it
trains fast enough to iterate on your annotations, which usually matters more
than the last point of mAP.

The **Suggest** button next to `Batch` fills in a batch size that fits your
actual VRAM at the chosen image size. Setting `Batch` to `-1` hands the decision
to ultralytics' autobatch instead.

**Freeze layers** is worth knowing about: setting it to `10` freezes the backbone
and trains only the head. On a few hundred images that often generalises better
than finetuning everything, and it trains considerably faster.

---

## Project layout

```
my-project/
    project.json          classes, colours, image registry, training defaults
    labels/               one YOLO .txt per image, named by a stable image id
    datasets/train_set/   generated split + data.yaml (rebuilt on each run)
    runs/detect_.../      ultralytics output: weights/best.pt, plots, results.csv
```

Pretrained checkpoints are cached once per machine in
`~/.yolostudio/weights/`, not per project. If a download fails, the app tells
you the exact URL and folder so you can drop the file in by hand.

The worker also points ultralytics' own `weights_dir` setting at that folder, so
its internal asset lookups — the AMP check in particular — reuse the cache
instead of re-downloading. This is a global ultralytics setting, so the `yolo`
CLI will share the same cache afterwards.

`labels/` is plain YOLO text and is the real content of a project — it is worth
putting under version control. A line with 5 numbers is a box
(`cls cx cy w h`); a longer line is a polygon (`cls x1 y1 x2 y2 …`). All
coordinates are normalised to 0–1. Storing both kinds together is what lets one
set of annotations train either a detector or a segmenter.

Splits are materialised with hardlinks where the filesystem allows, so exporting
a dataset from a large project takes little time and almost no extra disk.

---

## Troubleshooting

**`CUDA out of memory`** — lower `Batch` (halve it), reduce `Image size` to 512
or 416, or drop a model scale. The app reports peak VRAM per epoch under the
progress bar, which tells you how much headroom you actually had.

**Training is very slow / no GPU shown** — the status bar says `CPU only` when
CUDA is unavailable. Re-run `.\bootstrap.ps1`; it installs torch from the CUDA
index explicitly. A CPU-only wheel already present in the venv is the usual
cause.

**Training sits at 0% GPU before the first epoch** — it is fetching a
checkpoint. The log shows which one and the progress bar shows the download.
If the connection drops repeatedly, download the file named in the error to
`~/.yolostudio/weights/` by hand and start again.

**Windows dataloader errors or a hang at epoch 1** — set `Dataloader workers` to
`0`. Windows spawns worker processes rather than forking, and some environments
do not cope with it.

**`No labels found`** — the split contained no annotated images. Check the
counts printed in the log after export, and that the class you annotated with
still exists.

**Poor mAP after training** — almost always annotation quality rather than
hyperparameters. Check for missed objects (an unlabelled object teaches the model
that region is background), inconsistent class boundaries, and too few examples
of your rarest class. The per-class instance counts in the Classes panel make
imbalance easy to spot.
