# Using YOLO Studio

A complete pass through the app: a folder of images in, a deployable model out.
It assumes nothing has been set up yet.

The [README](README.md) is the reference — every option, every shortcut, the
troubleshooting table. This document is the path through them in order.

---

## Before you start

You need images, and enough of them. A few rules of thumb:

| Images | What to expect |
| --- | --- |
| < 50 | Enough to check the app works, not to get a usable model |
| 100–300 | A real model for one easy class, with `freeze 10` |
| 500–2000 | The normal range for a solid single-purpose detector |
| 2000+ | Worth it when classes are similar or objects are small |

Objects per image matters as much as image count. Fifty images with twenty
labelled objects each will beat three hundred with one.

Check the status bar bottom-right before a long session. It names the GPU it
will train on, or says `CPU only` — which still works, just slowly.

---

## 1. Create a project

`File → New project…` (`Ctrl+N`).

- **Name** and **Location** decide the project folder on disk.
- **Task** — *Detect* for bounding boxes, *Segment* for polygon masks. This only
  sets the default export format. You can draw both kinds of shape in either
  project and choose the task when you train.
- **Classes** — one name per line.

Class names are worth a minute's thought. `drone` and `quadcopter` as separate
classes will fight each other unless you can state the difference in one
sentence and apply it consistently at three in the morning. Start with fewer
classes than you think you need; splitting one later is easier than merging two.

You can add, rename, recolour, and delete classes afterwards in the **Classes**
panel. Deleting rewrites every label file that referenced the class, dropping
those shapes and shifting higher class ids down.

---

## 2. Import images

`File → Import folder…` (`Ctrl+Shift+O`) registers a folder recursively.
`File → Import images…` picks individual files.

Images are referenced where they live, never copied. Importing a large folder is
instant and costs no disk — but if you move or delete the originals, the project
loses them. Keep them somewhere stable.

---

## 3. Annotate

This is where the time goes, and it decides how good the model gets.

| Key | Action |
| --- | --- |
| `V` / `W` / `E` | Select / draw box / draw polygon |
| `1`–`9` | Set active class — also relabels the current selection |
| `A` / `D` | Previous / next image |
| `Tab` | Next image with no labels yet |
| `Del` | Delete selected shapes |
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo |
| `F` / `Ctrl+0` | Fit to window / actual size |
| `Esc` | Cancel the shape being drawn |
| `F1` | The full list, from inside the app |

Wheel zooms at the cursor, middle-drag or space-drag pans. Double-click closes a
polygon, or adds a vertex to an existing one; `Alt`+click a vertex removes it.

Labels save automatically when you change images, so `Ctrl+S` is a reassurance
rather than a requirement.

**What to label.** Every instance of every class, in every image — including the
partly occluded and the awkwardly small. An unlabelled object is not neutral: it
teaches the model that region is background, which is worse than not having the
image at all. Box the visible extent, not the guessed full extent.

**Consistency beats precision.** A boundary rule applied the same way every time
trains better than a rule that is more correct but applied unevenly. Decide early
whether the propellers are part of the drone, and stick to it.

`Tab` is the loop to work in — it walks only the unannotated images, so you can
keep pressing it until it stops finding any.

---

## 4. Pre-label the rest

Once 50–100 images are done by hand, stop annotating and train a throwaway model
(§5, twenty epochs is plenty). Then `Tools → Auto-label with a model…`
(`Ctrl+L`) to run it over what remains.

- **Source** — one of your runs, a custom `.pt`, or a pretrained COCO model.
  Pretrained is worth trying first for anything COCO already knows.
- **Confidence** — lower catches more objects but writes more false positives.
  Correcting a spurious box is faster than noticing a missing one, so err low.
- **Shape** — boxes, or polygons if the model is a segmentation model.
- **Overwrite labels that already exist** — leave this off. It protects the work
  you did by hand.
- **Class mapping** — maps the model's classes onto yours, with `— skip —` for
  the ones you don't want. A COCO model has `airplane` and `bird`; you decide
  what, if anything, those become.

Then correct the output. This is usually several times faster than drawing from
scratch, and each round makes the next one faster.

The trap is trusting it. Auto-labels are confident and wrong in exactly the
places your model is weakest, which are the places the next round most needs
correct. Review every image it touched.

---

## 5. Train

The **Train** tab, or `Ctrl+T`.

**Model.** Pick a family and size, continue from one of your own runs, or load a
custom `.pt`. Start with **YOLO11 `s` at 640** — it trains fast enough to iterate
on annotations, which matters more early than the last point of mAP.

**The settings that matter:**

| Setting | Default | When to change it |
| --- | --- | --- |
| **Epochs** | 100 | 20 for a throwaway pre-labeller; 100+ for the real run |
| **Image size** | 640 | 960+ if objects are small in frame; 416 to fit memory |
| **Batch** | 16 | **Suggest** fits it to your VRAM; `-1` hands it to autobatch |
| **Freeze layers** | 0 | `10` freezes the backbone — often better under ~500 images |
| **Patience** | 50 | Stops early after N epochs with no improvement; 0 disables |
| **Dataloader workers** | 8 | **Set to 0 if Windows hangs or errors at epoch 1** |
| **Val split** | 0.2 | Fraction held out for validation |
| **Test split** | 0.0 | A second holdout, untouched by early stopping |

The dataset is exported and split automatically when you press **Start** — no
separate step. Splits use hardlinks where the filesystem allows, so this is fast
and costs almost no disk.

**While it runs**, the curves and the raw ultralytics log update live. Watch
mAP50-95 on the validation split, not the training loss. Loss falling while mAP
stalls means it is memorising your training images.

Output lands in `runs/detect_…/`, with `weights/best.pt` the one to keep —
`best.pt` is the best validation epoch, `last.pt` merely the final one.

**If it fails:** `CUDA out of memory` → halve **Batch**, or drop **Image size**
to 512. The app reports peak VRAM per epoch under the progress bar, so you can
see how much headroom you actually had. Everything else is in the README's
troubleshooting section.

---

## 6. Judge the result

Before exporting anything, decide whether the model is worth deploying.

Ultralytics writes a confusion matrix and PR curves into the run folder. The
number to trust is **mAP50-95 on data the model never saw**. If you set a test
split, that is what it is for.

Then look at actual predictions on real images. Aggregate metrics hide the
failure that matters — a class that is never detected, or a background object
that always triggers a false positive.

**Poor mAP is almost always the annotations, not the hyperparameters.** Before
touching a learning rate, check for missed objects, class boundaries applied
inconsistently, and too few examples of the rarest class. The per-class instance
counts in the **Classes** panel make imbalance obvious. Fixing labels and
retraining beats tuning nearly every time.

---

## 7. Export

`Tools → Export trained model…`, pick the checkpoint, pick a format.

| Format | Use it for | Cost |
| --- | --- | --- |
| **ONNX** | Portable — onnxruntime on CPU or GPU | The safe default |
| **TensorRT** | Fastest on your RTX card | Builds take minutes; see below |
| **TorchScript** | A PyTorch graph with no Python needed | Still needs libtorch |
| **OpenVINO** | Intel CPUs and iGPUs | — |

- **FP16** roughly halves size and speeds up inference, at a small accuracy cost.
  Measure it rather than assuming it is free.
- **Dynamic input shape** allows varying input sizes. It cannot be combined with
  FP16 — the app drops `half` automatically if you tick both.
- **Simplify ONNX graph** applies to ONNX only.

**The TensorRT caveat is worth repeating:** a `.engine` file is built for the
specific GPU and driver it was created on. It will not load on a different card,
and often not after a driver upgrade. Ship ONNX and build the engine on the
target machine — or accept that you are shipping to exactly one machine.

The exported file is written beside the checkpoint, in the run's `weights/`
folder.

---

## 8. Deploy

**Simplest — ultralytics, any format:**

```python
from ultralytics import YOLO

model = YOLO("runs/detect_.../weights/best.pt")    # or best.onnx, best.engine
for r in model("image.jpg"):
    for box in r.boxes:
        cls = r.names[int(box.cls)]
        print(cls, float(box.conf), box.xyxy[0].tolist())
```

Point it at a folder, a video file, `0` for a webcam, or an RTSP URL and it
handles the loop. Add `stream=True` for video so it yields per frame instead of
building the whole list in memory.

**Without ultralytics — ONNX Runtime:**

```python
import numpy as np, onnxruntime as ort
from PIL import Image

sess = ort.InferenceSession(
    "best.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
img = Image.open("image.jpg").resize((640, 640))
x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
```

This is the deployable path — no PyTorch, no ultralytics, a ~50 MB dependency
instead of several GB. The catch is that `out` is **raw predictions**, not boxes.
You have to do the rest yourself:

1. Undo the letterbox — scale coordinates back to the original image, accounting
   for the aspect-ratio padding the resize introduced.
2. Filter by confidence.
3. Run non-maximum suppression per class.

Getting the letterbox arithmetic wrong produces boxes that are subtly offset,
which looks like a bad model rather than a bad transform. Validate your
implementation against ultralytics' output on the same image before trusting it.

**Command line, no code:**

```bash
yolo predict model=best.pt source=image.jpg
yolo predict model=best.pt source=video.mp4 save=True
```

---

## What is worth keeping

```
my-project/
    project.json          classes, colours, image registry, training defaults
    labels/               one YOLO .txt per image  <- the real work
    datasets/train_set/   regenerated on every run
    runs/detect_.../      weights/best.pt, plots, results.csv
```

`labels/` is plain text and is the actual value of a project — it outlives every
model you train from it, and it is worth putting under version control. A line
with 5 numbers is a box (`cls cx cy w h`); a longer line is a polygon
(`cls x1 y1 x2 y2 …`), all coordinates normalised to 0–1. Storing both kinds
together is what lets one set of annotations train either a detector or a
segmenter.

`datasets/` is rebuilt on every run and does not need backing up.

Pretrained checkpoints are cached per machine in `~/.yolostudio/weights/`, not
per project, so a second project costs no extra download.

---

## The short version

1. `Ctrl+N` — project, task, class names.
2. `Ctrl+Shift+O` — import a folder.
3. `W`, draw, `D`, repeat. `Tab` for the next unannotated image. 50–100 images.
4. Train 20 epochs, `Ctrl+L` to auto-label the rest, correct it.
5. Train properly. YOLO11 `s`, 640, 100 epochs, `freeze 10` if under ~500 images.
6. Check mAP on held-out data, and look at real predictions.
7. `Tools → Export trained model…` — ONNX unless you have a reason.
8. `YOLO("best.onnx")`, or onnxruntime plus your own NMS.

Steps 3 and 4 are most of the work, and most of the final accuracy.
