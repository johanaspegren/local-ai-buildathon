# Skin Burn Degree Assessor (Raspberry Pi)

A small image classifier that takes a close-up photo of a burn and predicts
its degree (`degree_1` / `degree_2` / `degree_3`), quantized down to a
TensorFlow Lite model small and fast enough to run on a Raspberry Pi 4/5 CPU.

> **Not a medical device.** This is a hobby/educational project built on a
> public Kaggle dataset. It has not been clinically validated and must not be
> used to make real treatment decisions. If this is ever intended for actual
> patient-facing or clinical use, involve Capgemini's Legal/Compliance/Data
> Privacy teams first -- burn assessment tools sit close to regulated medical
> device and health-data territory (e.g. GDPR, and potentially medical device
> regulation depending on jurisdiction and intended use).

## How it works

The source dataset (`shubhambaid/skin-burn-dataset` on Kaggle) is annotated
in **YOLO object-detection format**: each `imgN.jpg` has a matching `imgN.txt`
with one line per burn region (`class cx cy w h`, normalized 0-1), and there
are 3 classes (`0`, `1`, `2`).

Rather than running full object detection on the Pi (heavier, slower, harder
to quantize well), this pipeline:

1. Crops out each annotated burn region (`data_prep.py`) into a plain
   image-classification folder layout.
2. Trains a MobileNetV2 transfer-learning classifier on those crops
   (`train.py`).
3. Quantizes it to a ~1-3 MB fully int8 `.tflite` file (`convert_to_tflite.py`).
4. Runs it on the Pi with a small script that works with LiteRT or
   TensorFlow Lite runtimes, plus Pillow and numpy (`infer_pi.py`).

**Assumption to verify:** class ids `0/1/2` are mapped to `degree_1/2/3` in
`data_prep.py`'s `CLASS_NAMES`. This is the standard convention for a 3-class
burn severity dataset, but the raw archive has no `classes.txt`/`data.yaml`
to confirm it against. Check the Kaggle dataset card/notebook for the
"Code" tab you linked, and if the order is different, just edit
`CLASS_NAMES` in `data_prep.py` and rerun step 1 -- nothing else depends on it.

The trained model expects a **fairly close-up photo of the burn**, similar in
framing to the cropped training images -- not a wide scene photo. If you
later want "point a camera at a person and find the burn," that needs an
object-detection model (e.g. a quantized YOLO/EfficientDet), which is a
heavier lift on a Pi; ask if you want that version built instead.

## Current status

The full pipeline has already been run once end-to-end on this machine and
verified working:

- `dataset/` -- already generated (1,559 train / 334 val / 326 test crops)
- `models_v2/` -- a trained model that works, converted to TFLite, and
  confirmed to give correct predictions via `infer_pi.py` on sample images.
  Test-set accuracy: **76%** overall (`degree_1` 0.82 F1, `degree_2` 0.68 F1,
  `degree_3` 0.77 F1 -- `degree_2` is weakest, likely because its crops look
  visually similar to both neighbors and it's the most-confused class in the
  confusion matrix).
- `models/` -- **stale**, left over from a first training attempt whose
  export included data-augmentation ops that TFLite couldn't convert (fixed
  in `train.py`, see the `make_augmenter()` docstring). Safe to delete;
  left in place rather than deleted automatically since removing files
  wasn't explicitly confirmed.

To retrain from scratch (e.g. with more epochs, or after editing
`CLASS_NAMES`), just rerun steps 1-4 below with `--out models` (or any
folder name) and use that folder's outputs in step 5.

76% accuracy is a reasonable starting point for a first pass on ~1,900 total
crops, but is not production/clinical grade. To push it higher: more
epochs/data, trying `--fine-tune-at` with a lower value (unfreezes more of
MobileNetV2), or simply more labeled images (particularly more `degree_3`
examples, the smallest class).

## 1. Set up a training environment (on this PC, not the Pi)

TensorFlow does not yet support Python 3.14 (this machine's default). Use
the Python 3.12 install found on this system:

```bash
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements-train.txt
```

## 2. Prepare the dataset

```bash
.venv\Scripts\python data_prep.py --source "C:\Users\jokirk\Downloads\archive" --output dataset
```

This writes `dataset/train`, `dataset/val`, `dataset/test`, each with
`degree_1/`, `degree_2/`, `degree_3/` subfolders of cropped burn images.
Splitting happens at the source-photo level so crops from the same photo
never leak across splits.

## 3. Train

```bash
.venv\Scripts\python train.py --data dataset --epochs 15 --fine-tune-epochs 15 --out models
```

Trains in two phases (frozen MobileNetV2 head, then fine-tunes the top
layers at a low learning rate), applies class weighting to correct for the
imbalanced classes (~998/1212/408 crops), and prints a classification report
+ confusion matrix on the held-out test set. Outputs go to `models/`:

- `burn_classifier_saved_model/` -- the trained SavedModel
- `labels.txt` -- class names in the index order the model outputs (keep
  this next to the `.tflite` file, `infer_pi.py` reads it)

## 4. Convert to TFLite for the Pi

```bash
.venv\Scripts\python convert_to_tflite.py --data dataset --model models/burn_classifier_saved_model
```

Produces `models/burn_classifier.tflite`, fully int8-quantized (uint8 image
in, uint8 scores out) using a sample of training images for calibration.
This is what actually goes on the Pi.

## 5. Deploy to the Raspberry Pi

Copy these three files to the Pi (e.g. via `scp`):

```
models/burn_classifier.tflite
models/labels.txt
infer_pi.py
requirements-pi.txt
```

On the Pi:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-pi.txt

# If the package install is picky on your Pi/Python combo, try one of:
# pip install ai-edge-litert
# pip install tflite-runtime
# sudo apt install python3-tflite-runtime
# pip install tensorflow-cpu

python3 infer_pi.py path/to/photo.jpg --model burn_classifier.tflite --labels labels.txt
```

Output looks like:

```
Predicted: degree_2  (confidence: 87.3%)

All classes:
  degree_1     4.1%
  degree_2     87.3%
  degree_3     8.6%
```

Inference on a Pi 4/5 CPU should take well under a second per image -- plenty
fast for "take a photo, get a classification," which is the input mode you
specified (no live camera feed / real-time requirement).

## Files

| File | Runs on | Purpose |
|---|---|---|
| `data_prep.py` | PC | YOLO annotations -> cropped classification dataset |
| `train.py` | PC | Trains MobileNetV2 classifier |
| `convert_to_tflite.py` | PC | Quantizes to int8 `.tflite` |
| `infer_pi.py` | **Pi** | Loads `.tflite` + labels, classifies one image |
| `requirements-train.txt` | PC | TensorFlow training deps |
| `requirements-pi.txt` | **Pi** | Minimal inference deps |

---
If any part of this is turned into a client-facing deliverable or deployed
where real patients' images/data are involved, please review it with
Legal/Compliance/the Data Privacy Officer before use, and have a clinician
validate model outputs -- this note itself is not legal or medical advice.
