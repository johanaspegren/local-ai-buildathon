"""
Converts the raw YOLO-format skin burn dataset (img*.jpg + img*.txt, one line
per box as "class cx cy w h" normalized 0-1) into a plain image-classification
folder layout suitable for tf.keras.utils.image_dataset_from_directory:

    dataset/train/degree_1/*.jpg
    dataset/train/degree_2/*.jpg
    dataset/train/degree_3/*.jpg
    dataset/val/...
    dataset/test/...

Each labeled box is cropped out (with a padding margin) and saved as its own
training example. Splitting is done at the SOURCE IMAGE level (not per-crop)
so that crops from the same photo never leak across train/val/test.

ASSUMPTION: class id 0/1/2 in the annotations correspond to burn degree
1/2/3 (first/second/third degree). This is the standard convention for a
3-class burn severity dataset, but it is not explicitly confirmed by a
classes.txt/data.yaml file in the downloaded archive. If your Kaggle
notebook/dataset card documents a different order, just edit CLASS_NAMES
below and rerun -- nothing else depends on the mapping.
"""

import argparse
import random
from pathlib import Path

from PIL import Image

CLASS_NAMES = {0: "degree_1", 1: "degree_2", 2: "degree_3"}

PADDING_FRAC = 0.15        # extra context around each box, as a fraction of box size
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test
SEED = 1337


def find_image_for_stem(source_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png"):
        candidate = source_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def crop_box(image: Image.Image, cx, cy, w, h, padding_frac):
    img_w, img_h = image.size
    box_w, box_h = w * img_w, h * img_h
    cx_px, cy_px = cx * img_w, cy * img_h

    pad_w, pad_h = box_w * padding_frac, box_h * padding_frac
    left = max(0, cx_px - box_w / 2 - pad_w)
    top = max(0, cy_px - box_h / 2 - pad_h)
    right = min(img_w, cx_px + box_w / 2 + pad_w)
    bottom = min(img_h, cy_px + box_h / 2 + pad_h)

    if right - left < 4 or bottom - top < 4:
        return None
    return image.crop((left, top, right, bottom))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=r"C:\Users\jokirk\Downloads\archive",
                         help="Folder containing the raw imgN.jpg / imgN.txt files")
    parser.add_argument("--output", default="dataset",
                         help="Output folder for the classification-ready dataset")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    txt_files = sorted(source_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} annotation files in {source_dir}")

    # Build list of (image_path, [ (class_id, cx, cy, w, h), ... ])
    samples = []
    skipped_missing_image = 0
    for txt_path in txt_files:
        image_path = find_image_for_stem(source_dir, txt_path.stem)
        if image_path is None:
            skipped_missing_image += 1
            continue

        boxes = []
        for line in txt_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
            boxes.append((class_id, cx, cy, w, h))

        if boxes:
            samples.append((image_path, boxes))

    print(f"Usable source images: {len(samples)} "
          f"(skipped {skipped_missing_image} annotation files with no matching image)")

    # Split at the source-image level to avoid leakage between splits.
    rng = random.Random(SEED)
    rng.shuffle(samples)
    n = len(samples)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    counts = {split: {name: 0 for name in CLASS_NAMES.values()} for split in splits}

    for split_name, split_samples in splits.items():
        for class_name in CLASS_NAMES.values():
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

        for image_path, boxes in split_samples:
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    for i, (class_id, cx, cy, w, h) in enumerate(boxes):
                        class_name = CLASS_NAMES.get(class_id)
                        if class_name is None:
                            continue
                        crop = crop_box(img, cx, cy, w, h, PADDING_FRAC)
                        if crop is None:
                            continue
                        out_path = output_dir / split_name / class_name / f"{image_path.stem}_{i}.jpg"
                        crop.save(out_path, quality=95)
                        counts[split_name][class_name] += 1
            except Exception as e:
                print(f"  Warning: failed to process {image_path.name}: {e}")

    print("\nCrop counts per split/class:")
    for split_name, class_counts in counts.items():
        total = sum(class_counts.values())
        print(f"  {split_name} (total {total}): {class_counts}")

    print(f"\nDone. Classification-ready dataset written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
