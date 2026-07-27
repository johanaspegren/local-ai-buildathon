"""
Raspberry Pi inference script. Works with LiteRT or TensorFlow Lite runtimes
plus Pillow and numpy (see requirements-pi.txt) -- no full TensorFlow install
is required.

Usage on the Pi:
    python3 infer_pi.py path/to/photo.jpg
    python3 infer_pi.py path/to/photo.jpg --model burn_classifier.tflite --labels labels.txt

Expects a reasonably close-up photo of the burn area (similar framing to the
cropped training images) rather than a wide scene shot -- this model does
classification only, not detection/localization.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            # Falls back to the bundled interpreter from TensorFlow, useful for
            # development machines that have TensorFlow installed but not the
            # smaller LiteRT/tflite-runtime packages.
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "Unable to import a TensorFlow Lite runtime. Install one of:\n"
                "  pip install ai-edge-litert\n"
                "  pip install tflite-runtime\n"
                "  pip install tensorflow-cpu"
            ) from exc


def load_labels(labels_path: Path) -> list[str]:
    return [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]


def preprocess(image_path: Path, size) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize(size)
        arr = np.asarray(img, dtype=np.uint8)
    return np.expand_dims(arr, axis=0)


def dequantize(raw_output: np.ndarray, quant_params) -> np.ndarray:
    scale, zero_point = quant_params
    if scale == 0:
        return raw_output.astype(np.float32)
    return (raw_output.astype(np.float32) - zero_point) * scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to the burn photo to classify")
    parser.add_argument("--model", default="burn_classifier.tflite")
    parser.add_argument("--labels", default="labels.txt")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    labels = load_labels(Path(args.labels))

    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    height, width = input_details["shape"][1], input_details["shape"][2]
    input_data = preprocess(image_path, (width, height))

    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details["index"])[0]

    probs = dequantize(raw_output, output_details["quantization"])
    probs = probs / probs.sum() if probs.sum() > 0 else probs

    top_idx = int(np.argmax(probs))
    print(f"Predicted: {labels[top_idx]}  (confidence: {probs[top_idx]:.1%})")
    print("\nAll classes:")
    for i, label in enumerate(labels):
        print(f"  {label:12s} {probs[i]:.1%}")


if __name__ == "__main__":
    main()
