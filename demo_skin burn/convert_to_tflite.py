"""
Converts the trained SavedModel into a fully int8-quantized .tflite file
for fast, low-memory inference on a Raspberry Pi CPU.

Run this on the PC, after train.py. Copy the resulting burn_classifier.tflite
and labels.txt onto the Pi -- that's the only thing the Pi needs.

Usage:
    python convert_to_tflite.py --data dataset --model models/burn_classifier_saved_model
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

IMG_SIZE = (160, 160)
NUM_CALIBRATION_IMAGES = 200


def representative_dataset_gen(data_dir: Path):
    image_paths = list((data_dir / "train").rglob("*.jpg"))
    rng = np.random.default_rng(1337)
    rng.shuffle(image_paths)

    def gen():
        for path in image_paths[:NUM_CALIBRATION_IMAGES]:
            img = tf.io.read_file(str(path))
            img = tf.io.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.cast(img, tf.float32)
            img = tf.expand_dims(img, axis=0)
            yield [img]

    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--model", default="models/burn_classifier_saved_model")
    parser.add_argument("--out", default="models/burn_classifier.tflite")
    args = parser.parse_args()

    data_dir = Path(args.data)
    model_dir = Path(args.model)
    out_path = Path(args.out)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(data_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"Wrote fully quantized int8 model to {out_path} ({size_kb:.0f} KB)")

    interpreter = tf.lite.Interpreter(model_path=str(out_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f"Input:  shape={input_details['shape']}, dtype={input_details['dtype']}, "
          f"quant={input_details['quantization']}")
    print(f"Output: shape={output_details['shape']}, dtype={output_details['dtype']}, "
          f"quant={output_details['quantization']}")


if __name__ == "__main__":
    main()
