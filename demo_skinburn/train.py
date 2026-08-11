"""
Trains a lightweight MobileNetV2-based classifier to predict burn degree
(degree_1 / degree_2 / degree_3) from a cropped burn-region image.

Run this on a regular PC (NOT the Raspberry Pi) after running data_prep.py.
The Pi only ever runs the quantized .tflite export produced by
convert_to_tflite.py -- it never needs TensorFlow itself.

Usage:
    python train.py --data dataset --epochs 15 --fine-tune-epochs 15
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (160, 160)
BATCH_SIZE = 32


def make_augmenter():
    # Applied only in the tf.data pipeline (training split), NOT inside the
    # exported model -- Keras's random-augmentation layers emit ops (e.g.
    # StatelessRandomUniformV2, ImageProjectiveTransformV3) that the default
    # TFLite converter can't translate without pulling in heavyweight Flex
    # ops, which we don't want on the Pi. Keeping augmentation purely in the
    # data pipeline means the exported model is just Rescaling + MobileNetV2
    # + head, which converts to int8 TFLite cleanly.
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")


def build_datasets(data_dir: Path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int", shuffle=True, seed=1337,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int", shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int", shuffle=False,
    )
    class_names = train_ds.class_names  # alphabetical: degree_1, degree_2, degree_3

    autotune = tf.data.AUTOTUNE
    augmenter = make_augmenter()
    # Cache the decoded (un-augmented) images so disk/decode work isn't
    # repeated each epoch, then shuffle and re-apply random augmentation
    # fresh every epoch (map runs after cache, so it isn't baked in).
    train_ds = train_ds.cache().shuffle(500)
    train_ds = train_ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=autotune)
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(data_dir: Path, class_names):
    labels = []
    for idx, name in enumerate(class_names):
        n = len(list((data_dir / "train" / name).glob("*.jpg")))
        labels += [idx] * n
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)), y=np.array(labels))
    return {i: w for i, w in enumerate(weights)}


def build_model(num_classes: int):
    # Rescaling baked into the model so the exported/quantized model accepts
    # raw 0-255 pixel images directly -- no manual normalization needed on the Pi.
    # (Augmentation deliberately lives in the tf.data pipeline, not here --
    # see make_augmenter()'s docstring for why.)
    preprocessing = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescale_to_-1_1")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), dtype=tf.float32)
    x = preprocessing(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-at", type=int, default=100,
                         help="Unfreeze base model layers from this index onward for fine-tuning")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = build_datasets(data_dir)
    print(f"Class order (index -> name): {list(enumerate(class_names))}")

    class_weights = compute_class_weights(data_dir, class_names)
    print(f"Class weights (imbalance correction): {class_weights}")

    model, base_model = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    checkpoint_path = out_dir / "best_head.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    print("\n=== Phase 1: training classification head (base frozen) ===")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=class_weights, callbacks=callbacks)

    print("\n=== Phase 2: fine-tuning top of MobileNetV2 ===")
    base_model.trainable = True
    for layer in base_model.layers[:args.fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    ft_checkpoint_path = out_dir / "best_finetuned.keras"
    ft_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(ft_checkpoint_path), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
              class_weight=class_weights, callbacks=ft_callbacks)

    print("\n=== Evaluating on held-out test set ===")
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true += labels.numpy().tolist()
        y_pred += np.argmax(preds, axis=1).tolist()

    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

    saved_model_dir = out_dir / "burn_classifier_saved_model"
    model.export(str(saved_model_dir))
    print(f"\nSavedModel exported to {saved_model_dir}")

    labels_path = out_dir / "labels.txt"
    labels_path.write_text("\n".join(class_names))
    print(f"Class labels (index order) written to {labels_path}")

    with open(out_dir / "train_config.json", "w") as f:
        json.dump({"img_size": IMG_SIZE, "class_names": class_names}, f, indent=2)


if __name__ == "__main__":
    main()
