"""Sequential domain-configured experiments; validation selection precedes any test inference.

The Colab notebook embeds an exact copy of this module so no repository checkout
is needed in Colab. Completed trials are reused only after artifact verification.
Interrupted attempts are preserved and restarted, not resumed mid-epoch.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import uuid
from pathlib import Path

import keras
import numpy as np

PROTOCOL = "mediflow_common_v1"
TRIALS = [
    {"id": "b0_224_ce", "backbone": "B0", "size": 224, "loss": "ce"},
    {"id": "b0_256_ce", "backbone": "B0", "size": 256, "loss": "ce"},
    {"id": "b0_256_ls005", "backbone": "B0", "size": 256, "loss": "ls005"},
    {"id": "b0_256_focal15", "backbone": "B0", "size": 256, "loss": "focal15"},
    {"id": "b1_256_ls005", "backbone": "B1", "size": 256, "loss": "ls005"},
]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(".json-" + uuid.uuid4().hex[:12] + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def classification_metrics(truth, probabilities, count=5):
    truth = np.asarray(truth, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    if (
        truth.ndim != 1
        or not len(truth)
        or probabilities.shape != (len(truth), count)
        or not np.isfinite(probabilities).all()
        or np.any(truth < 0)
        or np.any(truth >= count)
    ):
        raise ValueError("Invalid evaluation arrays")
    predictions = probabilities.argmax(axis=1)
    cm = np.bincount(count * truth + predictions, minlength=count * count).reshape(count, count)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(count), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(count), where=cm.sum(1) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(count),
        where=precision + recall != 0,
    )
    return {
        "accuracy": float(np.mean(truth == predictions)),
        "macro_f1": float(f1.mean()),
        "class_f1": f1.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "support": cm.sum(1).tolist(),
        "confusion_matrix": cm.tolist(),
        "count": len(truth),
    }


def predict_dataset(model, dataset):
    truth, probabilities = [], []
    for images, labels in dataset:
        probabilities.extend(model(images, training=False).numpy())
        truth.extend(np.argmax(labels.numpy(), axis=1))
    return np.asarray(truth, dtype=np.int64), np.asarray(probabilities)


def save_predictions(path, paths, truth, probabilities):
    if len(paths) != len(truth):
        raise ValueError("File order and prediction count differ")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "path",
                "true_index",
                "pred_index",
                *[f"prob_C{i}" for i in range(probabilities.shape[1])],
            ]
        )
        for name, target, probs in zip(paths, truth, probabilities, strict=True):
            writer.writerow([name, int(target), int(probs.argmax()), *map(float, probs)])


def loss_function(name):
    if name == "ce":
        return keras.losses.CategoricalCrossentropy()
    if name == "ls005":
        return keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    if name == "focal15":
        return keras.losses.CategoricalFocalCrossentropy(alpha=1.0, gamma=1.5)
    raise ValueError(name)


def build_model(spec):
    builder = {
        "B0": keras.applications.EfficientNetB0,
        "B1": keras.applications.EfficientNetB1,
        "V2S": keras.applications.EfficientNetV2S,
    }
    size = spec["size"]
    backbone = builder[spec["backbone"]](
        include_top=False, weights="imagenet", input_shape=(size, size, 3)
    )
    backbone.trainable = False
    inputs = keras.Input((size, size, 3))
    features = backbone(inputs, training=False)
    features = keras.layers.GlobalAveragePooling2D()(features)
    features = keras.layers.Dropout(0.3)(features)
    outputs = keras.layers.Dense(spec["class_count"], activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=loss_function(spec["loss"]),
        metrics=["accuracy"],
    )
    return model


def configure_partial(model, loss_name):
    backbones = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower()
    ]
    if len(backbones) != 1:
        raise ValueError("Expected one EfficientNet backbone")
    backbone = backbones[0]
    backbone.trainable = True
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= len(backbone.layers) - 30 and not isinstance(
            layer, keras.layers.BatchNormalization
        )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5), loss=loss_function(loss_name), metrics=["accuracy"]
    )
    return [layer.name for layer in backbone.layers if layer.trainable]


class HistoryBackup(keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.values = {}

    def on_epoch_end(self, epoch, logs=None):
        for key, value in (logs or {}).items():
            self.values.setdefault(key, []).append(float(value))
        write_json(self.path, self.values)


def fit_stage(model, train, val, directory, name, epochs):
    best = directory / (name + "_best.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(best), monitor="val_accuracy", mode="max", save_best_only=True
        ),
        keras.callbacks.CSVLogger(str(directory / (name + "_log.csv"))),
        HistoryBackup(directory / (name + "_history.json")),
        keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks, verbose=2)
    values = {key: [float(v) for v in seq] for key, seq in history.history.items()}
    if len(values.get("val_accuracy", [])) != epochs or not all(
        np.isfinite(seq).all() for seq in values.values()
    ):
        raise RuntimeError("Incomplete or non-finite training; attempt retained")
    model.save(directory / (name + "_last.keras"))
    return values, best


def checkpoint_choice(baseline_score, new_score):
    """Keep the earlier/simpler checkpoint on ties."""
    return new_score > baseline_score


def cached_record(root, trial_id, signature):
    trial_dir = Path(root) / trial_id
    marker = trial_dir / "completed.json"
    if not marker.exists():
        return None
    record = read_json(marker)
    if record["signature"] != signature:
        raise ValueError("Resume settings differ; use a new suite directory")
    for relative, digest in record["artifact_hashes"].items():
        target = (trial_dir / relative).resolve()
        if not target.is_relative_to(trial_dir.resolve()) or file_hash(target) != digest:
            raise ValueError("Completed artifact changed or corrupted: " + relative)
    return record


def evaluate_to_files(model, dataset, paths, directory, prefix):
    truth, probabilities = predict_dataset(model, dataset)
    metrics = classification_metrics(truth, probabilities, probabilities.shape[1])
    write_json(directory / (prefix + "_metrics.json"), metrics)
    save_predictions(directory / (prefix + "_predictions.csv"), paths, truth, probabilities)
    return metrics


def run_trial(spec, dataset_factory, root, signature, seed=42, epochs1=15, epochs2=10):
    cached = cached_record(root, spec["id"], signature)
    if cached:
        return cached
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    directory = Path(root) / spec["id"] / ("attempt_" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "spec.json", spec)
    train, _ = dataset_factory("train", spec["size"], True)
    val, paths = dataset_factory("val", spec["size"], False)
    started = time.monotonic()
    model = build_model(spec)
    h1, best1 = fit_stage(model, train, val, directory, "stage1", epochs1)
    del model
    keras.backend.clear_session()
    model = keras.models.load_model(best1, compile=False)
    trainable = []
    h2 = {key: [] for key in h1}
    best2 = best1
    if epochs2:
        trainable = configure_partial(model, spec["loss"])
        h2, best2 = fit_stage(model, train, val, directory, "stage2", epochs2)
    del model
    selected_stage = (
        "stage2"
        if checkpoint_choice(max(h1["val_accuracy"]), max(h2["val_accuracy"], default=-1.0))
        else "stage1"
    )
    selected = best2 if selected_stage == "stage2" else best1
    model = keras.models.load_model(selected, compile=False)
    metrics = evaluate_to_files(model, val, paths, directory, "validation")
    record = {
        "id": spec["id"],
        "spec": spec,
        "signature": signature,
        "attempt": directory.name,
        "selected_model": selected.name,
        "selected_stage": selected_stage,
        "validation": metrics,
        "stage1_best_val": max(h1["val_accuracy"]),
        "stage2_best_val": max(h2["val_accuracy"]) if h2["val_accuracy"] else None,
        "training_seconds": time.monotonic() - started,
        "parameters": model.count_params(),
        "model_bytes": selected.stat().st_size,
        "trainable_backbone_layers": trainable,
        "history": {key: h1[key] + h2[key] for key in h1},
        "stage_boundary": len(h1["accuracy"]),
    }
    finish_record(directory, record)
    return record


def finish_record(directory, record):
    write_json(directory / "record.json", record)
    record["artifact_hashes"] = {
        str(path.relative_to(directory.parent)): file_hash(path)
        for path in directory.iterdir()
        if path.is_file()
    }
    write_json(directory.parent / "completed.json", record)


def selected_model_path(root, record):
    return Path(root) / record["id"] / record["attempt"] / record["selected_model"]


def extend_b1(parent, dataset_factory, root, signature, seed=42, epochs=5):
    trial_id = "b1_256_ls005_extend5"
    cached = cached_record(root, trial_id, signature)
    if cached:
        return cached
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    directory = Path(root) / trial_id / ("attempt_" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    parent_dir = Path(root) / parent["id"] / parent["attempt"]
    # Continue from epoch 10's LAST checkpoint, including optimizer state.
    source = parent_dir / "stage2_last.keras"
    model = keras.models.load_model(source)
    if model.optimizer is None:
        raise ValueError("Extension requires saved optimizer")
    train, _ = dataset_factory("train", 256, True)
    val, paths = dataset_factory("val", 256, False)
    started = time.monotonic()
    history, best = fit_stage(model, train, val, directory, "extension", epochs)
    del model
    parent_best = selected_model_path(root, parent)
    parent_score = max(parent["stage1_best_val"], parent["stage2_best_val"])
    keep_extension = checkpoint_choice(parent_score, max(history["val_accuracy"]))
    selected = directory / "selected.keras"
    import shutil

    shutil.copyfile(best if keep_extension else parent_best, selected)
    model = keras.models.load_model(selected, compile=False)
    metrics = evaluate_to_files(model, val, paths, directory, "validation")
    record = {
        "id": trial_id,
        "spec": {**parent["spec"], "id": trial_id},
        "signature": signature,
        "attempt": directory.name,
        "selected_model": selected.name,
        "selected_stage": "extension" if keep_extension else "parent_" + parent["selected_stage"],
        "validation": metrics,
        "parameters": model.count_params(),
        "model_bytes": selected.stat().st_size,
        "training_seconds": time.monotonic() - started,
        "history": {key: parent["history"][key] + history[key] for key in history},
        "stage_boundary": parent["stage_boundary"],
        "extension_boundary": len(parent["history"]["accuracy"]),
        "parent_last_sha256": file_hash(source),
        "optimizer_restored": True,
        "extension_best_val": max(history["val_accuracy"]),
        "stage1_best_val": parent["stage1_best_val"],
        "stage2_best_val": parent["stage2_best_val"],
    }
    finish_record(directory, record)
    return record


def select_winner(records):
    # Stable order preserves earlier experiments on exact ties.
    return max(records, key=lambda record: record["validation"]["accuracy"])
