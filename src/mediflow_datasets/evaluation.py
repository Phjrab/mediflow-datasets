from __future__ import annotations

from pathlib import Path

import numpy as np

from .inference import load_model, preprocess_image
from .models import get_model_spec

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, class_names: list[str]
) -> dict[str, object]:
    count = len(class_names)
    confusion = np.zeros((count, count), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, class_name in enumerate(class_names):
        true_positive = int(confusion[index, index])
        predicted_total = int(confusion[:, index].sum())
        actual_total = int(confusion[index, :].sum())
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_total,
        }

    total = int(confusion.sum())
    return {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_dataset(
    model_type: str,
    dataset_path: str | Path,
    variant: str = "augmented",
    batch_size: int = 32,
) -> dict[str, object]:
    spec = get_model_spec(model_type, variant)
    class_names = spec.class_names()
    dataset_root = Path(dataset_path).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Test dataset directory not found: {dataset_root}")

    actual_directories = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    if set(actual_directories) != set(class_names):
        raise ValueError(
            "Dataset class folders do not match the model classes. "
            f"Expected: {class_names}; found: {actual_directories}"
        )

    samples: list[tuple[Path, int]] = []
    for label, class_name in enumerate(class_names):
        class_images = sorted(
            path
            for path in (dataset_root / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not class_images:
            raise ValueError(f"No images found for class: {class_name}")
        samples.extend((path, label) for path in class_images)

    model = load_model(spec)
    all_labels: list[int] = []
    all_predictions: list[int] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        inputs = np.concatenate([preprocess_image(path, spec) for path, _ in batch], axis=0)
        probabilities = np.asarray(model.predict(inputs, verbose=0))
        all_labels.extend(label for _, label in batch)
        all_predictions.extend(np.argmax(probabilities, axis=1).astype(int).tolist())

    metrics = classification_metrics(
        np.asarray(all_labels), np.asarray(all_predictions), class_names
    )
    return {
        "model_type": model_type,
        "variant": variant,
        "dataset": str(dataset_root),
        "image_count": len(samples),
        "classes": class_names,
        **metrics,
    }
