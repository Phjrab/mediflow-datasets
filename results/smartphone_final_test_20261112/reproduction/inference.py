from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .models import ModelSpec, get_model_spec


def preprocess_image(image_path: str | Path, spec: ModelSpec) -> np.ndarray:
    """Load one RGB image as the 0..255 float32 tensor used during training.

    Keras EfficientNetB0 includes a Rescaling(1/255) layer in the saved model.
    Dividing here would normalize twice and is therefore intentionally omitted.
    """

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    with Image.open(path) as opened:
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
        resized = rgb.resize(spec.image_size, Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)

    return np.expand_dims(array, axis=0)


def load_model(spec: ModelSpec) -> Any:
    """Load a saved Keras model without restoring its training optimizer."""

    if not spec.model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {spec.model_path}")

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for inference. Install the project with `pip install -e .`."
        ) from error

    return tf.keras.models.load_model(spec.model_path, compile=False)


def predict_image(
    image_path: str | Path, model_type: str, variant: str = "augmented"
) -> dict[str, object]:
    spec = get_model_spec(model_type, variant)
    class_names = spec.class_names()
    model = load_model(spec)

    output_size = int(model.output_shape[-1])
    if output_size != len(class_names):
        raise ValueError(
            f"Model output size ({output_size}) does not match class count ({len(class_names)})"
        )

    probabilities = np.asarray(model.predict(preprocess_image(image_path, spec), verbose=0)[0])
    index = int(np.argmax(probabilities))
    return {
        "model_type": model_type,
        "variant": variant,
        "predicted_class": class_names[index],
        "confidence": float(probabilities[index]),
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(class_names, probabilities, strict=True)
        },
    }
