"""Standalone single-package inference, copied as inference.py into each delivery ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError("Package file is outside its directory")
    return path


def read_json(path):
    return json.loads(Path(path).read_bytes())


def verify_package(package_dir):
    root = Path(package_dir).resolve()
    manifest = read_json(root / "package_manifest.json")
    if manifest.get("schema") != "mediflow_adaptation_package_manifest_v1":
        raise ValueError("Unknown package manifest")
    if not {"model_card.json", "class_names.json", "preprocessing.json"}.issubset(
        manifest["files"]
    ):
        raise ValueError("Required package metadata is not covered by the manifest")
    for relative, expected in manifest["files"].items():
        if file_sha256(package_path(root, relative)) != expected:
            raise ValueError("Package file hash mismatch: " + relative)
    card = read_json(root / "model_card.json")
    classes = read_json(root / "class_names.json")
    preprocessing = read_json(root / "preprocessing.json")
    if (
        not isinstance(classes, list)
        or not classes
        or not all(isinstance(name, str) and name for name in classes)
        or len(classes) != len(set(classes))
        or card["class_names"] != classes
        or card["class_count"] != len(classes)
        or card["preprocessing"] != preprocessing
        or manifest["package_id"] != card["package_id"]
        or card["model_sha256"] != manifest["files"][card["model_file"]]
    ):
        raise ValueError("Class order, model or preprocessing metadata mismatch")
    return root, card, classes, preprocessing


def load_image(image_path, preprocessing):
    """Match the frozen Test: RGB, no EXIF rotation, optional center crop, TF resize."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError("Image file not found: " + str(path))
    try:
        image = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    except (tf.errors.InvalidArgumentError, tf.errors.DataLossError) as error:
        raise ValueError("Cannot decode the supplied image") from error
    if preprocessing["input_policy"] == "center_square":
        shape = tf.shape(image)
        side = tf.minimum(shape[0], shape[1])
        image = tf.image.crop_to_bounding_box(
            image, (shape[0] - side) // 2, (shape[1] - side) // 2, side, side
        )
    elif preprocessing["input_policy"] != "legacy_resize":
        raise ValueError("Unknown input policy")
    if (
        preprocessing["dtype"] != "float32"
        or preprocessing["color"] != "RGB"
        or preprocessing["layout"] != "NHWC"
        or preprocessing["exif_rotation_applied"] is not False
        or preprocessing["pixel_range"] != [0, 255]
        or preprocessing["external_division_by_255"] is not False
        or preprocessing["resize_method"] != "bilinear"
        or preprocessing["antialias"] is not False
    ):
        raise ValueError("Unsupported preprocessing contract")
    size = preprocessing["input_size"]
    image = tf.image.resize(image, (size, size), method="bilinear", antialias=False)
    return tf.expand_dims(tf.cast(image, tf.float32), 0)


class Predictor:
    """Load this package once and reuse it for individual images."""

    def __init__(self, package_dir=None):
        root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
        self.root, self.card, self.classes, self.preprocessing = verify_package(root)
        self.model = keras.models.load_model(
            package_path(self.root, self.card["model_file"]), compile=False, safe_mode=True
        )
        size, count = self.preprocessing["input_size"], len(self.classes)
        if tuple(self.model.input_shape) != (None, size, size, 3):
            raise ValueError("Model input shape differs from the package")
        adapter, shapes = self.card["output_adapter"], self.model.output_shape
        if adapter == "sum_logits_softmax":
            valid = isinstance(shapes, list) and len(shapes) == 4
            valid = valid and all(tuple(shape) == (None, count) for shape in shapes)
        else:
            valid = adapter == "direct_softmax" and tuple(shapes) == (None, count)
        if not valid:
            raise ValueError("Model class count or PMG output contract differs")
        stack = list(self.model.layers)
        rescaling_found = False
        while stack:
            layer = stack.pop()
            if isinstance(layer, keras.Model):
                stack.extend(layer.layers)
            if isinstance(layer, keras.layers.Rescaling):
                scale = np.asarray(layer.scale)
                rescaling_found |= (
                    scale.size == 1
                    and np.isclose(float(scale.item()), 1 / 255)
                    and np.allclose(layer.offset, 0)
                )
        if not rescaling_found:
            raise ValueError("Expected internal Rescaling(1/255) is missing")

    def scores(self, inputs):
        outputs = self.model(inputs, training=False)
        if self.card["output_adapter"] == "sum_logits_softmax":
            outputs = tf.nn.softmax(tf.add_n(outputs), axis=-1)
        values = np.asarray(outputs, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.classes)
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values > 1)
            or not np.allclose(values.sum(1), 1, atol=1e-5, rtol=0)
        ):
            raise ValueError("Model returned invalid classification scores")
        return values

    def predict_file(self, image_path):
        values = self.scores(load_image(image_path, self.preprocessing))[0]
        index = int(values.argmax())
        return {
            "schema": "mediflow_single_package_prediction_v1",
            "package_id": self.card["package_id"],
            "domain": self.card["domain"],
            "adaptation_stage": self.card["adaptation_stage"],
            "model_sha256": self.card["model_sha256"],
            "predicted_index": index,
            "predicted_class": self.classes[index],
            "top_score": float(values[index]),
            "scores": [
                {"index": i, "class_name": name, "score": float(values[i])}
                for i, name in enumerate(self.classes)
            ],
            "normal_class_included": self.card["normal_class_included"],
            "score_meaning": "uncalibrated_model_score_not_probability_of_correctness",
            "input_scope_checked": False,
            "clinical_diagnosis": False,
        }


def load_predictor(package_dir=None):
    return Predictor(package_dir)


def predict_file(image_path, package_dir=None):
    return load_predictor(package_dir).predict_file(image_path)


def main():
    parser = argparse.ArgumentParser(description="MediFlow 패키지 한 개의 이미지 분류")
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        result = predict_file(args.image)
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
