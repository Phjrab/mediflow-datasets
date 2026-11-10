"""Check actual saved-model feature gradients; never save or update model weights."""

from __future__ import annotations

import argparse
import gc
import io
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from PIL import Image

from mediflow_datasets import camera_extended as camera
from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets import domain_alignment as alignment
from mediflow_datasets import domain_nine as nine
from mediflow_datasets.candidate_reproduction import ROOT


def check(preparation):
    config = data.read_json(preparation / "nine_experiments.json")
    result = {}
    for domain in nine.DOMAINS:
        keras.backend.clear_session()
        spec, checkpoint, classes = runner.task_candidate(domain, config)
        before = data.sha256(checkpoint)
        profile = data.read_json(preparation / domain / "extended_profile.json")
        _, target = nine.target_inputs(preparation / domain, domain, spec)
        old = ROOT / "camera_capture/_review/preflight_v2_fbe37d44" / domain / "transform_qa"
        rows = data.read_json(preparation / domain / "qa_measurements.json")
        chosen = [next(r for r in rows if r["label"] == label) for label in (0, 1)]
        images = []
        for row in chosen:
            with Image.open(old / (row["id"] + "_before.png")) as image:
                stream = io.BytesIO()
                image.convert("RGB").save(stream, format="PNG")
            # Previews already have the agreed preprocessing, so do not crop them again.
            image = exp.image_tensor(stream.getvalue(), spec["input_size"], "legacy_resize")
            images.append(camera.transform(image, profile, "aligned", 42))
        images = np.stack(images)
        target_meta = data.read_json(preparation / domain / "target_inputs.json")
        indices = [
            next(i for i, r in enumerate(target_meta["rows"]) if r["person"] == p)
            for p in sorted({r["person"] for r in target_meta["rows"]})
        ]
        model = keras.models.load_model(checkpoint, compile=False)
        exp.configure_partial(model)
        extractor = exp.feature_model(model, spec["output_adapter"])
        features = extractor(images, training=False)[-1].numpy()
        reference = exp.feature_reference(features)
        reference["bandwidth_multipliers"] = [0.25, 1, 4]
        with tf.GradientTape() as tape:
            source_output = extractor(images, training=False)
            target_output = extractor(target[indices], training=False)
            mmd, coral = alignment.losses(source_output[-1], target_output[-1], reference)
            loss = mmd + coral
        gradients = [g for g in tape.gradient(loss, model.trainable_variables) if g is not None]
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Non-finite feature alignment gradient")
        norm = float(tf.linalg.global_norm(gradients))
        if not np.isfinite(norm) or norm <= 0 or data.sha256(checkpoint) != before:
            raise ValueError("Model gradient or preservation check failed: " + domain)
        outputs = source_output[:-1]
        if len(outputs) != (4 if domain == "web_skin" else 1):
            raise ValueError("Unexpected model output contract")
        if any(tuple(v.shape) != (2, len(classes)) for v in outputs):
            raise ValueError("Class order dimension mismatch")
        result[domain] = {
            "model_sha256": before,
            "model_size": spec["input_size"],
            "class_count": len(classes),
            "feature_dimensions": int(features.shape[1]),
            "source_batch": 2,
            "target_batch": 4,
            "source_input_paths": [r["path"] for r in chosen],
            "source_input_range": [float(images.min()), float(images.max())],
            "mmd": float(mmd),
            "coral": float(coral),
            "alignment_gradient_global_norm": norm,
            "weights_updated": False,
            "checkpoint_unchanged": True,
            "meaning": "Compatibility check, not model performance or a training experiment",
        }
        print(domain, result[domain], flush=True)
        del tape, gradients, model, extractor, source_output, target_output
        keras.backend.clear_session()
        gc.collect()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runner.write_json(args.output, check(args.preparation))
