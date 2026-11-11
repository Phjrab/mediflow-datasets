"""Private Target input and feature alignment for the staged camera experiment."""

import base64
import binascii
import hashlib
import io
import json

import numpy as np

from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets import domain_alignment


def target_inputs(runtime, domain, spec, task, key_text):
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        key = base64.b64decode(key_text.strip(), validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("3차 복호화 키 한 줄을 확인하세요.") from None
    if len(key) != 32:
        raise ValueError("3차 복호화 키 길이가 맞지 않습니다.")
    stem = f"target_encrypted/{domain}_{spec['size']}"
    member = stem + ".bin"
    metadata = json.loads((runtime / (stem + ".json")).read_text(encoding="utf-8"))
    rows = metadata["rows"]
    if (
        metadata["cipher"] != "AES-256-GCM"
        or metadata["aad_utf8"] != member
        or metadata["input_policy"] != task["input_policy"]
        or metadata["input_size"] != spec["size"]
        or len({r["id"] for r in rows}) != len(rows)
    ):
        raise ValueError(f"Unexpected Target image metadata: {domain}")
    if not domain_alignment.balanced_schedule(rows, [0], 42, 0):
        raise ValueError(f"Missing Target participants: {domain}")
    payload = (runtime / member).read_bytes()
    try:
        plaintext = AESGCM(key).decrypt(payload[:12], payload[12:], member.encode("utf-8"))
    except InvalidTag:
        raise ValueError("3차 복호화 키가 공통 ZIP v3와 일치하지 않습니다.") from None
    if hashlib.sha256(plaintext).hexdigest() != metadata["plaintext_sha256"]:
        raise ValueError(f"Target data hash differs: {domain}")
    with np.load(io.BytesIO(plaintext), allow_pickle=False) as arrays:
        images = arrays["images"]
    size = spec["size"]
    if (
        images.shape != (len(rows), size, size, 3)
        or images.dtype != np.float32
        or not np.isfinite(images).all()
        or images.min() < 0
        or images.max() > 255
    ):
        raise ValueError(f"Invalid Target tensor: {domain}")
    return {"rows": rows, "images": images, "sha256": metadata["plaintext_sha256"]}


def sampled_indices(rows, count):
    indexes = [
        i
        for label in range(count)
        for i in [j for j, row in enumerate(rows) if row.label == label][:8]
    ]
    if len(indexes) < 2 or {rows[i].label for i in indexes} != set(range(count)):
        raise ValueError("The alignment reference requires samples from every class")
    return indexes


def feature_outputs(model, images, spec, batch_size):
    import tensorflow as tf

    pmg = spec["architecture"] == "pmg"
    extractor = exp.feature_model(model, "sum_logits_softmax" if pmg else "single_softmax")
    scores, features = [], []
    for start in range(0, len(images), batch_size):
        outputs = extractor(images[start : start + batch_size], training=False)
        scores.append((tf.nn.softmax(tf.add_n(outputs[:-1]), -1) if pmg else outputs[0]).numpy())
        features.append(outputs[-1].numpy())
    scores, features = np.concatenate(scores), np.concatenate(features)
    if not np.isfinite(scores).all() or not np.isfinite(features).all():
        raise ValueError("Non-finite feature diagnostic inputs")
    return scores, features


def calibrate(model, cache, rows, task, spec, target):
    import keras

    count, batch = len(task["classes"]), task["batch_size"]
    indices = sampled_indices(rows, count)
    scores, source_features = feature_outputs(model, cache[indices], spec, batch)
    _, target_features = feature_outputs(model, target["images"], spec, batch)
    reference = exp.feature_reference(source_features)
    people = np.asarray([row["person"] for row in target["rows"]])
    means = np.stack([target_features[people == person].mean(0) for person in np.unique(people)])
    a = (source_features - reference["mean"]) / reference["std"]
    b = (means - reference["mean"]) / reference["std"]
    distances = ((a[:, None] - b[None]) ** 2).sum(-1)
    positive = distances[distances > 1e-8]
    if len(positive):
        reference["bandwidth_sq"] = float(np.median(positive))
    reference["bandwidth_multipliers"] = [0.25, 1.0, 4.0]
    mmd, coral = map(float, domain_alignment.losses(source_features, target_features, reference))
    labels = np.eye(count, dtype=np.float32)[[rows[i].label for i in indices]]
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=spec["label_smoothing"])
    classification = max(0.05, float(loss_fn(labels, scores)))
    if min(mmd, coral) <= 1e-8:
        raise ValueError("An initial alignment loss is too small to calibrate both terms")
    weights = {
        name: min(10.0, 0.05 * classification / value)
        for name, value in (("mmd", mmd), ("coral", coral))
    }
    return {
        "reference": {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in reference.items()
        },
        "weights": weights,
        "source_train_paths": [rows[i].path for i in indices],
        "target_sha256": target["sha256"],
        "target_role": "unlabeled_adaptation_train_and_diagnostic",
        "initial_mmd": mmd,
        "initial_coral": coral,
        "initial_classification_loss": classification,
        "weight_policy": "5% of starting Source Train CE per term, coefficient capped at 10",
    }


def diagnostic(model, expanded_val, val, task, spec, target, reference):
    count, batch = len(task["classes"]), task["batch_size"]
    indices = sampled_indices(val, count)
    _, source_features = feature_outputs(model, expanded_val[indices], spec, batch)
    scores, target_features = feature_outputs(model, target["images"], spec, batch)
    mmd, coral = map(float, domain_alignment.losses(source_features, target_features, reference))
    labels = scores.argmax(axis=1)
    return {
        "mmd": mmd,
        "coral": coral,
        "source_validation_count": len(indices),
        "target_training_count": len(target["rows"]),
        "target_predicted_class_counts": np.bincount(labels, minlength=count).tolist(),
    }, labels
