"""Source-controlled training and label-aware evaluation for measured camera shifts."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import asdict

import cv2
import numpy as np

from mediflow_datasets import common_engine as engine
from mediflow_datasets.candidate_reproduction import sha256
from mediflow_datasets.domain_adaptation_io import write_progress

ARMS = ("source_only", "brightness_only", "target_guided")


def image_tensor(data, size, input_policy="legacy_resize"):
    import tensorflow as tf

    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    if input_policy == "center_square":
        shape = tf.shape(image)
        side = tf.minimum(shape[0], shape[1])
        image = tf.image.crop_to_bounding_box(
            image, (shape[0] - side) // 2, (shape[1] - side) // 2, side, side
        )
    elif input_policy != "legacy_resize":
        raise ValueError(f"Unknown input policy: {input_policy}")
    return tf.image.resize(image, (size, size), method="bilinear", antialias=False).numpy()


def item_seed(row, seed, epoch=0):
    value = f"{seed}|{epoch}|{row.path}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def augmentation_applies(seed, probability):
    """Use an independent stream, leaving the transform's full parameter range available."""
    if not 0 <= probability <= 1:
        raise ValueError("Augmentation probability must be between zero and one")
    return np.random.default_rng(np.random.SeedSequence([seed, 0x4D464150])).random() < probability


def arm_plan(profile):
    """Do not present numerically identical transformations as different experiments."""
    extra = any(
        list(profile.get(key, identity)) != identity
        for key, identity in (
            ("contrast", [1.0, 1.0]),
            ("left_light", [0.0, 0.0]),
            ("scale_x", [1.0, 1.0]),
            ("scale_y", [1.0, 1.0]),
        )
    )
    return (
        (ARMS, {})
        if extra
        else (("source_only", "brightness_only"), {"target_guided": "brightness_only"})
    )


def bounded_photometric_update(before, proposal):
    """Approach the sensor bounds smoothly instead of flattening out-of-range updates.

    This operates in RGB 0-255, and is not external normalization for the model.
    The correction agrees with an additive update to first order away from the bounds.
    """
    original = np.asarray(before, dtype=np.float64)
    delta = np.asarray(proposal, dtype=np.float64) - original
    headroom = np.maximum(255.0 - original, 1e-12)
    footroom = np.maximum(original, 1e-12)
    positive = original + headroom * (-np.expm1(-np.maximum(delta, 0) / headroom))
    negative = original * np.exp(np.minimum(delta, 0) / footroom)
    result = np.where(delta >= 0, positive, negative)
    # Keep originally interior float32 values interior after the final cast as well.
    interior = (original > 0) & (original < 255)
    lower = float(np.nextafter(np.float32(0), np.float32(1)))
    upper = float(np.nextafter(np.float32(255), np.float32(0)))
    result = np.where(interior, np.clip(result, lower, upper), result)
    return result.clip(0, 255).astype(np.float32)


def transform_image(image, profile, arm, seed):
    if arm in ("baseline", "source_only"):
        return image.copy()
    if arm not in ARMS:
        raise ValueError(arm)
    rng = np.random.default_rng(seed)
    before = image.copy()
    offset = rng.uniform(*profile["brightness_offset"])
    if arm == "target_guided":
        h, w = image.shape[:2]
        if any(list(profile.get(key, [1, 1])) != [1, 1] for key in ("scale_x", "scale_y")):
            raise ValueError("Unsafe legacy geometry profile; regenerate a photometric v2 profile")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        contrast = rng.uniform(*profile["contrast"])
        image = image + ((gray - np.median(gray)) * (contrast - 1))[:, :, None]
        light = rng.uniform(*profile["left_light"])
        ramp = np.linspace(0.5, -0.5, w, dtype=np.float32)
        image = image + light * ramp[None, :, None]
    proposal = image + offset
    operator = profile.get("photometric_operator", "affine_clip_v2")
    if operator == "bounded_additive_v3":
        transformed = bounded_photometric_update(before, proposal)
    elif operator == "affine_clip_v2":
        transformed = np.clip(proposal, 0, 255).astype(np.float32)
    else:
        raise ValueError(f"Unknown photometric operator: {operator}")
    if transformed.shape != before.shape or not np.isfinite(transformed).all():
        raise ValueError("Transform contract failed")
    return transformed


def feature_model(model, adapter):
    import keras

    if adapter == "sum_logits_softmax":
        features = model.get_layer("pmg_fusion").output
    else:
        pools = [
            layer
            for layer in model.layers
            if isinstance(layer, keras.layers.GlobalAveragePooling2D)
        ]
        if len(pools) != 1:
            raise ValueError("Expected one feature pooling layer")
        features = pools[0].output
    return keras.Model(model.input, [*model.outputs, features])


def infer(
    model, rows, reader, size, adapter, batch_size, profile=None, input_policy="legacy_resize"
):
    import tensorflow as tf

    extractor = feature_model(model, adapter)
    scores, features = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = [image_tensor(reader(row), size, input_policy) for row in batch]
        if profile is not None:
            images = [
                transform_image(image, profile, "target_guided", item_seed(row, 707))
                for row, image in zip(batch, images, strict=True)
            ]
        outputs = extractor(np.stack(images), training=False)
        features.extend(outputs[-1].numpy())
        if adapter == "sum_logits_softmax":
            score = tf.nn.softmax(tf.add_n(outputs[:-1]), axis=-1)
        else:
            score = outputs[0]
        scores.extend(score.numpy())
    scores, features = np.asarray(scores), np.asarray(features)
    if not np.isfinite(scores).all() or not np.isfinite(features).all():
        raise ValueError("Non-finite model output")
    return scores, features


def classification(rows, scores, count):
    labeled = [i for i, row in enumerate(rows) if row.label is not None]
    if not labeled:
        return {"status": "unlabeled", "count": len(rows), "accuracy": None, "macro_f1": None}
    metrics = engine.classification_metrics(
        np.asarray([rows[i].label for i in labeled]), scores[labeled], count
    )
    return {"status": "measured", "labeled_count": len(labeled), **metrics}


def feature_reference(source):
    mean, std = source.mean(0), source.std(0)
    std = np.maximum(std, 1e-3)
    normalized = (source - mean) / std
    distances = np.maximum(0, ((normalized[:, None] - normalized[None, :]) ** 2).sum(-1))
    nonzero = distances[distances > 1e-8]
    return {
        "mean": mean,
        "std": std,
        "bandwidth_sq": float(np.median(nonzero)) if len(nonzero) else 1.0,
    }


def feature_distance(source, target, reference):
    """Biased RBF MMD² and Deep-CORAL covariance loss in fixed baseline coordinates."""
    if min(len(source), len(target)) < 2 or source.shape[1] != target.shape[1]:
        raise ValueError(
            "Feature distance needs at least two rows per domain and matching dimensions"
        )
    a, b = ((values - reference["mean"]) / reference["std"] for values in (source, target))

    def kernel(x, y):
        distance = np.maximum(0, (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T)
        return np.exp(-distance / (2 * reference["bandwidth_sq"]))

    mmd = float(kernel(a, a).mean() + kernel(b, b).mean() - 2 * kernel(a, b).mean())
    ac, bc = a - a.mean(0), b - b.mean(0)
    covariance_frobenius = (
        ((ac @ ac.T) ** 2).sum() / (len(ac) - 1) ** 2
        + ((bc @ bc.T) ** 2).sum() / (len(bc) - 1) ** 2
        - 2 * ((ac @ bc.T) ** 2).sum() / ((len(ac) - 1) * (len(bc) - 1))
    )
    coral = float(max(0, covariance_frobenius) / (4 * a.shape[1] ** 2))
    return {
        "mmd2_biased_rbf": max(0.0, mmd),
        "coral_covariance": coral,
        "source_n": len(source),
        "target_n": len(target),
        "meaning": "distribution distance only; does not measure target classification",
    }


def configure_partial(model):
    import keras

    backbones = [layer for layer in model.layers if isinstance(layer, keras.Model)]
    if len(backbones) != 1:
        raise ValueError("Expected one saved backbone")
    backbone = backbones[0]
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= len(backbone.layers) - 30 and not isinstance(
            layer, keras.layers.BatchNormalization
        )


def train_epoch(model, rows, reader, spec, config, profile, arm, epoch, optimizer):
    import keras
    import tensorflow as tf

    rng = np.random.default_rng(config["seed"] + epoch)
    order = rng.permutation(len(rows))
    size, adapter = spec["input_size"], spec["output_adapter"]
    smoothing = 0.05 if spec["loss"] == "ls005" else 0.0
    loss_fn = keras.losses.CategoricalCrossentropy(
        from_logits=adapter == "sum_logits_softmax", label_smoothing=smoothing
    )
    count = model.outputs[0].shape[-1]
    losses = []

    def update(images, labels, branch=None, grid=None, factor=1.0):
        if grid is not None:
            from mediflow_datasets.web_skin_paper_suite import _jigsaw

            images = _jigsaw(tf.convert_to_tensor(images), grid)
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            loss = factor * loss_fn(labels, outputs[branch] if branch is not None else outputs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            [
                (g, v)
                for g, v in zip(gradients, model.trainable_variables, strict=True)
                if g is not None
            ]
        )
        return float(loss)

    for start in range(0, len(order), config["batch_size"]):
        batch = [rows[i] for i in order[start : start + config["batch_size"]]]
        images = []
        for row in batch:
            image = image_tensor(reader(row), size, spec.get("input_policy", "legacy_resize"))
            seed = item_seed(row, config["seed"], epoch)
            if augmentation_applies(seed, config["augmentation_probability"]):
                image = transform_image(image, profile, arm, seed)
            images.append(image)
        images = np.stack(images)
        labels = tf.one_hot([row.label for row in batch], count)
        if adapter == "sum_logits_softmax":
            loss = sum(
                update(images, labels, branch, grid) for branch, grid in enumerate((8, 4, 2))
            )
            loss += update(images, labels, branch=3, factor=2.0)
        else:
            loss = update(images, labels)
        if not np.isfinite(loss):
            raise ValueError("Non-finite training loss")
        losses.append(loss)
    return float(np.mean(losses))


def select_candidate(baseline, scores, config):
    eligible = []
    for arm, result in scores.items():
        clean, shifted = result["clean"], result["shifted"]
        if (
            clean["macro_f1"]
            >= baseline["clean"]["macro_f1"] - config["validation_clean_macro_f1_tolerance"]
            and min(np.asarray(clean["recall"]) - baseline["clean"]["recall"])
            >= -config["validation_clean_recall_tolerance"]
            and shifted["macro_f1"]
            >= baseline["shifted"]["macro_f1"] + config["validation_min_shift_gain"]
        ):
            eligible.append((shifted["macro_f1"], arm))
    return max(eligible)[1] if eligible else "baseline"


def training_run(context):
    """Run all validation-controlled arms without reading source Test pixels."""
    import keras

    archive_path, source = context["source_zip"], context["source"]
    directory, config = context["output"], context["config"]
    spec, profile = context["spec"], context["profile"]
    validation = source["original", "val"]
    records = {}
    arms, aliases = arm_plan(profile)
    training_rows = source[config.get("train_variant", "augmented"), "train"]
    with zipfile.ZipFile(archive_path) as archive:
        reader = lambda row: archive.read(row.path)  # noqa: E731
        model = keras.models.load_model(context["model"], compile=False)
        baseline = {}
        for key, change in (("clean", None), ("shifted", profile)):
            scores, _ = infer(
                model,
                validation,
                reader,
                spec["input_size"],
                spec["output_adapter"],
                config["batch_size"],
                change,
                input_policy=spec.get("input_policy", "legacy_resize"),
            )
            baseline[key] = classification(validation, scores, len(context["classes"]))
        del model
        for arm in arms:
            keras.backend.clear_session()
            keras.utils.set_random_seed(config["seed"])
            model = keras.models.load_model(context["model"], compile=False)
            configure_partial(model)
            optimizer = keras.optimizers.Adam(config["learning_rate"])
            optimizer.build(model.trainable_variables)
            arm_dir = directory / arm
            arm_dir.mkdir()
            best = None
            history = []
            for epoch in range(config["epochs"]):
                loss = train_epoch(
                    model,
                    training_rows,
                    reader,
                    spec,
                    config,
                    profile,
                    arm,
                    epoch,
                    optimizer,
                )
                current = {}
                for key, change in (("clean", None), ("shifted", profile)):
                    scores, _ = infer(
                        model,
                        validation,
                        reader,
                        spec["input_size"],
                        spec["output_adapter"],
                        config["batch_size"],
                        change,
                        input_policy=spec.get("input_policy", "legacy_resize"),
                    )
                    current[key] = classification(validation, scores, len(context["classes"]))
                history.append({"epoch": epoch + 1, "loss": loss, **current})
                score = min(current["clean"]["macro_f1"], current["shifted"]["macro_f1"])
                if best is None or score > best[0]:
                    best = (score, current)
                    model.save(arm_dir / "best.keras")
                write_progress(arm_dir / "history.json", history)
                print(
                    arm,
                    epoch + 1,
                    loss,
                    current["clean"]["macro_f1"],
                    current["shifted"]["macro_f1"],
                    flush=True,
                )
            records[arm] = {
                **best[1],
                "model": str((arm_dir / "best.keras").resolve()),
                "model_sha256": sha256(arm_dir / "best.keras"),
            }
            del model
    selected = select_candidate(baseline, records, config)
    result = {
        "baseline": baseline,
        "arms": records,
        "equivalent_arms": aliases,
        "randomness_policy": "independent_apply_stream_v2",
        "selected": selected,
        "source_test_evaluated": False,
        "target_accuracy": None,
        "domain_adaptation": {"status": "deferred", "reason": context["da_reason"]},
        "trainable_policy": "last 30 backbone layers; BatchNorm frozen",
        "source_train_rows": [asdict(r) for r in training_rows],
    }
    write_progress(directory / "validation_selection.json", result)
    return result
