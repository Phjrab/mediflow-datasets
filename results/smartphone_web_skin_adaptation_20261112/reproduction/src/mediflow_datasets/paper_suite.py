"""Validation-only, multi-seed paper experiment suite for Hair."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import uuid
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from mediflow_datasets import common_engine as engine
from mediflow_datasets import common_workflow as flow

PROTOCOL = "hair_paper_suite_v1"
DEFAULT_EXPERIMENTS = [
    "baseline_b1_256",
    "supcon_b1_256",
    "dinov2_small_224",
    "efficientnetv2s_256",
    "multires_b1_384",
]
EXPERIMENTS = {
    "baseline_b1_256": {
        "method": "standard",
        "backbone": "B1",
        "size": 256,
        "loss": "ls005",
        "epochs1": 15,
        "epochs2": 15,
        "paper": "Current MediFlow validation baseline",
        "hypothesis": "Reference condition for measuring gains beyond seed variation",
    },
    "supcon_b1_256": {
        "method": "supcon",
        "backbone": "B1",
        "size": 256,
        "loss": "ls005",
        "contrastive_epochs": 15,
        "classifier_epochs": 15,
        "paper": "Khosla et al., NeurIPS 2020",
        "hypothesis": "Supervised contrastive features separate visually similar scalp classes",
    },
    "dinov2_small_224": {
        "method": "dinov2",
        "preset": "dinov2_small",
        "size": 224,
        "loss": "ls005",
        "epochs1": 15,
        "epochs2": 0,
        "paper": "Oquab et al., TMLR 2024",
        "hypothesis": (
            "Self-supervised foundation features transfer better than ImageNet CNN features"
        ),
    },
    "efficientnetv2s_256": {
        "method": "standard",
        "backbone": "V2S",
        "size": 256,
        "loss": "ls005",
        "epochs1": 15,
        "epochs2": 15,
        "paper": "Tan and Le, ICML 2021",
        "hypothesis": "A newer efficient CNN backbone improves discriminative features",
    },
    "multires_b1_384": {
        "method": "standard",
        "backbone": "B1",
        "size": 384,
        "loss": "ls005",
        "epochs1": 15,
        "epochs2": 15,
        "paper": "Gessert et al., MethodsX 2020",
        "hypothesis": "Higher resolution preserves subtle scalp texture cues",
    },
}


def validate_suite(context):
    config = context["config"]
    selected = config.get("experiments", DEFAULT_EXPERIMENTS)
    if config["domain"] != "hair" or config["mode"] not in (
        "paper_suite",
        "paper_screen",
    ):
        raise ValueError("이 suite는 Hair 논문 실험 전용입니다.")
    if config["train_variant"] != "augmented":
        raise ValueError("Hair 현재 후보와 같은 augmented Train을 사용해야 합니다.")
    expected_seeds = [42, 43, 44] if config["mode"] == "paper_suite" else [42]
    if config["seeds"] != expected_seeds:
        raise ValueError(f"{config['mode']}의 SEEDS는 {expected_seeds}여야 합니다.")
    if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
        raise ValueError("RUN_EXPERIMENTS는 중복 없는 실험 이름 목록이어야 합니다.")
    unknown = sorted(set(selected) - set(EXPERIMENTS))
    if unknown:
        raise ValueError("지원하지 않는 실험입니다: " + ", ".join(unknown))
    return selected


def _trial_signature(context, spec):
    payload = json.dumps(spec, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256((context["signature"] + payload).encode()).hexdigest()


def _trial_spec(context, experiment_id, seed):
    condition = EXPERIMENTS[experiment_id]
    return {
        **condition,
        "id": f"{experiment_id}_seed_{seed}",
        "experiment_id": experiment_id,
        "variant": context["config"]["train_variant"],
        "class_count": len(context["classes"]),
        "class_names": context["classes"],
        "seed": seed,
        "protocol": PROTOCOL,
    }


def _run_standard(context, spec):
    condition = EXPERIMENTS[spec["experiment_id"]]
    return engine.run_trial(
        spec,
        flow.factory(context, spec["variant"], seed=spec["seed"]),
        context["output"],
        _trial_signature(context, spec),
        spec["seed"],
        condition["epochs1"],
        condition["epochs2"],
    )


def supervised_contrastive_loss(labels, projections, temperature=0.1):
    """Khosla et al. loss; two views ensure every anchor has a positive."""
    labels = tf.argmax(labels, axis=1, output_type=tf.int32)
    projections = tf.math.l2_normalize(projections, axis=1)
    logits = tf.matmul(projections, projections, transpose_b=True) / temperature
    count = tf.shape(logits)[0]
    self_mask = tf.eye(count, dtype=tf.bool)
    logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
    exp_logits = tf.exp(logits) * tf.cast(~self_mask, logits.dtype)
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-12)
    positive = tf.equal(labels[:, None], labels[None, :]) & ~self_mask
    positive_count = tf.reduce_sum(tf.cast(positive, logits.dtype), axis=1)
    mean_log_prob = tf.reduce_sum(
        tf.cast(positive, logits.dtype) * log_prob, axis=1
    ) / tf.maximum(positive_count, 1.0)
    return -tf.reduce_mean(mean_log_prob)


class SupConTrainer(keras.Model):
    def __init__(self, encoder_projector, augmenter):
        super().__init__()
        self.encoder_projector = encoder_projector
        self.augmenter = augmenter
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        images, labels = data
        first = self.augmenter(images, training=True)
        second = self.augmenter(images, training=True)
        joined_images = tf.concat([first, second], axis=0)
        joined_labels = tf.concat([labels, labels], axis=0)
        with tf.GradientTape() as tape:
            projections = self.encoder_projector(joined_images, training=True)
            loss = supervised_contrastive_loss(joined_labels, projections)
            if self.encoder_projector.losses:
                loss += tf.add_n(self.encoder_projector.losses)
        variables = self.encoder_projector.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables, strict=True))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def _partial_backbone(backbone, count=30):
    backbone.trainable = True
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= len(backbone.layers) - count and not isinstance(
            layer, keras.layers.BatchNormalization
        )


def _run_supcon(context, spec):
    signature = _trial_signature(context, spec)
    cached = engine.cached_record(context["output"], spec["id"], signature)
    if cached:
        return cached
    keras.backend.clear_session()
    keras.utils.set_random_seed(spec["seed"])
    directory = Path(context["output"]) / spec["id"] / ("attempt_" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    engine.write_json(directory / "spec.json", spec)
    train, _ = flow.factory(context, spec["variant"], seed=spec["seed"])(
        "train", spec["size"], True
    )
    val, paths = flow.factory(context, spec["variant"], seed=spec["seed"])(
        "val", spec["size"], False
    )
    started = time.monotonic()
    backbone = keras.applications.EfficientNetB1(
        include_top=False,
        weights="imagenet",
        input_shape=(spec["size"], spec["size"], 3),
    )
    _partial_backbone(backbone)
    inputs = keras.Input((spec["size"], spec["size"], 3))
    features = backbone(inputs, training=False)
    pooled = keras.layers.GlobalAveragePooling2D(name="supcon_features")(features)
    feature_encoder = keras.Model(inputs, pooled, name="supcon_feature_encoder")
    projected = keras.layers.Dense(256, activation="relu")(pooled)
    projected = keras.layers.Dense(128)(projected)
    encoder_projector = keras.Model(inputs, projected, name="supcon_encoder_projector")
    augmenter = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=spec["seed"]),
            keras.layers.RandomRotation(0.03, seed=spec["seed"] + 1),
            keras.layers.RandomZoom(0.05, seed=spec["seed"] + 2),
            keras.layers.RandomContrast(0.1, seed=spec["seed"] + 3),
        ],
        name="weak_medical_views",
    )
    trainer = SupConTrainer(encoder_projector, augmenter)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-5))
    contrastive_history = trainer.fit(
        train,
        epochs=spec["contrastive_epochs"],
        callbacks=[
            keras.callbacks.CSVLogger(str(directory / "contrastive_log.csv")),
            engine.HistoryBackup(directory / "contrastive_history.json"),
            keras.callbacks.TerminateOnNaN(),
        ],
        verbose=2,
    ).history
    contrastive_history = {
        key: [float(value) for value in values]
        for key, values in contrastive_history.items()
    }
    if len(contrastive_history.get("loss", [])) != spec["contrastive_epochs"]:
        raise RuntimeError("SupCon 사전학습이 완료되지 않았습니다.")
    classifier_inputs = keras.Input((spec["size"], spec["size"], 3))
    classifier_features = feature_encoder(classifier_inputs, training=False)
    classifier_features = keras.layers.Dropout(0.3)(classifier_features)
    classifier_outputs = keras.layers.Dense(
        spec["class_count"], activation="softmax"
    )(classifier_features)
    classifier = keras.Model(classifier_inputs, classifier_outputs)
    classifier.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=engine.loss_function(spec["loss"]),
        metrics=["accuracy"],
    )
    history, best = engine.fit_stage(
        classifier,
        train,
        val,
        directory,
        "classifier",
        spec["classifier_epochs"],
    )
    del classifier, trainer, encoder_projector
    keras.backend.clear_session()
    model = keras.models.load_model(best, compile=False)
    metrics = engine.evaluate_to_files(model, val, paths, directory, "validation")
    record = {
        "id": spec["id"],
        "spec": spec,
        "signature": signature,
        "attempt": directory.name,
        "selected_model": best.name,
        "selected_stage": "supcon_then_classifier",
        "validation": metrics,
        "training_seconds": time.monotonic() - started,
        "parameters": model.count_params(),
        "model_bytes": best.stat().st_size,
        "trainable_backbone_layers": [layer.name for layer in backbone.layers if layer.trainable],
        "history": history,
        "contrastive_history": contrastive_history,
        "stage_boundary": 0,
        "test_evaluated": False,
    }
    engine.finish_record(directory, record)
    return record


def _build_dinov2_classifier(spec):
    import keras_hub

    converter = keras_hub.layers.DINOV2ImageConverter.from_preset(
        spec["preset"],
        image_size=(spec["size"], spec["size"]),
        crop_to_aspect_ratio=False,
    )
    backbone = keras_hub.models.DINOV2Backbone.from_preset(
        spec["preset"], image_shape=(spec["size"], spec["size"], 3)
    )
    backbone.trainable = False
    feature_model = keras.Model(
        inputs=backbone.inputs,
        outputs=list(backbone.pyramid_outputs.values())[-1],
        name="dinov2_final_features",
    )
    inputs = keras.Input((spec["size"], spec["size"], 3), name="raw_rgb_0_255")
    converted = converter(inputs)
    features = feature_model({"images": converted}, training=False)
    # DINOv2 emits [batch, tokens, channels], not a CNN feature map. Following
    # the official linear-classifier representation, combine the CLS token
    # with the mean of the patch tokens.
    token_count = int(features.shape[1])
    class_token = keras.layers.Cropping1D(
        cropping=(0, token_count - 1), name="keep_cls_token"
    )(features)
    class_token = keras.layers.Flatten(name="flatten_cls_token")(class_token)
    patch_tokens = keras.layers.Cropping1D(
        cropping=(1, 0), name="remove_cls_token"
    )(features)
    patch_mean = keras.layers.GlobalAveragePooling1D(name="mean_patch_tokens")(
        patch_tokens
    )
    features = keras.layers.Concatenate(name="cls_and_mean_patch")(
        [class_token, patch_mean]
    )
    features = keras.layers.Dropout(0.3)(features)
    outputs = keras.layers.Dense(spec["class_count"], activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=engine.loss_function(spec["loss"]),
        metrics=["accuracy"],
        # Keras 3 may enable XLA automatically on GPU. DINOV2ImageConverter
        # uses ResizeBicubic, which the Colab XLA_GPU_JIT path does not support.
        jit_compile=False,
    )
    return model


def _run_dinov2(context, spec):
    signature = _trial_signature(context, spec)
    cached = engine.cached_record(context["output"], spec["id"], signature)
    if cached:
        return cached
    keras.backend.clear_session()
    keras.utils.set_random_seed(spec["seed"])
    directory = Path(context["output"]) / spec["id"] / ("attempt_" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    engine.write_json(directory / "spec.json", spec)
    train, _ = flow.factory(context, spec["variant"], seed=spec["seed"])(
        "train", spec["size"], True
    )
    val, paths = flow.factory(context, spec["variant"], seed=spec["seed"])(
        "val", spec["size"], False
    )
    started = time.monotonic()
    model = _build_dinov2_classifier(spec)
    history, best = engine.fit_stage(
        model, train, val, directory, "linear_probe", spec["epochs1"]
    )
    del model
    keras.backend.clear_session()
    import keras_hub  # noqa: F401

    model = keras.models.load_model(best, compile=False)
    metrics = engine.evaluate_to_files(model, val, paths, directory, "validation")
    record = {
        "id": spec["id"],
        "spec": spec,
        "signature": signature,
        "attempt": directory.name,
        "selected_model": best.name,
        "selected_stage": "frozen_linear_probe",
        "validation": metrics,
        "training_seconds": time.monotonic() - started,
        "parameters": model.count_params(),
        "model_bytes": best.stat().st_size,
        "trainable_backbone_layers": [],
        "history": history,
        "stage_boundary": len(history["accuracy"]),
        "test_evaluated": False,
    }
    engine.finish_record(directory, record)
    return record


def run(context):
    selected = validate_suite(context)
    records = []
    handlers = {
        "standard": _run_standard,
        "supcon": _run_supcon,
        "dinov2": _run_dinov2,
    }
    try:
        for experiment_id in selected:
            for seed in context["config"]["seeds"]:
                spec = _trial_spec(context, experiment_id, seed)
                record = handlers[spec["method"]](context, spec)
                records.append(record)
                engine.write_json(
                    context["output"] / "paper_suite_progress.json",
                    {"completed": [item["id"] for item in records]},
                )
        engine.write_json(context["output"] / "all_validation_results.json", records)
        return records
    except Exception as exc:
        engine.write_json(
            context["output"] / ("paper_suite_failure_" + uuid.uuid4().hex[:8] + ".json"),
            {"error": repr(exc), "completed": [item["id"] for item in records]},
        )
        raise


def _mean_std(values):
    values = np.asarray(values, dtype="float64")
    return {"mean": float(values.mean()), "std_population": float(values.std(ddof=0))}


def summarize(context, records):
    import matplotlib.pyplot as plt

    selected = validate_suite(context)
    grouped = {experiment_id: [] for experiment_id in selected}
    for record in records:
        grouped[record["spec"]["experiment_id"]].append(record)
    summary_rows, class_rows = [], []
    summary = {
        "protocol": PROTOCOL,
        "domain": "hair",
        "data_sha256": context["data_hash"],
        "class_names": context["classes"],
        "seeds": context["config"]["seeds"],
        "selection_metric": (
            "validation_macro_f1_3seed_mean"
            if len(context["config"]["seeds"]) == 3
            else "validation_macro_f1_single_seed_screen"
        ),
        "test_evaluated": False,
        "experiments": {},
    }
    for experiment_id in selected:
        items = sorted(grouped[experiment_id], key=lambda item: item["spec"]["seed"])
        if [item["spec"]["seed"] for item in items] != context["config"]["seeds"]:
            raise ValueError(experiment_id + "의 설정된 seed 결과가 완전하지 않습니다.")
        accuracy = [item["validation"]["accuracy"] for item in items]
        macro_f1 = [item["validation"]["macro_f1"] for item in items]
        class_f1 = np.asarray([item["validation"]["class_f1"] for item in items])
        details = {
            "paper": EXPERIMENTS[experiment_id]["paper"],
            "hypothesis": EXPERIMENTS[experiment_id]["hypothesis"],
            "validation_accuracy": _mean_std(accuracy),
            "validation_macro_f1": _mean_std(macro_f1),
            "class_f1": {
                name: _mean_std(class_f1[:, index])
                for index, name in enumerate(context["classes"])
            },
            "training_seconds": _mean_std(
                [item["training_seconds"] for item in items]
            ),
        }
        summary["experiments"][experiment_id] = details
        summary_rows.append(
            {
                "experiment": experiment_id,
                "validation_accuracy_mean": details["validation_accuracy"]["mean"],
                "validation_accuracy_std": details["validation_accuracy"]["std_population"],
                "validation_macro_f1_mean": details["validation_macro_f1"]["mean"],
                "validation_macro_f1_std": details["validation_macro_f1"]["std_population"],
                "training_seconds_mean": details["training_seconds"]["mean"],
            }
        )
        for class_name, values in details["class_f1"].items():
            class_rows.append(
                {
                    "experiment": experiment_id,
                    "class_name": class_name,
                    "mean_f1": values["mean"],
                    "std_population": values["std_population"],
                }
            )
    leader = max(summary_rows, key=lambda row: row["validation_macro_f1_mean"])
    summary["provisional_validation_leader"] = leader["experiment"]
    output = Path(context["output"])
    engine.write_json(output / "paper_suite_summary.json", summary)
    for name, rows in (
        ("paper_suite_comparison.csv", summary_rows),
        ("paper_suite_class_f1.csv", class_rows),
    ):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    labels = [row["experiment"] for row in summary_rows]
    means = np.asarray([row["validation_macro_f1_mean"] for row in summary_rows])
    errors = np.asarray([row["validation_macro_f1_std"] for row in summary_rows])
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    x = np.arange(len(labels))
    axes[0].bar(x, means, yerr=errors, capsize=5, color="#2f6fed")
    axes[0].set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1),
        ylabel="Validation Macro F1",
        title=(
            "Hair paper experiments: 3-seed mean ± population std"
            if len(context["config"]["seeds"]) == 3
            else "Hair paper experiments: seed 42 screening"
        ),
    )
    axes[0].tick_params(axis="x", rotation=15)
    matrix = np.asarray(
        [
            [summary["experiments"][exp]["class_f1"][name]["mean"] for name in context["classes"]]
            for exp in labels
        ]
    )
    image = axes[1].imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axes[1].set(
        xticks=np.arange(len(context["classes"])),
        xticklabels=[f"C{i}" for i in range(len(context["classes"]))],
        yticks=np.arange(len(labels)),
        yticklabels=labels,
        title="Class F1 mean (C0–C4 follow class_names.json)",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1].text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[1], fraction=0.02)
    fig.tight_layout()
    fig.savefig(output / "paper_suite_dashboard.png", dpi=180)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(len(records), 2, figsize=(15, 4 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        history = record["history"]
        epochs = np.arange(1, len(history["accuracy"]) + 1)
        for column, metric in enumerate(("accuracy", "loss")):
            axes[row, column].plot(epochs, history[metric], label="Train")
            axes[row, column].plot(epochs, history["val_" + metric], label="Validation")
            boundary = record.get("stage_boundary", 0)
            if 0 < boundary < len(epochs):
                axes[row, column].axvline(boundary + 0.5, color="gray", linestyle="--")
            axes[row, column].set(title=record["id"] + " / " + metric, xlabel="Epoch")
            axes[row, column].grid(alpha=0.25)
            axes[row, column].legend()
    fig.tight_layout()
    fig.savefig(output / "paper_suite_training_curves.png", dpi=160)
    plt.show()
    plt.close(fig)
    archive = flow.archive_results(context)
    print("Validation 전용 논문 실험 완료:", output)
    print("잠정 Validation 선두:", summary["provisional_validation_leader"])
    print("Test는 평가하지 않았습니다.")
    return summary, archive
