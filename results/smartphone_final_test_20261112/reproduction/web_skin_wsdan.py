"""Validation-only WS-DAN-inspired screening for Web Skin.

The implementation adapts Hu et al. (2019) to EfficientNet-B0 with bilinear
attention pooling and attention-guided crop/drop augmentation.  It uses image
labels only and deliberately does not access Test during screening.
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
import tensorflow as tf

from mediflow_datasets import common_engine as engine
from mediflow_datasets import common_workflow as flow

PROTOCOL = "web_skin_wsdan_screen_v1"
TRIAL_ID = "wsdan_b0_256_ce_seed_42"
PAPER_URL = "https://arxiv.org/abs/1901.09891"
EXPECTED_CLASSES = ["건선", "아토피", "여드름", "정상", "주사"]
BASELINE = {
    "id": "b0_256_ce",
    "validation_accuracy": 0.796,
    "validation_macro_f1": 0.7915394647566528,
    "validation_count": 500,
    "class_f1": [
        0.7606837606837609,
        0.6847826086956522,
        0.6818181818181819,
        0.9345794392523363,
        0.8958333333333334,
    ],
    "confusion_matrix": [
        [89, 3, 3, 3, 2],
        [18, 63, 8, 8, 3],
        [22, 14, 60, 3, 1],
        [0, 0, 0, 100, 0],
        [5, 4, 5, 0, 86],
    ],
    "data_sha256": "f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d",
    "source": (
        "results/web_skin/experiments/suite_20260908_014452_72768a42/"
        "b0_256_ce/attempt_8977c45d04bc/validation_metrics.json"
    ),
}


def validate(context):
    config = context["config"]
    if config["domain"] != "web_skin" or config["mode"] not in (
        "web_skin_wsdan",
        "web_skin_paper_suite",
    ):
        raise ValueError("이 노트북은 Web Skin WS-DAN 선별 전용입니다.")
    if config["train_variant"] != "augmented":
        raise ValueError("기존 기준선과 같은 augmented Train을 사용해야 합니다.")
    if config["seed"] != 42 or config["seeds"] != [42]:
        raise ValueError("WS-DAN 선별 seed는 42로 고정합니다.")
    experiments = config.get("experiments", [])
    if config["mode"] == "web_skin_wsdan" and experiments != [TRIAL_ID]:
        raise ValueError("이번 실행은 WS-DAN 한 조건만 허용합니다.")
    if config["mode"] == "web_skin_paper_suite" and TRIAL_ID not in experiments:
        raise ValueError("통합 실험에 WS-DAN 조건이 없습니다.")
    if context["classes"] != EXPECTED_CLASSES:
        raise ValueError("Web Skin 클래스 순서가 기존 모델 계약과 다릅니다.")
    if context.get("data_hash") != BASELINE["data_sha256"]:
        raise ValueError("저장된 B0·256·CE 기준선과 데이터 SHA-256이 다릅니다.")
    maps = config.get("attention_maps")
    if not isinstance(maps, int) or maps <= 0:
        raise ValueError("ATTENTION_MAPS는 양의 정수여야 합니다.")
    for key in ("crop_threshold", "drop_threshold"):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ValueError(key + "는 0과 1 사이여야 합니다.")


@keras.saving.register_keras_serializable(package="MediFlow")
class BilinearAttentionPooling(keras.layers.Layer):
    """Pool one feature vector per learned attention map."""

    def call(self, inputs):
        features, attention = inputs
        pooled = keras.ops.einsum("bhwc,bhwm->bmc", features, attention)
        normalizer = keras.ops.sum(attention, axis=(1, 2))
        pooled = pooled / (keras.ops.expand_dims(normalizer, -1) + 1e-6)
        pooled = keras.ops.reshape(pooled, (keras.ops.shape(pooled)[0], -1))
        pooled = keras.ops.sign(pooled) * keras.ops.sqrt(keras.ops.abs(pooled) + 1e-8)
        norm = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(pooled), axis=-1, keepdims=True) + 1e-8
        )
        return pooled / norm

    def compute_output_shape(self, input_shape):
        feature_shape, attention_shape = input_shape
        return (feature_shape[0], feature_shape[-1] * attention_shape[-1])


def build_model(class_count, size=256, attention_maps=8, weights="imagenet"):
    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(size, size, 3)
    )
    backbone.trainable = False
    inputs = keras.Input((size, size, 3), name="image")
    features = backbone(inputs, training=False)
    attention = keras.layers.Conv2D(
        attention_maps, 1, activation="sigmoid", name="attention_maps"
    )(features)
    pooled = BilinearAttentionPooling(name="bilinear_attention_pooling")(
        [features, attention]
    )
    pooled = keras.layers.Dropout(0.3, name="attention_dropout")(pooled)
    outputs = keras.layers.Dense(class_count, activation="softmax", name="predictions")(
        pooled
    )
    model = keras.Model(inputs, outputs, name="web_skin_wsdan_b0")
    attention_probe = keras.Model(inputs, [outputs, attention], name="attention_probe")
    return model, attention_probe


def configure_partial(model):
    backbones = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower()
    ]
    if len(backbones) != 1:
        raise ValueError("EfficientNet Backbone 하나가 필요합니다.")
    backbone = backbones[0]
    backbone.trainable = True
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= len(backbone.layers) - 30 and not isinstance(
            layer, keras.layers.BatchNormalization
        )
    return [layer.name for layer in backbone.layers if layer.trainable]


def _one_attention_augmentation(item, size, crop_threshold, drop_threshold):
    image, maps = item
    score = tf.reduce_mean(maps, axis=-1, keepdims=True)
    score = tf.image.resize(score, (size, size), method="bilinear")
    score = score / (tf.reduce_max(score) + 1e-6)
    crop_mask = score[..., 0] >= crop_threshold
    coordinates = tf.cast(tf.where(crop_mask), tf.int32)

    def crop_region():
        top_left = tf.reduce_min(coordinates, axis=0)
        bottom_right = tf.reduce_max(coordinates, axis=0) + 1
        height = tf.maximum(bottom_right[0] - top_left[0], 1)
        width = tf.maximum(bottom_right[1] - top_left[1], 1)
        crop = tf.image.crop_to_bounding_box(
            image, top_left[0], top_left[1], height, width
        )
        return tf.image.resize(crop, (size, size), method="bilinear")

    crop = tf.cond(tf.shape(coordinates)[0] > 0, crop_region, lambda: image)
    keep = tf.cast(score < drop_threshold, image.dtype)
    dropped = image * keep
    choose_crop = tf.random.uniform(()) < 0.5
    return tf.cond(choose_crop, lambda: crop, lambda: dropped)


def attention_augment(images, maps, size, crop_threshold, drop_threshold):
    maps = tf.stop_gradient(maps)
    return tf.map_fn(
        lambda item: _one_attention_augmentation(
            item, size, crop_threshold, drop_threshold
        ),
        (images, maps),
        fn_output_signature=tf.TensorSpec((size, size, 3), images.dtype),
    )


def _one_attention_crop(item, size, crop_threshold):
    image, maps = item
    score = tf.reduce_mean(maps, axis=-1, keepdims=True)
    score = tf.image.resize(score, (size, size), method="bilinear")
    score = score / (tf.reduce_max(score) + 1e-6)
    coordinates = tf.cast(tf.where(score[..., 0] >= crop_threshold), tf.int32)

    def crop_region():
        top_left = tf.reduce_min(coordinates, axis=0)
        bottom_right = tf.reduce_max(coordinates, axis=0) + 1
        crop = tf.image.crop_to_bounding_box(
            image,
            top_left[0],
            top_left[1],
            tf.maximum(bottom_right[0] - top_left[0], 1),
            tf.maximum(bottom_right[1] - top_left[1], 1),
        )
        return tf.image.resize(crop, (size, size), method="bilinear")

    return tf.cond(tf.shape(coordinates)[0] > 0, crop_region, lambda: image)


def attention_crop(images, maps, size, crop_threshold):
    return tf.map_fn(
        lambda item: _one_attention_crop(item, size, crop_threshold),
        (images, maps),
        fn_output_signature=tf.TensorSpec((size, size, 3), images.dtype),
    )


class AttentionEnsemblePredictor:
    def __init__(self, model, size, crop_threshold):
        self.model = model
        self.size = size
        self.crop_threshold = crop_threshold
        self.probe = keras.Model(
            model.input,
            [model.output, model.get_layer("attention_maps").output],
        )

    def __call__(self, images, training=False):
        raw, maps = self.probe(images, training=False)
        cropped_images = attention_crop(
            images, maps, self.size, self.crop_threshold
        )
        cropped = self.model(cropped_images, training=False)
        return (raw + cropped) / 2.0


def _fit_stage(
    model,
    attention_probe,
    train,
    val,
    directory,
    name,
    epochs,
    learning_rate,
    size,
    crop_threshold,
    drop_threshold,
):
    optimizer = keras.optimizers.Adam(learning_rate)
    loss_function = keras.losses.CategoricalCrossentropy()
    history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}
    best_score = -1.0
    best = directory / f"{name}_best.keras"
    log_path = directory / f"{name}_log.csv"

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            raw_predictions, maps = attention_probe(images, training=True)
            augmented = attention_augment(
                images, maps, size, crop_threshold, drop_threshold
            )
            augmented_predictions = model(augmented, training=True)
            raw_loss = loss_function(labels, raw_predictions)
            augmented_loss = loss_function(labels, augmented_predictions)
            loss = 0.5 * (raw_loss + augmented_loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients, model.trainable_variables, strict=True
            )
            if gradient is not None
        ]
        optimizer.apply_gradients(pairs)
        return loss, raw_predictions

    @tf.function
    def validation_step(images, labels):
        raw_predictions, maps = attention_probe(images, training=False)
        cropped_images = attention_crop(images, maps, size, crop_threshold)
        cropped_predictions = model(cropped_images, training=False)
        predictions = (raw_predictions + cropped_predictions) / 2.0
        return loss_function(labels, predictions), predictions

    with log_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["epoch", "accuracy", "loss", "val_accuracy", "val_loss"]
        )
        writer.writeheader()
        for epoch in range(epochs):
            train_loss = keras.metrics.Mean()
            train_accuracy = keras.metrics.CategoricalAccuracy()
            val_loss = keras.metrics.Mean()
            val_accuracy = keras.metrics.CategoricalAccuracy()
            for images, labels in train:
                loss, predictions = train_step(images, labels)
                train_loss.update_state(loss)
                train_accuracy.update_state(labels, predictions)
            for images, labels in val:
                loss, predictions = validation_step(images, labels)
                val_loss.update_state(loss)
                val_accuracy.update_state(labels, predictions)
            values = {
                "accuracy": float(train_accuracy.result()),
                "loss": float(train_loss.result()),
                "val_accuracy": float(val_accuracy.result()),
                "val_loss": float(val_loss.result()),
            }
            for key, value in values.items():
                history[key].append(value)
            writer.writerow({"epoch": epoch + 1, **values})
            stream.flush()
            engine.write_json(directory / f"{name}_history.json", history)
            print(
                f"{name} Epoch {epoch + 1}/{epochs} - loss: {values['loss']:.4f} "
                f"- accuracy: {values['accuracy']:.4f} - val_loss: "
                f"{values['val_loss']:.4f} - val_accuracy: {values['val_accuracy']:.4f}"
            )
            if values["val_accuracy"] > best_score:
                best_score = values["val_accuracy"]
                model.save(best)
    if not best.is_file() or len(history["val_accuracy"]) != epochs:
        raise RuntimeError("WS-DAN 학습이 완전하게 끝나지 않았습니다.")
    model.save(directory / f"{name}_last.keras")
    return history, best


def run(context):
    validate(context)
    config = context["config"]
    spec = {
        "id": TRIAL_ID,
        "method": "wsdan_inspired_attention_crop_drop",
        "paper": PAPER_URL,
        "backbone": "B0",
        "size": 256,
        "loss": "ce",
        "variant": "augmented",
        "epochs1": config["epochs1"],
        "epochs2": config["epochs2"],
        "attention_maps": config["attention_maps"],
        "crop_threshold": float(config["crop_threshold"]),
        "drop_threshold": float(config["drop_threshold"]),
        "seed": 42,
        "class_count": len(context["classes"]),
        "class_names": context["classes"],
        "protocol": PROTOCOL,
        "test_evaluated": False,
    }
    signature = hashlib.sha256(
        (context["signature"] + json.dumps(spec, sort_keys=True)).encode()
    ).hexdigest()
    cached = engine.cached_record(context["output"], TRIAL_ID, signature)
    if cached:
        return cached
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    directory = Path(context["output"]) / TRIAL_ID / ("attempt_" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    engine.write_json(directory / "spec.json", spec)
    engine.write_json(Path(context["output"]) / "baseline_reference.json", BASELINE)
    train, _ = flow.factory(context, "augmented", seed=42)("train", 256, True)
    val, paths = flow.factory(context, "augmented", seed=42)("val", 256, False)
    started = time.monotonic()
    model, probe = build_model(
        len(context["classes"]), 256, config["attention_maps"]
    )
    h1, best1 = _fit_stage(
        model,
        probe,
        train,
        val,
        directory,
        "stage1",
        config["epochs1"],
        1e-4,
        256,
        config["crop_threshold"],
        config["drop_threshold"],
    )
    del model, probe
    keras.backend.clear_session()
    model = keras.models.load_model(best1, compile=False)
    probe = keras.Model(
        model.input,
        [model.output, model.get_layer("attention_maps").output],
        name="attention_probe",
    )
    trainable = configure_partial(model)
    h2, best2 = _fit_stage(
        model,
        probe,
        train,
        val,
        directory,
        "stage2",
        config["epochs2"],
        1e-5,
        256,
        config["crop_threshold"],
        config["drop_threshold"],
    )
    selected_stage = (
        "stage2"
        if engine.checkpoint_choice(max(h1["val_accuracy"]), max(h2["val_accuracy"]))
        else "stage1"
    )
    selected = best2 if selected_stage == "stage2" else best1
    del model, probe
    keras.backend.clear_session()
    model = keras.models.load_model(selected, compile=False)
    predictor = AttentionEnsemblePredictor(
        model, 256, float(config["crop_threshold"])
    )
    metrics = engine.evaluate_to_files(
        predictor, val, paths, directory, "validation"
    )
    record = {
        "id": TRIAL_ID,
        "spec": spec,
        "signature": signature,
        "attempt": directory.name,
        "selected_model": selected.name,
        "selected_stage": selected_stage,
        "validation": metrics,
        "stage1_best_val": max(h1["val_accuracy"]),
        "stage2_best_val": max(h2["val_accuracy"]),
        "training_seconds": time.monotonic() - started,
        "parameters": model.count_params(),
        "model_bytes": selected.stat().st_size,
        "trainable_backbone_layers": trainable,
        "history": {key: h1[key] + h2[key] for key in h1},
        "stage_boundary": len(h1["accuracy"]),
        "test_evaluated": False,
        "inference": "mean of original and deterministic attention-crop predictions",
    }
    engine.finish_record(directory, record)
    return record


def _attention_figure(model, dataset, directory, classes):
    import matplotlib.pyplot as plt

    probe = keras.Model(
        model.input,
        [model.output, model.get_layer("attention_maps").output],
    )
    images, labels = next(iter(dataset))
    predictions, maps = probe(images[:12], training=False)
    heatmaps = tf.reduce_mean(maps, axis=-1, keepdims=True)
    heatmaps = tf.image.resize(heatmaps, (256, 256)).numpy()
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for index, axis in enumerate(axes.flat):
        image = np.clip(images[index].numpy() / 255.0, 0, 1)
        heat = heatmaps[index, ..., 0]
        heat = heat / (heat.max() + 1e-6)
        axis.imshow(image)
        axis.imshow(heat, cmap="jet", alpha=0.38, vmin=0, vmax=1)
        true_name = classes[int(tf.argmax(labels[index]))]
        pred_name = classes[int(tf.argmax(predictions[index]))]
        axis.set_title(f"정답: {true_name}\n예측: {pred_name}", fontsize=9)
        axis.axis("off")
    fig.suptitle("Web Skin WS-DAN Attention Map", fontsize=15)
    fig.tight_layout()
    fig.savefig(directory / "wsdan_attention_examples.png", dpi=180)
    plt.show()
    plt.close(fig)


def _validation_figures(record, output, classes):
    import matplotlib.pyplot as plt

    wsdan = record["validation"]
    methods = ["Saved B0 256 CE", "WS-DAN B0 256 CE"]
    accuracy = [BASELINE["validation_accuracy"], wsdan["accuracy"]]
    macro_f1 = [BASELINE["validation_macro_f1"], wsdan["macro_f1"]]
    positions = np.arange(len(methods))
    width = 0.34
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    axes[0].bar(positions - width / 2, accuracy, width, label="Accuracy")
    axes[0].bar(positions + width / 2, macro_f1, width, label="Macro F1")
    for index, value in enumerate(accuracy):
        axes[0].text(index - width / 2, value + 0.004, f"{value:.4f}", ha="center")
    for index, value in enumerate(macro_f1):
        axes[0].text(index + width / 2, value + 0.004, f"{value:.4f}", ha="center")
    axes[0].set_xticks(positions, methods)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Validation score")
    axes[0].set_title("Web Skin Validation Performance")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    class_positions = np.arange(len(classes))
    axes[1].bar(
        class_positions - width / 2,
        BASELINE["class_f1"],
        width,
        label="Saved B0 256 CE",
    )
    axes[1].bar(
        class_positions + width / 2,
        wsdan["class_f1"],
        width,
        label="WS-DAN B0 256 CE",
    )
    axes[1].set_xticks(class_positions, [f"C{i}" for i in class_positions])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Validation class F1")
    axes[1].set_title("Class F1 (C0-C4 follow class_mapping.json)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "wsdan_validation_dashboard.png", dpi=180)
    plt.show()
    plt.close(fig)

    matrix = np.asarray(wsdan["confusion_matrix"], dtype=int)
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    labels = [f"C{i}" for i in range(len(classes))]
    axis.set_xticks(range(len(classes)), labels)
    axis.set_yticks(range(len(classes)), labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("WS-DAN Validation Confusion Matrix")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output / "wsdan_validation_confusion_matrix.png", dpi=180)
    plt.show()
    plt.close(fig)


def summarize(context, record):
    import matplotlib.pyplot as plt

    validate(context)
    output = Path(context["output"])
    attempt = output / record["id"] / record["attempt"]
    model = keras.models.load_model(attempt / record["selected_model"], compile=False)
    val, _ = flow.factory(context, "augmented", seed=42)("val", 256, False)
    _attention_figure(model, val, output, context["classes"])
    _validation_figures(record, output, context["classes"])
    rows = [
        {
            "method": "saved_b0_256_ce_baseline",
            "validation_accuracy": BASELINE["validation_accuracy"],
            "validation_macro_f1": BASELINE["validation_macro_f1"],
            "trained_in_this_run": False,
        },
        {
            "method": "wsdan_b0_256_ce",
            "validation_accuracy": record["validation"]["accuracy"],
            "validation_macro_f1": record["validation"]["macro_f1"],
            "trained_in_this_run": True,
        },
    ]
    with (output / "wsdan_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": PROTOCOL,
        "paper": PAPER_URL,
        "implementation": (
            "EfficientNet-B0 adaptation with bilinear attention pooling and "
            "attention-guided crop/drop; not a byte-identical reproduction"
        ),
        "fixed_conditions": {
            "data_sha256": context["data_hash"],
            "split": "same original Validation 500 images",
            "train_variant": "augmented",
            "backbone": "EfficientNet-B0",
            "input_size": 256,
            "loss": "categorical_crossentropy",
            "seed": 42,
            "stage1_epochs": context["config"]["epochs1"],
            "stage2_epochs": context["config"]["epochs2"],
            "test_access": False,
        },
        "changed_condition": (
            "global average pooling and random stored augmentation versus WS-DAN-inspired "
            "bilinear attention pooling plus online attention crop/drop"
        ),
        "baseline_retrained": False,
        "baseline": BASELINE,
        "wsdan": record["validation"],
        "delta_wsdan_minus_baseline": {
            "validation_accuracy": record["validation"]["accuracy"]
            - BASELINE["validation_accuracy"],
            "validation_macro_f1": record["validation"]["macro_f1"]
            - BASELINE["validation_macro_f1"],
        },
        "screening_only": True,
        "test_evaluated": False,
    }
    engine.write_json(output / "wsdan_comparison_summary.json", summary)
    history = record["history"]
    epochs = np.arange(1, len(history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(epochs, history["accuracy"], label="Train")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].axvline(record["stage_boundary"] + 0.5, color="gray", linestyle="--")
    axes[0].set(title="WS-DAN Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].plot(epochs, history["loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].axvline(record["stage_boundary"] + 0.5, color="gray", linestyle="--")
    axes[1].set(title="WS-DAN Loss", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "wsdan_training_curves.png", dpi=180)
    plt.show()
    plt.close(fig)
    archive = flow.archive_results(context)
    print("Web Skin WS-DAN Validation 선별 완료:", output)
    print("기존 B0·256·CE는 다시 학습하지 않았고 Test도 평가하지 않았습니다.")
    return summary, archive
