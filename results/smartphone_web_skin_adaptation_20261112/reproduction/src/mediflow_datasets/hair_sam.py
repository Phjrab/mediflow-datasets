"""Validation-only SAM screening from the exact B1/384 stage-1 checkpoint."""

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

PROTOCOL = "hair_sam_screen_v1"
TRIAL_ID = "sam_b1_384_seed_42"


def validate(context):
    config = context["config"]
    if config["domain"] != "hair" or config["mode"] != "sam_screen":
        raise ValueError("이 노트북은 Hair B1·384 SAM 선별 전용입니다.")
    if config["train_variant"] != "augmented":
        raise ValueError("기존 Hair 후보와 같은 augmented Train을 사용해야 합니다.")
    if config["seed"] != 42 or config["seeds"] != [42]:
        raise ValueError("SAM 선별 seed는 42로 고정합니다.")
    if config.get("experiments") != ["sam_b1_384"]:
        raise ValueError("이번 실행은 sam_b1_384 한 조건만 허용합니다.")
    if not isinstance(config.get("sam_rho"), (int, float)) or config["sam_rho"] <= 0:
        raise ValueError("SAM_RHO는 양수여야 합니다.")
    if not isinstance(config.get("parent_run_dir"), str) or not config[
        "parent_run_dir"
    ].strip():
        raise ValueError("PARENT_RUN_DIR을 입력하세요.")
    parent_hash = config.get("parent_stage1_sha256")
    if (
        not isinstance(parent_hash, str)
        or len(parent_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in parent_hash)
    ):
        raise ValueError("PARENT_STAGE1_SHA256은 64자리 SHA-256이어야 합니다.")


def _parent(context):
    config = context["config"]
    parent = Path(config["parent_run_dir"])
    marker = parent / "multires_b1_384_seed_42" / "completed.json"
    if not marker.is_file():
        raise FileNotFoundError("B1·384 완료 기록을 찾을 수 없습니다: " + str(marker))
    completed = engine.read_json(marker)
    attempt = marker.parent / completed["attempt"]
    checkpoint = attempt / "stage1_best.keras"
    expected = config["parent_stage1_sha256"].strip().lower()
    actual = engine.file_hash(checkpoint)
    if actual != expected:
        raise ValueError("B1·384 Stage 1 체크포인트 SHA-256이 다릅니다.")
    relative = completed["attempt"] + "/stage1_best.keras"
    if completed.get("artifact_hashes", {}).get(relative) != actual:
        raise ValueError("완료 기록의 Stage 1 체크포인트 해시와 다릅니다.")
    run_config = engine.read_json(parent / "run_config.json")
    settings = run_config["settings"]
    if (
        settings["data_sha256"] != context["data_hash"]
        or settings["classes"] != context["classes"]
        or settings["seed"] != 42
    ):
        raise ValueError("부모 실행의 데이터·클래스·seed가 현재 실행과 다릅니다.")
    return completed, checkpoint, actual


def _configure_partial(model):
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


def _fit_sam(model, train, val, directory, epochs, rho):
    loss_function = engine.loss_function("ls005")
    optimizer = keras.optimizers.Adam(1e-5)
    train_loss = keras.metrics.Mean()
    train_accuracy = keras.metrics.CategoricalAccuracy()
    val_loss = keras.metrics.Mean()
    val_accuracy = keras.metrics.CategoricalAccuracy()
    variables = model.trainable_variables
    history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}
    best_score = -1.0
    best = directory / "sam_best.keras"

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as first_tape:
            first_predictions = model(images, training=True)
            first_loss = loss_function(labels, first_predictions)
        first_gradients = first_tape.gradient(first_loss, variables)
        usable = [g for g in first_gradients if g is not None]
        norm = tf.linalg.global_norm(usable)
        perturbations = []
        for variable, gradient in zip(variables, first_gradients, strict=True):
            if gradient is None:
                perturbations.append(None)
                continue
            perturbation = tf.cast(rho, gradient.dtype) * gradient / (
                tf.cast(norm, gradient.dtype) + tf.cast(1e-12, gradient.dtype)
            )
            variable.assign_add(perturbation)
            perturbations.append(perturbation)
        with tf.GradientTape() as second_tape:
            predictions = model(images, training=True)
            loss = loss_function(labels, predictions)
        gradients = second_tape.gradient(loss, variables)
        for variable, perturbation in zip(variables, perturbations, strict=True):
            if perturbation is not None:
                variable.assign_sub(perturbation)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables, strict=True)
            if gradient is not None
        ]
        optimizer.apply_gradients(gradient_pairs)
        train_loss.update_state(loss)
        train_accuracy.update_state(labels, predictions)

    @tf.function
    def validation_step(images, labels):
        predictions = model(images, training=False)
        loss = loss_function(labels, predictions)
        val_loss.update_state(loss)
        val_accuracy.update_state(labels, predictions)

    log_path = directory / "sam_log.csv"
    with log_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["epoch", "accuracy", "loss", "val_accuracy", "val_loss"]
        )
        writer.writeheader()
        for epoch in range(epochs):
            for metric in (train_loss, train_accuracy, val_loss, val_accuracy):
                metric.reset_state()
            for images, labels in train:
                train_step(images, labels)
            for images, labels in val:
                validation_step(images, labels)
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
            engine.write_json(directory / "sam_history.json", history)
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {values['loss']:.4f} - "
                f"accuracy: {values['accuracy']:.4f} - val_loss: "
                f"{values['val_loss']:.4f} - val_accuracy: {values['val_accuracy']:.4f}"
            )
            if values["val_accuracy"] > best_score:
                best_score = values["val_accuracy"]
                model.save(best)
    if not best.is_file() or len(history["val_accuracy"]) != epochs:
        raise RuntimeError("SAM 학습이 완전하게 끝나지 않았습니다.")
    model.save(directory / "sam_last.keras")
    return history, best


def run(context):
    validate(context)
    baseline, checkpoint, checkpoint_hash = _parent(context)
    spec = {
        "id": TRIAL_ID,
        "method": "sam",
        "backbone": "B1",
        "size": 384,
        "loss": "ls005",
        "seed": 42,
        "epochs": context["config"]["epochs2"],
        "rho": float(context["config"]["sam_rho"]),
        "parent_stage1_sha256": checkpoint_hash,
        "protocol": PROTOCOL,
    }
    signature = hashlib.sha256(
        (context["signature"] + json.dumps(spec, sort_keys=True)).encode()
    ).hexdigest()
    cached = engine.cached_record(context["output"], TRIAL_ID, signature)
    if cached:
        return baseline, cached
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    directory = Path(context["output"]) / TRIAL_ID / ("attempt_" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    engine.write_json(directory / "spec.json", spec)
    engine.write_json(
        context["output"] / "sam_parent.json",
        {
            "parent_run_dir": str(Path(context["config"]["parent_run_dir"])),
            "parent_trial": "multires_b1_384_seed_42",
            "parent_stage1_sha256": checkpoint_hash,
        },
    )
    train, _ = flow.factory(context, "augmented", seed=42)("train", 384, True)
    val, paths = flow.factory(context, "augmented", seed=42)("val", 384, False)
    started = time.monotonic()
    model = keras.models.load_model(checkpoint, compile=False)
    trainable = _configure_partial(model)
    history, best = _fit_sam(
        model,
        train,
        val,
        directory,
        context["config"]["epochs2"],
        float(context["config"]["sam_rho"]),
    )
    del model
    keras.backend.clear_session()
    model = keras.models.load_model(best, compile=False)
    metrics = engine.evaluate_to_files(model, val, paths, directory, "validation")
    record = {
        "id": TRIAL_ID,
        "spec": spec,
        "signature": signature,
        "attempt": directory.name,
        "selected_model": best.name,
        "selected_stage": "sam_partial_finetuning",
        "validation": metrics,
        "training_seconds": time.monotonic() - started,
        "parameters": model.count_params(),
        "model_bytes": best.stat().st_size,
        "trainable_backbone_layers": trainable,
        "history": history,
        "stage_boundary": 0,
        "test_evaluated": False,
    }
    engine.finish_record(directory, record)
    return baseline, record


def summarize(context, baseline, sam):
    import matplotlib.pyplot as plt

    validate(context)
    output = Path(context["output"])
    rows = []
    for name, record in (("Adam", baseline), ("SAM", sam)):
        metrics = record["validation"]
        rows.append(
            {
                "optimizer": name,
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
            }
        )
    with (output / "sam_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": PROTOCOL,
        "fixed_conditions": {
            "data_sha256": context["data_hash"],
            "split": "same original Validation",
            "backbone": "EfficientNet-B1",
            "input_size": 384,
            "loss": "label_smoothing_0.05",
            "seed": 42,
            "parent_stage1_sha256": sam["spec"]["parent_stage1_sha256"],
            "partial_trainable_layers": 30,
            "batch_normalization_trainable": False,
            "fine_tuning_epochs": context["config"]["epochs2"],
            "test_access": False,
        },
        "changed_condition": "Adam partial fine-tuning versus SAM with Adam base optimizer",
        "adam": baseline["validation"],
        "sam": sam["validation"],
        "delta_sam_minus_adam": {
            "validation_accuracy": (
                sam["validation"]["accuracy"] - baseline["validation"]["accuracy"]
            ),
            "validation_macro_f1": (
                sam["validation"]["macro_f1"] - baseline["validation"]["macro_f1"]
            ),
        },
        "screening_only": True,
        "test_evaluated": False,
    }
    engine.write_json(output / "sam_comparison_summary.json", summary)
    labels = ["Adam", "SAM"]
    x = np.arange(2)
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    axes[0].bar(
        x - 0.18,
        [row["validation_accuracy"] for row in rows],
        0.36,
        label="Validation Accuracy",
        color="#2f6fed",
    )
    axes[0].bar(
        x + 0.18,
        [row["validation_macro_f1"] for row in rows],
        0.36,
        label="Validation Macro F1",
        color="#18a558",
    )
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 1), title="B1·384: Adam vs SAM")
    axes[0].legend()
    epochs = np.arange(1, len(sam["history"]["accuracy"]) + 1)
    axes[1].plot(epochs, sam["history"]["accuracy"], label="Train Accuracy")
    axes[1].plot(epochs, sam["history"]["val_accuracy"], label="Validation Accuracy")
    axes[1].set(title="SAM fine-tuning curve", xlabel="Epoch", ylabel="Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "sam_performance_dashboard.png", dpi=180)
    plt.show()
    plt.close(fig)
    archive = flow.archive_results(context)
    print("B1·384 Adam vs SAM Validation 비교 완료:", output)
    print("Test는 평가하지 않았습니다.")
    return summary, archive
