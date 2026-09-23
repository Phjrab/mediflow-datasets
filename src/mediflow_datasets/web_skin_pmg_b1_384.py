"""Single Web Skin PMG B1/384 validation experiment; Test remains unopened."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import keras
import numpy as np

from mediflow_datasets import common_engine as engine
from mediflow_datasets import common_workflow as flow
from mediflow_datasets import web_skin_paper_suite as pmg
from mediflow_datasets import web_skin_wsdan as contract

PROTOCOL = "web_skin_pmg_b1_384_screen_v1"
TRIAL_ID = "pmg_b1_384_ce_seed_42"
EXPERIMENTS = [TRIAL_ID]
PAPERS = {
    "pmg": "https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650154.pdf",
    "efficientnet": "https://proceedings.mlr.press/v97/tan19a.html",
}
BASELINE = {
    "id": "pmg_b0_256_ce_seed_42",
    "validation_accuracy": 0.85,
    "validation_macro_f1": 0.8473279632397033,
    "validation_count": 500,
    "class_f1": [
        0.8252427184466019,
        0.7835051546391754,
        0.7634408602150538,
        0.9615384615384615,
        0.9029126213592233,
    ],
    "confusion_matrix": [
        [85, 5, 4, 3, 3],
        [7, 76, 8, 1, 8],
        [12, 12, 71, 3, 2],
        [0, 0, 0, 100, 0],
        [2, 1, 3, 1, 93],
    ],
    "data_sha256": contract.BASELINE["data_sha256"],
    "source": (
        "web_skin_paper_suite_20260922_124758_717b465e/"
        "pmg_b0_256_ce_seed_42/attempt_a8364775fcb8/validation_metrics.json"
    ),
    "test_evaluated": False,
}


def validate(context):
    config = context["config"]
    if config["domain"] != "web_skin" or config["mode"] != "web_skin_pmg_b1_384":
        raise ValueError("이 코드는 Web Skin PMG·B1·384 선별 실험 전용입니다.")
    if config["train_variant"] != "augmented":
        raise ValueError("기존 PMG와 같은 augmented Train을 사용해야 합니다.")
    if config["seed"] != 42 or config["seeds"] != [42]:
        raise ValueError("선별 seed는 42로 고정합니다.")
    if config.get("experiments") != EXPERIMENTS:
        raise ValueError("PMG·B1·384 단일 조건만 실행해야 합니다.")
    if config.get("pmg_jigsaw_grids") != [8, 4, 2]:
        raise ValueError("PMG jigsaw 순서는 8, 4, 2로 고정합니다.")
    if config["batch_size"] != 16:
        raise ValueError("384 입력의 Colab GPU 메모리를 위해 batch size는 16입니다.")
    if config["epochs1"] != 15 or config["epochs2"] != 10:
        raise ValueError("기존 PMG와 같이 Stage 1=15, Stage 2=10으로 고정합니다.")
    if context["classes"] != contract.EXPECTED_CLASSES:
        raise ValueError("Web Skin 클래스 순서가 기존 모델 계약과 다릅니다.")
    if context["data_hash"] != BASELINE["data_sha256"]:
        raise ValueError("기존 PMG와 데이터 SHA-256이 다릅니다.")


def _signature(context, spec):
    return hashlib.sha256(
        (context["signature"] + json.dumps(spec, sort_keys=True)).encode()
    ).hexdigest()


def _attempt(context):
    directory = (
        Path(context["output"]) / TRIAL_ID / ("attempt_" + uuid.uuid4().hex[:12])
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def run(context):
    validate(context)
    config = context["config"]
    spec = {
        "id": TRIAL_ID,
        "method": "efficientnet_b1_384_pmg_adaptation",
        "papers": PAPERS,
        "backbone": "B1",
        "input_size": 384,
        "loss": "ce",
        "jigsaw_grids": [8, 4, 2],
        "batch_size": 16,
        "epochs1": 15,
        "epochs2": 10,
        "seed": 42,
        "selection_split": "validation",
        "test_evaluated": False,
    }
    signature = _signature(context, spec)
    cached = engine.cached_record(context["output"], TRIAL_ID, signature)
    if cached:
        return cached

    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    directory = _attempt(context)
    engine.write_json(directory / "spec.json", spec)
    engine.write_json(Path(context["output"]) / "baseline_reference.json", BASELINE)
    train, _ = flow.factory(context, "augmented", seed=42)("train", 384, True)
    val, paths = flow.factory(context, "augmented", seed=42)("val", 384, False)
    started = time.monotonic()
    try:
        model = pmg.build_pmg_model(
            len(context["classes"]), size=384, backbone_name="B1"
        )
        h1, best1 = pmg._fit_pmg_stage(
            model, train, val, directory, "stage1", config["epochs1"], 1e-4
        )
        del model
        keras.backend.clear_session()
        model = keras.models.load_model(best1, compile=False)
        trainable = pmg._configure_pmg_partial(model)
        h2, best2 = pmg._fit_pmg_stage(
            model, train, val, directory, "stage2", config["epochs2"], 1e-5
        )
        selected_stage = (
            "stage2"
            if engine.checkpoint_choice(
                max(h1["val_accuracy"]), max(h2["val_accuracy"])
            )
            else "stage1"
        )
        selected = best2 if selected_stage == "stage2" else best1
        del model
        keras.backend.clear_session()
        model = keras.models.load_model(selected, compile=False)
        predictor = pmg._pmg_predictor(model)
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
            "inference": (
                "one 384x384 image; sum of three branch logits and fused logits"
            ),
            "test_evaluated": False,
        }
        engine.finish_record(directory, record)
        return record
    except Exception as exc:
        engine.write_json(
            Path(context["output"]) / ("failure_" + uuid.uuid4().hex[:8] + ".json"),
            {
                "error": repr(exc),
                "completed": [],
                "resume_dir": str(context["output"]),
                "attempt": directory.name,
            },
        )
        print("실패 attempt는 보존됐습니다. RESUME_DIR:", context["output"])
        raise


def _draw_confusion(axis, matrix, title):
    matrix = np.asarray(matrix, dtype=int)
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axis.set(
        title=title,
        xlabel="Predicted",
        ylabel="True",
        xticks=range(len(matrix)),
        yticks=range(len(matrix)),
        xticklabels=[f"C{i}" for i in range(len(matrix))],
        yticklabels=[f"C{i}" for i in range(len(matrix))],
    )
    return image


def summarize(context, record):
    import matplotlib.pyplot as plt
    import pandas as pd

    validate(context)
    if record["id"] != TRIAL_ID or record.get("test_evaluated") is not False:
        raise ValueError("PMG·B1·384 Validation 완료 기록이 아닙니다.")
    output = Path(context["output"])
    rows = [
        {
            "experiment": BASELINE["id"],
            "validation_accuracy": BASELINE["validation_accuracy"],
            "validation_macro_f1": BASELINE["validation_macro_f1"],
            "trained_in_this_run": False,
        },
        {
            "experiment": record["id"],
            "validation_accuracy": record["validation"]["accuracy"],
            "validation_macro_f1": record["validation"]["macro_f1"],
            "trained_in_this_run": True,
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(output / "pmg_scale_comparison.csv", index=False, encoding="utf-8-sig")

    labels = ["PMG B0·256", "PMG B1·384"]
    x = np.arange(2)
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.bar(x - 0.2, table.validation_accuracy, 0.4, label="Validation Accuracy")
    axis.bar(x + 0.2, table.validation_macro_f1, 0.4, label="Validation Macro F1")
    for index, value in enumerate(table.validation_accuracy):
        axis.text(index - 0.2, value + 0.005, f"{value:.4f}", ha="center")
    for index, value in enumerate(table.validation_macro_f1):
        axis.text(index + 0.2, value + 0.005, f"{value:.4f}", ha="center")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_title("Web Skin PMG Scale Comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "pmg_scale_performance.png", dpi=180)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    _draw_confusion(axes[0], BASELINE["confusion_matrix"], labels[0])
    _draw_confusion(axes[1], record["validation"]["confusion_matrix"], labels[1])
    fig.suptitle("Web Skin PMG Validation Confusion Matrices (C0-C4)")
    fig.tight_layout()
    fig.savefig(output / "pmg_scale_confusion_matrices.png", dpi=180)
    plt.show()
    plt.close(fig)

    history = record["history"]
    epochs = np.arange(1, len(history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history["accuracy"], label="Train")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].plot(epochs, history["loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    for axis, title, ylabel in zip(
        axes, ("Accuracy", "Loss"), ("Accuracy", "Loss"), strict=True
    ):
        axis.axvline(record["stage_boundary"] + 0.5, ls="--", color="gray")
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "pmg_b1_384_training_curves.png", dpi=180)
    plt.show()
    plt.close(fig)

    class_frame = pd.DataFrame(
        [BASELINE["class_f1"], record["validation"]["class_f1"]],
        index=labels,
        columns=context["classes"],
    )
    class_frame.to_csv(output / "pmg_scale_class_f1.csv", encoding="utf-8-sig")
    winner = max(rows, key=lambda item: item["validation_macro_f1"])
    summary = {
        "protocol": PROTOCOL,
        "data_sha256": context["data_hash"],
        "classes": context["classes"],
        "baseline_retrained": False,
        "completed_experiments": [TRIAL_ID],
        "comparison": rows,
        "winner_by_validation_macro_f1": winner,
        "test_evaluated": False,
        "necessary_accompanying_change": (
            "Batch size 16 is used for B1/384 GPU memory; the earlier B0/256 PMG used 32."
        ),
    }
    engine.write_json(output / "pmg_scale_summary.json", summary)
    archive = flow.archive_results(context)
    return summary, archive
