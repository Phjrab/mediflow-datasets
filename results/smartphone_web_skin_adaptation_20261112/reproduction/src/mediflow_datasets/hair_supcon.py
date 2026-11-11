"""Single-seed Hair baseline versus supervised contrastive learning experiment."""

from __future__ import annotations

import csv
import uuid
from pathlib import Path

import numpy as np

from mediflow_datasets import common_engine as engine
from mediflow_datasets import common_workflow as flow
from mediflow_datasets import paper_suite

PROTOCOL = "hair_supcon_comparison_v1"
CONDITIONS = ["baseline_b1_256", "supcon_b1_256"]
PAPER = {
    "title": "Supervised Contrastive Learning",
    "authors": "Khosla et al.",
    "venue": "NeurIPS 2020",
    "url": (
        "https://proceedings.neurips.cc/paper/2020/hash/"
        "d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html"
    ),
}


def validate(context):
    config = context["config"]
    if config["domain"] != "hair" or config["mode"] != "supcon_compare":
        raise ValueError("이 노트북은 Hair SupCon 비교 전용입니다.")
    if config["train_variant"] != "augmented":
        raise ValueError("기존 Hair 후보와 같은 augmented Train을 사용해야 합니다.")
    if config["seeds"] != [42] or config["seed"] != 42:
        raise ValueError("첫 선별 실험은 seed 42 한 번만 수행합니다.")
    if config.get("experiments") != CONDITIONS:
        raise ValueError("비교 조건은 baseline과 SupCon 두 가지로 고정합니다.")


def run(context):
    """Run exactly one baseline and one SupCon condition without Test access."""
    validate(context)
    records = []
    handlers = {
        "baseline_b1_256": paper_suite._run_standard,
        "supcon_b1_256": paper_suite._run_supcon,
    }
    try:
        for condition in CONDITIONS:
            spec = paper_suite._trial_spec(context, condition, 42)
            spec["protocol"] = PROTOCOL
            record = handlers[condition](context, spec)
            records.append(record)
            engine.write_json(
                context["output"] / "supcon_progress.json",
                {"completed": [item["spec"]["experiment_id"] for item in records]},
            )
        engine.write_json(context["output"] / "all_validation_results.json", records)
        return records
    except Exception as exc:
        engine.write_json(
            context["output"] / ("supcon_failure_" + uuid.uuid4().hex[:8] + ".json"),
            {"error": repr(exc), "completed": [item["id"] for item in records]},
        )
        raise


def _confusion(ax, metrics, title):
    matrix = np.asarray(metrics["confusion_matrix"])
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(title=title, xlabel="Predicted class index", ylabel="True class index")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    return image


def summarize(context, records):
    """Create a validation-only comparison report for the single hypothesis."""
    import matplotlib.pyplot as plt

    validate(context)
    by_condition = {item["spec"]["experiment_id"]: item for item in records}
    if list(by_condition) != CONDITIONS:
        raise ValueError("기준선과 SupCon 결과가 모두 필요합니다.")
    baseline = by_condition[CONDITIONS[0]]
    supcon = by_condition[CONDITIONS[1]]
    output = Path(context["output"])
    rows = []
    for condition, record in by_condition.items():
        rows.append(
            {
                "condition": condition,
                "seed": record["spec"]["seed"],
                "validation_accuracy": record["validation"]["accuracy"],
                "validation_macro_f1": record["validation"]["macro_f1"],
                "training_seconds": record["training_seconds"],
                "parameters": record["parameters"],
                "model_bytes": record["model_bytes"],
            }
        )
    class_rows = []
    for index, class_name in enumerate(context["classes"]):
        base_f1 = baseline["validation"]["class_f1"][index]
        supcon_f1 = supcon["validation"]["class_f1"][index]
        class_rows.append(
            {
                "class_index": index,
                "class_name": class_name,
                "baseline_f1": base_f1,
                "supcon_f1": supcon_f1,
                "delta_supcon_minus_baseline": supcon_f1 - base_f1,
            }
        )
    for name, values in (
        ("supcon_comparison.csv", rows),
        ("supcon_class_f1_comparison.csv", class_rows),
    ):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = {
        "protocol": PROTOCOL,
        "paper": PAPER,
        "hypothesis": (
            "Supervised contrastive representation learning reduces confusion between "
            "visually similar Hair classes compared with the fixed B1/256 baseline."
        ),
        "fixed_conditions": {
            "data_sha256": context["data_hash"],
            "split": "same immutable Train and Validation folders",
            "train_variant": "augmented",
            "validation_variant": "original",
            "class_names": context["classes"],
            "backbone": "EfficientNet-B1",
            "input_size": 256,
            "seed": 42,
            "test_access": False,
        },
        "changed_condition": (
            "B1 supervised head/partial fine-tuning versus 15-epoch SupCon representation "
            "training followed by 15-epoch supervised classifier fine-tuning"
        ),
        "necessary_accompanying_change": (
            "SupCon creates two weak augmented views per image because positive pairs are "
            "required by the method"
        ),
        "baseline": baseline["validation"],
        "supcon": supcon["validation"],
        "delta": {
            "validation_accuracy": (
                supcon["validation"]["accuracy"] - baseline["validation"]["accuracy"]
            ),
            "validation_macro_f1": (
                supcon["validation"]["macro_f1"] - baseline["validation"]["macro_f1"]
            ),
            "class_f1": {
                row["class_name"]: row["delta_supcon_minus_baseline"] for row in class_rows
            },
        },
        "screening_only": True,
        "repeat_policy": (
            "Run seeds 43 and 44 only if seed 42 shows a useful Validation improvement "
            "without a material regression in another core class."
        ),
        "test_evaluated": False,
    }
    engine.write_json(output / "supcon_comparison_summary.json", summary)

    labels = ["Baseline", "SupCon"]
    accuracy = [item["validation_accuracy"] for item in rows]
    macro_f1 = [item["validation_macro_f1"] for item in rows]
    class_base = [row["baseline_f1"] for row in class_rows]
    class_supcon = [row["supcon_f1"] for row in class_rows]
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    x = np.arange(2)
    axes[0].bar(x - 0.18, accuracy, 0.36, label="Validation Accuracy", color="#2f6fed")
    axes[0].bar(x + 0.18, macro_f1, 0.36, label="Validation Macro F1", color="#18a558")
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 1), title="Hair: Baseline vs SupCon")
    axes[0].legend()
    class_x = np.arange(len(context["classes"]))
    axes[1].bar(class_x - 0.2, class_base, 0.4, label="Baseline")
    axes[1].bar(class_x + 0.2, class_supcon, 0.4, label="SupCon")
    axes[1].set(
        xticks=class_x,
        xticklabels=[f"C{i}" for i in class_x],
        ylim=(0, 1),
        title="Class F1 (see class_names.json)",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "supcon_performance_dashboard.png", dpi=180)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _confusion(axes[0], baseline["validation"], "Baseline Validation")
    image = _confusion(axes[1], supcon["validation"], "SupCon Validation")
    fig.colorbar(image, ax=axes, fraction=0.02)
    fig.savefig(output / "supcon_confusion_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for column, record in enumerate((baseline, supcon)):
        history = record["history"]
        epochs = np.arange(1, len(history["accuracy"]) + 1)
        for row, metric in enumerate(("accuracy", "loss")):
            axes[row, column].plot(epochs, history[metric], label="Train")
            axes[row, column].plot(epochs, history["val_" + metric], label="Validation")
            boundary = record.get("stage_boundary", 0)
            if 0 < boundary < len(epochs):
                axes[row, column].axvline(boundary + 0.5, color="gray", linestyle="--")
            axes[row, column].set(
                title=f"{labels[column]} / {metric}", xlabel="Epoch", ylabel=metric
            )
            axes[row, column].grid(alpha=0.25)
            axes[row, column].legend()
    fig.tight_layout()
    fig.savefig(output / "supcon_training_curves.png", dpi=180)
    plt.show()
    plt.close(fig)
    archive = flow.archive_results(context)
    print("Hair 기준선 vs SupCon Validation 비교 완료:", output)
    print("Validation Macro F1 변화:", summary["delta"]["validation_macro_f1"])
    print("Test는 평가하지 않았습니다.")
    return summary, archive
