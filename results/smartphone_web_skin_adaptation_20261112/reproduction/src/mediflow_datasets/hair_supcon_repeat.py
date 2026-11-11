"""Hair baseline versus SupCon confirmation at seeds 43 and 44."""

from __future__ import annotations

import csv
import uuid
from pathlib import Path

import numpy as np

from mediflow_datasets import common_engine as engine
from mediflow_datasets import common_workflow as flow
from mediflow_datasets import paper_suite

PROTOCOL = "hair_supcon_repeat_v1"
CONDITIONS = ["baseline_b1_256", "supcon_b1_256"]
SEEDS = [43, 44]


def validate(context):
    config = context["config"]
    if config["domain"] != "hair" or config["mode"] != "supcon_repeat":
        raise ValueError("이 노트북은 Hair SupCon 반복 검증 전용입니다.")
    if config["train_variant"] != "augmented":
        raise ValueError("기존 Hair 후보와 같은 augmented Train을 사용해야 합니다.")
    if config["seeds"] != SEEDS or config["seed"] != SEEDS[0]:
        raise ValueError("반복 검증 seed는 [43, 44]로 고정합니다.")
    if config.get("experiments") != CONDITIONS:
        raise ValueError("비교 조건은 baseline과 SupCon 두 가지로 고정합니다.")


def run(context):
    """Run paired baseline and SupCon trials for seeds 43 and 44."""
    validate(context)
    handlers = {
        "baseline_b1_256": paper_suite._run_standard,
        "supcon_b1_256": paper_suite._run_supcon,
    }
    records = []
    try:
        for seed in SEEDS:
            for condition in CONDITIONS:
                spec = paper_suite._trial_spec(context, condition, seed)
                spec["protocol"] = PROTOCOL
                record = handlers[condition](context, spec)
                records.append(record)
                engine.write_json(
                    context["output"] / "supcon_repeat_progress.json",
                    {"completed": [item["id"] for item in records]},
                )
        engine.write_json(context["output"] / "all_validation_results.json", records)
        return records
    except Exception as exc:
        engine.write_json(
            context["output"] / ("supcon_repeat_failure_" + uuid.uuid4().hex[:8] + ".json"),
            {"error": repr(exc), "completed": [item["id"] for item in records]},
        )
        raise


def _mean_std(values):
    values = np.asarray(values, dtype="float64")
    return {"mean": float(values.mean()), "std_population": float(values.std(ddof=0))}


def _ordered(records):
    lookup = {(r["spec"]["seed"], r["spec"]["experiment_id"]): r for r in records}
    expected = [(seed, condition) for seed in SEEDS for condition in CONDITIONS]
    if set(lookup) != set(expected) or len(records) != len(expected):
        raise ValueError("seed 43·44의 기준선과 SupCon 결과가 모두 필요합니다.")
    return [lookup[key] for key in expected]


def summarize(context, records):
    """Save the two-seed confirmation report without accessing Test."""
    import matplotlib.pyplot as plt

    validate(context)
    records = _ordered(records)
    output = Path(context["output"])
    run_rows = []
    class_rows = []
    per_seed = {}
    for seed in SEEDS:
        baseline, supcon = [
            r for r in records if r["spec"]["seed"] == seed
        ]
        per_seed[str(seed)] = {
            "baseline": baseline["validation"],
            "supcon": supcon["validation"],
            "delta": {
                "validation_accuracy": (
                    supcon["validation"]["accuracy"] - baseline["validation"]["accuracy"]
                ),
                "validation_macro_f1": (
                    supcon["validation"]["macro_f1"] - baseline["validation"]["macro_f1"]
                ),
            },
        }
        for record in (baseline, supcon):
            run_rows.append(
                {
                    "seed": seed,
                    "condition": record["spec"]["experiment_id"],
                    "validation_accuracy": record["validation"]["accuracy"],
                    "validation_macro_f1": record["validation"]["macro_f1"],
                    "training_seconds": record["training_seconds"],
                    "parameters": record["parameters"],
                    "model_bytes": record["model_bytes"],
                }
            )
        for index, name in enumerate(context["classes"]):
            base = baseline["validation"]["class_f1"][index]
            sup = supcon["validation"]["class_f1"][index]
            class_rows.append(
                {
                    "seed": seed,
                    "class_index": index,
                    "class_name": name,
                    "baseline_f1": base,
                    "supcon_f1": sup,
                    "delta_supcon_minus_baseline": sup - base,
                }
            )

    condition_summary = {}
    for condition in CONDITIONS:
        items = [r for r in records if r["spec"]["experiment_id"] == condition]
        class_values = np.asarray([r["validation"]["class_f1"] for r in items])
        condition_summary[condition] = {
            "validation_accuracy": _mean_std(
                [r["validation"]["accuracy"] for r in items]
            ),
            "validation_macro_f1": _mean_std(
                [r["validation"]["macro_f1"] for r in items]
            ),
            "class_f1": {
                name: _mean_std(class_values[:, index])
                for index, name in enumerate(context["classes"])
            },
            "training_seconds": _mean_std([r["training_seconds"] for r in items]),
        }
    summary = {
        "protocol": PROTOCOL,
        "paper": paper_suite.EXPERIMENTS["supcon_b1_256"]["paper"],
        "fixed_conditions": {
            "data_sha256": context["data_hash"],
            "split": "same immutable Train and Validation folders as seed 42",
            "train_variant": "augmented",
            "validation_variant": "original",
            "class_names": context["classes"],
            "backbone": "EfficientNet-B1",
            "input_size": 256,
            "seeds": SEEDS,
            "test_access": False,
        },
        "changed_condition": "baseline classification versus SupCon representation learning",
        "per_seed": per_seed,
        "repeat_only_statistics": condition_summary,
        "requires_seed_42_merge_for_final_statistics": True,
        "test_evaluated": False,
    }
    engine.write_json(output / "supcon_repeat_summary.json", summary)
    for name, rows in (
        ("supcon_repeat_runs.csv", run_rows),
        ("supcon_repeat_class_f1.csv", class_rows),
    ):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    x = np.arange(len(SEEDS))
    base_f1 = [per_seed[str(seed)]["baseline"]["macro_f1"] for seed in SEEDS]
    sup_f1 = [per_seed[str(seed)]["supcon"]["macro_f1"] for seed in SEEDS]
    axes[0].bar(x - 0.18, base_f1, 0.36, label="Baseline", color="#6b7280")
    axes[0].bar(x + 0.18, sup_f1, 0.36, label="SupCon", color="#2f6fed")
    axes[0].set(
        xticks=x,
        xticklabels=[f"seed {seed}" for seed in SEEDS],
        ylim=(0, 1),
        ylabel="Validation Macro F1",
        title="Hair SupCon confirmation: paired seeds 43 and 44",
    )
    axes[0].legend()
    class_matrix = np.asarray(
        [
            [condition_summary[c]["class_f1"][name]["mean"] for name in context["classes"]]
            for c in CONDITIONS
        ]
    )
    image = axes[1].imshow(class_matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axes[1].set(
        xticks=np.arange(len(context["classes"])),
        xticklabels=[f"C{i}" for i in range(len(context["classes"]))],
        yticks=np.arange(2),
        yticklabels=["Baseline", "SupCon"],
        title="Seeds 43–44 mean class F1 (C0–C4 follow class_names.json)",
    )
    for row in range(2):
        for column in range(class_matrix.shape[1]):
            axes[1].text(column, row, f"{class_matrix[row, column]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[1], fraction=0.02)
    fig.tight_layout()
    fig.savefig(output / "supcon_repeat_dashboard.png", dpi=180)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(15, 18), squeeze=False)
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
    fig.savefig(output / "supcon_repeat_training_curves.png", dpi=160)
    plt.show()
    plt.close(fig)

    archive = flow.archive_results(context)
    print("Hair SupCon seed 43·44 반복 검증 완료:", output)
    print("이 보고서는 seed 43·44만 집계합니다. 최종 판단에는 seed 42 결과를 합칩니다.")
    print("Test는 평가하지 않았습니다.")
    return summary, archive
