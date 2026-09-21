"""Three-seed validation baselines for the paper-based enhancement phase."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mediflow_datasets import common_engine as engine
from mediflow_datasets import common_workflow as flow

PROTOCOL = "paper_baseline3_v1"
BASELINES = {
    "hair": {
        "backbone": "B1",
        "size": 256,
        "loss": "ls005",
        "epochs1": 15,
        "epochs2": 15,
    },
    "web_skin": {
        "backbone": "B0",
        "size": 256,
        "loss": "ce",
        "epochs1": 15,
        "epochs2": 10,
    },
    "skin": {
        "backbone": "B0",
        "size": 224,
        "loss": "ce",
        "epochs1": 15,
        "epochs2": 0,
    },
}


def validate_protocol(context):
    config = context["config"]
    expected = BASELINES[config["domain"]]
    if config["mode"] != "baseline3":
        raise ValueError("MODE는 baseline3이어야 합니다.")
    if config["train_variant"] != "augmented":
        raise ValueError("현재 후보 기준선은 augmented Train을 사용합니다.")
    if config["epochs1"] != expected["epochs1"]:
        raise ValueError(f"STAGE1_EPOCHS는 {expected['epochs1']}이어야 합니다.")
    if config["epochs2"] != max(expected["epochs2"], 1):
        raise ValueError(
            "STAGE2_EPOCHS 설정이 현재 도메인의 기준선과 다릅니다: "
            + str(max(expected["epochs2"], 1))
        )
    return expected


def run(context):
    """Train one fixed baseline for each seed; never access Test."""
    expected = validate_protocol(context)
    config, output = context["config"], context["output"]
    records = []
    for seed in config["seeds"]:
        spec = {
            "id": f"baseline_seed_{seed}",
            "backbone": expected["backbone"],
            "size": expected["size"],
            "loss": expected["loss"],
            "variant": config["train_variant"],
            "class_count": len(context["classes"]),
            "class_names": context["classes"],
            "seed": seed,
            "protocol": PROTOCOL,
        }
        record = engine.run_trial(
            spec,
            flow.factory(context, config["train_variant"], seed=seed),
            output,
            context["signature"],
            seed,
            expected["epochs1"],
            expected["epochs2"],
        )
        records.append(record)
        engine.write_json(
            output / "progress.json",
            {"completed_seeds": [item["spec"]["seed"] for item in records]},
        )
    engine.write_json(output / "all_validation_results.json", records)
    return records


def _mean_std(values):
    array = np.asarray(values, dtype="float64")
    return {"mean": float(array.mean()), "std_population": float(array.std(ddof=0))}


def summarize(context, records):
    """Save validation-only tables and figures; Test remains unopened."""
    import matplotlib.pyplot as plt

    output = Path(context["output"])
    seeds = [record["spec"]["seed"] for record in records]
    if seeds != context["config"]["seeds"]:
        raise ValueError("seed 실행 순서가 설정과 다릅니다.")
    accuracy = [record["validation"]["accuracy"] for record in records]
    macro_f1 = [record["validation"]["macro_f1"] for record in records]
    class_f1 = np.asarray([record["validation"]["class_f1"] for record in records])
    summary = {
        "protocol": PROTOCOL,
        "domain": context["config"]["domain"],
        "data_sha256": context["data_hash"],
        "class_names": context["classes"],
        "seeds": seeds,
        "selection_metric": "validation_macro_f1",
        "validation_accuracy": _mean_std(accuracy),
        "validation_macro_f1": _mean_std(macro_f1),
        "class_f1": {
            name: _mean_std(class_f1[:, index])
            for index, name in enumerate(context["classes"])
        },
        "test_evaluated": False,
    }
    engine.write_json(output / "multiseed_summary.json", summary)
    with (output / "multiseed_runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["seed", "validation_accuracy", "validation_macro_f1", "seconds"],
        )
        writer.writeheader()
        for seed, record in zip(seeds, records, strict=True):
            writer.writerow(
                {
                    "seed": seed,
                    "validation_accuracy": record["validation"]["accuracy"],
                    "validation_macro_f1": record["validation"]["macro_f1"],
                    "seconds": record["training_seconds"],
                }
            )
    with (output / "class_f1_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["class_name", "mean_f1", "std_population"],
        )
        writer.writeheader()
        for name, values in summary["class_f1"].items():
            writer.writerow(
                {
                    "class_name": name,
                    "mean_f1": values["mean"],
                    "std_population": values["std_population"],
                }
            )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(seeds))
    axes[0].bar(x - 0.18, accuracy, 0.36, label="Validation Accuracy")
    axes[0].bar(x + 0.18, macro_f1, 0.36, label="Validation Macro F1")
    axes[0].axhline(np.mean(macro_f1), color="green", linestyle="--", label="Macro F1 mean")
    axes[0].set(xticks=x, xticklabels=[f"seed {seed}" for seed in seeds], ylim=(0, 1))
    axes[0].set_title("Three-seed validation baseline")
    axes[0].legend()
    means = class_f1.mean(axis=0)
    deviations = class_f1.std(axis=0, ddof=0)
    axes[1].bar(np.arange(len(means)), means, yerr=deviations, capsize=4)
    axes[1].set(
        xticks=np.arange(len(means)),
        xticklabels=[f"C{i}" for i in range(len(means))],
        ylim=(0, 1),
        title="Class F1 mean ± population std",
    )
    fig.tight_layout()
    fig.savefig(output / "multiseed_validation_dashboard.png", dpi=180)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(len(records), 2, figsize=(15, 4.5 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        history = record["history"]
        epochs = np.arange(1, len(history["accuracy"]) + 1)
        for column, metric in enumerate(("accuracy", "loss")):
            ax = axes[row, column]
            ax.plot(epochs, history[metric], label="Train")
            ax.plot(epochs, history["val_" + metric], label="Validation")
            ax.axvline(record["stage_boundary"] + 0.5, color="gray", linestyle="--")
            ax.set(title=f"seed {record['spec']['seed']} / {metric}", xlabel="Epoch")
            ax.grid(alpha=0.25)
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / "multiseed_training_curves.png", dpi=180)
    plt.show()
    plt.close(fig)
    archive = flow.archive_results(context)
    print("3-seed Validation 기준선 저장 완료:", output)
    print("Test는 평가하지 않았습니다.")
    return summary, archive
