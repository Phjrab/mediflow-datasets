"""Run the locked camera-adaptation comparison on the original labeled Test split."""

from __future__ import annotations

import argparse
import csv
import io
import platform
import tempfile
import zipfile
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import cv2
import keras
import numpy as np
import tensorflow as tf

from mediflow_datasets import adaptation_stage1 as training
from mediflow_datasets import camera_extended as extended
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as experiment
from mediflow_datasets.adaptation_stage1_colab import copy_source
from mediflow_datasets.domain_adaptation_batch import digest, find_source
from mediflow_datasets.domain_adaptation_io import write_progress

CONDITIONS = ("clean", "lighting", "expanded")
MODEL_LABELS = {
    "initial": "Before adaptation",
    "stage1_lighting": "Stage 1",
    "stage2_expanded": "Stage 2",
    "stage3_aligned": "Stage 3",
}
DOMAIN_LABELS = {"skin": "Microscope skin", "web_skin": "Webcam face", "hair": "Microscope scalp"}


def prepare_test(archive, task, lock, manifest):
    if digest(manifest) != lock["manifest_sha256"]:
        raise ValueError(f"{task['domain']}: Test manifest SHA-256 mismatch")
    rows, _ = data.source_index(archive, task["classes"], task["source_sha256"])
    original_test = rows["original", "test"]
    declared = data.read_json(manifest)["original"]["test"]
    actual = [(row.path, row.label) for row in original_test]
    expected = [(row["path"], row["label"]) for row in declared]
    if actual != expected or len(actual) != lock["test_count"]:
        raise ValueError(f"{task['domain']}: original Test rows differ from the locked manifest")
    return original_test


def evaluate_zip(archive_path, rows, task, spec, reference, predictors, seed):
    """Read each Test image once; feed the exact same arrays to all model arms."""
    scores = {name: {condition: [] for condition in CONDITIONS} for name in predictors}
    size, batch_size = spec["size"], task["batch_size"]
    with zipfile.ZipFile(archive_path) as archive:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            clean, lighting, expanded = [], [], []
            for row in batch:
                image = experiment.image_tensor(archive.read(row.path), size, task["input_policy"])
                item_seed = experiment.item_seed(row, seed)
                clean.append(image)
                lighting.append(
                    experiment.transform_image(
                        image, reference["extended_profile"]["lighting"], "target_guided", item_seed
                    )
                )
                expanded.append(
                    extended.transform(image, reference["extended_profile"], "expanded", item_seed)
                )
            for condition, inputs in zip(CONDITIONS, (clean, lighting, expanded), strict=True):
                tensor = tf.convert_to_tensor(np.stack(inputs), dtype=tf.float32)
                for name, predict in predictors.items():
                    result = predict(tensor).numpy()
                    if result.shape != (len(batch), len(task["classes"])):
                        raise ValueError(f"{name}: unexpected model output shape {result.shape}")
                    scores[name][condition].append(result)
            done = start + len(batch)
            if start == 0 or start % (batch_size * 20) == 0 or done == len(rows):
                training.show(
                    f"[{task['domain']}] Test 예측 · 모델 4개 × 조건 3개", done, len(rows)
                )
    return {
        name: {
            condition: experiment.classification(
                rows, np.concatenate(batches), len(task["classes"])
            )
            for condition, batches in results.items()
        }
        for name, results in scores.items()
    }


def domain_result(runtime, stages, domain, task, lock, seed, archive):
    print(f"[{domain}] 원본 ZIP 해시와 Test 목록 확인", flush=True)
    rows = prepare_test(archive, task, lock, stages[0] / domain / "source_manifest.json")
    reference = data.read_json(runtime / "references" / f"{domain}.json")
    if training.object_hash(reference) != lock["reference_sha256"]:
        raise ValueError(f"{domain}: camera profile mismatch")
    spec = task["models"][lock["start_role"]]
    predictors = {}
    for name, model in lock["models"].items():
        root = runtime if model["stage"] == 0 else stages[model["stage"] - 1]
        path = data.local_path(root, model["path"])
        if digest(path) != model["sha256"]:
            raise ValueError(f"{domain}/{name}: checkpoint SHA-256 mismatch")
        print(f"[{domain}] {name} 모델 로딩", flush=True)
        predictors[name] = training.predictor(keras.models.load_model(path, compile=False), spec)
    metrics = evaluate_zip(archive, rows, task, spec, reference, predictors, seed)
    return {
        "domain": domain,
        "classes": task["classes"],
        "test_count": len(rows),
        "source_sha256": task["source_sha256"],
        "reference_sha256": lock["reference_sha256"],
        "input_policy": task["input_policy"],
        "selected_on_validation": lock["selected_on_validation"],
        "metrics": metrics,
    }


def write_comparison(output, results):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "domain",
            "model",
            "selected_on_validation",
            "condition",
            "count",
            "accuracy",
            "macro_f1",
            "delta_accuracy_vs_initial",
            "delta_macro_f1_vs_initial",
        ]
    )
    for result in results:
        for model, conditions in result["metrics"].items():
            for condition, metrics in conditions.items():
                baseline = result["metrics"]["initial"][condition]
                writer.writerow(
                    [
                        result["domain"],
                        model,
                        model == result["selected_on_validation"],
                        condition,
                        metrics["labeled_count"],
                        metrics["accuracy"],
                        metrics["macro_f1"],
                        metrics["accuracy"] - baseline["accuracy"],
                        metrics["macro_f1"] - baseline["macro_f1"],
                    ]
                )
    (output / "test_comparison.csv").write_text(stream.getvalue(), encoding="utf-8", newline="")


def write_class_metrics(output, results):
    with (output / "class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "domain",
                "model",
                "condition",
                "class_index",
                "class_name",
                "support",
                "precision",
                "recall",
                "f1",
            ]
        )
        for result in results:
            for model, conditions in result["metrics"].items():
                for condition, metrics in conditions.items():
                    for index, name in enumerate(result["classes"]):
                        writer.writerow(
                            [
                                result["domain"],
                                model,
                                condition,
                                index,
                                name,
                                metrics["support"][index],
                                metrics["precision"][index],
                                metrics["recall"][index],
                                metrics["class_f1"][index],
                            ]
                        )


def write_plots(output, results):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    try:
        for row, result in enumerate(results):
            for column, metric in enumerate(("accuracy", "macro_f1")):
                axis = axes[row, column]
                for model, conditions in result["metrics"].items():
                    label = MODEL_LABELS[model]
                    if model == result["selected_on_validation"]:
                        label += " (fixed on Validation)"
                    axis.plot(
                        range(3),
                        [conditions[c][metric] for c in CONDITIONS],
                        marker="o",
                        label=label,
                    )
                axis.set_xticks(range(3), ["Original", "Lighting", "Expanded"])
                axis.set_ylim(0, 1.01)
                axis.set_title(f"{DOMAIN_LABELS[result['domain']]} · {metric}")
                axis.grid(alpha=0.2)
                axis.legend(fontsize=8)
        fig.suptitle("MediFlow · Original labeled Test comparison", fontsize=16)
        fig.savefig(plots / "test_comparison.png", dpi=180)
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), layout="constrained")
    try:
        for row, result in enumerate(results):
            model = result["selected_on_validation"]
            for column, condition in enumerate(CONDITIONS):
                axis = axes[row, column]
                matrix = np.asarray(result["metrics"][model][condition]["confusion_matrix"])
                axis.imshow(matrix, cmap="Blues", vmin=0)
                indexes = range(len(result["classes"]))
                axis.set_xticks(indexes)
                axis.set_yticks(indexes)
                axis.set_xlabel("Predicted class index")
                axis.set_ylabel("True class index")
                axis.set_title(
                    f"{DOMAIN_LABELS[result['domain']]} · {MODEL_LABELS[model]} · {condition}"
                )
                for i, j in np.ndindex(matrix.shape):
                    axis.text(
                        j,
                        i,
                        str(matrix[i, j]),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                    )
        fig.suptitle("Validation-fixed candidates · Class names in class_metrics.csv", fontsize=16)
        fig.savefig(plots / "selected_confusion_matrices.png", dpi=180)
    finally:
        plt.close(fig)


def run(runtime, stages, dataset_root, plan_path, output, work=None):
    plan = data.read_json(plan_path)
    config = data.read_json(runtime / "configs" / "adaptation_stage1.json")
    if training.object_hash(config) != plan["configuration_sha256"]:
        raise ValueError("Common bundle configuration differs from the locked plan")
    if digest(training.__file__) != plan["predictor_code_sha256"]:
        raise ValueError("Predictor differs from the Stage 3 validation code")
    identity = {"plan_sha256": digest(plan_path), "evaluator_sha256": digest(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "evaluation_plan.json"
    record = {**identity, "plan": plan}
    if marker.exists() and data.read_json(marker) != record:
        raise ValueError("Output belongs to another evaluation plan")
    write_progress(marker, record)
    code = output / "code"
    code.mkdir(exist_ok=True)
    for path in (Path(__file__), Path(training.__file__)):
        (code / path.name).write_bytes(path.read_bytes())
    results = []
    for domain in training.DOMAINS:
        task, lock = config["tasks"][domain], plan["domains"][domain]
        destination = output / f"{domain}.json"
        if destination.is_file():
            result = data.read_json(destination)
            if result["identity"] != identity:
                raise ValueError(f"{domain}: saved result identity mismatch")
            print(f"[{domain}] 동일 계획의 완료 결과 재사용", flush=True)
        else:
            source = find_source(dataset_root, task)
            with tempfile.TemporaryDirectory(dir=work) if work else nullcontext() as scratch:
                archive = source
                if scratch:
                    archive = Path(scratch) / "source.zip"
                    copy_source(source, archive)
                result = domain_result(
                    runtime, stages, domain, task, lock, plan["transform_seed"], archive
                )
            result.update(
                identity=identity,
                environment={
                    "python": platform.python_version(),
                    "tensorflow": tf.__version__,
                    "keras": keras.__version__,
                    "numpy": np.__version__,
                    "opencv": cv2.__version__,
                    "matplotlib": version("matplotlib"),
                    "gpu": [
                        tf.config.experimental.get_device_details(d)
                        for d in tf.config.list_physical_devices("GPU")
                    ],
                },
            )
            write_progress(destination, result)
            keras.backend.clear_session()
        results.append(result)
    write_comparison(output, results)
    write_class_metrics(output, results)
    write_plots(output, results)
    write_progress(
        output / "summary.json",
        {
            **identity,
            "completed_domains": [r["domain"] for r in results],
            "comparisons": 36,
            "source_test_evaluated": True,
            "real_device_accuracy": None,
        },
    )
    print(f"최종 Test 결과: {output / 'test_comparison.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--stages", type=Path, nargs=3, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    run(args.runtime, args.stages, args.datasets, args.plan, args.output, args.work)


if __name__ == "__main__":
    main()
