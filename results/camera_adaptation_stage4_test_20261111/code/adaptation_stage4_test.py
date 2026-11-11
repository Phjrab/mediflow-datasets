"""Evaluate Stage 4 only; reuse verified historical Test metrics without new inference."""

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

from mediflow_datasets import adaptation_final_test as original
from mediflow_datasets import adaptation_stage1 as training
from mediflow_datasets import camera_extended as extended
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as experiment
from mediflow_datasets.adaptation_stage1_colab import copy_source
from mediflow_datasets.domain_adaptation_batch import digest, find_source
from mediflow_datasets.domain_adaptation_io import write_progress

CONDITIONS = ("clean", "lighting", "expanded", "stage4_lighting", "stage4_expanded")
MODEL_LABELS = {**original.MODEL_LABELS, "stage4_revised": "Stage 4 (before refit)"}
CONDITION_LABELS = (
    "Original",
    "Old lighting",
    "Old expanded",
    "Stage 4 lighting",
    "Stage 4 expanded",
)


def evaluate_zip(archive_path, rows, task, spec, old_profile, stage4_profile, predictors, seed):
    """Keep the same pixel arrays, class order and item seed for every checkpoint."""
    scores = {name: {condition: [] for condition in CONDITIONS} for name in predictors}
    batch_size = task["batch_size"]
    with zipfile.ZipFile(archive_path) as archive:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs = {condition: [] for condition in CONDITIONS}
            for row in batch:
                image = experiment.image_tensor(
                    archive.read(row.path), spec["size"], task["input_policy"]
                )
                item_seed = experiment.item_seed(row, seed)
                inputs["clean"].append(image)
                for name, profile, mode in (
                    ("lighting", old_profile, "lighting"),
                    ("expanded", old_profile, "expanded"),
                    ("stage4_lighting", stage4_profile, "lighting"),
                    ("stage4_expanded", stage4_profile, "expanded"),
                ):
                    inputs[name].append(extended.transform(image, profile, mode, item_seed))
            for condition, arrays in inputs.items():
                tensor = tf.convert_to_tensor(np.stack(arrays), dtype=tf.float32)
                for name, predict in predictors.items():
                    result = np.asarray(predict(tensor).numpy(), dtype=np.float32)
                    if result.shape != (len(batch), len(task["classes"])):
                        raise ValueError(f"{name}: unexpected model output shape {result.shape}")
                    if not np.isfinite(result).all():
                        raise ValueError(f"{name}: nonfinite model output")
                    scores[name][condition].append(result)
            done = start + len(batch)
            if start == 0 or start % (batch_size * 20) == 0 or done == len(rows):
                training.show(
                    f"[{task['domain']}] Test 예측 · 모델 {len(predictors)}개 × 조건 5개",
                    done,
                    len(rows),
                )
    arrays = {
        name: {condition: np.concatenate(parts) for condition, parts in values.items()}
        for name, values in scores.items()
    }
    metrics = {
        name: {
            condition: experiment.classification(rows, values, len(task["classes"]))
            for condition, values in conditions.items()
        }
        for name, conditions in arrays.items()
    }
    return metrics, arrays


def verify_inputs(runtime, stages, plan, domain):
    lock = plan["domains"][domain]
    task = data.read_json(runtime / "configs/adaptation_stage1.json")["tasks"][domain]
    reference = data.read_json(runtime / "references" / f"{domain}.json")
    if training.object_hash(reference) != lock["reference_sha256"]:
        raise ValueError(f"{domain}: original camera profile changed")
    manifest = stages[0] / domain / "source_manifest.json"
    if digest(manifest) != lock["manifest_sha256"]:
        raise ValueError(f"{domain}: split manifest changed")
    study = data.read_json(stages[3] / "study.json")
    if (
        digest(stages[3] / "study.json") != plan["stage4_study_file_sha256"]
        or study["study_sha256"] != plan["stage4_study_sha256"]
        or study["study"]["domains"][domain]["profile"] != plan["stage4_profiles"][domain]
    ):
        raise ValueError(f"{domain}: expected completed pre-refit Stage 4 study")
    spec = task["models"][lock["start_role"]]
    if task["classes"] != lock["classes"] or spec != lock["model_spec"]:
        raise ValueError(f"{domain}: class order or input/model configuration changed")
    paths = {}
    for name, model in lock["models"].items():
        if name not in plan["evaluate_arms"]:
            continue
        root = runtime if model["stage"] == 0 else stages[model["stage"] - 1]
        path = data.local_path(root, model["path"])
        if digest(path) != model["sha256"]:
            raise ValueError(f"{domain}/{name}: checkpoint SHA-256 mismatch")
        paths[name] = path
    return task, spec, reference["extended_profile"], paths


def previous_result(runtime, output, plan, domain):
    """Keep the original result bytes and reject mismatched comparison inputs."""
    prior = plan["previous_test"]
    folder = output / "previous_test"
    folder.mkdir(exist_ok=True)
    for name in ("evaluation_plan.json", "summary.json", f"{domain}.json"):
        source = runtime / "previous_test" / name
        expected = prior["files"][name]
        if digest(source) != expected:
            raise ValueError(f"Previous Test result bytes changed: {name}")
        destination = folder / name
        if destination.exists() and digest(destination) != expected:
            raise ValueError(f"Saved previous Test result bytes changed: {name}")
        if not destination.exists():
            destination.write_bytes(source.read_bytes())
    recorded_plan = data.read_json(folder / "evaluation_plan.json")
    result = data.read_json(folder / f"{domain}.json")
    lock = plan["domains"][domain]
    if (
        recorded_plan["plan_sha256"] != prior["plan_sha256"]
        or result["identity"]["plan_sha256"] != prior["plan_sha256"]
        or result["classes"] != lock["classes"]
        or result["test_count"] != lock["test_count"]
        or result["source_sha256"] != lock["source_sha256"]
        or result["reference_sha256"] != lock["reference_sha256"]
        or result["input_policy"] != plan["input_policies"][domain]
        or recorded_plan["plan"]["transform_seed"] != plan["transform_seed"]
        or recorded_plan["plan"]["domains"][domain]["manifest_sha256"] != lock["manifest_sha256"]
        or set(result["metrics"]) != set(original.MODEL_LABELS)
        or any(set(values) != set(original.CONDITIONS) for values in result["metrics"].values())
    ):
        raise ValueError(f"{domain}: previous Test conditions or data changed")
    return result


def combine_results(fresh, prior):
    return {
        **fresh,
        "metrics": {**prior["metrics"], **fresh["metrics"]},
        "metric_origins": {
            **dict.fromkeys(prior["metrics"], "reused_previous_test"),
            **dict.fromkeys(fresh["metrics"], "new_stage4_inference"),
        },
    }


def write_comparison(output, results):
    with (output / "test_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "domain",
                "model",
                "measurement_source",
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
            for name, conditions in result["metrics"].items():
                for condition, metrics in conditions.items():
                    baseline = result["metrics"]["initial"].get(condition)
                    writer.writerow(
                        [
                            result["domain"],
                            name,
                            result["metric_origins"][name],
                            name == result["selected_on_validation"],
                            condition,
                            metrics["labeled_count"],
                            metrics["accuracy"],
                            metrics["macro_f1"],
                            metrics["accuracy"] - baseline["accuracy"] if baseline else "",
                            metrics["macro_f1"] - baseline["macro_f1"] if baseline else "",
                        ]
                    )


def domain_result(runtime, stages, plan, domain, archive, output):
    task, spec, old_profile, paths = verify_inputs(runtime, stages, plan, domain)
    lock = plan["domains"][domain]
    rows = original.prepare_test(archive, task, lock, stages[0] / domain / "source_manifest.json")
    predictors = {}
    for name, path in paths.items():
        print(f"[{domain}] {MODEL_LABELS[name]} 모델 로딩", flush=True)
        predictors[name] = training.predictor(keras.models.load_model(path, compile=False), spec)
    metrics, scores = evaluate_zip(
        archive,
        rows,
        task,
        spec,
        old_profile,
        plan["stage4_profiles"][domain],
        predictors,
        plan["transform_seed"],
    )
    predictions = output / f"{domain}_predictions.npz"
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        **{
            f"{name}__{condition}": values
            for name, c in scores.items()
            for condition, values in c.items()
        },
        labels=np.asarray([row.label for row in rows], dtype=np.int32),
    )
    predictions.write_bytes(payload.getvalue())
    row_path = output / f"{domain}_test_rows.json"
    write_progress(row_path, [{"path": row.path, "label": row.label} for row in rows])
    return {
        "domain": domain,
        "classes": task["classes"],
        "test_count": len(rows),
        "source_sha256": task["source_sha256"],
        "reference_sha256": lock["reference_sha256"],
        "stage4_study_sha256": plan["stage4_study_sha256"],
        "input_policy": task["input_policy"],
        "selected_on_validation": lock["selected_on_validation"],
        "selection_scope": plan["selection_scope"],
        "model_sha256": {name: value["sha256"] for name, value in lock["models"].items()},
        "prediction_files": {p.name: digest(p) for p in (predictions, row_path)},
        "metrics": metrics,
    }


def write_plots(output, results):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), layout="constrained")
    try:
        for row, result in enumerate(results):
            for column, metric in enumerate(("accuracy", "macro_f1")):
                axis = axes[row, column]
                for name, conditions in result["metrics"].items():
                    axis.plot(
                        range(5),
                        [conditions.get(c, {}).get(metric, np.nan) for c in CONDITIONS],
                        marker="o",
                        label=MODEL_LABELS[name],
                    )
                axis.set_xticks(range(5), CONDITION_LABELS, rotation=15, ha="right")
                axis.set_ylim(0, 1.01)
                axis.set_title(f"{original.DOMAIN_LABELS[result['domain']]} · {metric}")
                axis.legend(fontsize=7)
                axis.grid(alpha=0.2)
        fig.suptitle("Stage 4 Test + reused historical results (missing conditions not measured)")
        fig.savefig(plots / "test_comparison.png", dpi=180)
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), layout="constrained")
    try:
        for row, result in enumerate(results):
            selected = "stage4_revised"
            for column, condition in enumerate(("clean", "expanded", "stage4_expanded")):
                axis = axes[row, column]
                matrix = np.asarray(result["metrics"][selected][condition]["confusion_matrix"])
                axis.imshow(matrix, cmap="Blues", vmin=0)
                axis.set_xticks(range(len(matrix)))
                axis.set_yticks(range(len(matrix)))
                axis.set_xlabel("Predicted class index")
                axis.set_ylabel("True class index")
                axis.set_title(f"{result['domain']} · {MODEL_LABELS[selected]} · {condition}")
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
        fig.suptitle("New Stage 4 Test · Class names in class_metrics.csv")
        fig.savefig(plots / "stage4_confusion_matrices.png", dpi=180)
    finally:
        plt.close(fig)


def run(runtime, stages, dataset_root, plan_path, output, work=None):
    plan = data.read_json(plan_path)
    config = data.read_json(runtime / "configs/adaptation_stage1.json")
    if training.object_hash(config) != plan["configuration_sha256"]:
        raise ValueError("Common bundle configuration differs from locked plan")
    for name, expected in plan["runtime_sha256"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError(f"Evaluation code changed: {name}")
    identity = {"plan_sha256": digest(plan_path), "evaluator_sha256": digest(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "evaluation_plan.json"
    record = {**identity, "plan": plan}
    if marker.exists() and data.read_json(marker) != record:
        raise ValueError("Output belongs to another evaluation plan")
    write_progress(marker, record)
    code = output / "code"
    code.mkdir(exist_ok=True)
    for name in plan["runtime_sha256"]:
        (code / name).write_bytes((Path(__file__).parent / name).read_bytes())
    results = []
    for domain in training.DOMAINS:
        verify_inputs(runtime, stages, plan, domain)
        prior = previous_result(runtime, output, plan, domain)
        destination = output / f"{domain}.json"
        if destination.exists():
            result = data.read_json(destination)
            if result["identity"] != identity or any(
                digest(output / name) != expected
                for name, expected in result["prediction_files"].items()
            ):
                raise ValueError(f"{domain}: saved evaluation identity or prediction file changed")
            print(f"[{domain}] 동일 계획의 완료 결과 재사용", flush=True)
        else:
            source = find_source(dataset_root, config["tasks"][domain])
            with tempfile.TemporaryDirectory(dir=work) if work else nullcontext() as scratch:
                archive = source
                if scratch:
                    archive = Path(scratch) / "source.zip"
                    copy_source(source, archive)
                result = domain_result(runtime, stages, plan, domain, archive, output)
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
        results.append(combine_results(result, prior))
    write_comparison(output, results)
    original.write_class_metrics(output, results)
    write_plots(output, results)
    write_progress(
        output / "summary.json",
        {
            **identity,
            "completed_domains": [r["domain"] for r in results],
            "comparisons": sum(
                len(r["metrics"][arm]) for r in results for arm in plan["evaluate_arms"]
            ),
            "new_inference_arms": plan["evaluate_arms"],
            "reused_comparisons": 36,
            "combined_comparisons": sum(len(c) for r in results for c in r["metrics"].values()),
            "source_test_evaluated": True,
            "real_device_accuracy": None,
            "prior_source_test_inspected": True,
            "training_executed": False,
            "test_used_for_model_selection": False,
        },
    )
    print(f"완료: {output / 'test_comparison.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--stages", type=Path, nargs=4, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    run(args.runtime, args.stages, args.datasets, args.plan, args.output, args.work)


if __name__ == "__main__":
    main()
