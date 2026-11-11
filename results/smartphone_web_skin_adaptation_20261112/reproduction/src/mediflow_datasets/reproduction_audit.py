"""Independently check saved final-Test predictions and selected model files.

This verifies stored artifacts. It does not rerun image decoding or model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

PREDICTION_SOURCES = {
    "skin": "skin/experiments/comparison_20260909_075056_72d865bf/final_test_predictions.csv",
    "web_skin": (
        "web_skin/candidates/public_candidate_v2_pmg_b0_256_ce_20260922_235840_093d10de/"
        "final_test_predictions.csv"
    ),
    "hair": (
        "hair/candidates/public_candidate_v2_b1_384_ls005_adam_20260921_155906_82311da4/"
        "final_test_predictions.csv"
    ),
}

METRIC_SOURCES = {
    "skin": "skin/experiments/comparison_20260909_075056_72d865bf/final_test_metrics.json",
    "web_skin": (
        "web_skin/candidates/public_candidate_v2_pmg_b0_256_ce_20260922_235840_093d10de/"
        "final_test_metrics.json"
    ),
    "hair": (
        "hair/candidates/public_candidate_v2_b1_384_ls005_adam_20260921_155906_82311da4/"
        "final_test_metrics.json"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_prediction_record(
    prediction_path: Path, metrics_path: Path, class_names: list[str]
) -> dict[str, object]:
    count = len(class_names)
    truth: list[int] = []
    predicted: list[int] = []
    seen_paths: set[str] = set()
    with prediction_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        expected = {"path", "true_index", "pred_index"} | {
            f"prob_C{i}" for i in range(count)
        }
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Unexpected prediction columns: {prediction_path}")
        for row in reader:
            image_path = row["path"].replace("\\", "/")
            if image_path in seen_paths:
                raise ValueError(f"Duplicate prediction path: {image_path}")
            seen_paths.add(image_path)
            target = int(row["true_index"])
            guess = int(row["pred_index"])
            if not (0 <= target < count and 0 <= guess < count):
                raise ValueError(f"Class index outside range: {image_path}")
            parts = image_path.split("/")
            if len(parts) < 3 or parts[-2] != class_names[target]:
                raise ValueError(f"Path class does not match true_index: {image_path}")
            probabilities = np.array([float(row[f"prob_C{i}"]) for i in range(count)])
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
                raise ValueError(f"Invalid model output: {image_path}")
            if not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
                raise ValueError(f"Model output does not sum to one: {image_path}")
            if int(probabilities.argmax()) != guess:
                raise ValueError(f"pred_index does not match model output: {image_path}")
            truth.append(target)
            predicted.append(guess)
    if not truth:
        raise ValueError(f"No prediction rows: {prediction_path}")

    confusion = np.zeros((count, count), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    actual = confusion.sum(axis=1)
    predicted_total = confusion.sum(axis=0)
    diagonal = confusion.diagonal()
    precision = np.divide(
        diagonal, predicted_total, out=np.zeros(count, dtype=float), where=predicted_total != 0
    )
    recall = np.divide(diagonal, actual, out=np.zeros(count, dtype=float), where=actual != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(count, dtype=float),
        where=(precision + recall) != 0,
    )
    calculated = {
        "accuracy": float(diagonal.sum() / len(truth)),
        "macro_f1": float(f1.mean()),
        "class_f1": f1.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "support": actual.tolist(),
        "confusion_matrix": confusion.tolist(),
        "count": len(truth),
    }
    reported = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key, value in calculated.items():
        if key not in reported or not np.allclose(
            np.asarray(value, dtype=float),
            np.asarray(reported[key], dtype=float),
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(f"Reported {key} differs from saved predictions: {metrics_path}")
    return calculated


def audit_initial_reports(repository_root: Path) -> dict[str, object]:
    """Cross-check the six historical summaries without claiming fresh inference."""

    root = repository_root / "results"
    audited: dict[str, object] = {}
    expected_counts = {"skin": 700, "web_skin": 400, "hair": 1450}
    for domain in ("skin", "web_skin", "hair"):
        for variant in ("original", "augmented"):
            directory = root / domain / "1_training" / variant
            if domain == "hair":
                source = directory / "training_config.json"
                summary = json.loads(source.read_text(encoding="utf-8"))
                classes = summary["class_names"]
            else:
                source = directory / "results.json"
                summary = json.loads(source.read_text(encoding="utf-8"))
                classes = summary["classes"]
            report_path = directory / (
                "classification_report.txt" if domain == "skin" else "classification_report.csv"
            )
            if domain == "skin":
                rows = {}
                for line in report_path.read_text(encoding="utf-8").splitlines():
                    match = re.match(
                        r"^\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", line
                    )
                    if match and match.group(1) in classes:
                        rows[match.group(1)] = {
                            "recall": float(match.group(3)),
                            "f1": float(match.group(4)),
                            "support": int(match.group(5)),
                        }
            else:
                with report_path.open(newline="", encoding="utf-8-sig") as stream:
                    report_rows = list(csv.DictReader(stream))
                label_key = "" if domain == "hair" else "class"
                f1_key = "f1-score" if domain == "hair" else "f1_score"
                rows = {
                    row[label_key]: {
                        "recall": float(row["recall"]),
                        "f1": float(row[f1_key]),
                        "support": int(float(row["support"])),
                    }
                    for row in report_rows
                    if row[label_key] in classes
                }
            if set(rows) != set(classes):
                raise ValueError(f"Historical report classes differ: {directory}")
            total = sum(rows[name]["support"] for name in classes)
            if total != expected_counts[domain]:
                raise ValueError(f"Historical Test count differs: {directory}")
            accuracy_from_report = sum(
                rows[name]["recall"] * rows[name]["support"] for name in classes
            ) / total
            # Skin's text report is rounded to four places; Web Skin's Keras
            # accuracy was stored as float32 while CSV recalls are full precision.
            tolerance = 1e-4 if domain == "skin" else 1e-6
            if not np.isclose(
                accuracy_from_report, summary["test_accuracy"], atol=tolerance, rtol=0
            ):
                raise ValueError(f"Historical accuracy and class report disagree: {directory}")
            macro_f1_from_report = sum(rows[name]["f1"] for name in classes) / len(classes)
            if "macro_f1" in summary and not np.isclose(
                macro_f1_from_report, summary["macro_f1"], atol=tolerance, rtol=0
            ):
                raise ValueError(f"Historical Macro F1 and class report disagree: {directory}")
            model_path = directory / "best_model.keras"
            audited[f"{domain}_{variant}"] = {
                "model": str(model_path.relative_to(root)),
                "model_sha256": sha256_file(model_path),
                "summary_file": str(source.relative_to(root)),
                "class_report_file": str(report_path.relative_to(root)),
                "class_order": classes,
                "reported_test_accuracy": summary["test_accuracy"],
                "reported_test_macro_f1": summary.get("macro_f1"),
                "test_image_count_from_report": total,
                "accuracy_from_class_report": accuracy_from_report,
                "macro_f1_from_class_report": macro_f1_from_report,
                "report_precision": "4 decimals" if domain == "skin" else "full saved precision",
            }
    return audited


def audit_repository(repository_root: Path) -> dict[str, object]:
    results = repository_root / "results"
    index = json.loads((results / "CANDIDATE_INDEX.json").read_text(encoding="utf-8"))
    audited: dict[str, object] = {}
    for domain in ("skin", "web_skin", "hair"):
        candidate = index["candidates"][domain]
        model = results / candidate["model"]
        if sha256_file(model) != candidate["model_sha256"]:
            raise ValueError(f"Selected model SHA-256 mismatch: {domain}")
        classes_path = results / candidate["selected_directory"] / "class_names.json"
        classes = json.loads(classes_path.read_text(encoding="utf-8"))
        if not isinstance(classes, list) or not classes:
            raise ValueError(f"Invalid class order: {classes_path}")
        prediction_path = results / PREDICTION_SOURCES[domain]
        metrics_path = results / METRIC_SOURCES[domain]
        metrics = check_prediction_record(prediction_path, metrics_path, classes)
        for key in ("accuracy", "macro_f1"):
            if not np.isclose(
                metrics[key], candidate[f"test_{key}"], atol=1e-12, rtol=0
            ):
                raise ValueError(f"Candidate index {key} mismatch: {domain}")
        audited[domain] = {
            "selected_version": candidate["selected_version"],
            "model": candidate["model"],
            "model_sha256": candidate["model_sha256"],
            "class_order": classes,
            "prediction_file": PREDICTION_SOURCES[domain],
            "prediction_sha256": sha256_file(prediction_path),
            "metrics_file": METRIC_SOURCES[domain],
            "metrics_sha256": sha256_file(metrics_path),
            "image_count": metrics["count"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "support": metrics["support"],
            "confusion_matrix": metrics["confusion_matrix"],
        }
    return {
        "audit_type": "saved_prediction_and_model_integrity",
        "new_model_inference_performed": False,
        "original_six_model_test_reproduced": False,
        "reason": (
            "Original pre-cleaning Test splits are unavailable; cleaned splits are different."
        ),
        "historical_initial_reports": audit_initial_reports(repository_root),
        "candidates": audited,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = audit_repository(args.repository_root.resolve())
    rendered = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
