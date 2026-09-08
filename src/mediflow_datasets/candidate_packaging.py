"""Preserve and package the already selected Web Skin candidate, without training."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

CLASSES = ["건선", "아토피", "여드름", "정상", "주사"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def within(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Package path escapes its root")
    return path


def verify_source(source, policy):
    """Only accept the report bytes reviewed before packaging was requested."""
    source = Path(source)
    for relative, expected in policy["report_hashes"].items():
        if sha256_file(within(source, relative)) != expected:
            raise ValueError("Reviewed report changed: " + relative)
    selection = read_json(source / "selection_before_test.json")
    config = read_json(source / "suite_config.json")
    card = read_json(source / "model_card.json")
    records = read_json(source / "all_validation_results.json")
    winner = max(records, key=lambda r: r["validation"]["accuracy"])
    if (
        selection["winner"] != "b0_256_ce"
        or winner["id"] != selection["winner"]
        or winner["validation"] != selection["validation"]
        or card["model_sha256"] != policy["model_sha256"]
        or selection["model_sha256"] != policy["model_sha256"]
        or card["model_relative_path"] != policy["model_relative_path"]
        or config["settings"]["class_names"] != CLASSES
        or card["class_names"] != CLASSES
        or config["settings"]["data_sha256"] != policy["data_sha256"]
        or card["test"] != read_json(source / "final_test_metrics.json")
    ):
        raise ValueError("Source candidate contract mismatch")
    model_path = within(source, policy["model_relative_path"])
    if sha256_file(model_path) != policy["model_sha256"]:
        raise ValueError("Selected model hash differs from the reviewed candidate")
    return model_path, card


def check_model_contract(path):
    model = keras.models.load_model(path, compile=False)
    if (
        tuple(model.input_shape) != (None, 256, 256, 3)
        or tuple(model.output_shape) != (None, 5)
        or model.count_params() != 4055976
    ):
        raise ValueError("Model shape or parameter count mismatch")
    bases = [layer for layer in model.layers if isinstance(layer, keras.Model)]
    if len(bases) != 1 or "efficientnetb0" not in bases[0].name.lower():
        raise ValueError("Expected EfficientNet-B0")
    rescaling = [layer for layer in bases[0].layers if isinstance(layer, keras.layers.Rescaling)]
    if not any(
        np.asarray(layer.scale).size == 1
        and np.isclose(float(np.asarray(layer.scale).item()), 1 / 255)
        and np.allclose(layer.offset, 0)
        for layer in rescaling
    ):
        raise ValueError("Internal Rescaling(1/255) missing")
    values = np.broadcast_to(
        np.array([0, 127.5, 255], dtype="float32")[:, None, None, None], (3, 256, 256, 3)
    ).copy()
    scores = np.asarray(model(values, training=False))
    if (
        scores.shape != (3, 5)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
        or not np.allclose(scores.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Model output check failed")
    return {
        "input_shape": [256, 256, 3],
        "output_count": 5,
        "parameters": model.count_params(),
        "dummy_forward_passed": True,
        "note": "Artificial inputs test execution only; no new performance evaluation",
    }


def build_package(source, output_parent, policy, packaging_source=None):
    """Build locally; source files are immutable and no test images are loaded."""
    source = Path(source)
    model_path, card = verify_source(source, policy)
    contract = check_model_contract(model_path)
    name = (
        "public_candidate_v1_b0_256_ce_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    package = Path(output_parent) / name
    package.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(model_path, package / "web_skin_model.keras")
    if sha256_file(package / "web_skin_model.keras") != policy["model_sha256"]:
        raise ValueError("Copied model hash mismatch")
    reports = package / "source_reports"
    reports.mkdir()
    write_json(package / "packaging_policy.json", policy)
    if packaging_source is not None:
        (package / "packaging_source.py").write_text(packaging_source, encoding="utf-8")
    for relative, expected in policy["report_hashes"].items():
        target = within(reports, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(within(source, relative), target)
        if sha256_file(target) != expected:
            raise ValueError("Copied report hash mismatch: " + relative)
    write_json(package / "class_names.json", CLASSES)
    write_json(
        package / "preprocessing.json",
        {
            "input_shape": [256, 256, 3],
            "color_order": "RGB",
            "input_dtype": "float32",
            "input_pixel_range": [0, 255],
            "external_normalization": False,
            "internal_rescaling": "1/255",
            "resize": "TensorFlow bilinear, antialias=False",
            "aspect_ratio": "resize to 256x256, no crop/pad",
            "exif_transpose": False,
            "output": "5 softmax scores in class_names.json order; not calibrated correctness",
        },
    )
    write_json(package / "model_contract_check.json", contract)
    manifest = {
        "status": "public_data_candidate_v1_not_device_validated",
        "domain": "web_skin",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_suite": source.name,
        "model_file": "web_skin_model.keras",
        "model_sha256": policy["model_sha256"],
        "model_size_bytes": (package / "web_skin_model.keras").stat().st_size,
        "model_parameter_count": contract["parameters"],
        "backbone": "EfficientNet-B0",
        "image_size": [256, 256],
        "loss": "Categorical Crossentropy",
        "stage1_epochs": 15,
        "stage2_epochs_run": 10,
        "selected_stage2_epoch": 8,
        "selection_metric": "validation_accuracy; ties keep earlier experiment",
        "validation_accuracy": card["validation"]["accuracy"],
        "test_accuracy": card["test"]["accuracy"],
        "macro_f1": card["test"]["macro_f1"],
        "data_zip_sha256": policy["data_sha256"],
        "normal_class_included": True,
        "class_names_file": "class_names.json",
        "preprocessing_file": "preprocessing.json",
        "validation": card["validation"],
        "test": card["test"],
        "limitations": card["limitations"],
        "new_training_or_test_evaluation": False,
        "packaging_environment": {
            "python": platform.python_version(),
            "keras": keras.__version__,
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
        },
    }
    (package / "MODEL_CARD.md").write_text(
        "# MediFlow Web Skin 공개 데이터 후보 v1\n\n"
        "EfficientNet-B0 / 256 / CE. 실제 웹캠 검증 전 공개 데이터 후보입니다.\n\n"
        f"클래스 순서: {CLASSES}\n\n"
        f"Validation Accuracy: {manifest['validation_accuracy']}\n\n"
        f"Test Accuracy: {manifest['test_accuracy']}\n\n"
        f"Test Macro F1: {manifest['macro_f1']}\n\n"
        "위 수치는 완료된 실험의 보고값을 그대로 보존한 것으로 새 평가가 아닙니다.\n"
        "정상 클래스는 포함하지만 범위 밖 입력 거부 기능은 없습니다.\n"
        "사람·병변·촬영 세션 누수 및 실제 웹캠 성능은 미검증입니다.\n"
        "입력은 얼굴 정면 RGB float32 0~255이며 외부 /255를 적용하지 않습니다.\n"
        "모델 출력은 보정되지 않은 점수입니다. 자세한 계약은 preprocessing.json을 보세요.\n",
        encoding="utf-8",
    )
    manifest["artifact_hashes"] = {
        p.relative_to(package).as_posix(): sha256_file(p)
        for p in sorted(package.rglob("*"))
        if p.is_file()
    }
    write_json(package / "package_manifest.json", manifest)
    archive_path = package.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(package.rglob("*")):
            if p.is_file():
                archive.write(p, arcname=name + "/" + p.relative_to(package).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Package ZIP CRC failure")
        expected = {
            **manifest["artifact_hashes"],
            "package_manifest.json": sha256_file(package / "package_manifest.json"),
        }
        for relative, digest in expected.items():
            with archive.open(name + "/" + relative) as stream:
                actual = hashlib.sha256()
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    actual.update(chunk)
                if actual.hexdigest() != digest:
                    raise ValueError("ZIP content hash mismatch: " + relative)
    return package, archive_path


def publish_package(package, archive_path, drive_parent):
    """Copy only verified outputs, preserving every existing Drive package."""
    package, archive_path, drive_parent = map(Path, (package, archive_path, drive_parent))
    drive_parent.mkdir(parents=True, exist_ok=True)
    target = drive_parent / package.name
    destination = drive_parent / archive_path.name
    if target.exists() or destination.exists():
        raise FileExistsError("Destination already exists; nothing overwritten")
    shutil.copytree(package, target)
    for path in package.rglob("*"):
        if path.is_file() and sha256_file(path) != sha256_file(target / path.relative_to(package)):
            raise ValueError("Drive package copy verification failed")
    with archive_path.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    digest = sha256_file(archive_path)
    if sha256_file(destination) != digest:
        raise ValueError("Drive ZIP copy verification failed")
    with destination.with_suffix(".zip.sha256").open("x", encoding="utf-8") as stream:
        stream.write(digest + "  " + destination.name + "\n")
    return target, destination, digest
