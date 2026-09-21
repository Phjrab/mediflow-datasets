"""One-time Test evaluation and packaging of the fixed Hair B1/384 candidate."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np

from mediflow_datasets import common_engine as engine
from mediflow_datasets import common_workflow as flow

PROTOCOL = "hair_b1_384_final_candidate_v1"
PARENT_TRIAL = "multires_b1_384_seed_42"
MODEL_NAME = "stage2_best.keras"


def validate(context):
    config = context["config"]
    if config["domain"] != "hair" or config["mode"] != "final_candidate":
        raise ValueError("이 노트북은 Hair B1·384 최종평가 전용입니다.")
    if config["train_variant"] != "augmented":
        raise ValueError("고정된 후보와 같은 augmented 데이터 계약이 필요합니다.")
    if config["seed"] != 42 or config["seeds"] != [42]:
        raise ValueError("고정된 후보의 seed 계약이 다릅니다.")
    if not isinstance(config.get("parent_run_dir"), str) or not config[
        "parent_run_dir"
    ].strip():
        raise ValueError("PARENT_RUN_DIR을 입력하세요.")
    digest = config.get("parent_model_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in digest)
    ):
        raise ValueError("PARENT_MODEL_SHA256은 64자리 SHA-256이어야 합니다.")


def _parent(context):
    config = context["config"]
    parent = Path(config["parent_run_dir"])
    marker = parent / PARENT_TRIAL / "completed.json"
    if not marker.is_file():
        raise FileNotFoundError("B1·384 완료 기록을 찾을 수 없습니다: " + str(marker))
    record = engine.read_json(marker)
    if (
        record["id"] != PARENT_TRIAL
        or record["selected_stage"] != "stage2"
        or record["selected_model"] != MODEL_NAME
        or record.get("test_evaluated", False) is not False
    ):
        raise ValueError("고정한 B1·384 Validation 후보 기록과 다릅니다.")
    model = marker.parent / record["attempt"] / MODEL_NAME
    expected = config["parent_model_sha256"].lower()
    actual = engine.file_hash(model)
    relative = record["attempt"] + "/" + MODEL_NAME
    if actual != expected or record.get("artifact_hashes", {}).get(relative) != actual:
        raise ValueError("B1·384 최종 후보 모델 SHA-256이 다릅니다.")
    settings = engine.read_json(parent / "run_config.json")["settings"]
    if (
        settings["data_sha256"] != context["data_hash"]
        or settings["classes"] != context["classes"]
        or settings["seed"] != 42
    ):
        raise ValueError("부모 실행의 데이터·클래스·seed가 현재 실행과 다릅니다.")
    return record, model, actual


def _model_contract(model):
    if (
        tuple(model.input_shape) != (None, 384, 384, 3)
        or tuple(model.output_shape) != (None, 5)
        or model.count_params() != 6_581_644
    ):
        raise ValueError("B1·384 모델 입출력 또는 파라미터 계약이 다릅니다.")
    backbones = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model) and "efficientnetb1" in layer.name.lower()
    ]
    if len(backbones) != 1:
        raise ValueError("EfficientNet-B1 Backbone 하나가 필요합니다.")
    rescaling = [
        layer for layer in backbones[0].layers if isinstance(layer, keras.layers.Rescaling)
    ]
    if not any(
        np.asarray(layer.scale).size == 1
        and np.isclose(float(np.asarray(layer.scale).item()), 1 / 255)
        and np.allclose(layer.offset, 0)
        for layer in rescaling
    ):
        raise ValueError("모델 내부 Rescaling(1/255)을 찾지 못했습니다.")
    values = np.broadcast_to(
        np.array([0, 127.5, 255], dtype="float32")[:, None, None, None],
        (3, 384, 384, 3),
    ).copy()
    scores = np.asarray(model(values, training=False))
    if (
        scores.shape != (3, 5)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
        or not np.allclose(scores.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("모델 출력 계약 검사가 실패했습니다.")
    return {
        "input_shape": [384, 384, 3],
        "output_count": 5,
        "parameters": model.count_params(),
        "dummy_forward_passed": True,
    }


def _evaluate_once(context, record, model_path, model_hash):
    output = Path(context["output"])
    selection = {
        "protocol": PROTOCOL,
        "winner": PARENT_TRIAL,
        "parent_attempt": record["attempt"],
        "parent_selected_model": MODEL_NAME,
        "model_sha256": model_hash,
        "validation": record["validation"],
        "test_was_unread_at_selection": True,
    }
    selection_path = output / "selection_before_test.json"
    if selection_path.exists() and engine.read_json(selection_path) != selection:
        raise ValueError("이미 고정한 후보와 현재 후보가 다릅니다.")
    if not selection_path.exists():
        engine.write_json(selection_path, selection)
    marker = output / "test_completed.json"
    if marker.exists():
        completed = engine.read_json(marker)
        if completed["selection"] != selection:
            raise ValueError("기존 Test 평가의 후보가 다릅니다.")
        for name, digest in completed["hashes"].items():
            if engine.file_hash(output / name) != digest:
                raise ValueError("저장된 Test 결과가 변경됐습니다: " + name)
        return completed["metrics"], completed["model_contract"]
    keras.backend.clear_session()
    model = keras.models.load_model(model_path, compile=False)
    contract = _model_contract(model)
    test, paths = flow.factory(context, "augmented", seed=42)("test", 384, False)
    metrics = engine.evaluate_to_files(model, test, paths, output, "final_test")
    del model
    keras.backend.clear_session()
    names = ("final_test_metrics.json", "final_test_predictions.csv")
    engine.write_json(
        marker,
        {
            "selection": selection,
            "metrics": metrics,
            "model_contract": contract,
            "hashes": {name: engine.file_hash(output / name) for name in names},
        },
    )
    return metrics, contract


def _figures(context, metrics):
    import matplotlib.pyplot as plt

    output = Path(context["output"])
    fig, ax = plt.subplots(figsize=(8, 8))
    flow.confusion(ax, metrics, "Hair B1·384 Adam / Final Test")
    fig.tight_layout()
    fig.savefig(output / "final_test_confusion_matrix.png", dpi=180)
    plt.show()
    plt.close(fig)
    flow.errors(
        context,
        output / "final_test_predictions.csv",
        output / "final_test_errors.png",
    )


def _package(context, record, model_path, model_hash, metrics, contract):
    output = Path(context["output"])
    marker_path = output / "package_completed.json"
    if marker_path.exists():
        marker = engine.read_json(marker_path)
        for path_text, digest in marker["hashes"].items():
            if engine.file_hash(Path(path_text)) != digest:
                raise ValueError("완료된 후보 패키지가 변경됐습니다: " + path_text)
        return Path(marker["package"]), Path(marker["archive"])
    card = {
        "protocol": PROTOCOL,
        "status": "public_data_candidate_not_device_validated",
        "domain": "hair",
        "class_names": context["classes"],
        "normal_class_included": False,
        "backbone": "EfficientNet-B1",
        "input_size": [384, 384],
        "loss": "Categorical Crossentropy with Label Smoothing 0.05",
        "optimizer": "Adam",
        "stage1_epochs": 15,
        "stage2_epochs": 15,
        "selected_model": PARENT_TRIAL,
        "model_sha256": model_hash,
        "model_parameter_count": contract["parameters"],
        "validation": record["validation"],
        "test": metrics,
        "data_zip_sha256": context["data_hash"],
        "limitations": context["audit"]["limitations"]
        + [
            "Actual USB microscope patient images were not evaluated",
            "No normal or out-of-scope rejection class",
            "Softmax scores are not calibrated correctness probabilities",
        ],
    }
    engine.write_json(output / "model_card.json", card)
    package_parent = (
        context["project"] / "2_results" / "hair" / "selected_models"
    )
    suffix = output.name.removeprefix("final_candidate_")
    name = "public_candidate_v2_b1_384_ls005_adam_" + suffix
    package = package_parent / name
    package.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(model_path, package / "hair_model.keras")
    if engine.file_hash(package / "hair_model.keras") != model_hash:
        raise OSError("후보 모델 복사 해시가 다릅니다.")
    for filename in (
        "class_names.json",
        "audit_summary.json",
        "run_config.json",
        "selection_before_test.json",
        "test_completed.json",
        "final_test_metrics.json",
        "final_test_predictions.csv",
        "final_test_confusion_matrix.png",
        "final_test_errors.png",
        "model_card.json",
        "common_engine.py",
        "common_audit.py",
        "common_workflow.py",
        "hair_final.py",
    ):
        shutil.copyfile(output / filename, package / filename)
    preprocessing = {
        "input_shape": [384, 384, 3],
        "color_order": "RGB",
        "input_dtype": "float32",
        "input_pixel_range": [0, 255],
        "external_normalization": False,
        "internal_rescaling": "1/255",
        "resize": "TensorFlow bilinear, antialias=False",
        "aspect_ratio": "resize to 384x384, no crop/pad",
        "exif_transpose": False,
        "output": "5 softmax scores in class_names.json order",
    }
    engine.write_json(package / "preprocessing.json", preprocessing)
    engine.write_json(package / "model_contract_check.json", contract)
    (package / "MODEL_CARD.md").write_text(
        "# MediFlow Hair 공개 데이터 후보 v2\n\n"
        "EfficientNet-B1 / 384 / Label Smoothing 0.05 / Adam.\n\n"
        f"클래스 순서: {context['classes']}\n\n"
        f"Validation Accuracy: {record['validation']['accuracy']}\n\n"
        f"Test Accuracy: {metrics['accuracy']}\n\n"
        f"Test Macro F1: {metrics['macro_f1']}\n\n"
        "입력은 두피 RGB float32 0~255를 384×384로 resize합니다. 외부 /255는 금지합니다.\n"
        "정상 및 범위 밖 입력 거부 기능과 실제 USB 현미경 환자 검증은 포함되지 않습니다.\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_file": "hair_model.keras",
        "model_sha256": model_hash,
        "source_run": Path(context["config"]["parent_run_dir"]).name,
        "source_trial": PARENT_TRIAL,
        "validation": record["validation"],
        "test": metrics,
        "data_zip_sha256": context["data_hash"],
        "test_evaluated_in_this_run": True,
        "artifact_hashes": {
            path.relative_to(package).as_posix(): engine.file_hash(path)
            for path in sorted(package.rglob("*"))
            if path.is_file()
        },
    }
    engine.write_json(package / "package_manifest.json", manifest)
    archive_path = package.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, name + "/" + path.relative_to(package).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise OSError("후보 ZIP CRC 검사가 실패했습니다.")
        expected = {
            **manifest["artifact_hashes"],
            "package_manifest.json": engine.file_hash(package / "package_manifest.json"),
        }
        for relative, digest in expected.items():
            actual = hashlib.sha256(archive.read(name + "/" + relative)).hexdigest()
            if actual != digest:
                raise OSError("후보 ZIP 내용 해시가 다릅니다: " + relative)
    archive_hash = engine.file_hash(archive_path)
    archive_path.with_suffix(".zip.sha256").write_text(archive_hash, encoding="ascii")
    engine.write_json(
        marker_path,
        {
            "package": str(package),
            "archive": str(archive_path),
            "hashes": {
                str(package / "package_manifest.json"): engine.file_hash(
                    package / "package_manifest.json"
                ),
                str(archive_path): archive_hash,
            },
        },
    )
    return package, archive_path


def finalize(context):
    validate(context)
    record, model_path, model_hash = _parent(context)
    metrics, contract = _evaluate_once(
        context, record, model_path, model_hash
    )
    _figures(context, metrics)
    package, archive = _package(
        context, record, model_path, model_hash, metrics, contract
    )
    report = flow.archive_results(context)
    print("최종 후보:", package)
    print("배포 ZIP:", archive)
    print("보고서 ZIP:", report)
    return metrics, package, archive, report
