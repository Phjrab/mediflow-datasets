"""One-time Test evaluation and packaging of the fixed Web Skin PMG B0/256 model."""

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

PROTOCOL = "web_skin_pmg_b0_256_final_candidate_v1"
PARENT_TRIAL = "pmg_b0_256_ce_seed_42"
MODEL_NAME = "stage2_best.keras"
EXPECTED_MODEL_SHA256 = "83e659dd09a9355135ee0de0197037e81ece962777f59dedae8d9a37b49a3fc4"
EXPECTED_DATA_SHA256 = "f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d"
EXPECTED_CLASSES = ["건선", "아토피", "여드름", "정상", "주사"]
EXPECTED_VALIDATION = {
    "accuracy": 0.85,
    "macro_f1": 0.8473279632397033,
    "count": 500,
}


def validate(context):
    config = context["config"]
    if config["domain"] != "web_skin" or config["mode"] != "web_skin_pmg_final":
        raise ValueError("이 노트북은 Web Skin PMG·B0·256 최종평가 전용입니다.")
    if config["train_variant"] != "augmented":
        raise ValueError("고정된 PMG 후보와 같은 augmented 데이터 계약이 필요합니다.")
    if config["seed"] != 42 or config["seeds"] != [42]:
        raise ValueError("고정된 PMG 후보의 seed 계약이 다릅니다.")
    if context["classes"] != EXPECTED_CLASSES:
        raise ValueError("Web Skin 클래스 순서가 고정 후보와 다릅니다.")
    if context["data_hash"] != EXPECTED_DATA_SHA256:
        raise ValueError("Web Skin 데이터 SHA-256이 고정 후보와 다릅니다.")
    parent = config.get("parent_run_dir")
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError("PARENT_RUN_DIR을 입력하세요.")
    digest = config.get("parent_model_sha256")
    if digest != EXPECTED_MODEL_SHA256:
        raise ValueError("고정된 PMG·B0·256 모델 SHA-256과 다릅니다.")


def _parent(context):
    parent = Path(context["config"]["parent_run_dir"])
    marker = parent / PARENT_TRIAL / "completed.json"
    if not marker.is_file():
        raise FileNotFoundError("PMG·B0·256 완료 기록을 찾을 수 없습니다: " + str(marker))
    record = engine.read_json(marker)
    spec = record.get("spec", {})
    validation = record.get("validation", {})
    if (
        record.get("id") != PARENT_TRIAL
        or record.get("selected_stage") != "stage2"
        or record.get("selected_model") != MODEL_NAME
        or record.get("test_evaluated", False) is not False
        or spec.get("method") != "efficientnet_b0_pmg_adaptation"
        or spec.get("input_size") != 256
        or spec.get("jigsaw_grids") != [8, 4, 2]
        or validation.get("accuracy") != EXPECTED_VALIDATION["accuracy"]
        or validation.get("macro_f1") != EXPECTED_VALIDATION["macro_f1"]
        or validation.get("count") != EXPECTED_VALIDATION["count"]
    ):
        raise ValueError("고정한 PMG·B0·256 Validation 후보 기록과 다릅니다.")
    model_path = marker.parent / record["attempt"] / MODEL_NAME
    actual = engine.file_hash(model_path)
    relative = record["attempt"] + "/" + MODEL_NAME
    if (
        actual != EXPECTED_MODEL_SHA256
        or record.get("artifact_hashes", {}).get(relative) != actual
    ):
        raise ValueError("PMG·B0·256 최종 후보 모델 SHA-256이 다릅니다.")
    settings = engine.read_json(parent / "run_config.json")["settings"]
    if (
        settings["data_sha256"] != context["data_hash"]
        or settings["classes"] != context["classes"]
        or settings["seed"] != 42
    ):
        raise ValueError("부모 실행의 데이터·클래스·seed가 현재 실행과 다릅니다.")
    return record, model_path, actual


def _predictor(model):
    if not isinstance(model.outputs, list) or len(model.outputs) != 4:
        raise ValueError("PMG는 네 개의 logit 출력이 필요합니다.")
    total = keras.layers.Add(name="pmg_logit_sum")(model.outputs)
    probabilities = keras.layers.Activation("softmax", name="predictions")(total)
    return keras.Model(model.input, probabilities, name="web_skin_pmg_inference")


def _model_contract(model):
    shapes = [tuple(shape) for shape in model.output_shape]
    if (
        tuple(model.input_shape) != (None, 256, 256, 3)
        or shapes != [(None, 5)] * 4
        or model.count_params() != 8_872_375
    ):
        raise ValueError("PMG·B0·256 모델 입출력 또는 파라미터 계약이 다릅니다.")
    try:
        backbone = model.get_layer("pmg_backbone")
    except ValueError as exc:
        raise ValueError("PMG Backbone을 찾지 못했습니다.") from exc
    rescaling = [
        layer for layer in backbone.layers if isinstance(layer, keras.layers.Rescaling)
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
        (3, 256, 256, 3),
    ).copy()
    raw = model(values, training=False)
    if len(raw) != 4 or any(
        np.asarray(output).shape != (3, 5) or not np.isfinite(output).all()
        for output in raw
    ):
        raise ValueError("PMG 내부 logit 출력 검사가 실패했습니다.")
    scores = np.asarray(_predictor(model)(values, training=False))
    if (
        scores.shape != (3, 5)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
        or not np.allclose(scores.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("PMG 합산 softmax 출력 검사가 실패했습니다.")
    return {
        "input_shape": [256, 256, 3],
        "internal_output_count": 4,
        "public_output_count": 5,
        "parameters": model.count_params(),
        "dummy_forward_passed": True,
    }


def _evaluate_once(context, record, model_path, model_hash):
    output = Path(context["output"])
    selection = {
        "protocol": PROTOCOL,
        "winner": PARENT_TRIAL,
        "selection_reason": "deployment balance of validation quality and inference cost",
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
    predictor = _predictor(model)
    test, paths = flow.factory(context, "augmented", seed=42)("test", 256, False)
    metrics = engine.evaluate_to_files(predictor, test, paths, output, "final_test")
    del predictor, model
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
    fig, axis = plt.subplots(figsize=(8, 8))
    flow.confusion(axis, metrics, "Web Skin PMG·B0·256 / Final Test")
    fig.tight_layout()
    fig.savefig(output / "final_test_confusion_matrix.png", dpi=180)
    plt.show()
    plt.close(fig)
    flow.errors(
        context,
        output / "final_test_predictions.csv",
        output / "final_test_errors.png",
    )


def _inference_source():
    return '''"""MediFlow Web Skin PMG inference helper."""
from pathlib import Path

import keras
import tensorflow as tf


def load_predictor(model_path):
    model = keras.models.load_model(Path(model_path), compile=False)
    if not isinstance(model.outputs, list) or len(model.outputs) != 4:
        raise ValueError("Expected four PMG logit outputs")
    total = keras.layers.Add(name="pmg_logit_sum")(model.outputs)
    probabilities = keras.layers.Activation("softmax", name="predictions")(total)
    return keras.Model(model.input, probabilities, name="web_skin_pmg_inference")


def load_image(image_path):
    content = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(content, channels=3, expand_animations=False)
    image = tf.image.resize(tf.cast(image, tf.float32), (256, 256), method="bilinear")
    return image[None, ...]


def predict_file(predictor, image_path):
    return predictor(load_image(image_path), training=False).numpy()[0]
'''


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
        "domain": "web_skin",
        "class_names": context["classes"],
        "normal_class_included": True,
        "architecture": "PMG with ImageNet EfficientNet-B0",
        "input_size": [256, 256],
        "loss": "Categorical Crossentropy",
        "optimizer": "Adam",
        "stage1_epochs": 15,
        "stage2_epochs": 10,
        "selected_model": PARENT_TRIAL,
        "selection_reason": "validation quality and deployment cost balance",
        "model_sha256": model_hash,
        "model_parameter_count": contract["parameters"],
        "validation": record["validation"],
        "test": metrics,
        "data_zip_sha256": context["data_hash"],
        "limitations": context["audit"]["limitations"]
        + [
            "Actual webcam patient images were not evaluated",
            "No out-of-scope rejection mechanism",
            "Softmax scores are not calibrated correctness probabilities",
        ],
    }
    engine.write_json(output / "model_card.json", card)
    package_parent = context["project"] / "2_results" / "web_skin" / "selected_models"
    suffix = output.name.removeprefix("web_skin_pmg_final_")
    name = "public_candidate_v2_pmg_b0_256_ce_" + suffix
    package = package_parent / name
    package.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(model_path, package / "web_skin_pmg_model.keras")
    if engine.file_hash(package / "web_skin_pmg_model.keras") != model_hash:
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
        "common_workflow.py",
        "web_skin_pmg_final.py",
    ):
        shutil.copyfile(output / filename, package / filename)
    (package / "inference.py").write_text(_inference_source(), encoding="utf-8")
    preprocessing = {
        "input_shape": [256, 256, 3],
        "color_order": "RGB",
        "input_dtype": "float32",
        "input_pixel_range": [0, 255],
        "external_normalization": False,
        "internal_rescaling": "1/255",
        "resize": "TensorFlow bilinear, antialias=False",
        "aspect_ratio": "resize to 256x256, no crop/pad",
        "exif_transpose": False,
        "internal_outputs": "four PMG logits",
        "public_output": "sum logits then 5-way softmax in class_names.json order",
    }
    engine.write_json(package / "preprocessing.json", preprocessing)
    engine.write_json(package / "model_contract_check.json", contract)
    (package / "MODEL_CARD.md").write_text(
        "# MediFlow Web Skin 공개 데이터 후보 v2\n\n"
        "PMG / EfficientNet-B0 / 256 / Cross Entropy / Adam.\n\n"
        f"클래스 순서: {context['classes']}\n\n"
        f"Validation Accuracy: {record['validation']['accuracy']}\n\n"
        f"Test Accuracy: {metrics['accuracy']}\n\n"
        f"Test Macro F1: {metrics['macro_f1']}\n\n"
        "입력은 얼굴 피부 RGB float32 0~255를 256×256으로 resize합니다. 외부 /255는 "
        "금지합니다. 모델의 네 logit 출력을 직접 사용하지 말고 inference.py처럼 합산 후 "
        "softmax를 적용해야 합니다.\n\n"
        "실제 웹캠 환자 검증과 범위 밖 입력 거부 기능은 포함되지 않습니다.\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_file": "web_skin_pmg_model.keras",
        "inference_file": "inference.py",
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
    metrics, contract = _evaluate_once(context, record, model_path, model_hash)
    _figures(context, metrics)
    package, archive = _package(
        context, record, model_path, model_hash, metrics, contract
    )
    report = flow.archive_results(context)
    print("최종 후보:", package)
    print("배포 ZIP:", archive)
    print("보고서 ZIP:", report)
    return metrics, package, archive, report
