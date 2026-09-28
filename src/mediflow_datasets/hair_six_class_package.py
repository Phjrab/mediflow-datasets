"""Package the already selected Hair six-class model without re-evaluating Test."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_CLASSES = ["모낭사이홍반", "미세각질", "비듬", "탈모", "피지과다", "양호"]
EXPECTED_MODEL_SHA256 = "5ee54f51e65d8656fc81e88de96ad82a6d119991aa74684f921e4cd7e8b7efbb"
EXPECTED_DATA_SHA256 = "e2a6b4347d94bd7050d17bd3aca404161693083b12afb51d7e20aa9b284f0f93"
RUN_NAME = "hair_six_class_four_20260928_011824_ad21e8ae"
WINNER = "b1_384_original"
MODEL_NAME = "stage2_best.keras"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_records(parent: Path) -> dict:
    """Check the fixed validation choice, Test record and all package inputs."""
    _check(parent.is_dir() and parent.name == RUN_NAME, "6클래스 실험 결과 폴더가 다릅니다.")
    names = _read(parent / "class_names.json")
    contract = _read(parent / "dataset_contract.json")
    config = _read(parent / "run_config.json")
    summary = _read(parent / "comparison_summary.json")
    test = _read(parent / "selected_test_metrics.json")
    completed = _read(parent / WINNER / "completed.json")
    _check(names == EXPECTED_CLASSES == contract["class_names"], "6클래스 순서가 다릅니다.")
    _check(config["settings"]["dataset_sha256"] == EXPECTED_DATA_SHA256, "데이터 버전이 다릅니다.")
    _check(config["signature"] == completed["signature"], "실험 서명이 다릅니다.")
    _check(
        config["settings"]["protocol"] == summary["protocol"] == "hair_six_class_four_v1",
        "실험 종류가 다릅니다.",
    )
    _check(
        summary["selection"] == "validation_macro_f1_then_accuracy" and summary["winner"] == WINNER,
        "Validation 선정 기록이 다릅니다.",
    )
    _check(
        summary["test_evaluated"] is True and test["selected_model"] == WINNER,
        "선정된 Test 평가 기록이 없습니다.",
    )
    _check(
        summary["selected_test_accuracy"] == test["accuracy"]
        and summary["selected_test_macro_f1"] == test["macro_f1"],
        "Test 결과가 요약과 다릅니다.",
    )
    _check(test["count"] == sum(test["support"]) == 1305, "Test 장수가 다릅니다.")
    _check(
        len(test["class_f1"]) == len(test["confusion_matrix"]) == 6, "Test 클래스 수가 다릅니다."
    )
    _check(
        completed["id"] == WINNER
        and completed["selected_stage"] == "stage2"
        and completed["selected_model"] == MODEL_NAME,
        "선정 체크포인트가 다릅니다.",
    )
    _check(
        completed["spec"]
        == {
            "id": WINNER,
            "backbone": "B1",
            "size": 384,
            "loss": "ls005",
            "variant": "original",
            "class_count": 6,
        },
        "모델 학습 조건이 다릅니다.",
    )
    _check(completed["parameters"] == 6582925, "기록된 파라미터 수가 다릅니다.")
    experiment = next((row for row in summary["experiments"] if row["id"] == WINNER), None)
    _check(
        experiment is not None
        and experiment["validation_accuracy"] == completed["validation"]["accuracy"]
        and experiment["validation_macro_f1"] == completed["validation"]["macro_f1"],
        "Validation 결과가 다릅니다.",
    )
    relative = f"{completed['attempt']}/{MODEL_NAME}"
    model_path = parent / WINNER / relative
    _check(model_path.is_file(), f"선정 모델이 없습니다: {model_path}")
    digest = _sha(model_path)
    _check(
        digest
        == EXPECTED_MODEL_SHA256
        == summary["selected_model_sha256"]
        == completed["artifact_hashes"][relative],
        "선정 모델 SHA-256이 다릅니다.",
    )
    _check(model_path.stat().st_size == completed["model_bytes"], "모델 크기가 기록과 다릅니다.")
    return {
        "model_path": model_path,
        "model_sha256": digest,
        "classes": names,
        "contract": contract,
        "config": config,
        "summary": summary,
        "test": test,
        "completed": completed,
    }


def verify_model(model_path: Path) -> dict:
    """Load the actual checkpoint and check its inference contract."""
    import keras
    import numpy as np

    keras.backend.clear_session()
    model = keras.models.load_model(model_path, compile=False)
    _check(tuple(model.input_shape) == (None, 384, 384, 3), "모델 입력 크기가 다릅니다.")
    _check(tuple(model.output_shape) == (None, 6), "모델 출력 클래스 수가 다릅니다.")
    _check(model.count_params() == 6582925, "실제 파라미터 수가 다릅니다.")
    backbones = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model) and "efficientnetb1" in layer.name.lower()
    ]
    _check(len(backbones) == 1, "EfficientNet-B1 본체를 확인할 수 없습니다.")
    rescaling = [
        layer for layer in backbones[0].layers if isinstance(layer, keras.layers.Rescaling)
    ]
    _check(
        any(
            np.isclose(float(np.asarray(layer.scale).item()), 1 / 255)
            and np.allclose(layer.offset, 0)
            for layer in rescaling
        ),
        "모델 내부 픽셀 변환이 다릅니다.",
    )
    pixels = np.broadcast_to(
        np.array([0, 127.5, 255], dtype="float32")[:, None, None, None], (3, 384, 384, 3)
    ).copy()
    scores = np.asarray(model(pixels, training=False))
    _check(
        scores.shape == (3, 6)
        and np.isfinite(scores).all()
        and np.all((scores >= 0) & (scores <= 1))
        and np.allclose(scores.sum(axis=1), 1, atol=1e-5),
        "모델 예측 출력이 올바르지 않습니다.",
    )
    del model
    keras.backend.clear_session()
    return {
        "input_shape": [384, 384, 3],
        "output_count": 6,
        "parameters": 6582925,
        "dummy_forward_passed": True,
    }


def package(
    parent_run_dir: str, project_root: str, *, check_model: bool = True
) -> tuple[Path, Path]:
    """Create a new versioned model directory and ZIP under Drive 2_results."""
    parent = Path(parent_run_dir)
    root = Path(project_root)
    records = verify_records(parent)
    model_contract = (
        verify_model(records["model_path"]) if check_model else {"verified_in_colab": False}
    )
    name = "public_candidate_6class_v1_b1_384_original_" + datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    target_parent = root / "2_results" / "hair" / "selected_models"
    target_parent.mkdir(parents=True, exist_ok=True)
    output = target_parent / name
    output.mkdir(exist_ok=False)
    files = {
        "hair_model.keras": records["model_path"],
        "class_names.json": parent / "class_names.json",
        "dataset_contract.json": parent / "dataset_contract.json",
        "run_config.json": parent / "run_config.json",
        "comparison_summary.json": parent / "comparison_summary.json",
        "selected_test_metrics.json": parent / "selected_test_metrics.json",
        "selected_test_predictions.csv": parent / "selected_test_predictions.csv",
        "selected_test_confusion.png": parent / "selected_test_confusion.png",
        "four_training_curves.png": parent / "four_training_curves.png",
        "validation_dashboard.png": parent / "validation_dashboard.png",
        "winner_completed.json": parent / WINNER / "completed.json",
        "winner_validation_metrics.json": parent
        / WINNER
        / records["completed"]["attempt"]
        / "validation_metrics.json",
    }
    for filename, source in files.items():
        _check(source.is_file(), f"패키지 입력 파일이 없습니다: {source}")
        shutil.copyfile(source, output / filename)
        _check(_sha(output / filename) == _sha(source), f"복사 검증 실패: {filename}")
    preprocessing = {
        "input_shape": [384, 384, 3],
        "color_order": "RGB",
        "input_dtype": "float32",
        "input_pixel_range": [0, 255],
        "external_normalization": False,
        "internal_rescaling": "1/255",
        "resize": "TensorFlow bilinear, antialias=False",
        "aspect_ratio": "direct resize, no crop/pad",
        "exif_transpose": False,
        "output": "six softmax scores in class_names.json order",
    }
    card = {
        "version": "hair_6class_v1",
        "domain": "hair",
        "status": "public_data_candidate_not_device_validated",
        "backbone": "EfficientNet-B1",
        "train_variant": "original",
        "input_size": [384, 384],
        "loss": "categorical crossentropy with label smoothing 0.05",
        "optimizer": "Adam",
        "stage1_epochs": 15,
        "stage2_epochs": 15,
        "classes": records["classes"],
        "selected_checkpoint": f"{WINNER}/{records['completed']['attempt']}/{MODEL_NAME}",
        "model_sha256": records["model_sha256"],
        "data_zip_sha256": EXPECTED_DATA_SHA256,
        "selection": records["summary"]["selection"],
        "validation": records["completed"]["validation"],
        "test": records["test"],
        "model_contract": model_contract,
        "limitations": records["contract"]["limitations"]
        + [
            "Actual USB microscope patient images were not evaluated.",
            "Softmax scores are not calibrated correctness probabilities.",
            "The good class has all six symptom values zero; source JSON was unavailable.",
        ],
    }
    for filename, data in (
        ("preprocessing.json", preprocessing),
        ("model_card.json", card),
        ("model_contract_check.json", model_contract),
    ):
        (output / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    validation = records["completed"]["validation"]
    test = records["test"]
    (output / "MODEL_INFO.md").write_text(
        "# Hair 6클래스 공개 데이터 후보 v1\n\n"
        "선정 모델: B1·384, 원본 학습, Label Smoothing 0.05, Stage 1/2 각 15 epoch.\n\n"
        "입력은 두피 RGB 사진 한 장을 384×384로 직접 크기 조정한 float32 픽셀(0–255)입니다. "
        "외부에서 /255를 적용하지 마세요. 모델에 해당 변환이 포함되어 있습니다.\n\n"
        f"출력 순서: {', '.join(records['classes'])}\n\n"
        f"Validation Accuracy {validation['accuracy']}; Macro F1 {validation['macro_f1']}\n"
        f"Test Accuracy {test['accuracy']}; Macro F1 {test['macro_f1']} "
        f"(n={test['count']})\n\n"
        "`양호`는 이번 6클래스 데이터에서 여섯 증상 값이 모두 0인 사진입니다. "
        "다른 부위·촬영 장비·품질 불량 입력은 이 분류기의 검증 범위 밖입니다. "
        "출력 점수를 진단 확률로 해석하지 마세요. 실제 장비 데이터 검증은 아직 하지 않았습니다.\n",
        encoding="utf-8",
    )
    manifest = {path.name: _sha(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    archive = target_parent / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in sorted(output.iterdir()):
            target.write(path, f"{name}/{path.name}")
    with zipfile.ZipFile(archive) as target:
        _check(target.testzip() is None, "생성된 ZIP 검사에 실패했습니다.")
    (target_parent / f"{name}.zip.sha256").write_text(
        f"{_sha(archive)}  {archive.name}\n", encoding="utf-8"
    )
    return output, archive
