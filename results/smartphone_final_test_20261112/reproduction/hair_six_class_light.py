"""Evaluate and package the fixed six-class Hair B0 lightweight alternate."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

RUN_NAME = "hair_six_class_four_20260928_011824_ad21e8ae"
DATA_NAME = "hair_clean_v2_6class_d6ac77b7d507.zip"
DATA_SHA256 = "e2a6b4347d94bd7050d17bd3aca404161693083b12afb51d7e20aa9b284f0f93"
TRIAL = "b0_256_augmented"
MODEL_SHA256 = "38e91fee1c514b75165d07f5dfbb0c52fdb7ac92313b40d47abb50c71367d432"
HEAVY_SHA256 = "5ee54f51e65d8656fc81e88de96ad82a6d119991aa74684f921e4cd7e8b7efbb"
CLASSES = ["모낭사이홍반", "미세각질", "비듬", "탈모", "피지과다", "양호"]
OUTPUT_NAME = "public_candidate_6class_light_b0_256_augmented_v1"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def verify_candidate(parent: Path) -> dict:
    _check(parent.is_dir() and parent.name == RUN_NAME, "6클래스 실험 폴더가 다릅니다.")
    names = _read(parent / "class_names.json")
    config = _read(parent / "run_config.json")
    contract = _read(parent / "dataset_contract.json")
    summary = _read(parent / "comparison_summary.json")
    b0_original = _read(parent / "b0_256_original" / "completed.json")
    candidate = _read(parent / TRIAL / "completed.json")
    _check(names == contract["class_names"] == CLASSES, "클래스 순서가 다릅니다.")
    _check(config["settings"]["dataset_sha256"] == DATA_SHA256, "데이터 버전이 다릅니다.")
    _check(config["signature"] == candidate["signature"], "실험 설정이 다릅니다.")
    _check(summary["winner"] == "b1_384_original", "기존 성능 우선 후보가 다릅니다.")
    _check(summary["selected_model_sha256"] == HEAVY_SHA256, "기존 성능 우선 모델이 다릅니다.")
    _check(
        candidate["validation"]["macro_f1"] > b0_original["validation"]["macro_f1"],
        "B0 경량 후보의 Validation 선정 근거가 다릅니다.",
    )
    _check(
        candidate["id"] == TRIAL
        and candidate["selected_stage"] == "stage2"
        and candidate["selected_model"] == "stage2_best.keras"
        and candidate["spec"]
        == {
            "id": TRIAL,
            "backbone": "B0",
            "size": 256,
            "loss": "ce",
            "variant": "augmented",
            "class_count": 6,
        },
        "B0 후보 학습 조건이 다릅니다.",
    )
    relative = f"{candidate['attempt']}/{candidate['selected_model']}"
    model_path = parent / TRIAL / relative
    _check(model_path.is_file(), "B0 선정 모델 파일이 없습니다.")
    _check(
        _sha(model_path) == candidate["artifact_hashes"][relative] == MODEL_SHA256,
        "B0 선정 모델 해시가 다릅니다.",
    )
    _check(model_path.stat().st_size == candidate["model_bytes"], "B0 모델 크기가 다릅니다.")
    return {
        "model_path": model_path,
        "candidate": candidate,
        "contract": contract,
        "config": config,
        "summary": summary,
        "classes": names,
    }


def _verify_model(path: Path):
    import keras
    import numpy as np

    keras.backend.clear_session()
    model = keras.models.load_model(path, compile=False)
    _check(tuple(model.input_shape) == (None, 256, 256, 3), "B0 입력 크기가 다릅니다.")
    _check(tuple(model.output_shape) == (None, 6), "B0 클래스 수가 다릅니다.")
    _check(model.count_params() == 4057257, "B0 파라미터 수가 다릅니다.")
    backbones = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model) and "efficientnetb0" in layer.name.lower()
    ]
    _check(len(backbones) == 1, "EfficientNet-B0 본체를 찾을 수 없습니다.")
    rescaling = [
        layer for layer in backbones[0].layers if isinstance(layer, keras.layers.Rescaling)
    ]
    _check(
        any(
            np.isclose(float(np.asarray(layer.scale).item()), 1 / 255)
            and np.allclose(layer.offset, 0)
            for layer in rescaling
        ),
        "B0 내부 픽셀 변환이 다릅니다.",
    )
    pixels = np.broadcast_to(
        np.array([0, 127.5, 255], dtype="float32")[:, None, None, None],
        (3, 256, 256, 3),
    ).copy()
    scores = np.asarray(model(pixels, training=False))
    _check(
        scores.shape == (3, 6)
        and np.isfinite(scores).all()
        and np.all((scores >= 0) & (scores <= 1))
        and np.allclose(scores.sum(axis=1), 1, atol=1e-5),
        "B0 예측 출력이 올바르지 않습니다.",
    )
    return model


def _extract_test(dataset_zip: Path, destination: Path, contract: dict):
    # The dataset builder archived its local `dataset/` directory as ZIP root.
    prefix = "dataset"
    expected = contract["original_counts"]["test"]
    entries = []
    with zipfile.ZipFile(dataset_zip) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            _check(
                not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename,
                "데이터 ZIP 경로가 안전하지 않습니다.",
            )
            if (
                not info.is_dir()
                and len(path.parts) == 5
                and path.parts[:3] == (prefix, "original", "test")
                and path.parts[3] in CLASSES
                and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ):
                entries.append((info, path.parts[3], path))
        counts = {name: sum(label == name for _, label, _ in entries) for name in CLASSES}
        _check(counts == expected and sum(counts.values()) == 1305, "Test 분할 구성이 다릅니다.")
        for info, _label, path in entries:
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    return destination / prefix


def _test_dataset(root: Path, batch_size: int):
    import numpy as np
    import tensorflow as tf

    paths = [
        path
        for name in CLASSES
        for path in sorted((root / "original" / "test" / name).iterdir())
        if path.is_file()
    ]
    labels = [CLASSES.index(path.parent.name) for path in paths]
    relative = [path.relative_to(root).as_posix() for path in paths]
    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(path) for path in paths], np.asarray(labels, dtype=np.int32))
    )

    def decode(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(tf.cast(image, tf.float32), (256, 256))
        return image, tf.one_hot(label, 6)

    return dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    ), relative


def _evaluate_once(records: dict, dataset_zip: Path, output: Path, batch_size: int):
    from mediflow_datasets import common_engine as engine

    marker = output / "test_completed.json"
    if marker.exists():
        completed = _read(marker)
        _check(
            completed["model_sha256"] == MODEL_SHA256 and completed["data_sha256"] == DATA_SHA256,
            "저장된 Test 평가의 모델·데이터가 다릅니다.",
        )
        for filename, digest in completed["artifact_hashes"].items():
            _check(_sha(output / filename) == digest, f"기존 Test 결과가 변경됐습니다: {filename}")
        return completed["metrics"]
    _check(_sha(dataset_zip) == DATA_SHA256, "데이터 ZIP SHA-256이 다릅니다.")
    with tempfile.TemporaryDirectory(prefix="hair6_light_test_") as temporary:
        root = _extract_test(dataset_zip, Path(temporary), records["contract"])
        dataset, paths = _test_dataset(root, batch_size)
        model = _verify_model(records["model_path"])
        truth, probabilities = engine.predict_dataset(model, dataset)
        del model
        metrics = engine.classification_metrics(truth, probabilities, 6)
        metrics["selected_model"] = TRIAL
        engine.save_predictions(output / "test_predictions.csv", paths, truth, probabilities)
    _check(
        metrics["support"] == list(records["contract"]["original_counts"]["test"].values()),
        "평가 클래스별 장수가 다릅니다.",
    )
    _write(output / "test_metrics.json", metrics)
    completed = {
        "model_sha256": MODEL_SHA256,
        "data_sha256": DATA_SHA256,
        "metrics": metrics,
        "artifact_hashes": {
            name: _sha(output / name) for name in ("test_metrics.json", "test_predictions.csv")
        },
    }
    _write(marker, completed)
    return metrics


def package_light(parent_run_dir: str, project_root: str, batch_size: int = 32):
    """Read-only source check, one fixed Test evaluation, then separate package ZIP."""
    parent = Path(parent_run_dir)
    project = Path(project_root)
    records = verify_candidate(parent)
    dataset_zip = project / "datasets" / DATA_NAME
    _check(dataset_zip.is_file(), f"데이터 ZIP을 찾을 수 없습니다: {dataset_zip}")
    target_parent = project / "2_results" / "hair" / "selected_models"
    target_parent.mkdir(parents=True, exist_ok=True)
    output = target_parent / OUTPUT_NAME
    output.mkdir(exist_ok=True)
    archive = target_parent / f"{OUTPUT_NAME}.zip"
    if archive.exists():
        manifest = _read(output / "manifest.json")
        _check(
            all(_sha(output / name) == digest for name, digest in manifest.items()),
            "기존 경량 패키지 파일이 변경됐습니다.",
        )
        with zipfile.ZipFile(archive) as existing:
            _check(existing.testzip() is None, "기존 ZIP이 손상됐습니다.")
        return output, archive, _read(output / "test_metrics.json")
    metrics = _evaluate_once(records, dataset_zip, output, batch_size)
    sources = {
        "hair_model.keras": records["model_path"],
        "class_names.json": parent / "class_names.json",
        "dataset_contract.json": parent / "dataset_contract.json",
        "run_config.json": parent / "run_config.json",
        "comparison_summary.json": parent / "comparison_summary.json",
        "winner_completed.json": parent / TRIAL / "completed.json",
        "winner_validation_metrics.json": parent
        / TRIAL
        / records["candidate"]["attempt"]
        / "validation_metrics.json",
        "four_training_curves.png": parent / "four_training_curves.png",
        "validation_dashboard.png": parent / "validation_dashboard.png",
    }
    for name, source in sources.items():
        _check(source.is_file(), f"패키지 입력 파일이 없습니다: {source}")
        shutil.copyfile(source, output / name)
        _check(_sha(output / name) == _sha(source), f"복사 검증 실패: {name}")
    _write(
        output / "preprocessing.json",
        {
            "input_shape": [256, 256, 3],
            "color_order": "RGB",
            "input_dtype": "float32",
            "input_pixel_range": [0, 255],
            "external_normalization": False,
            "internal_rescaling": "1/255",
            "resize": "TensorFlow bilinear, antialias=False",
            "aspect_ratio": "direct resize, no crop/pad",
            "exif_transpose": False,
            "output": "six softmax scores in class_names.json order",
        },
    )
    _write(
        output / "model_card.json",
        {
            "version": "hair_6class_light_v1",
            "role": "lightweight_alternate_for_cost_comparison",
            "domain": "hair",
            "status": "public_data_candidate_not_device_validated",
            "backbone": "EfficientNet-B0",
            "input_size": [256, 256],
            "train_variant": "augmented",
            "loss": "categorical crossentropy",
            "stage1_epochs": 15,
            "stage2_epochs": 15,
            "classes": CLASSES,
            "model_sha256": MODEL_SHA256,
            "data_zip_sha256": DATA_SHA256,
            "validation": records["candidate"]["validation"],
            "test": metrics,
            "comparison_model_sha256": HEAVY_SHA256,
            "limitations": records["contract"]["limitations"]
            + [
                "Actual USB microscope patient images were not evaluated.",
                "Latency, energy and peak memory have not been measured on deployment hardware.",
                "Softmax scores are not calibrated correctness probabilities.",
            ],
        },
    )
    (output / "MODEL_INFO.md").write_text(
        "# Hair 6클래스 경량 비교 후보\n\n"
        "B0·256·원본+증강·CE. 두 B0 실험 중 Validation Macro F1이 더 높아 "
        "경량 후보로 고정했습니다. 성능 우선 B1·384·원본 모델은 별도로 보존합니다.\n\n"
        "입력: 두피 RGB 사진을 256×256으로 직접 조정한 float32 픽셀 0–255. "
        "외부 /255 정규화는 하지 않습니다.\n\n"
        f"클래스 출력 순서: {', '.join(CLASSES)}\n\n"
        f"Validation Accuracy {records['candidate']['validation']['accuracy']}; "
        f"Macro F1 {records['candidate']['validation']['macro_f1']}\n"
        f"Test Accuracy {metrics['accuracy']}; Macro F1 {metrics['macro_f1']} "
        f"(n={metrics['count']})\n\n"
        "경량 여부는 모델 파일 크기·입력 크기에 근거한 구분입니다. "
        "실제 장비의 지연 시간과 메모리는 아직 측정하지 않았습니다.\n",
        encoding="utf-8",
    )
    manifest = {path.name: _sha(path) for path in sorted(output.iterdir()) if path.is_file()}
    _write(output / "manifest.json", manifest)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(output.iterdir()):
            target.write(path, f"{OUTPUT_NAME}/{path.name}")
    with zipfile.ZipFile(archive) as target:
        _check(target.testzip() is None, "새 경량 ZIP이 손상됐습니다.")
    (target_parent / f"{OUTPUT_NAME}.zip.sha256").write_text(
        f"{_sha(archive)}  {archive.name}\n", encoding="utf-8"
    )
    return output, archive, metrics
