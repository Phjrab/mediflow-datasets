"""Deliver the Validation-fixed smartphone Stage4 model without changing official routing."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import keras
import numpy as np
from PIL import Image, ImageCms

from mediflow_datasets import adaptation_packaging as shared
from mediflow_datasets import adaptation_stage1 as training
from mediflow_datasets import domain_adaptation_experiment as experiment
from mediflow_datasets.domain_adaptation_data import read_json, sha256
from mediflow_datasets.smartphone_final_test import CONDITIONS
from mediflow_datasets.smartphone_profile import canonical_input

PACKAGE_ID = "web_skin_smartphone_adaptation_stage4_20261112"
PRIOR = "results/smartphone_web_skin_adaptation_20261112"
JOINT = "results/smartphone_joint_adaptation_20261112"
TEST = "results/smartphone_final_test_20261112"


def verify_selection(envelope, result, selection, model_sha):
    plan = envelope["plan"]
    if (
        selection["model"] != "stage4"
        or selection["sha256"] != model_sha
        or plan["models"]["stage4"]["sha256"] != model_sha
        or plan["selected_on_validation"] != "stage4"
        or plan["test_used_for_selection"] is not False
        or result["selected_on_validation"] != "stage4"
        or result["test_used_for_selection"] is not False
        or result["classes"] != plan["task"]["classes"]
        or result["test_count"] != plan["test_count"]
        or tuple(result["metrics"]["stage4"]) != CONDITIONS
    ):
        raise ValueError("스마트폰 Validation 후보·모델과 Test 기록이 다릅니다.")
    return result["metrics"]["stage4"]


def preprocessing():
    return {
        "input_size": 256,
        "input_policy": "center_square",
        "color": "RGB",
        "dtype": "float32",
        "pixel_range": [0, 255],
        "layout": "NHWC",
        "resize_method": "bilinear",
        "antialias": False,
        "exif_rotation_applied": True,
        "icc_to_srgb": True,
        "missing_icc_policy": "assume_sRGB",
        "invalid_icc_policy": "error",
        "external_division_by_255": False,
        "internal_rescaling": "Rescaling(1/255) in model",
        "training_augmentation_applied_during_inference": False,
        "source_test_input_kind": "source: TensorFlow decoder without EXIF/ICC correction",
        "smartphone_input_kind": "smartphone: PIL EXIF orientation + ICC to sRGB",
    }


def verify_execution(folder, model_path, spec, work):
    helper = shared.load_standalone(folder)
    predictor = helper.load_predictor(folder)
    model = keras.models.load_model(model_path, compile=False, safe_mode=True)
    oracle = training.predictor(model, spec)
    rng = np.random.default_rng(42)
    inputs = work / "synthetic_inputs"
    inputs.mkdir()
    wide, rotated, colored = (inputs / n for n in ("wide.png", "rotated.jpg", "icc.png"))
    Image.fromarray(rng.integers(0, 256, (61, 89, 3), dtype=np.uint8)).save(wide)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(rng.integers(0, 256, (91, 63, 3), dtype=np.uint8)).save(rotated, exif=exif)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(rng.integers(0, 256, (67, 93, 3), dtype=np.uint8)).save(
        colored, icc_profile=profile
    )
    cases = []
    for path in (wide, rotated, colored):
        for kind in ("smartphone", "source"):
            expected = (
                canonical_input(path, 256)
                if kind == "smartphone"
                else (experiment.image_tensor(path.read_bytes(), 256, "center_square"))
            )
            actual = helper.load_image(path, predictor.preprocessing, kind).numpy()[0]
            if not np.array_equal(actual, expected):
                raise ValueError("스마트폰/Source 전처리가 기존 계약과 다릅니다.")
            before = oracle(expected[None]).numpy()[0]
            prediction = predictor.predict_file(path, kind)
            after = np.asarray([r["score"] for r in prediction["scores"]])
            if not np.allclose(before, after, atol=1e-6, rtol=1e-5) or (
                int(before.argmax()) != prediction["predicted_index"]
            ):
                raise ValueError("패키지와 원본 모델 예측이 다릅니다.")
            cases.append(
                {
                    "synthetic_input": path.name,
                    "input_kind": kind,
                    "preprocessing_exactly_equal": True,
                    "max_score_absolute_difference": float(np.max(np.abs(before - after))),
                }
            )
    return {
        "status": "passed",
        "scope": "Synthetic execution parity only; no new clinical evaluation",
        "cases": cases,
        "model_sha256": sha256(model_path),
    }, wide


def build(root, output, work):
    root, output, work = map(lambda p: Path(p).resolve(), (root, output, work))
    if output.exists() or work.exists():
        raise FileExistsError("기존 폴더를 보존하도록 새 패키지·작업 폴더를 지정하세요.")
    source, test, joint = root / PRIOR, root / TEST, root / JOINT
    complete = read_json(source / "stage4/complete.json")
    run = read_json(source / "stage4/run_configuration.json")
    history = read_json(source / "stage4/history.json")
    envelope, result = (
        read_json(test / "evaluation_plan.json"),
        read_json(test / "test_results.json"),
    )
    selection = read_json(joint / "selection.json")
    receipt = read_json(test / "receipt_verification.json")
    model = source / "stage4/best.keras"
    model_sha = sha256(model)
    if (
        receipt["status"] != "passed"
        or model_sha != complete["best_model"]["sha256"]
        or len(history) != 5
        or complete["epochs_completed"] != 5
    ):
        raise ValueError("완료 학습·Test 수신 검사를 확인하세요.")
    metrics = verify_selection(envelope, result, selection, model_sha)
    preserved_paths = [
        root / "results/CANDIDATE_INDEX.json",
        model,
        source / "stage4/complete.json",
        source / "stage4/run_configuration.json",
        source / "stage4/history.json",
        joint / "selection.json",
        test / "evaluation_plan.json",
        test / "test_results.json",
    ]
    before = {p.relative_to(root).as_posix(): sha256(p) for p in preserved_paths}
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    package = output / "package"
    package.mkdir()
    classes = result["classes"]
    card = {
        "schema": "mediflow_smartphone_adaptation_package_v1",
        "package_id": PACKAGE_ID,
        "status": "research_followup_candidate_not_device_validated",
        "domain": "web_skin",
        "adaptation_stage": 4,
        "architecture": "pmg",
        "model_file": "model.keras",
        "model_sha256": model_sha,
        "class_names": classes,
        "class_count": len(classes),
        "normal_class_included": True,
        "normal_class_index": classes.index("정상"),
        "output_adapter": "sum_logits_softmax",
        "preprocessing": preprocessing(),
        "input_scope": {
            "device_type": "smartphone",
            "body_region": "face",
            "automatic_domain_selection": False,
        },
        "unknown_input_handling": {
            "automatic_quality_or_scope_detection": False,
            "validated_abstention_threshold": None,
            "scores_are_calibrated_correctness_probabilities": False,
            "behavior": (
                "Readable out-of-scope inputs may be assigned a known class; not a diagnosis"
            ),
        },
        "training": {
            "epochs_completed": 5,
            "best_epoch": complete["best_epoch"],
            "model_spec": complete["model_spec"],
            "study": run["study"],
            "applied_profile": run["applied_profile"],
            "alignment_loss": False,
            "environment": {k: run[k] for k in ("tensorflow", "keras", "numpy", "gpu")},
        },
        "selection": {
            "made_using": "Source Validation before final Source Test",
            "model": "stage4",
            "rule": read_json(joint / "reproduction/study.json")["joint_experiment"][
                "comparison_selection"
            ],
            "final_five_condition_selection": selection,
            "test_used_for_selection": False,
            "official_public_candidate_index_replaced": False,
        },
        "source_test": {
            "plan_sha256": envelope["plan_sha256"],
            "metrics": metrics,
            "count": 400,
            "input_kind": "source",
            "historical_checkpoint_independence_verified": False,
            "target_disease_accuracy": None,
        },
        "limitations": [
            "Five people with five phones; person/device effects are confounded",
            "No disease labels for actual phone captures; no Target Accuracy",
            "Source Test: public-data images with synthetic transforms; no phone disease labels",
            "Prior checkpoint exposure and person/lesion/session split independence are unverified",
            "Pixel similarity and feature MMD/CORAL improvement were not jointly demonstrated",
        ],
    }
    shutil.copyfile(model, package / "model.keras")
    shutil.copyfile(
        Path(__file__).with_name("smartphone_package_inference.py"), package / "inference.py"
    )
    for name, value in (
        ("model_card.json", card),
        ("class_names.json", classes),
        ("preprocessing.json", card["preprocessing"]),
    ):
        shared.write_json(package / name, value)
    (package / "requirements.txt").write_text(
        "tensorflow==2.20.0\nkeras==3.13.2\nnumpy>=1.26,<3\npillow>=10,<13\n", encoding="utf-8"
    )
    (package / "example_usage.py").write_text(
        "from inference import load_predictor\nimport json,sys\n"
        "predictor=load_predictor()\nfor path in sys.argv[1:]:\n"
        "    print(json.dumps(predictor.predict_file(path),ensure_ascii=False))\n",
        encoding="utf-8",
    )
    reports = package / "reports"
    reports.mkdir()
    report_sources = {
        "training_complete.json": source / "stage4/complete.json",
        "training_run_configuration.json": source / "stage4/run_configuration.json",
        "training_history.json": source / "stage4/history.json",
        "validation_selection.json": joint / "selection.json",
        "source_test_results.json": test / "test_results.json",
        "source_test_evaluation_plan.json": test / "evaluation_plan.json",
    }
    for name, path in report_sources.items():
        shutil.copyfile(path, reports / name)
    shared.write_json(
        reports / "source_files.json",
        {
            name: {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
            for name, path in report_sources.items()
        },
    )
    readme = """# MediFlow 스마트폰 얼굴 · 4차 연구용 패키지

Validation에서 미리 선정한 PMG·EfficientNet-B0·256 저장 모델이다. 기존 웹캠 패키지와 별도다.
스마트폰으로 촬영한 얼굴 입력을 위해 EXIF 방향 보정·ICC→sRGB·중앙 자르기를 포함한다.
정상 포함 5클래스: **건선·아토피·여드름·정상·주사**. 실제 질환 정답이 있는 휴대폰 평가 결과는 없다.

## 실행

Python 3.10~3.13 환경에서 `python -m pip install -r requirements.txt` 후
`python inference.py --image "휴대폰사진.jpg"`를 실행한다. 사진은 실제 파일 경로로 바꾼다.
여러 장은 `python example_usage.py "사진1.jpg" "사진2.jpg"`로 모델을 한 번만 로드한다.
JPEG·PNG 입력을 권장한다. 현재 Pillow에서 읽지 못하는 HEIC는 JPEG/PNG로 준비해야 한다.

스마트폰 기본 경로는 EXIF 방향을 반영하고 내장 ICC 색상 프로필을 sRGB로 변환한다.
프로필이 없으면 sRGB로 가정하며 손상된 프로필은 오류다. 중앙 정사각형 256px,
RGB float32 0–255, TensorFlow bilinear·antialias=False다. 외부 `/255`를 추가하지 않는다.
학습 때의 조명·색·흐림 변환은 실제 사진에 다시 적용하지 않는다.

연구 재현용 Source Test 입력은 `python inference.py --image "원본데이터.png" --input-kind source`다.
이 경로는 고정 Test와 같이 TensorFlow로 읽으며 EXIF·ICC 보정을 하지 않는다.
아래 Source Test 수치는 이 `source` 경로의 기록이다.
기본 스마트폰 경로의 실제 질환 성능으로 쓰지 않는다.

PMG 네 logit을 더한 뒤 softmax를 적용하며 클래스별 점수는 정답 확률로 보정되지 않았다.
JSON에 모델·분야·예측 인덱스/이름·5개 점수·입력 종류를 기록한다.
스마트폰+얼굴인지 호출자가 확인해 선택한다. 품질·촬영 부위·범위 밖 입력은 자동 판정하지 않는다.
범위 밖 이미지도 기존 클래스 하나로 나올 수 있으며 질환 진단·치료 판단용으로 검증하지 않았다.
파일 없음·해독/ICC 실패·해시/클래스/모델 계약 불일치는 JSON 오류와 종료 코드 1로 알린다.

## 고정 모델의 Source Test 기록

| 조건 | Accuracy | Macro-F1 |
| --- | --- | --- |
"""
    for condition, values in metrics.items():
        readme += f"| {condition} | {values['accuracy']} | {values['macro_f1']} |\n"
    readme += (
        "\n모델·클래스·전처리·학습/검증/Test 원본·해시를 함께 보존했다. "
        "사진·복호화 키는 포함하지 않는다.\n"
    )
    (package / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    shared.manifest_for(package, PACKAGE_ID)
    verification, sample = verify_execution(package, model, complete["model_spec"], work)
    shared.write_json(package / "contract_verification.json", verification)
    shared.manifest_for(package, PACKAGE_ID)
    archive = output / (PACKAGE_ID + ".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(package.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zipped.write(path, path.relative_to(package).as_posix())
    extracted = work / "extracted"
    shared.extract_zip(archive, extracted)
    helper = shared.load_standalone(extracted)
    helper.verify_package(extracted)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    for script in ("inference.py", "example_usage.py"):
        command = [sys.executable, str(extracted / script)]
        command += ["--image", str(sample)] if script == "inference.py" else [str(sample)]
        process = subprocess.run(
            command, cwd=extracted, env=env, capture_output=True, text=True, encoding="utf-8"
        )
        if process.returncode or json.loads(process.stdout)["model_sha256"] != model_sha:
            raise RuntimeError("압축 해제본 단독 실행 실패: " + process.stderr[-1500:])
    if any(sha256(root / name) != digest for name, digest in before.items()):
        raise ValueError("원본 모델·결과·후보 색인이 변경됐습니다.")
    archive.with_suffix(".zip.sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n", encoding="utf-8"
    )
    index = {
        "schema": "mediflow_smartphone_package_index_v1",
        "package_id": PACKAGE_ID,
        "model_sha256": model_sha,
        "package_directory": "package",
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "selected_on": "Source Validation",
        "stage": 4,
        "device": "smartphone",
        "body_region": "face",
        "source_test_metrics": metrics,
        "isolated_cli_passed": True,
        "example_usage_passed": True,
        "preserved_source_file_sha256": before,
        "official_candidate_index_changed": False,
        "new_training_executed": False,
        "real_device_disease_accuracy_measured": False,
        "code_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "code_sha256": {
            p.name: sha256(p)
            for p in (Path(__file__), Path(__file__).with_name("smartphone_package_inference.py"))
        },
    }
    shared.write_json(output / "package_index.json", index)
    print(
        json.dumps(
            {
                k: index[k]
                for k in ("package_id", "archive_bytes", "model_sha256", "isolated_cli_passed")
            },
            ensure_ascii=False,
        )
    )
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output, args.work)


if __name__ == "__main__":
    main()
