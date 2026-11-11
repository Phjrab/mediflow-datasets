"""Package frozen adaptation follow-up candidates without training or service integration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

from mediflow_datasets.adaptation_package_inference import file_sha256

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = "configs/adaptation_stage4_test_20261111.json"
STAGE4 = "results/camera_adaptation_stage4_before_refit_20261111"
STAGE2 = "results/camera_adaptation_stage2_20261110"
TEST4 = "results/camera_adaptation_stage4_test_20261111"
TEST3 = "results/camera_adaptation_final_test_20261110"
DECISIONS = {
    "skin": (4, "stage4_revised"),
    "web_skin": (4, "stage4_revised"),
    "hair": (2, "stage2_expanded"),
}
CONDITION_NAMES = {
    "clean": "원본",
    "lighting": "기존 조명",
    "expanded": "기존 확장",
    "stage4_lighting": "4차 조명",
    "stage4_expanded": "4차 확장",
}


def read(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def metrics_for_checkpoint(record, result, domain, arm, model_sha256, classes):
    """Reject metrics belonging to another checkpoint, especially Hair Stage 4."""
    lock = record["plan"]["domains"][domain]
    if (
        result["domain"] != domain
        or result["classes"] != classes
        or result["identity"]["plan_sha256"] != record["plan_sha256"]
        or lock["models"][arm]["sha256"] != model_sha256
        or result["test_count"] != lock["test_count"]
        or arm not in result["metrics"]
        or result.get("model_sha256", {}).get(arm, model_sha256) != model_sha256
    ):
        raise ValueError("Source Test metrics do not belong to the packaged checkpoint")
    return result["metrics"][arm]


def source_specs(root=ROOT):
    root = Path(root)
    plan = read(root / PLAN_PATH)
    test4 = read(root / TEST4 / "evaluation_plan.json")
    if test4["plan"] != plan or test4["plan_sha256"] != file_sha256(root / PLAN_PATH):
        raise ValueError("Frozen Test plan changed")
    chosen = read(root / STAGE4 / "selection.json")["selected_on_validation"]
    names = {"stronger_expanded": "stage4_revised", "previous_expanded": "stage2_expanded"}
    specs = []
    for domain, (stage, arm) in DECISIONS.items():
        if names[chosen[domain]] != arm:
            raise ValueError("Validation follow-up selection changed")
        source = root / (STAGE4 if stage == 4 else STAGE2) / domain
        run, complete = read(source / "run_configuration.json"), read(source / "complete.json")
        task = run["training_configuration"]["tasks"][domain]
        lock = plan["domains"][domain]
        model = source / complete["best_model"]["file"]
        sha = file_sha256(model)
        if (
            sha != complete["best_model"]["sha256"]
            or sha != lock["models"][arm]["sha256"]
            or task["classes"] != lock["classes"]
            or task["input_policy"] != plan["input_policies"][domain]
            or complete["model_config"] != {**lock["model_spec"], "role": lock["start_role"]}
        ):
            raise ValueError(f"{domain}: model/class/input contract changed")
        test_root = root / (TEST4 if stage == 4 else TEST3)
        record = read(test_root / "evaluation_plan.json")
        result = read(test_root / f"{domain}.json")
        metrics = metrics_for_checkpoint(record, result, domain, arm, sha, task["classes"])
        selection_report = root / STAGE4 / domain / "complete.json"
        selection_arm = chosen[domain]
        manifest = root / f"results/camera_adaptation_stage1_20261110/{domain}/source_manifest.json"
        if file_sha256(manifest) != lock["manifest_sha256"]:
            raise ValueError("Frozen split manifest changed")
        history = source / "expanded/history.json"
        if len(read(history)) != run["training_configuration"]["epochs"]:
            raise ValueError("Training history is incomplete")
        specs.append(
            {
                "domain": domain,
                "stage": stage,
                "arm": arm,
                "source": source,
                "model": model,
                "model_sha256": sha,
                "run": run,
                "complete": complete,
                "task": task,
                "lock": lock,
                "test_root": test_root,
                "test_record": record,
                "metrics": metrics,
                "selection_report": selection_report,
                "selection_metrics": read(selection_report)["validation"][selection_arm],
                "manifest": manifest,
                "history": history,
            }
        )
    return specs


def preprocessing_for(spec):
    task, model = spec["task"], spec["complete"]["model_config"]
    return {
        "input_size": model["size"],
        "input_policy": task["input_policy"],
        "color": "RGB",
        "dtype": "float32",
        "pixel_range": [0, 255],
        "layout": "NHWC",
        "resize_method": "bilinear",
        "antialias": False,
        "exif_rotation_applied": False,
        "external_division_by_255": False,
        "internal_rescaling": "saved model includes Rescaling(1/255)",
        "training_augmentation_applied_during_inference": False,
    }


def model_card(spec, package_id, root):
    domain, stage, classes = spec["domain"], spec["stage"], spec["task"]["classes"]
    source, test_root = spec["source"], spec["test_root"]
    config = spec["run"]["training_configuration"]
    manifest = read(spec["manifest"])["original"]
    normal = domain == "web_skin"
    return {
        "schema": "mediflow_adaptation_followup_package_v1",
        "package_id": package_id,
        "status": "research_followup_candidate_not_device_validated",
        "domain": domain,
        "adaptation_stage": stage,
        "model_file": "model.keras",
        "model_sha256": spec["model_sha256"],
        "class_names": classes,
        "class_count": len(classes),
        "normal_class_included": normal,
        "normal_class_index": classes.index("정상") if normal else None,
        "architecture": spec["complete"]["model_config"]["architecture"],
        "output_adapter": "sum_logits_softmax" if domain == "web_skin" else "direct_softmax",
        "preprocessing": preprocessing_for(spec),
        "input_scope": {
            "device_type": "webcam" if normal else "usb_microscope",
            "body_region": {"skin": "skin", "web_skin": "face", "hair": "scalp"}[domain],
            "automatic_domain_selection": False,
        },
        "unknown_input_handling": {
            "automatic_quality_or_scope_detection": False,
            "validated_abstention_threshold": None,
            "behavior": (
                "A readable out-of-scope image may be assigned a known class; not a diagnosis"
            ),
            "scores_are_calibrated_correctness_probabilities": False,
        },
        "training": {
            "epochs_completed": config["epochs"],
            "best_epoch": spec["complete"]["best_epoch"],
            "seed": config["seed"],
            "learning_rate": config["learning_rate"],
            "augmentation_probability": config["augmentation_probability"],
            "batch_size": spec["task"]["batch_size"],
            "label_smoothing": spec["complete"]["model_config"]["label_smoothing"],
            "starting_model": spec["complete"]["model_config"],
            "source_zip_sha256": spec["task"]["source_sha256"],
            "source_manifest_sha256": spec["lock"]["manifest_sha256"],
            "split_counts": {key: len(rows) for key, rows in manifest.items()},
            "profile": spec["run"]["profile"],
            "recorded_environment": {
                k: spec["run"][k] for k in ("tensorflow", "keras", "numpy", "opencv", "gpu")
            },
        },
        "selection": {
            "made_using": "Validation, before the Stage 4 Source Test",
            "scope": (
                "initial vs Stage2 vs Stage4, not all five models under the new Stage4 profile"
            ),
            "rule": "max worst(clean, previous expanded, stronger expanded) F1; clean; earlier arm",
            "metrics": spec["selection_metrics"],
            "condition_meanings": {
                "clean": "original Validation",
                "weak": "Stage2 profile Validation",
                "expanded": "Stage4 profile Validation",
            },
            "official_public_candidate_index_replaced": False,
        },
        "source_test": {
            "plan_sha256": spec["test_record"]["plan_sha256"],
            "model_arm": spec["arm"],
            "metrics": spec["metrics"],
            "condition_meanings": {c: CONDITION_NAMES[c] for c in spec["metrics"]},
            "unmeasured_conditions": [c for c in CONDITION_NAMES if c not in spec["metrics"]],
            "previously_inspected_test": True,
            "historical_checkpoint_independence_verified": False,
            "target_disease_accuracy": None,
        },
        "provenance": {
            "repository_model": spec["model"].relative_to(root).as_posix(),
            "repository_training_report": (source / "complete.json").relative_to(root).as_posix(),
            "repository_source_test": (test_root / f"{domain}.json").relative_to(root).as_posix(),
            "source_test_report_sha256": file_sha256(test_root / f"{domain}.json"),
            "source_selection_report_sha256": file_sha256(spec["selection_report"]),
        },
        "limitations": [
            "Source Test uses original public-data images and synthetic camera transformations",
            "Four-person unlabeled pilot captures are not an independent labeled Target Test",
            "Person/lesion/session and historical checkpoint exposure are not fully verified",
            "Hair package is five-class; the separate six-class packages are not included",
            "No new training, clinical accuracy evaluation or device benchmark during packaging",
        ],
    }


def manifest_for(folder, package_id):
    manifest = {
        "schema": "mediflow_adaptation_package_manifest_v1",
        "package_id": package_id,
        "files": {
            p.relative_to(folder).as_posix(): file_sha256(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != "package_manifest.json" and "__pycache__" not in p.parts
        },
    }
    path = folder / "package_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def readme_for(card):
    names = {"skin": "현미경 피부", "web_skin": "웹캠 얼굴", "hair": "현미경 두피"}
    text = f"""# MediFlow {names[card["domain"]]} · {card["adaptation_stage"]}차 사용 패키지

이 폴더 하나에서 모델을 불러와 사진 한 장을 분류합니다.
연구용 후속 검토 후보이며 실제 장비 질환 정확도를 검증한 배포 모델은 아닙니다.

모델: `{card["package_id"]}`
정상 클래스 {"포함" if card["normal_class_included"] else "없음"}, 클래스 {card["class_count"]}개.
장비와 촬영 부위를 확인해 이 분야의 패키지를 직접 선택합니다. 입력 분야를 자동 판별하지 않습니다.

## 실행

프로젝트와 같은 TensorFlow 2.20.0·Keras 3.13.2 환경을 사용합니다.
새 환경은 이 폴더에서 아래처럼 준비합니다.
기본 Python이 프로젝트 지원 범위인 3.10~3.13인지 확인합니다.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe inference.py --image "촬영사진.jpg"
```

사진 경로는 실제 사용할 파일로 바꿉니다.
이미 준비된 환경에서는 `python inference.py --image "사진경로"`로 실행할 수 있습니다.
같은 모델로 여러 장을 처리하는 예시는 `example_usage.py`에 있습니다.
`python example_usage.py "사진1.jpg" "사진2.jpg"`로 실행합니다.

## 입력 처리와 출력

입력은 RGB float32 0~255이고 모델 내부 Rescaling을 사용합니다. 외부 `/255`는 추가하지 않습니다.
전처리는 `{card["preprocessing"]["input_policy"]}`입니다.
입력 크기는 {card["preprocessing"]["input_size"]}입니다.
얼굴은 중앙 정사각형으로 자르고 피부·두피는 전체 영상을 리사이즈합니다.
TensorFlow bilinear·antialias=False이며 학습/Test와 같이 EXIF 회전을 자동 적용하지 않습니다.
학습 때 사용한 밝기·색·흐림 증강을 실제 촬영 사진에 다시 적용하지 않습니다.

JSON에는 분야·모델 버전·클래스 인덱스와 이름·클래스별 점수가 나옵니다.
점수는 정답 확률로 검증되지 않았습니다.
PMG 얼굴 모델은 네 logit을 합산한 뒤 softmax를 적용합니다.
파일 없음·이미지 해독 실패·패키지 해시 불일치·모델 출력 불일치는 오류로 종료합니다.
CLI 오류는 표준 오류에 JSON으로 출력하고 종료 코드 1을 반환합니다.

촬영 부위·질환 범위·사진 품질을 자동 판정하지 않으므로
범위 밖 사진도 기존 클래스 하나로 분류될 수 있습니다.
질환 확정이나 치료 판단에 사용하지 않습니다.
두피는 5클래스이며 별도 6클래스 모델과 다릅니다.

## 평가 기록

아래 수치는 새 측정값이 아니라 이 모델 파일에 대응하는 저장된 Source Test 원본입니다.

| 조건 | Accuracy | Macro-F1 |
| --- | ---: | ---: |
"""
    for condition, values in card["source_test"]["metrics"].items():
        text += f"| {CONDITION_NAMES[condition]} | {values['accuracy']} | {values['macro_f1']} |\n"
    text += """
두피 2차의 새 4차 조명·확장 Test는 미측정입니다.
그 점수를 두피 4차 결과에서 가져오지 않았습니다.
Validation의 `expanded`는 4차 변환, Source Test의 `expanded`는 기존 2차 변환입니다.
같은 이름을 같은 조건으로 섞지 않습니다.

## 파일

- `model.keras`: 원본과 바이트가 같은 저장 모델.
- `class_names.json`, `preprocessing.json`, `model_card.json`:
  클래스 순서·입력·버전·기록된 조건과 지표.
- `inference.py`, `example_usage.py`, `requirements.txt`: 단독 사용 코드와 환경.
- `reports`: 원본 학습·선정·Test 기록과 평가 계획.
  원본 기록의 과거 경로는 실행 입력으로 사용하지 않습니다.
- `contract_verification.json`: 패키징 때의 로드·전처리·예측 일치 검사. 새로운 성능 측정이 아닙니다.
- `package_manifest.json`: 파일별 SHA-256.
  ZIP 자체의 SHA-256은 함께 제공한 `.sha256` 파일을 확인합니다.

이 패키지는 사용을 위한 자료입니다.
원본 학습 데이터·실제 참여자 사진·복호화 키는 포함하지 않습니다.
모델은 원본 저장본을 그대로 복사했으며 학습 재개를 보장하는 패키지는 아닙니다.
원본 연구 자료와 기존 공식 후보는 프로젝트에 보존돼 있습니다.
"""
    return text


def populate(folder, spec, root):
    package_id = f"{spec['domain']}_camera_adaptation_stage{spec['stage']}_20261111"
    if spec["domain"] == "hair":
        package_id += "_5class"
    folder.mkdir(exist_ok=False)
    card = model_card(spec, package_id, root)
    shutil.copyfile(spec["model"], folder / "model.keras")
    shutil.copyfile(
        Path(__file__).with_name("adaptation_package_inference.py"), folder / "inference.py"
    )
    write_json(folder / "class_names.json", card["class_names"])
    write_json(folder / "preprocessing.json", card["preprocessing"])
    write_json(folder / "model_card.json", card)
    (folder / "requirements.txt").write_text(
        "tensorflow==2.20.0\nkeras==3.13.2\nnumpy>=1.26,<3\n", encoding="utf-8"
    )
    (folder / "README.md").write_text(readme_for(card), encoding="utf-8", newline="\n")
    (folder / "example_usage.py").write_text(
        '"""Reuse this package model; no application/service integration."""\n'
        "import argparse\nimport json\nfrom inference import load_predictor\n\n"
        'if __name__ == "__main__":\n'
        "    parser = argparse.ArgumentParser()\n"
        '    parser.add_argument("images", nargs="+")\n'
        "    args = parser.parse_args()\n"
        "    predictor = load_predictor()\n"
        "    for image in args.images:\n"
        "        print(json.dumps(predictor.predict_file(image), ensure_ascii=False))\n",
        encoding="utf-8",
        newline="\n",
    )
    reports = folder / "reports"
    reports.mkdir()
    sources = {
        "training_complete.json": spec["source"] / "complete.json",
        "training_run_configuration.json": spec["source"] / "run_configuration.json",
        "training_history.json": spec["history"],
        "selection_stage4.json": root / STAGE4 / "selection.json",
        "source_test_original.json": spec["test_root"] / f"{spec['domain']}.json",
        "source_test_evaluation_plan.json": spec["test_root"] / "evaluation_plan.json",
    }
    selection_target = (
        "training_complete.json" if spec["stage"] == 4 else "selection_comparison.json"
    )
    if spec["stage"] != 4:
        sources[selection_target] = spec["selection_report"]
    for name, source in sources.items():
        shutil.copyfile(source, reports / name)
    write_json(
        reports / "source_files.json",
        {
            "files": {
                name: {
                    "repository_relative_path": source.relative_to(root).as_posix(),
                    "sha256": file_sha256(source),
                }
                for name, source in sources.items()
            },
            "selection_validation_report": selection_target,
        },
    )
    manifest_for(folder, package_id)
    return card


def load_standalone(folder):
    spec = importlib.util.spec_from_file_location("_packaged_inference", folder / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_package(folder, spec, inputs):
    import keras
    import numpy as np
    import tensorflow as tf

    from mediflow_datasets import adaptation_stage1 as training
    from mediflow_datasets import domain_adaptation_experiment as experiment

    helper = load_standalone(folder)
    predictor = helper.load_predictor(folder)
    original = keras.models.load_model(spec["model"], compile=False, safe_mode=True)
    oracle = training.predictor(original, spec["complete"]["model_config"])
    cases = []
    for path in inputs:
        expected = experiment.image_tensor(
            path.read_bytes(),
            spec["complete"]["model_config"]["size"],
            spec["task"]["input_policy"],
        )
        actual = helper.load_image(path, predictor.preprocessing).numpy()[0]
        if not np.array_equal(expected, actual):
            raise ValueError("Packaged preprocessing differs from the frozen evaluation")
        source_scores = oracle(expected[None, ...]).numpy()[0]
        packaged = predictor.predict_file(path)
        scores = np.asarray([row["score"] for row in packaged["scores"]])
        if (
            not np.allclose(source_scores, scores, atol=1e-6, rtol=1e-5)
            or int(source_scores.argmax()) != packaged["predicted_index"]
        ):
            raise ValueError("Packaged predictions differ from the original model")
        cases.append(
            {
                "input": path.name,
                "image_sha256": file_sha256(path),
                "preprocessing_exactly_equal": True,
                "max_score_absolute_difference": float(np.max(np.abs(source_scores - scores))),
                "predicted_index": packaged["predicted_index"],
            }
        )
    result = {
        "status": "passed",
        "purpose": "execution parity only; synthetic inputs are not clinical ground truth",
        "model_sha256": spec["model_sha256"],
        "cases": cases,
        "score_atol": 1e-6,
        "score_rtol": 1e-5,
        "source_test_recomputed": False,
        "new_training_executed": False,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
        },
    }
    del predictor, original, oracle
    keras.backend.clear_session()
    return result


def extract_zip(archive_path, destination):
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        names = set()
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in info.filename
                or ":" in info.filename
                or info.filename.casefold() in names
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise ValueError("Unsafe package archive entry")
            names.add(info.filename.casefold())
        if archive.testzip() is not None:
            raise ValueError("Package ZIP CRC failed")
        archive.extractall(destination)


def build_packages(output, work, root=ROOT):
    import numpy as np
    from PIL import Image

    root, output, work = map(lambda p: Path(p).resolve(), (root, output, work))
    if output.exists() or work.exists():
        raise FileExistsError(
            "Choose new output and work directories; existing files are preserved"
        )
    specs = source_specs(root)
    public_index_before = file_sha256(root / "results/CANDIDATE_INDEX.json")
    preserved = {root / PLAN_PATH, root / STAGE4 / "selection.json"}
    for spec in specs:
        preserved.update(
            [
                spec["model"],
                spec["source"] / "complete.json",
                spec["source"] / "run_configuration.json",
                spec["history"],
                spec["manifest"],
                spec["selection_report"],
                spec["test_root"] / f"{spec['domain']}.json",
                spec["test_root"] / "evaluation_plan.json",
            ]
        )
    before = {path.relative_to(root).as_posix(): file_sha256(path) for path in sorted(preserved)}
    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("adaptation_package_inference.py").resolve(),
    ]
    code_names = [p.relative_to(root).as_posix() for p in code_paths]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    code_status = subprocess.run(
        ["git", "status", "--porcelain", "--", *code_names],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    work.mkdir(parents=True)
    output.mkdir(parents=True)
    inputs_dir = work / "synthetic_inputs"
    inputs_dir.mkdir()
    rng = np.random.default_rng(42)
    wide = inputs_dir / "synthetic_wide.png"
    portrait = inputs_dir / "synthetic_portrait.jpg"
    Image.fromarray(rng.integers(0, 256, (61, 89, 3), dtype=np.uint8)).save(wide)
    Image.fromarray(rng.integers(0, 256, (91, 63, 3), dtype=np.uint8)).save(portrait)
    catalog = {
        "schema": "mediflow_adaptation_package_index_v1",
        "status": "research_followup_packages",
        "selection": "skin Stage4, web_skin Stage4, hair Stage2; fixed using Validation",
        "official_public_index_replaced": False,
        "packaging_code": {
            "repository_head": head,
            "uncommitted_code_status": code_status,
            "file_sha256": {p.relative_to(root).as_posix(): file_sha256(p) for p in code_paths},
        },
        "packages": {},
    }
    for spec in specs:
        domain = spec["domain"]
        print(f"[{domain}] 원본·클래스·평가 기록 확인, 패키지 작성", flush=True)
        folder = work / domain
        card = populate(folder, spec, root)
        verification = check_package(folder, spec, (wide, portrait))
        write_json(folder / "contract_verification.json", verification)
        manifest_for(folder, card["package_id"])
        archive = output / (card["package_id"] + ".zip")
        with zipfile.ZipFile(
            archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zipped:
            for path in sorted(folder.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    zipped.write(path, path.relative_to(folder).as_posix())
        extracted = work / (domain + "_extracted")
        extract_zip(archive, extracted)
        load_standalone(extracted).verify_package(extracted)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        env.pop("PYTHONPATH", None)
        process = subprocess.run(
            [sys.executable, str(extracted / "inference.py"), "--image", str(wide)],
            cwd=extracted,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if process.returncode:
            raise RuntimeError(f"{domain}: standalone ZIP CLI failed: {process.stderr[-2000:]}")
        cli = json.loads(process.stdout)
        if (
            cli["model_sha256"] != spec["model_sha256"]
            or cli["predicted_index"] != verification["cases"][0]["predicted_index"]
        ):
            raise ValueError("Extracted ZIP CLI result mismatch")
        example = subprocess.run(
            [sys.executable, str(extracted / "example_usage.py"), str(wide)],
            cwd=extracted,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if (
            example.returncode
            or json.loads(example.stdout)["predicted_index"] != cli["predicted_index"]
        ):
            raise ValueError("Extracted ZIP example did not match the inference CLI")
        sha = file_sha256(archive)
        archive.with_suffix(".zip.sha256").write_text(f"{sha}  {archive.name}\n", encoding="utf-8")
        if file_sha256(spec["model"]) != spec["model_sha256"]:
            raise ValueError("Source model changed while packaging")
        catalog["packages"][domain] = {
            "package_id": card["package_id"],
            "archive": archive.name,
            "archive_sha256": sha,
            "archive_bytes": archive.stat().st_size,
            "model_sha256": card["model_sha256"],
            "adaptation_stage": spec["stage"],
            "class_names": card["class_names"],
            "normal_class_included": card["normal_class_included"],
            "preprocessing": card["preprocessing"],
            "verification": verification,
            "zip_file_verification_passed": True,
            "isolated_cli_passed": True,
            "example_usage_passed": True,
        }
        print(f"[{domain}] ZIP·원본 예측 일치·단독 실행 예시 확인 완료", flush=True)
    if file_sha256(root / "results/CANDIDATE_INDEX.json") != public_index_before:
        raise ValueError("Official candidate index changed")
    if any(file_sha256(root / relative) != sha for relative, sha in before.items()):
        raise ValueError("Original model, split or report changed while packaging")
    write_json(output / "package_index.json", catalog)
    write_json(
        output / "packaging_verification.json",
        {
            "status": "passed",
            "completed_domains": list(catalog["packages"]),
            "source_models_unchanged": True,
            "preserved_source_file_sha256": before,
            "packaging_code": catalog["packaging_code"],
            "official_candidate_index_unchanged": True,
            "new_training_executed": False,
            "new_source_test_accuracy_measured": False,
            "real_device_accuracy_measured": False,
            "service_integration_implemented": False,
        },
    )
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    build_packages(args.output, args.work)


if __name__ == "__main__":
    main()
