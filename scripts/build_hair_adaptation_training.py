"""Package the reviewed Hair pilot for one-run Colab training and Source evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_batch as batch
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_training as training
from mediflow_datasets.candidate_reproduction import ROOT

DRIVER = '''
def run_hair_training(expected_hash):
    from google.colab import files, drive, output
    from IPython.display import display, HTML
    print("mediflow_hair_training_runtime.zip 하나를 선택하세요.")
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("이번 학습용 실행 ZIP 하나만 선택하세요.")
    bundle = Path(next(iter(uploaded))).resolve()
    del uploaded
    workspace = Path("/content") / ("mediflow_hair_training_" + uuid.uuid4().hex[:8])
    workspace.mkdir()
    runtime = unpack_bundle(bundle, workspace / "runtime", expected_hash)
    drive.mount("/content/drive")
    python, versions = prepare_python(workspace)
    if not versions["gpu"]:
        raise RuntimeError("GPU가 없습니다. 런타임 유형을 GPU로 변경한 뒤 모두 실행하세요.")
    config_path = runtime / "configs/domain_adaptation_hair_training.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = find_source(Path("/content/drive/MyDrive"), config["tasks"]["hair"])
    local_source = workspace / "hair_source.zip"
    print("Drive의 기존 두피 데이터셋을 실행 공간으로 복사합니다. 원본은 보존합니다.", flush=True)
    shutil.copyfile(source, local_source)
    parent = Path("/content/drive/MyDrive") / ("mediflow_hair_adaptation_" + uuid.uuid4().hex[:8])
    parent.mkdir()
    save_json(parent / "environment.json", versions)
    env = python_environment(python, runtime)
    result = {"status": "failed", "training_completed": False, "target_accuracy": None}
    try:
        run_logged([str(python), "-m", "mediflow_datasets.domain_adaptation_training",
            "--config", str(config_path), "--source-zip", str(local_source),
            "--calibration-file", str(runtime / "calibration/hair.json"),
            "--analysis", str(runtime / "review/hair"),
            "--plan", str(runtime / "review/hair/training_plan.json"),
            "--output", str(parent / "hair")], env=env, log_path=parent / "training.log")
        result = json.loads((parent / "hair/run_status.json").read_text())
        result["training_completed"] = True
        print((parent / "hair/RESULTS.md").read_text(), flush=True)
    except Exception as error:
        result["error"] = str(error)
        save_json(parent / "failure.json", {
            "error": str(error), "traceback": traceback.format_exc()})
        print("학습 또는 평가가 완료되지 않았습니다. 결과 ZIP의 로그를 확인해 주세요.", error)
    save_json(parent / "summary.json", result)
    for path in workspace.glob("environment_*.log"):
        shutil.copyfile(path, parent / path.name)
    destination = parent.parent / (parent.name + ".zip")
    export_review(parent, destination)
    local = workspace / destination.name
    shutil.copyfile(destination, local)
    def download_again():
        files.download(str(local))
    callback = "mediflow.hair_training_download_" + workspace.name
    output.register_callback(callback, download_again)
    display(HTML('<button onclick="google.colab.kernel.invokeFunction(\\'' + callback +
        '\\', [], {})">학습 결과 ZIP 다시 다운로드</button>'))
    print("실행 상태:", result["status"])
    print("Drive 보관 위치:", parent)
    print("실제 촬영 원본은 업로드하지 않았습니다. Target 정확도는 계산하지 않습니다.")
    files.download(str(local))
    return result
'''


def build(output, bundle, review):
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(bundle) as archive:
        content = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("src/")
            or name.startswith("results/hair/selected_models/v2/")
            or name in ("calibration/hair.json", "results/CANDIDATE_INDEX.json")
        }
    for path in (ROOT / "src/mediflow_datasets").glob("*.py"):
        content[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    config = data.read_json(ROOT / "configs/domain_adaptation.json")
    config["tasks"] = {"hair": config["tasks"]["hair"]}
    config["tasks"]["hair"].update(input_policy="legacy_resize", geometry=False)
    config["experiment_revision"] = "reviewed_hair_bounded_v3_pilot"
    spec, _, classes = runner.task_candidate("hair", config)
    profile = data.read_json(review / "profile_v3.json")
    if profile["model_sha256"] != spec["model_sha256"]:
        raise ValueError("Reviewed model and selected Hair checkpoint differ")
    summary = data.read_json(review / "summary.json")
    if summary["samples"] != 120 or len(summary["pages"]) != 15:
        raise ValueError("The reviewed 120 Hair previews are missing")
    profile.update(
        train_visual_review="approved",
        review_note=(
            "Codex inspected all 120 fixed Train before/after previews (pages 1-15). "
            "No gross hair/scale boundary loss or artificial geometry observed "
            "at preview resolution. "
            "Bounded brightness operator retained for a limited pilot, not a clinical validation. "
            "The full Source leakage audit is repeated before training; "
            "model inputs and augmentation use float32 RGB 0-255."
        ),
    )
    evidence = {
        "replay_summary.json": (review / "summary.json").read_bytes(),
        "replay_measurements.json": (review / "replay_measurements.json").read_bytes(),
    }
    for name in summary["pages"]:
        evidence["contact_pages/" + name] = (review / name).read_bytes()
    for name, body in evidence.items():
        content["review/hair/" + name] = body
    plan = {
        "schema": "mediflow_reviewed_training_v1",
        "domain": "hair",
        "status": "ready_for_pilot_training",
        "reviewed_train_previews": 120,
        "reviewed_contact_pages": list(range(1, 16)),
        "profile_sha256": training.object_hash(profile),
        "config_sha256": training.object_hash(config),
        "model_sha256": spec["model_sha256"],
        "source_sha256": config["tasks"]["hair"]["source_sha256"],
        "class_names": classes,
        "input_policy": spec["input_policy"],
        "evidence_sha256": {
            name: hashlib.sha256(body).hexdigest() for name, body in evidence.items()
        },
        "target_labels_verified": False,
        "identity_independence_verified": False,
        "identity_limitations": (
            "Source person, lesion and session IDs absent; some augmented origins missing. "
            "No current image-hash candidates found; all leakage is not ruled out."
        ),
        "held_domains": {
            "web_skin": "159 image-derivation candidate groups and ROI/transform review remain",
            "skin": (
                "1618 candidate groups; two near-copy concerns require data and baseline review"
            ),
        },
        "fixed": [
            "source ZIP, class order and splits",
            "starting checkpoint, seed 42, three epochs, batch 16, Adam learning rate 1e-5",
            "last 30 backbone layers, BatchNorm frozen, Label Smoothing 0.05",
            "RGB float32 0-255, 384 input, legacy_resize",
        ],
        "tested_variable": "apply measured bounded brightness augmentation with probability 0.5",
        "alias": {"target_guided": "brightness_only"},
        "target_comparison": "local after download; raw Target photos excluded from bundle",
        "target_accuracy": None,
        "code_provenance": {
            "repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "uncommitted_source_changes_included": True,
            "source_sha256": {
                name: hashlib.sha256(body).hexdigest()
                for name, body in content.items() if name.startswith("src/")
            },
        },
    }
    def encode(value):
        return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")

    content["configs/domain_adaptation_hair_training.json"] = encode(config)
    content["review/hair/measured_profile.json"] = encode(profile)
    content["review/hair/training_plan.json"] = encode(plan)
    destination = output / "mediflow_hair_training_runtime.zip"
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, body in content.items():
            archive.writestr(name, body)
        archive.writestr(
            "bundle_manifest.json",
            encode({name: hashlib.sha256(body).hexdigest() for name, body in content.items()}),
        )
    code = (
        (ROOT / "src/mediflow_datasets/domain_adaptation_batch.py").read_text(encoding="utf-8")
        + "\n" + DRIVER
        + f'\nTRAINING_RESULT = run_hair_training("{batch.digest(destination)}")\n'
    )
    compile(code, "Hair_Adaptation_Training_Colab.py", "exec")
    guide = (
        "# MediFlow 두피 실제 밝기 반영 비교 학습\n\n"
        "**이번에는 기존 Hair 5클래스 모델을 실제로 추가 학습하고 평가합니다.** "
        "얼굴·피부는 유사 사진 검토가 남아 이 노트북에서 학습하지 않습니다.\n\n"
        "Colab에서 GPU를 선택하고 **런타임 → 모두 실행**을 누르세요. "
        "새 `mediflow_hair_training_runtime.zip` 하나를 선택하고 Drive를 연결하세요. "
        "코드 수정이나 REVIEW_NOTE 입력은 필요 없습니다. 기존 데이터셋은 Drive에서 찾습니다.\n\n"
        "실행 순서: 원본·분할 재점검 → 기존 방식 추가 학습 3회 → "
        "촬영 밝기 반영 추가 학습 3회 → Validation으로 선택 고정 → "
        "원본/촬영 변화 Source Test 평가 → 모델·결과 ZIP 하나 다운로드. "
        "각 회는 전체 학습 사진을 한 번 학습하는 단위입니다.\n\n"
        "기존 저장 모델도 기준으로 평가합니다. Target-guided는 밝기 군과 같아 중복 실행하지 "
        "않습니다. Source 사람·병변·세션과 일부 증강 원본 정보는 확인할 수 없습니다. "
        "실제 촬영 질환 정답이 없으므로 실제 장비 정확도는 계산하지 않습니다. "
        "실제 촬영 원본은 ZIP에 포함되지 않습니다. 내려받은 새 모델로 PC에서 예측 변화와 "
        "특징 거리를 후속 비교합니다.\n\n"
        "결과는 Drive의 새 `mediflow_hair_adaptation_…` 폴더와 ZIP에 보관합니다. "
        "실패하면 summary·failure·training.log를 포함한 ZIP을 내려받습니다. "
        "정확한 실행 시간은 GPU와 Drive 속도에 따라 달라집니다. 기존 결과는 보존합니다.\n"
    )
    book = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3",
        }},
        "cells": [
            {"cell_type": "markdown", "id": "hair-training-guide", "metadata": {},
             "source": guide.splitlines(keepends=True)},
            {"cell_type": "code", "id": "hair-training-run", "metadata": {},
             "source": code.splitlines(keepends=True), "execution_count": None, "outputs": []},
        ],
    }
    with (output / "Hair_Adaptation_Training_Colab.ipynb").open("x", encoding="utf-8") as stream:
        json.dump(book, stream, ensure_ascii=False, indent=2)
    (output / "Hair_Adaptation_Training_Colab.py").write_text(code, encoding="utf-8")
    (output / "README.md").write_text(guide, encoding="utf-8")
    runner.write_json(output / "training_plan.json", plan)
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle", type=Path, default=ROOT / (
        "output/adaptation_preflight_v2_20261109/mediflow_preflight_runtime.zip"
    ))
    parser.add_argument("--review", type=Path, default=ROOT / (
        "camera_capture/_review/checks/transform_review/hair"
    ))
    args = parser.parse_args()
    build(args.output, args.bundle, args.review)
