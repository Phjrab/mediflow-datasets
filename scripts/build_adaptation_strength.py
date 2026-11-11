"""Freeze measured follow-up profiles and export a single Colab training notebook."""

import argparse
import hashlib
import inspect
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile

from mediflow_datasets import domain_adaptation_batch as batch
from mediflow_datasets.camera_strength import stronger_profile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "MediFlow_Adaptation_Common_v3.zip"
BUNDLE = ROOT / "camera_capture/_review/stage1_release_v3_20261110" / BUNDLE_NAME
MODULES = (
    "adaptation_strength.py",
    "adaptation_stage1.py",
    "adaptation_stage1_colab.py",
    "adaptation_stage2.py",
)


def read(path):
    return json.loads(path.read_bytes())


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def study_configuration():
    study = {
        "schema": "mediflow_measured_strength_v1",
        "bundle_sha256": batch.digest(BUNDLE),
        "hypothesis": (
            "Bounded Source/Target-directed stronger photometric transforms improve Validation"
        ),
        "prior_source_test_inspected": True,
        "source_test_evaluated": False,
        "target_accuracy": None,
        "fixed_conditions": [
            "starting checkpoint",
            "split",
            "5 epochs",
            "seed 42",
            "Adam 1e-5",
            "batch sizes",
            "augmentation probability 0.5",
        ],
        "changed_condition": "measured camera transform profile",
        "accompanying_changes": "brightness/contrast direction and calibration policy also change",
        "profile_builder_sha256": batch.digest(ROOT / "src/mediflow_datasets/camera_strength.py"),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "uncommitted_runtime_sha256": {
            name: batch.digest(ROOT / "src/mediflow_datasets" / name) for name in MODULES
        },
        "domains": {},
    }
    with ZipFile(BUNDLE) as archive:
        config = json.loads(archive.read("configs/adaptation_stage1.json"))
        study["configuration_sha256"] = digest_json(config)
        for domain in ("skin", "web_skin", "hair"):
            reference = json.loads(archive.read(f"references/{domain}.json"))
            manifest_path = (
                ROOT / f"results/camera_adaptation_stage1_20261110/{domain}/source_manifest.json"
            )
            manifest = read(manifest_path)
            keep = {row["path"]: row["label"] for row in manifest["original"]["train"]}
            source_path = (
                ROOT
                / "camera_capture/_review/domain_adaptation_colab_20261109"
                / domain
                / "analysis/analysis.json"
            )
            cached = read(source_path)
            source = [
                row for row in cached["images"] if row["kind"] == "source" and row["path"] in keep
            ]
            if (
                cached["source_sha256"] != config["tasks"][domain]["source_sha256"]
                or len(source) != len(keep)
                or any(row["label"] != keep[row["path"]] for row in source)
            ):
                raise ValueError(f"{domain}: retained Train measurements differ from the source")
            packet_path = (
                ROOT
                / "camera_capture/_review/training_reference_inputs_20261110"
                / domain
                / "calibration.json"
            )
            packet = read(packet_path)
            if packet["fingerprint"] != reference["capture_fingerprint"]:
                raise ValueError(f"{domain}: Target calibration differs from the common bundle")
            prior = read(ROOT / f"results/camera_adaptation_stage2_20261110/{domain}/complete.json")
            profile = stronger_profile(
                reference["extended_profile"],
                source,
                packet["measurements"],
                len(config["tasks"][domain]["classes"]),
            )
            study["domains"][domain] = {
                "profile": profile,
                "previous_reference_sha256": digest_json(reference),
                "source_manifest_sha256": batch.digest(manifest_path),
                "previous_best_sha256": prior["best_model"]["sha256"],
                "measurement_provenance": {
                    "source": source_path.relative_to(ROOT).as_posix(),
                    "source_sha256": batch.digest(source_path),
                    "target": packet_path.relative_to(ROOT).as_posix(),
                    "target_sha256": batch.digest(packet_path),
                    "capture_fingerprint": packet["fingerprint"],
                    "retained_train_count": len(source),
                    "target_count": len(packet["measurements"]),
                },
            }
    return study


def notebook(study_json):
    study = json.loads(study_json)
    study_sha = hashlib.sha256(study_json.encode()).hexdigest()
    overlays = {name: (ROOT / "src/mediflow_datasets" / name).read_bytes() for name in MODULES}
    code = """import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from google.colab import drive, files
"""
    code += "\n" + inspect.getsource(batch.digest)
    code += "\n" + inspect.getsource(batch.unpack_bundle).replace(
        "mediflow_da_runtime.zip", BUNDLE_NAME
    )
    code += f'''
drive.mount("/content/drive")
my_drive = Path("/content/drive/MyDrive")
project = my_drive / "mediflow_Project"
bundle_sha = "{study["bundle_sha256"]}"
parent = my_drive / "MediFlow_adaptation_runs" / bundle_sha[:16]
out = parent / "measured_strength_{study_sha[:12]}"
logs = out / "logs"
logs.mkdir(parents=True, exist_ok=True)
run_id = uuid.uuid4().hex[:12]
log_path = logs / f"run_{{run_id}}.log"
print("실측 기반 후속 학습 · 세 분야 각각 5 epoch", flush=True)
print("결과 저장 위치", out, flush=True)
with tempfile.TemporaryDirectory(prefix="mediflow_strength_", dir="/content") as temporary:
    work = Path(temporary)
    local_bundle = work / "{BUNDLE_NAME}"
    shutil.copyfile(project / "{BUNDLE_NAME}", local_bundle)
    runtime = unpack_bundle(local_bundle, work / "runtime", bundle_sha)
    local_bundle.unlink()
    for name, content in {overlays!r}.items():
        (runtime / "src/mediflow_datasets" / name).write_bytes(content)
    study_path = work / "study.json"
    study_path.write_bytes({study_json.encode()!r})
    env = {{**os.environ, "PYTHONPATH": str(runtime / "src"), "MPLBACKEND": "Agg",
           "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}}
    command = [sys.executable, "-u", "-m", "mediflow_datasets.adaptation_strength",
               "--launch", str(runtime), str(work), str(project), str(study_path), str(my_drive)]
    with (
        subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1,
                         start_new_session=True) as process,
        log_path.open("x", encoding="utf-8") as log,
    ):
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError(f"후속 학습 미완료 · 로그를 확인하세요. {{log_path}}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            for diagnostic in work.glob("*.log"):
                shutil.copyfile(diagnostic, logs / f"{{run_id}}_{{diagnostic.name}}")
    names = ["study.json", "summary.json", "selection.json", "validation_comparison.csv",
             "plots/learning_curves.png"]
    names.extend("code/" + name for name in (
        "adaptation_strength.py", "adaptation_stage1.py", "camera_extended.py"))
    for domain in ("skin", "web_skin", "hair"):
        names.extend(f"{{domain}}/{{name}}" for name in (
            "best.keras", "complete.json", "run_configuration.json", "data_audit.json",
            "source_manifest.json", "identity.json", "expanded/history.json"))
    local_zip = Path("/content/MediFlow_Stronger_Adaptation_Results_20261110.zip")
    with ZipFile(local_zip, "w", compression=8) as archive:
        for name in names:
            archive.write(out / name, name)
    shutil.copyfile(local_zip, out / local_zip.name)
print("세 분야 후속 학습 완료", out, flush=True)
import pandas as pd
from IPython.display import Image, display
display(pd.read_csv(out / "validation_comparison.csv", dtype=str, keep_default_na=False))
display(Image(filename=str(out / "plots/learning_curves.png")))
files.download(str(local_zip))
'''
    compile(code, "stronger-colab", "exec")
    guide = f"""# MediFlow 실측 기반 변환 확대 후속 학습

**GPU 선택 → 모두 실행 → Drive 연결**을 합니다. 기존 공통 ZIP v3와 원본 데이터 ZIP,
1·2차 결과를 읽습니다. 피부 초기 B0·224, 얼굴 선정 PMG·B0·256, 두피 선정 B1·384의
1차 학습 전 상태에서 각 5 epoch, 총 세 분야를 학습합니다.

밝기·대비는 Source Train과 실제 촬영 측정의 차이를 범위 제한 안에서 반영합니다.
색·미세 대비·초점은 참여자 내 측정 변화의 반영 비율을 0.5에서 1.0으로 올립니다.
기존 2차와 학습 전 모델을 같은 원본·기존 변환·새 변환 Validation에서 비교합니다.
이번 연구는 변환 방향과 강도를 함께 바꾸는 후속 실험입니다.

결과는 `MediFlow_adaptation_runs/{study["bundle_sha256"][:16]}/measured_strength_{study_sha[:12]}`에
저장합니다. 화면에 epoch·배치·검증 진행을 표시하며 회차별 저장 상태에서 이어 실행합니다.
마지막에는 최적 모델 3개·비교표·학습곡선이 든 결과 ZIP을 다운로드합니다.
이 노트북은 Validation으로 비교·선정하며 이미 확인한 Source Test는 추론하지 않습니다.
"""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "guide",
                "metadata": {},
                "source": guide.splitlines(True),
            },
            {
                "cell_type": "code",
                "id": "run-strength",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": code.splitlines(True),
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    study_json = json.dumps(study_configuration(), ensure_ascii=False, indent=2) + "\n"
    for path, content in (
        (ROOT / "configs/camera_strength_20261110.json", study_json),
        (
            ROOT
            / (
                "notebooks/camera_adaptation/archive/"
                "MediFlow_Measured_Stronger_Adaptation_20261110.ipynb"
            ),
            json.dumps(notebook(study_json), ensure_ascii=False, indent=1) + "\n",
        ),
    ):
        with path.open("w" if args.overwrite else "x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)


if __name__ == "__main__":
    main()
