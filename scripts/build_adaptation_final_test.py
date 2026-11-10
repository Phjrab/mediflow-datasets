"""Lock checkpoints using Validation and build the read-only Colab Test notebook."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile

from mediflow_datasets.candidate_reproduction import sha256

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "MediFlow_Adaptation_Common_v3.zip"
BUNDLE = ROOT / "camera_capture/_review/stage1_release_v3_20261110" / BUNDLE_NAME
STAGES = tuple(ROOT / f"results/camera_adaptation_stage{i}_20261110" for i in (1, 2, 3))
ARMS = ("initial", "stage1_lighting", "stage2_expanded", "stage3_aligned")
COUNTS = {"skin": 700, "web_skin": 400, "hair": 1252}
MODULE_DIR = "src/mediflow_datasets"
EVALUATOR = "adaptation_final_test.py"
SEED = 1707


def read(path):
    return json.loads(path.read_bytes())


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def lock_plan():
    bundle_hash = sha256(BUNDLE)
    for number, stage in enumerate(STAGES, 1):
        marker = read(stage / "bundle.json")
        if marker["stage"] != number or marker["sha256"] != bundle_hash:
            raise ValueError(f"Stage {number} does not match the common bundle")
    selection = read(STAGES[0] / "selection.json")["selected"]
    comparison = STAGES[2] / "validation_comparison.csv"
    with comparison.open(encoding="utf-8-sig", newline="") as source:
        validation = list(csv.DictReader(source))
    plan = {
        "scope": "source_test_only_historical_checkpoint_independence_unverified",
        "bundle_sha256": bundle_hash,
        "validation_sha256": sha256(comparison),
        "selection_rule": "Maximize minimum of clean, lighting and expanded Validation Macro-F1; "
        "then clean Validation Macro-F1; then earlier arm.",
        "transform_seed": SEED,
        "conditions": ["clean", "lighting", "expanded"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "domains": {},
    }
    with ZipFile(BUNDLE) as bundle:
        config = json.loads(bundle.read("configs/adaptation_stage1.json"))
        plan["configuration_sha256"] = object_hash(config)
        predictor_sha = sha256(ROOT / MODULE_DIR / "adaptation_stage1.py")
        plan["predictor_code_sha256"] = predictor_sha
        for name in ("camera_extended.py", "domain_adaptation_experiment.py"):
            path = f"{MODULE_DIR}/{name}"
            if bundle.read(path) != (ROOT / path).read_bytes():
                raise ValueError(f"Common bundle transform differs from {name}")
        for domain, count in COUNTS.items():
            choice = selection[domain]
            spec = choice["model_config"]
            role = choice["role"]
            if {**config["tasks"][domain]["models"][role], "role": role} != spec:
                raise ValueError(f"{domain}: invalid Stage 1 selection")
            manifest = STAGES[0] / domain / "source_manifest.json"
            if len(read(manifest)["original"]["test"]) != count:
                raise ValueError(f"{domain}: invalid Test count")
            reference = json.loads(bundle.read(f"references/{domain}.json"))
            prior = read(STAGES[2] / domain / "run_configuration.json")
            if (
                prior["training_configuration"] != config
                or prior["model"] != spec
                or prior["validation_seed"] != SEED
                or prior["profile"] != reference["extended_profile"]
                or prior["identity"]["reference_sha256"] != object_hash(reference)
                or prior["identity"]["source_manifest_sha256"] != sha256(manifest)
                or prior["identity"]["code_sha256"]["training"] != predictor_sha
            ):
                raise ValueError(f"{domain}: validation and Test inputs differ")
            models = {
                "initial": {"stage": 0, "path": spec["model"], "sha256": spec["model_sha256"]},
                "stage1_lighting": {"stage": 1, **choice["stage1_checkpoint"]},
            }
            for idx, arm in enumerate(ARMS[2:], 2):
                entry = read(STAGES[idx - 1] / domain / "complete.json")
                if (
                    entry["model_config"] != spec
                    or entry["stage1_best"] != choice["stage1_checkpoint"]
                ):
                    raise ValueError(f"{domain}: unexpected {arm} checkpoint")
                models[arm] = {
                    "stage": idx,
                    "path": f"{domain}/{entry['best_model']['file']}",
                    "sha256": entry["best_model"]["sha256"],
                }
            for info in models.values():
                root = ROOT if info["stage"] == 0 else STAGES[info["stage"] - 1]
                if sha256(root / info["path"]) != info["sha256"]:
                    raise ValueError(f"{domain}: unexpected model bytes")
            recorded = read(STAGES[2] / domain / "complete.json")["validation"]
            rows = {r["model"]: r for r in validation if r["domain"] == domain}
            if set(rows) != set(ARMS) or set(recorded) != set(ARMS):
                raise ValueError(f"{domain}: incomplete validation comparison")
            scores = {}
            for arm in ARMS:
                scores[arm] = {}
                for condition in plan["conditions"]:
                    score = float(rows[arm][f"{condition}_macro_f1"])
                    if score != recorded[arm][condition]["macro_f1"]:
                        raise ValueError(f"{domain}: inconsistent validation score for {arm}")
                    scores[arm][condition] = score
            selected = max(
                ARMS,
                key=lambda arm: (min(scores[arm].values()), scores[arm]["clean"], -ARMS.index(arm)),
            )
            plan["domains"][domain] = {
                "start_role": role,
                "models": models,
                "selected_on_validation": selected,
                "validation_macro_f1": scores,
                "reference_sha256": object_hash(reference),
                "manifest_sha256": sha256(manifest),
                "test_count": count,
            }
    return plan


def notebook(plan_json):
    overlay = {
        name: (ROOT / MODULE_DIR / name).read_bytes()
        for name in (EVALUATOR, "adaptation_stage1.py", "adaptation_stage1_colab.py")
    }
    plan = json.loads(plan_json)
    bundle_hash = plan["bundle_sha256"]
    plan_hash = hashlib.sha256(plan_json.encode()).hexdigest()
    code = f'''import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from google.colab import drive

print("MediFlow 최종 Test · 그래프 출력 설정 자동 처리 수정본", flush=True)
drive.mount("/content/drive")
my_drive = Path("/content/drive/MyDrive")
project = my_drive / "mediflow_Project"
bundle = project / "{BUNDLE_NAME}"
run_root = my_drive / "MediFlow_adaptation_runs" / "{bundle_hash[:16]}"
stages = [run_root / f"stage{{n}}" for n in (1, 2, 3)]
out = run_root / "final_test_{plan_hash[:12]}"
if not all((stage / "bundle.json").is_file() for stage in stages):
    raise FileNotFoundError("Drive에서 1·2·3차 완료 폴더를 확인하세요.")
log_root = out / "logs"
log_root.mkdir(parents=True, exist_ok=True)
run_id = uuid.uuid4().hex[:12]
log_path = log_root / f"run_{{run_id}}.log"
print("실행 로그 저장 위치", log_path, flush=True)

with tempfile.TemporaryDirectory(prefix="mediflow_final_test_", dir="/content") as temp:
    work = Path(temp)
    runtime = work / "runtime"
    local_bundle = work / bundle.name
    print("공통 ZIP v3 확인 중", flush=True)
    shutil.copyfile(bundle, local_bundle)
    with local_bundle.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != "{bundle_hash}":
            raise ValueError("공통 ZIP v3의 해시가 일치하지 않습니다.")
    with ZipFile(local_bundle) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or not path.parts or path.parts[0] not in (
                "src", "configs", "references", "results"
            ):
                continue
            if path.is_absolute() or ".." in path.parts or "\\\\" in member.filename:
                raise ValueError("공통 ZIP에 예상치 못한 경로가 있습니다.")
            archive.extract(member, runtime)
    local_bundle.unlink()
    for name, contents in {overlay!r}.items():
        (runtime / "{MODULE_DIR}" / name).write_bytes(contents)
    plan_path = work / "final_test_plan.json"
    plan_path.write_bytes({plan_json.encode()!r})
    env = {{**os.environ, "PYTHONPATH": str(runtime / "src"), "PYTHONIOENCODING": "utf-8",
           "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"}}
    launch = """from pathlib import Path
import sys
from mediflow_datasets import domain_adaptation_batch as batch
from mediflow_datasets.adaptation_stage1_colab import run_live
import subprocess
work, runtime = map(Path, sys.argv[1:3])
print("Python·TensorFlow 환경과 GPU 확인 중", flush=True)
python, versions = batch.prepare_python(work, executor=run_live)
print("확인된 실행 환경", versions, flush=True)
if not versions["gpu"]:
    raise RuntimeError("Colab에서 GPU 런타임을 선택하고 다시 실행하세요.")
env = batch.python_environment(python, runtime)
check = subprocess.run([str(python), "-c", "import matplotlib"], env=env, capture_output=True)
if check.returncode:
    subprocess.run([str(python), "-m", "pip", "install", "matplotlib>=3.8,<4"], env=env, check=True)
run_live(
    [str(python), "-m", "mediflow_datasets.adaptation_final_test", *sys.argv[3:]],
    env, work / "test.log"
)
"""
    command = [sys.executable, "-u", "-c", launch, str(work), str(runtime),
               "--runtime", str(runtime), "--stages", *map(str, stages),
               "--datasets", str(project / "datasets"), "--plan", str(plan_path),
               "--output", str(out), "--work", str(work)]
    with subprocess.Popen(
        command, env=env, start_new_session=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    ) as process, log_path.open("x", encoding="utf-8") as log:
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
            if return_code:
                raise RuntimeError(
                    f"최종 Test 실행 실패 · 종료 코드 {{return_code}} · "
                    f"위의 실제 오류 또는 Drive 로그를 확인하세요. {{log_path}}"
                )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            for diagnostic in work.glob("*.log"):
                shutil.copyfile(diagnostic, log_root / f"{{run_id}}_{{diagnostic.name}}")
    local_zip = Path("/content/MediFlow_Final_Test_Results_20261110.zip")
    shutil.make_archive(str(work / "results"), "zip", root_dir=out)
    shutil.copyfile(work / "results.zip", local_zip)
    shutil.copyfile(local_zip, out.with_suffix(".zip"))
print("완료:", out / "test_comparison.csv")
import pandas as pd
from IPython.display import Image, display
from google.colab import files
display(pd.read_csv(out / "test_comparison.csv", dtype=str, keep_default_na=False))
display(Image(filename=str(out / "plots/test_comparison.png")))
display(Image(filename=str(out / "plots/selected_confusion_matrices.png")))
files.download(str(local_zip))
'''
    compile(code, "final-test-colab", "exec")
    guide = f"""# MediFlow 기존 데이터 최종 Test 비교 · 실행 수정본

GPU 런타임을 선택하고 **모두 실행**한 뒤 Drive를 연결합니다. 이미 Drive에 있는 공통 ZIP v3,
세 원본 데이터 ZIP, 1·2·3차 모델을 읽습니다. 실제 촬영 사진의 복호화 키는 사용하지 않습니다.
별도 실행 환경의 그래프 출력 설정을 자동 처리합니다. 이 노트북을 열고 모두 실행하면 됩니다.

피부 700장, 얼굴 400장, 두피 1,252장의 원본 Test를 확인하고 학습 전·1차·2차·3차 모델을
같은 원본·조명 변환·확장 변환으로 평가합니다. 분야별 후보는 Validation으로 고정했습니다.
Test 점수로 모델을 다시 고르지 않습니다. 새 학습은 수행하지 않습니다.

결과는 Drive `MediFlow_adaptation_runs/{bundle_hash[:16]}/final_test_{plan_hash[:12]}`에 저장합니다.
CSV에는 정확도·Macro-F1·학습 전 대비 변화, 분야별 JSON에는 클래스별 지표와 혼동행렬이 있습니다.
클래스별 CSV와 비교 그래프도 저장하며, 완료되면 결과 ZIP 다운로드를 시작합니다.
중단되면 같은 노트북에서 이미 완료된 분야 결과를 재사용합니다.
Python 환경 준비부터 예측까지 실제 출력과 오류를 화면에 표시합니다.
실행 로그는 결과 폴더의 `logs`에 남습니다. 실패하면 마지막 실제 오류 문구를 확인하세요.
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
                "id": "run-test",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": code.splitlines(True),
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    plan = lock_plan()
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    nb = notebook(plan_json)
    if args.plan.exists() and args.plan.read_text(encoding="utf-8") != plan_json:
        raise ValueError("기존 고정 계획과 달라 새 평가 계획 경로가 필요합니다.")
    for path, value in (
        (args.plan, plan_json),
        (args.notebook, json.dumps(nb, ensure_ascii=False, indent=1) + "\n"),
    ):
        with path.open("w" if args.overwrite else "x", encoding="utf-8", newline="\n") as target:
            target.write(value)


if __name__ == "__main__":
    main()
