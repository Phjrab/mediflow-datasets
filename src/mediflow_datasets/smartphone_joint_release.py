"""Build one-trial Colab delivery reusing the previously uploaded phone runtime ZIP."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import subprocess
import textwrap
import zipfile
from copy import deepcopy
from pathlib import Path

from mediflow_datasets import smartphone_notebook
from mediflow_datasets.domain_adaptation_data import read_json, sha256

ROOT = Path(__file__).resolve().parents[2]
BASE_ZIP = "MediFlow_Smartphone_WebSkin_Runtime_20261111.zip"
RESULT_ZIP = "MediFlow_Smartphone_Joint_Results_20261112.zip"


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def build_notebook(base_sha, files, manifest_sha, run_id, previous_run_id):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    payload = base64.b64encode(stream.getvalue()).decode()
    book = smartphone_notebook.build_notebook(base_sha, stage2_reused=False)
    guide = f"""
    # MediFlow 휴대폰 실측 근접 후보 · 추가 1회 학습

    **기존 공통 ZIP은 그대로 → 이 새 노트북에서 GPU 선택 → 모두 실행**.
    새로운 큰 ZIP 업로드가 필요 없습니다. Drive의 mediflow_Project/{BASE_ZIP}을 재사용합니다.
    이전 1~4차 결과 폴더 `{previous_run_id}`가 Drive에 있어야 합니다. 기존 결과는 읽기만 합니다.

    - 출발: 휴대폰 적응 전 선정 PMG·B0·256 저장본. 4차 가중치를 이어 쓰지 않습니다.
    - 추가 1회, 5 epoch. Adam 0.00001, batch 8, seed 42, 변환 적용 확률 0.5.
    - 밝기 −19.624129908304198~−10.181201425385177, 대비 1~1.1808786167229048.
    - 좌우 조명차 −0.42238616943359375~0, Lab a 0~1.0117187500000009, Lab b −1.2126953125000002~0.
    - 추가 질감 배율 1, 흐림 σ 0. MMD·CORAL 학습 손실은 사용하지 않습니다.
    - 7개 영상 지표가 가까워진 탐색 후보입니다. 조명 불균일·개인차 기준은 미통과이며 실제 장비 동등성이 보장되지 않습니다.
    - 기존 Source Train/Validation/Test와 클래스 순서를 유지합니다. Source Test와 P05 사진은 평가하지 않습니다.
    - 기존 20개 Validation 결과는 그대로 재사용하고, 새 조건/새 모델 10개를 측정해 6모델×5조건으로 비교합니다.
    - 기존 키를 가려진 입력란에 한 번 넣습니다. 설정용 42장만 진단하며 P05 입력은 이 런타임에서 제외합니다.
    - 실제 epoch·배치·손실·검증·저장 로그와 학습곡선을 출력합니다. 중단되면 같은 노트북을 다시 실행합니다.

    결과는 `MyDrive/MediFlow_smartphone_runs/joint_{run_id}`와 `{RESULT_ZIP}`에 저장됩니다.
    실제 휴대폰 질환 정답이 없어 Target Accuracy는 계산하지 않습니다. 자동 패키징·공식 후보 변경은 하지 않습니다.
    """
    book["cells"][0]["source"] = textwrap.dedent(guide).strip().splitlines(keepends=True)
    setup = f'''
    from google.colab import drive
    from pathlib import Path, PurePosixPath
    import base64, hashlib, io, json, os, subprocess, sys, zipfile
    drive.mount("/content/drive")
    PROJECT=Path("/content/drive/MyDrive/mediflow_Project")
    ZIP_PATH=PROJECT/"{BASE_ZIP}"
    EXPECTED_SHA256="{base_sha}"
    MANIFEST_SHA256="{manifest_sha}"
    def digest(path):
        h=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
        return h.hexdigest()
    if not ZIP_PATH.is_file():raise FileNotFoundError("기존 공통 ZIP이 없습니다: "+str(ZIP_PATH))
    print("기존 공통 ZIP 확인 중",flush=True)
    if digest(ZIP_PATH)!=EXPECTED_SHA256:raise ValueError("기존 공통 ZIP 버전이 다릅니다. 11월12일 학습에 쓴 파일이 필요합니다.")
    RUNTIME=Path("/content/mediflow_phone_joint_{run_id}")
    patch=zipfile.ZipFile(io.BytesIO(base64.b64decode("{payload}")))
    frozen=json.loads(patch.read("runtime_manifest.json"))
    allowed={{*frozen['files'],'runtime_manifest.json'}}
    if hashlib.sha256(patch.read("runtime_manifest.json")).hexdigest()!=MANIFEST_SHA256:raise ValueError("노트북 내 실행 기록 불일치")
    RUNTIME.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as base:
        for name in allowed:
            part=PurePosixPath(name)
            if part.is_absolute() or '..' in part.parts or '\\\\' in name or ':' in name:raise ValueError("잘못된 런타임 경로")
            path=RUNTIME.joinpath(*part.parts)
            expected=MANIFEST_SHA256 if name=='runtime_manifest.json' else frozen['files'][name]
            if path.exists():
                if digest(path)!=expected:raise ValueError("기존 5차 런타임 파일이 다릅니다: "+name)
                continue
            content=patch.read(name) if name in patch.namelist() else base.read(name)
            if hashlib.sha256(content).hexdigest()!=expected:raise ValueError("실행 파일 해시 불일치: "+name)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(content)
    patch.close()
    PREVIOUS=Path("/content/drive/MyDrive/MediFlow_smartphone_runs/{previous_run_id}")
    if not PREVIOUS.is_dir():raise FileNotFoundError("이전 결과 폴더를 확인하세요: "+str(PREVIOUS))
    OUTPUT=Path("/content/drive/MyDrive/MediFlow_smartphone_runs/joint_{run_id}")
    LOCAL=Path("/content/mediflow_phone_joint_work_{run_id}")
    OUTPUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(exist_ok=True)
    print("추가 학습 결과 저장 위치:",OUTPUT)
    '''
    book["cells"][1]["source"] = textwrap.dedent(setup).strip().splitlines(keepends=True)
    cell = book["cells"][3]
    source = "".join(cell["source"]).replace(
        "mediflow_datasets.smartphone_training", "mediflow_datasets.smartphone_joint_training"
    )
    source = source.replace(
        '"--output",str(OUTPUT),', '"--previous",str(PREVIOUS), "--output",str(OUTPUT),'
    )
    cell["source"] = source.splitlines(keepends=True)
    cell["id"] = "run-one-joint-trial"
    result = f'''
    from IPython.display import display,Image
    summary=json.loads((OUTPUT/"summary.json").read_text())
    if summary.get("status")!="completed":raise RuntimeError("추가 학습이 완료되지 않았습니다.")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    display(Image(filename=str(OUTPUT/"stage5/learning_curves.png")))
    display(Image(filename=str(OUTPUT/"validation_comparison.png")))
    display(Image(filename=str(OUTPUT/"stage5_confusion_matrices.png")))
    RESULT_ZIP=OUTPUT/"{RESULT_ZIP}"
    print("Drive 결과 ZIP:",RESULT_ZIP)
    from google.colab import files
    files.download(str(RESULT_ZIP))
    '''
    book["cells"][4]["source"] = textwrap.dedent(result).strip().splitlines(keepends=True)
    return book


def build(root, output, notebook_path):
    root, output, notebook_path = (Path(p).resolve() for p in (root, output, notebook_path))
    if output.exists() or notebook_path.exists():
        raise FileExistsError("새 출력/노트북 경로를 지정하세요.")
    base = root / "camera_capture/_review/smartphone_web_skin_20261111/runtime"
    release = root / "output/smartphone_web_skin_20261111"
    prior = root / "results/smartphone_web_skin_adaptation_20261112"
    candidate_path = (
        root
        / "presentation/report_assets/smartphone_joint_review_20261112/broad_candidate_review_only.json"
    )
    candidate = read_json(candidate_path)
    if candidate["candidate_id"] != "f10926b662a4a43c" or candidate["accepted"] is not False:
        raise ValueError("검토한 탐색 후보가 아닙니다.")
    old = read_json(base / "study.json")
    study = deepcopy(old)
    receipt = read_json(prior / "receipt_verification.json")
    preparation = read_json(release / "preparation.json")
    if sha256(release / BASE_ZIP) != preparation["runtime_sha256"]:
        raise ValueError("공통 ZIP이 다릅니다.")
    if sha256(base / "study.json") != receipt["study_sha256"]:
        raise ValueError("실행 설정 연결 불일치")
    study["task"]["models"] = {"selected": study["task"]["models"]["selected"]}
    prior_files = [
        "runtime_identity.json",
        "summary.json",
        "validation_comparison.json",
        "source_manifest.json",
    ]
    previous_models = {}
    for stage, folder in [
        ("stage1", "stage1_selected"),
        ("stage2", "stage2"),
        ("stage3", "stage3"),
        ("stage4", "stage4"),
    ]:
        prior_files.append(folder + "/complete.json")
        record = read_json(prior / folder / "complete.json")
        model = prior / folder / "best.keras"
        if sha256(model) != record["best_model"]["sha256"]:
            raise ValueError("이전 모델 불일치")
        previous_models[stage] = {"file": folder + "/best.keras", "sha256": sha256(model)}
    study["joint_profile"] = candidate["profile"]
    study["joint_experiment"] = {
        "schema": "mediflow_phone_joint_single_trial_v1",
        "candidate_id": candidate["candidate_id"],
        "authorization": "User approved exploratory training of reviewed candidate",
        "candidate_all_similarity_guards_passed": False,
        "candidate_artifact_sha256": sha256(candidate_path),
        "hypothesis": "Reviewed joint photometric bundle improves robustness with less synthetic blur",
        "changed_variable": "Training image transformation bundle only",
        "training_trials": 1,
        "alignment_loss": False,
        "source_test_used": False,
        "target_holdout_used": False,
        "validation_conditions": [
            "clean",
            "phone_lighting",
            "phone_expanded",
            "phone_refit",
            "phone_joint",
        ],
        "checkpoint_selection": "max worst(clean,phone_joint) F1; clean F1; earlier epoch",
        "comparison_selection": "max worst(all five common conditions) F1; clean F1; earlier model",
        "previous_models": previous_models,
        "previous_result_files": {p: sha256(prior / p) for p in prior_files},
        "previous_runtime_identity": read_json(prior / "runtime_identity.json"),
        "previous_runtime_zip_sha256": preparation["runtime_sha256"],
        "fixed": {
            "epochs": 5,
            "seed": 42,
            "learning_rate": 0.00001,
            "augmentation_probability": 0.5,
            "batch_size": 8,
            "initialization": "original pre-adaptation selected PMG B0 256",
        },
        "prior_p05_already_inspected": True,
        "p05_reopened_in_this_run": False,
    }
    study["expected_training_trials"] = 1
    study["stage1_selection"] = "Prior selected role frozen; Stage1 not repeated"
    study["final_selection"] = study["joint_experiment"]["comparison_selection"]
    initial_manifest = read_json(base / "runtime_manifest.json")
    if sha256(base / "runtime_manifest.json") != receipt["runtime_manifest_sha256"]:
        raise ValueError("기존 실행 manifest 불일치")
    for name, expected in initial_manifest["files"].items():
        if sha256(base / name) != expected:
            raise ValueError("기존 런타임 파일이 다릅니다: " + name)
    keep = {
        name: digest
        for name, digest in initial_manifest["files"].items()
        if not name.startswith("target_encrypted/")
        or name in ("target_encrypted/calibration_256.bin", "target_encrypted/calibration_256.json")
    }
    basic = old["task"]["models"]["basic"]["model"]
    keep.pop(basic)
    patched = {
        "study.json": json_bytes(study),
        "joint_candidate.json": candidate_path.read_bytes(),
        "previous_validation.json": (prior / "validation_comparison.json").read_bytes(),
        "feature_reference.json": json_bytes(
            read_json(prior / "stage3/alignment_setup.json")["reference"]
        ),
        "previous_calibration_diagnostics.json": json_bytes(
            {
                n: v["calibration"]
                for n, v in read_json(prior / "feature_and_prediction_diagnostics.json")[
                    "models"
                ].items()
            }
        ),
    }
    reference = read_json(base / "reference.json")
    inventory = read_json(base.parent / "inventory.json")
    reference["pixel_hashes"] = [
        r["pixel_sha256"] for r in inventory if r["split"] == "calibration"
    ]
    patched["reference.json"] = json_bytes(reference)
    for name in ("smartphone_training.py", "smartphone_joint_training.py"):
        patched["src/mediflow_datasets/" + name] = (
            root / "src/mediflow_datasets" / name
        ).read_bytes()
    keep.update({p: hashlib.sha256(value).hexdigest() for p, value in patched.items()})
    manifest = {
        "schema": "mediflow_phone_runtime_manifest_v1",
        "files": keep,
        "derived_from_manifest_sha256": receipt["runtime_manifest_sha256"],
        "previous_zip_sha256": preparation["runtime_sha256"],
    }
    manifest_bytes = json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    patched["runtime_manifest.json"] = manifest_bytes
    run_id = manifest_sha[:12]
    book = build_notebook(
        preparation["runtime_sha256"],
        patched,
        manifest_sha,
        run_id,
        preparation["runtime_sha256"][:12],
    )
    output.mkdir(parents=True, exist_ok=False)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_bytes(json_bytes(book))
    (output / "study.json").write_bytes(json_bytes(study))
    (output / "runtime_manifest.json").write_bytes(manifest_bytes)
    with zipfile.ZipFile(output / "joint_code_patch_20261112.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for name, value in patched.items():
            z.writestr(name, value)
    build_record = {
        "status": "prepared_not_run",
        "notebook": notebook_path.relative_to(root).as_posix(),
        "notebook_sha256": sha256(notebook_path),
        "base_zip_sha256": preparation["runtime_sha256"],
        "base_zip_name": BASE_ZIP,
        "runtime_manifest_sha256": manifest_sha,
        "run_id": "joint_" + run_id,
        "training_trials": 1,
        "epochs": 5,
        "new_large_zip_upload_required": False,
        "embedded_patch_files": len(patched),
        "patch_zip_sha256": sha256(output / "joint_code_patch_20261112.zip"),
        "runtime_members": len(keep) + 1,
        "p05_payload_included": False,
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "runtime_code_sha256": {name: keep[name] for name in keep if name.endswith(".py")},
    }
    (output / "preparation.json").write_bytes(json_bytes(build_record))
    print(
        json.dumps(
            {k: v for k, v in build_record.items() if k != "runtime_code_sha256"},
            ensure_ascii=False,
        )
    )
    return build_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    args = parser.parse_args()
    build(ROOT, args.output, args.notebook)


if __name__ == "__main__":
    main()
