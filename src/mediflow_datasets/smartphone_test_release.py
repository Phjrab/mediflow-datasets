"""Freeze completed smartphone checkpoints and build a key-free Source Test notebook."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import subprocess
import textwrap
import zipfile
from copy import deepcopy
from pathlib import Path

from mediflow_datasets.domain_adaptation_data import read_json, sha256
from mediflow_datasets.smartphone_final_test import CONDITIONS, MODELS, RESULT_ZIP, object_hash
from mediflow_datasets.smartphone_joint_release import BASE_ZIP, build_notebook, json_bytes


def notebook(base_sha, patched, manifest_sha, run_id, previous_id, joint_id):
    book = build_notebook(base_sha, patched, manifest_sha, run_id, previous_id)

    def set_cell(index, content):
        book["cells"][index]["source"] = textwrap.dedent(content).strip().splitlines(keepends=True)

    set_cell(
        0,
        f"""
    # MediFlow 스마트폰 얼굴 · 고정 최종 Source Test

    **GPU 선택 → 모두 실행**. 기존 공통 실행 ZIP과 Drive에 저장된 1~5차 모델을 재사용합니다.
    **새 ZIP 업로드·복호화 키 입력·새 학습은 없습니다.**

    - 기존 원본 Test 400장, 건선·아토피·여드름·정상·주사 순서, PMG·B0·256을 유지합니다.
    - 적응 전·1·2·3·4·추가5차의 6개 모델 × 원본·1차 조명·2차 약한 확장·4차 강한 확장·새 공동 근접 변환의 5조건, 총 30개 비교입니다.
    - 모든 모델에 같은 픽셀 입력·seed 1707을 씁니다. 변환 조건은 평가 시 100% 적용합니다.
    - **Validation에서 고른 4차를 미리 고정**합니다. Test 결과로 모델·epoch·강도를 다시 고르지 않습니다.
    - 사진 원본·학습 가중치를 바꾸지 않습니다. 실제 휴대폰 사진은 읽지 않고 Target Accuracy를 계산하지 않습니다.
    - 원본과 변환별 Accuracy·Macro-F1·클래스 Recall·혼동행렬·개별 예측·비교 그래프가 나옵니다.
    - 모델·조건·배치 진행률을 표시합니다. 중단되면 같은 노트북을 모두 실행하면 검증된 완료 조건을 재사용합니다.

    결과는 `MyDrive/MediFlow_smartphone_runs/test_{run_id}`에 저장되며
    `{RESULT_ZIP}`을 Drive에 만들고 자동 다운로드합니다. 결과 ZIP에는 모델·사진·키가 들어가지 않습니다.
    Source의 사람/병변/세션 식별 누락과 과거 체크포인트의 Test 노출 미확인 한계를 유지한 평가입니다.
    """,
    )
    setup = "".join(book["cells"][1]["source"])
    setup = setup.replace("mediflow_phone_joint_", "mediflow_phone_test_")
    setup = setup.replace(
        "MediFlow_smartphone_runs/joint_" + run_id, "MediFlow_smartphone_runs/test_" + run_id
    )
    setup = setup.replace(
        'print("추가 학습 결과 저장 위치:",OUTPUT)', 'print("최종 Test 결과 저장 위치:",OUTPUT)'
    )
    setup += f'\nJOINT=Path("/content/drive/MyDrive/MediFlow_smartphone_runs/{joint_id}")\n'
    setup += (
        'if not JOINT.is_dir():raise FileNotFoundError("추가 5차 결과가 없습니다: "+str(JOINT))\n'
    )
    set_cell(1, setup)
    environment = "".join(book["cells"][2]["source"])
    environment = environment.replace('"cryptography>=44,<49", ', "")
    environment = environment.replace("import cryptography,cv2", "import cv2")
    environment = environment.replace("r/'study.json'", "r/'evaluation_plan.json'")
    environment = environment.replace("s['task']", "s['plan']['task']")
    set_cell(2, environment)
    set_cell(
        3,
        """
    import fcntl, uuid
    from datetime import datetime, timezone
    log_path=OUTPUT/("test_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:6]+".log")
    command=[str(PYTHON),"-m","mediflow_datasets.smartphone_final_test",
             "--runtime",str(RUNTIME),"--source",str(SOURCE),"--previous",str(PREVIOUS),
             "--joint",str(JOINT),"--output",str(OUTPUT),"--local",str(LOCAL)]
    print("실행 로그:",log_path,flush=True)
    with (LOCAL/"execution.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("같은 Test가 이미 실행 중입니다.") from None
        process=None
        try:
            with log_path.open("x",encoding="utf-8") as log:
                process=subprocess.Popen(command,env=ENV,stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in process.stdout:
                    print(line,end="",flush=True);log.write(line);log.flush()
                if process.wait():raise RuntimeError("최종 Test 미완료: 위 실제 로그를 확인하고 같은 노트북을 다시 실행하세요.")
        finally:
            if process is not None:
                if process.poll() is None:process.terminate();process.wait()
                process.stdout.close()
    """,
    )
    book["cells"][3]["id"] = "run-locked-source-test"
    set_cell(
        4,
        f"""
    from IPython.display import display, Image
    summary=json.loads((OUTPUT/"summary.json").read_text())
    if summary.get("status")!="completed":raise RuntimeError("최종 Test가 완료되지 않았습니다.")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    display(Image(filename=str(OUTPUT/"plots/test_comparison.png")))
    for name in ("initial","stage4","stage5"):
        display(Image(filename=str(OUTPUT/"plots"/(name+"_confusion_matrices.png"))))
    RESULT_ZIP=OUTPUT/"{RESULT_ZIP}"
    if not RESULT_ZIP.is_file():raise FileNotFoundError("결과 ZIP이 없습니다: "+str(RESULT_ZIP))
    print("Drive 결과 ZIP:",RESULT_ZIP)
    print("ZIP SHA-256:",digest(RESULT_ZIP))
    from google.colab import files
    files.download(str(RESULT_ZIP))
    """,
    )
    return book


def build(root, output, notebook_path):
    root, output, notebook_path = map(lambda p: Path(p).resolve(), (root, output, notebook_path))
    if output.exists() or notebook_path.exists():
        raise FileExistsError("새 Test 출력·노트북 경로를 지정하세요.")
    prior = root / "results/smartphone_web_skin_adaptation_20261112"
    joint = root / "results/smartphone_joint_adaptation_20261112"
    release = root / "output/smartphone_web_skin_20261111"
    base_path = release / BASE_ZIP
    preparation = read_json(release / "preparation.json")
    if sha256(base_path) != preparation["runtime_sha256"]:
        raise ValueError("기존 공통 ZIP 해시 불일치")
    study = read_json(joint / "reproduction/study.json")
    recipe = study["joint_experiment"]
    task = deepcopy(study["task"])
    spec = task["models"]["selected"]
    selected = read_json(joint / "selection.json")
    if selected["model"] != "stage4" or selected["source_test_evaluated"]:
        raise ValueError("Test 전 Validation 선정·완료 상태가 다릅니다.")
    for folder in (prior, joint):
        summary = read_json(folder / "summary.json")
        if summary["status"] != "completed" or summary["source_test_evaluated"]:
            raise ValueError("학습 완료·Source Test 미실행 상태를 확인하세요.")
    manifest_path = prior / "source_manifest.json"
    if sha256(manifest_path) != study["source_manifest_sha256"] or (
        manifest_path.read_bytes() != (joint / "source_manifest.json").read_bytes()
    ):
        raise ValueError("Source 고정 분할이 다릅니다.")
    test = read_json(manifest_path)["original"]["test"]
    if len(test) != 400 or any(r["split"] != "test" or r["variant"] != "original" for r in test):
        raise ValueError("기존 원본 Test 400장이어야 합니다.")
    models = {"initial": {"root": "runtime", "file": spec["model"], "sha256": spec["model_sha256"]}}
    models.update(
        {name: {"root": "prior", **info} for name, info in recipe["previous_models"].items()}
    )
    best = read_json(joint / "stage5/complete.json")
    models["stage5"] = {
        "root": "joint",
        "file": "stage5/best.keras",
        "sha256": best["best_model"]["sha256"],
    }
    if tuple(models) != MODELS or selected["sha256"] != models["stage4"]["sha256"]:
        raise ValueError("완료 모델·고정 후보 불일치")
    for name, info in models.items():
        folder = root if name == "initial" else prior if info["root"] == "prior" else joint
        if sha256(folder / info["file"]) != info["sha256"]:
            raise ValueError("저장 모델 해시 불일치: " + name)
    profiles = {
        condition: deepcopy(study["profiles"][stage])
        for condition, stage in zip(CONDITIONS[1:4], ("stage1", "stage2", "stage4"), strict=True)
    }
    profiles["phone_joint"] = deepcopy(study["joint_profile"])
    for profile in profiles.values():
        profile["blur_input_scale"] *= spec["size"] / 256
    inputs = []
    for group, folder, files in (
        (
            "prior",
            prior,
            [
                "summary.json",
                "runtime_identity.json",
                "source_manifest.json",
                "validation_comparison.json",
                "stage1_selected/complete.json",
                "stage2/complete.json",
                "stage3/complete.json",
                "stage4/complete.json",
            ],
        ),
        (
            "joint",
            joint,
            [
                "summary.json",
                "runtime_identity.json",
                "source_manifest.json",
                "validation_comparison.json",
                "selection.json",
                "stage5/complete.json",
                "reproduction/study.json",
            ],
        ),
    ):
        inputs.extend(
            {"root": group, "file": name, "sha256": sha256(folder / name)} for name in files
        )
    module = root / "src/mediflow_datasets/smartphone_final_test.py"
    plan = {
        "schema": "mediflow_phone_final_test_v1",
        "task": task,
        "model_spec": spec,
        "models": models,
        "conditions": list(CONDITIONS),
        "profiles": profiles,
        "manifest_sha256": sha256(manifest_path),
        "test_count": len(test),
        "test_support": [sum(r["label"] == i for r in test) for i in range(len(task["classes"]))],
        "transform_seed": 1707,
        "evaluation_transform_probability": 1.0,
        "selected_on_validation": "stage4",
        "test_used_for_selection": False,
        "target_photos_used": False,
        "input_records": inputs,
        "scope": "Source Test only; historical checkpoint exposure and person independence unverified",
        "primary_question": "Evaluate fixed Validation candidate stage4 against original PMG; describe stages1-5 without reselection",
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "uncommitted_evaluator_sha256": sha256(module),
        "prior_validation_sha256": sha256(joint / "validation_comparison.json"),
    }
    envelope = {"plan_sha256": object_hash(plan), "plan": plan}
    patched = {
        "evaluation_plan.json": json_bytes(envelope),
        "src/mediflow_datasets/smartphone_final_test.py": module.read_bytes(),
    }
    with zipfile.ZipFile(base_path) as base:
        previous = read_json(prior / "receipt_verification.json")
        old_manifest_bytes = base.read("runtime_manifest.json")
        if hashlib.sha256(old_manifest_bytes).hexdigest() != previous["runtime_manifest_sha256"]:
            raise ValueError("기존 runtime manifest 불일치")
        import json

        old_manifest = json.loads(old_manifest_bytes)
        keep = {
            name: digest
            for name, digest in old_manifest["files"].items()
            if name.startswith("src/")
            and name.endswith(".py")
            or name in (spec["model"], "source_manifest.json")
        }
        for name, expected in keep.items():
            if hashlib.sha256(base.read(name)).hexdigest() != expected:
                raise ValueError("공통 ZIP 실행 파일 불일치: " + name)
        if base.read("source_manifest.json") != manifest_path.read_bytes():
            raise ValueError("공통 ZIP 고정 분할이 다릅니다.")
    keep.update({name: hashlib.sha256(value).hexdigest() for name, value in patched.items()})
    runtime_manifest = {
        "schema": "mediflow_phone_test_runtime_v1",
        "files": keep,
        "base_zip_sha256": preparation["runtime_sha256"],
    }
    patched["runtime_manifest.json"] = json_bytes(runtime_manifest)
    manifest_sha = hashlib.sha256(patched["runtime_manifest.json"]).hexdigest()
    run_id = manifest_sha[:12]
    old_id = preparation["runtime_sha256"][:12]
    joint_id = "joint_" + read_json(joint / "runtime_identity.json")["runtime_manifest_sha256"][:12]
    book = notebook(preparation["runtime_sha256"], patched, manifest_sha, run_id, old_id, joint_id)
    output.mkdir(parents=True, exist_ok=False)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_bytes(json_bytes(book))
    (output / "evaluation_plan.json").write_bytes(patched["evaluation_plan.json"])
    (output / "runtime_manifest.json").write_bytes(patched["runtime_manifest.json"])
    record = {
        "status": "prepared_not_run",
        "source_test_evaluated": False,
        "new_training_executed": False,
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "comparisons": 30,
        "test_images": len(test),
        "selected_before_test": "stage4",
        "notebook": notebook_path.relative_to(root).as_posix(),
        "notebook_sha256": sha256(notebook_path),
        "plan_sha256": envelope["plan_sha256"],
        "runtime_manifest_sha256": manifest_sha,
        "run_id": "test_" + run_id,
        "old_run_id": old_id,
        "joint_run_id": joint_id,
        "new_zip_upload_required": False,
        "decryption_key_required": False,
        "target_payload_included": False,
        "runtime_members": len(keep) + 1,
    }
    (output / "preparation.json").write_bytes(json_bytes(record))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.root, args.output, args.notebook))


if __name__ == "__main__":
    main()
