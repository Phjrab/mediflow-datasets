"""Export reviewed quarantine/fresh-baseline experiments for one-run two-domain Colab."""

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
from mediflow_datasets.candidate_reproduction import ROOT
from mediflow_datasets.domain_adaptation_training import object_hash

DRIVER = '''
def run_resolved_training(expected_hash):
    from google.colab import files, drive, output
    from IPython.display import display, HTML
    print("mediflow_skin_web_training_runtime.zip 하나를 선택하세요.")
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("새 실행 ZIP 하나만 선택하세요.")
    workspace = Path("/content") / ("mediflow_resolved_" + uuid.uuid4().hex[:8])
    workspace.mkdir()
    bundle = Path(next(iter(uploaded))).resolve()
    del uploaded
    runtime = unpack_bundle(bundle, workspace / "runtime", expected_hash)
    drive.mount("/content/drive")
    python, versions = prepare_python(workspace)
    if not versions["gpu"]:
        raise RuntimeError("런타임 유형을 GPU로 변경한 뒤 모두 실행하세요.")
    config_path = runtime / "configs/skin_web_resolution.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent = Path("/content/drive/MyDrive") / (
        "mediflow_skin_web_adaptation_" + uuid.uuid4().hex[:8])
    parent.mkdir()
    save_json(parent / "environment.json", versions)
    env = python_environment(python, runtime)
    summary = {}
    for domain in ("skin", "web_skin"):
        print("\\n===", domain, "새 기준 모델 및 적응 비교 학습 ===", flush=True)
        try:
            source = find_source(Path("/content/drive/MyDrive"), config["tasks"][domain])
            local_source = workspace / (domain + "_source.zip")
            print("기존 Source ZIP을 실행 공간으로 복사합니다. 원본은 보존합니다.", flush=True)
            shutil.copyfile(source, local_source)
            run_logged([str(python), "-m", "mediflow_datasets.domain_adaptation_resolution",
                "--domain", domain, "--config", str(config_path),
                "--source-zip", str(local_source),
                "--calibration-file", str(runtime / "calibration" / (domain + ".json")),
                "--review", str(runtime / "review" / domain),
                "--output", str(parent / domain)], env=env, log_path=parent / (domain + ".log"))
            summary[domain] = json.loads((parent / domain / "status.json").read_text())
            display(HTML("<pre>" + (parent / domain / "RESULTS.md").read_text() + "</pre>"))
        except Exception as error:
            folder = parent / (domain + "_failure")
            folder.mkdir()
            summary[domain] = {"status": "failed", "error": str(error), "target_accuracy": None}
            save_json(folder / "failure.json", {
                "error": str(error), "traceback": traceback.format_exc()})
            print(domain, "미완료:", error, flush=True)
    save_json(parent / "summary.json", summary)
    for path in workspace.glob("environment_*.log"):
        shutil.copyfile(path, parent / path.name)
    destination = parent.parent / (parent.name + ".zip")
    export_review(parent, destination)
    local = workspace / destination.name
    shutil.copyfile(destination, local)
    def retry():
        files.download(str(local))
    callback = "mediflow.skin_web_training_" + workspace.name
    output.register_callback(callback, retry)
    display(HTML('<button onclick="google.colab.kernel.invokeFunction(\\'' + callback +
        '\\', [], {})">두 분야 결과 ZIP 다시 다운로드</button>'))
    print("분야별 상태:", summary)
    print("Drive 결과:", parent)
    print("실제 촬영 질환 정확도는 계산하지 않았습니다. 실제 사진 원본은 PC에 있습니다.")
    files.download(str(local))
    return summary
'''


def package(output, preparation, bundle, decisions):
    output.mkdir(parents=True, exist_ok=False)
    config = data.read_json(preparation / "resolution_config.json")
    decisions = data.read_json(decisions)
    with zipfile.ZipFile(bundle) as archive:
        content = {
            name: archive.read(name) for name in archive.namelist()
            if name.startswith(("src/", "results/skin/selected_models/v1/",
                                "results/web_skin/selected_models/v2/"))
            or name == "results/CANDIDATE_INDEX.json"
        }
        for domain in ("skin", "web_skin"):
            packet = json.loads(archive.read(f"calibration/{domain}.json"))
            packet.pop("scores", None)
            packet.pop("features", None)
            content[f"calibration/{domain}.json"] = encode(packet)
    for path in (ROOT / "src/mediflow_datasets").glob("*.py"):
        content[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    content["configs/skin_web_resolution.json"] = encode(config)
    for domain in ("skin", "web_skin"):
        directory = preparation / domain
        profile = data.read_json(directory / "profile_v4.json")
        decision = decisions[domain]
        qa = data.read_json(directory / "transform_qa/summary.json")
        if (
            decision["profile_sha256"] != object_hash(profile)
            or decision["decision"] != "approved_for_pilot"
            or not decision["reviewed_pages"] or not decision["reason"]
            or not set(decision["reviewed_pages"]) <= set(qa["pages"])
        ):
            raise ValueError("Missing or changed visual review: " + domain)
        approved = {**profile, "train_visual_review": "approved",
                    "review_note": decision["reason"]}
        files = [directory / name for name in (
            "quarantine_plan.json", "source_measurements.json", "class_names.json",
            "transform_qa/summary.json", "transform_qa/measurements.json",
            "transform_qa/distributions.svg", "transform_qa/before_after_target.svg",
        )]
        files.extend(directory / "transform_qa" / name for name in decision["reviewed_pages"])
        evidence = {}
        for path in files:
            relative = path.relative_to(directory).as_posix()
            body = path.read_bytes()
            content[f"review/{domain}/{relative}"] = body
            evidence[relative] = hashlib.sha256(body).hexdigest()
        receipt = {
            "domain": domain, "status": "ready_for_fresh_baseline_pilot",
            "profile_sha256": object_hash(approved),
            "quarantine_plan_sha256": object_hash(
                data.read_json(directory / "quarantine_plan.json")),
            "config_sha256": object_hash(config), "evidence_sha256": evidence,
            "visual_review": decision, "identity_independence_verified": False,
            "all_candidates_declared_duplicates": False,
            "old_checkpoint_weights_used_for_training": False,
            "target_accuracy": None,
            "repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "uncommitted_source_included": True,
            "code_sha256": {name: hashlib.sha256(body).hexdigest()
                            for name, body in content.items() if name.startswith("src/")},
        }
        content[f"review/{domain}/profile_v4.json"] = encode(approved)
        content[f"review/{domain}/readiness.json"] = encode(receipt)
    dest = output / "mediflow_skin_web_training_runtime.zip"
    with zipfile.ZipFile(dest, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, body in content.items():
            archive.writestr(name, body)
        archive.writestr("bundle_manifest.json", encode({
            name: hashlib.sha256(body).hexdigest() for name, body in content.items()
        }))
    code = (
        (ROOT / "src/mediflow_datasets/domain_adaptation_batch.py").read_text(encoding="utf-8")
        + "\n" + DRIVER + f'\nRESULTS=run_resolved_training("{batch.digest(dest)}")\n'
    )
    compile(code, "Skin_WebSkin_Resolved_Training_Colab.py", "exec")
    guide = (
        "# MediFlow 피부·얼굴 문제 처리 후 새 기준선 및 적응 학습\n\n"
        "**GPU를 선택하고 모두 실행**하세요. 새 `mediflow_skin_web_training_runtime.zip` 하나를 "
        "선택하고 Drive를 연결하면 됩니다. 코드 수정이나 검토 확인 입력은 필요 없습니다.\n\n"
        "원본은 보존하고, 유사 후보 연결 관계에 따라 학습·검증 목록만 보수적으로 제외합니다. "
        "기존 증강 데이터는 사용하지 않습니다. Test는 그대로 보존합니다. "
        "기존 MediFlow 체크포인트의 학습 이력이 남지 않도록 **ImageNet 초기값에서 새 기준 모델**을 "
        "학습합니다. 처음 이 단계에는 사전학습 가중치 다운로드가 필요할 수 있습니다.\n\n"
        "피부 → 얼굴 순서입니다. 각 분야에서 새 기준선 head 15회·후반부 10회 학습 후, "
        "같은 기준선으로 기존 방식·밝기만·측정 복합 변환을 각각 3회 추가 학습합니다. "
        "Validation 선택을 고정한 뒤 원본/촬영 변화 Source Test를 평가합니다. "
        "새 기준선을 만드는 만큼 이전 3회 추가 학습보다 오래 걸립니다.\n\n"
        "결과는 Drive의 새 `mediflow_skin_web_adaptation_…` 폴더에 저장하고 ZIP 하나를 "
        "다운로드합니다. 실제 촬영 원본은 포함하지 않았으며, 새 모델로 PC에서 예측과 특징을 "
        "후속 비교합니다. Target 질환 정답이 없어 실제 장비 정확도는 계산하지 않습니다. "
        "사람·병변·세션 단위 누수의 완전한 배제는 확인하지 못했습니다.\n"
    )
    book = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "accelerator": "GPU", "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {"cell_type": "markdown", "id": "resolution-guide", "metadata": {},
             "source": guide.splitlines(keepends=True)},
            {"cell_type": "code", "id": "resolution-run", "metadata": {},
             "source": code.splitlines(keepends=True), "outputs": [], "execution_count": None},
        ]}
    notebook = output / "Skin_WebSkin_Resolved_Training_Colab.ipynb"
    with notebook.open("x", encoding="utf-8") as stream:
        json.dump(book, stream, ensure_ascii=False, indent=2)
    (output / "Skin_WebSkin_Resolved_Training_Colab.py").write_text(code, encoding="utf-8")
    (output / "README.md").write_text(guide, encoding="utf-8")
    runner.write_json(output / "package_info.json", {
        "bytes": dest.stat().st_size, "sha256": batch.digest(dest),
        "training_executed": False, "domains": ["skin", "web_skin"],
    })
    print(output, flush=True)


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "preparation", "decisions"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--bundle", type=Path, default=ROOT / (
        "output/adaptation_preflight_v2_20261109/mediflow_preflight_runtime.zip"
    ))
    args = parser.parse_args()
    package(args.output, args.preparation, args.bundle, args.decisions)
