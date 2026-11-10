"""Package corrected runtime and all three cached Source analyses for one-run Colab QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_batch as batch
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets.candidate_reproduction import ROOT

COLAB_DRIVER = """
def run_preflight_colab(expected_hash):
    from google.colab import files, drive, output
    from IPython.display import display, HTML
    print("새 mediflow_preflight_runtime.zip 하나를 선택하세요.")
    uploaded=files.upload()
    if len(uploaded)!=1:
        raise ValueError("새 실행 ZIP 하나만 선택하세요.")
    bundle=Path(next(iter(uploaded))).resolve()
    del uploaded
    workspace=Path("/content")/("mediflow_preflight_"+uuid.uuid4().hex[:8])
    workspace.mkdir()
    runtime=unpack_bundle(bundle,workspace/"runtime",expected_hash)
    drive.mount("/content/drive")
    python,versions=prepare_python(workspace)
    config_path=runtime/"configs/domain_adaptation_preflight_v2.json"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    parent=Path("/content/drive/MyDrive")/("mediflow_preflight_v2_"+uuid.uuid4().hex[:8])
    parent.mkdir()
    env=python_environment(python,runtime)
    summary={}
    for domain in ("web_skin","hair","skin"):
        print("\\n===== "+domain+" 학습 전 검토 =====",flush=True)
        try:
            source=find_source(Path("/content/drive/MyDrive"),config["tasks"][domain])
            run_logged([str(python),"-m","mediflow_datasets.domain_adaptation_preflight",
                "--domain",domain,"--config",str(config_path),"--source-zip",str(source),
                "--calibration-file",str(runtime/"calibration"/(domain+".json")),
                "--previous",str(runtime/"previous_analysis"/domain),
                "--output",str(parent/domain)],env=env,log_path=parent/(domain+".log"))
            summary[domain]=json.loads((parent/domain/"preflight_status.json").read_text())
        except Exception as error:
            summary[domain]={"status":"failed","error":str(error),"training_executed":False}
            print(domain,"실패:",error,flush=True)
    save_json(parent/"summary.json",summary)
    save_json(parent/"environment.json",versions)
    for log in workspace.glob("environment_*.log"):
        shutil.copyfile(log,parent/log.name)
    destination=parent.parent/(parent.name+".zip")
    export_review(parent,destination)
    local=workspace/destination.name
    shutil.copyfile(destination,local)
    def download_again():
        files.download(str(local))
    callback="mediflow.preflight_download_"+workspace.name
    output.register_callback(callback,download_again)
    display(HTML('<button onclick="google.colab.kernel.invokeFunction(\\''+callback+
                 '\\', [], {})">검토 결과 ZIP 다시 다운로드</button>'))
    print("세 분야 상태:",summary)
    print("Drive 결과:",parent)
    print("검토 자료만 생성했습니다. 후보 자동 승인이나 추가 학습은 실행하지 않았습니다.")
    files.download(str(local))
    return summary
"""


def build(output, bundle, previous):
    import keras

    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(bundle) as archive:
        content = {
            name: archive.read(name)
            for name in archive.namelist()
            if name != "bundle_manifest.json"
        }
    for path in (ROOT / "src/mediflow_datasets").glob("*.py"):
        content[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    config = json.loads(content["configs/domain_adaptation.json"])
    config["experiment_revision"] = "preflight_photometric_v2"
    for domain, task in config["tasks"].items():
        task["input_policy"] = "center_square" if domain == "web_skin" else "legacy_resize"
        task["geometry"] = False
    content["configs/domain_adaptation_preflight_v2.json"] = json.dumps(config, indent=2).encode()
    local = output / "local_preparation"
    local.mkdir()
    for domain in ("web_skin", "hair", "skin"):
        packet = json.loads(content[f"calibration/{domain}.json"])
        spec, model, classes = data.candidate(domain)
        rows = [data.ImageRow(**r) for r in packet["rows"]]
        if runner.target_fingerprint(ROOT / "camera_capture", rows) != packet["fingerprint"]:
            raise ValueError("Target photos changed since the archived analysis")
        policy = config["tasks"][domain]["input_policy"]
        old_scores = np.asarray(packet["scores"])
        if policy == "center_square":
            detector = analysis.face_detector()
            packet["measurements"] = [
                {
                    **r,
                    "metrics": analysis.measure(
                        data.local_path(ROOT / "camera_capture", r["path"]).read_bytes(),
                        domain,
                        spec["input_size"],
                        detector,
                        r["face_box"],
                        input_policy=policy,
                    ),
                }
                for r in packet["measurements"]
            ]
            loaded = keras.models.load_model(model, compile=False)
            scores, features = exp.infer(
                loaded,
                rows,
                lambda r: data.local_path(ROOT / "camera_capture", r.path).read_bytes(),
                spec["input_size"],
                spec["output_adapter"],
                config["batch_size"],
                input_policy=policy,
            )
            packet["scores"], packet["features"] = scores.tolist(), features.tolist()
            runner.write_json(
                local / "web_skin_preprocessing_comparison.json",
                {
                    "model_sha256": spec["model_sha256"],
                    "photos": len(rows),
                    "changed_prediction_count": int(
                        np.sum(scores.argmax(1) != old_scores.argmax(1))
                    ),
                    "old_input_policy": "legacy_resize",
                    "new_input_policy": policy,
                    "target_accuracy": None,
                    "meaning": "Preprocessing changed; no model training or accuracy claim",
                },
            )
            del loaded
            keras.backend.clear_session()
        packet["input_policy"] = policy
        content[f"calibration/{domain}.json"] = json.dumps(packet, ensure_ascii=False).encode()
        for name in ("analysis.json", "source_audit.json"):
            path = previous / domain / "analysis" / name
            content[f"previous_analysis/{domain}/{name}"] = path.read_bytes()
        old = json.loads(content[f"previous_analysis/{domain}/analysis.json"])
        if old["source_sha256"] != config["tasks"][domain]["source_sha256"]:
            raise ValueError("Previous Source analysis fingerprint differs")
        measured = [r for r in old["images"] if r["kind"] == "source"]
        profile = analysis.calibrate(domain, measured, packet["measurements"])
        profile.update(
            input_policy=policy,
            train_visual_review="pending",
            source_sha256=old["source_sha256"],
            model_sha256=spec["model_sha256"],
            calibration_fingerprint=packet["fingerprint"],
        )
        runner.write_json(local / f"{domain}_profile_v2.json", profile)
        arms, aliases = exp.arm_plan(profile)
        print(domain, policy, "arms", arms, "aliases", aliases, flush=True)
    manifest = {name: hashlib.sha256(body).hexdigest() for name, body in content.items()}
    destination = output / "mediflow_preflight_runtime.zip"
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, body in content.items():
            archive.writestr(name, body)
        archive.writestr("bundle_manifest.json", json.dumps(manifest))
    helper = (ROOT / "src/mediflow_datasets/domain_adaptation_batch.py").read_text(encoding="utf-8")
    code = (
        helper
        + "\n"
        + COLAB_DRIVER
        + f'\nPREFLIGHT_RESULTS=run_preflight_colab("{batch.digest(destination)}")\n'
    )
    compile(code, "Adaptation_Preflight_Colab.py", "exec")
    guide = (
        "# MediFlow 세 분야 학습 전 검토 v2\n\n"
        "GPU를 선택하고 **모두 실행**을 누르세요. 새 `mediflow_preflight_runtime.zip`을 선택하고 "
        "Drive를 연결하면 됩니다. 코드 수정은 필요 없습니다.\n\n"
        "이전 전체 Train 측정값을 재사용하고, 원본·증강 이력으로 유사 후보를 묶어 "
        "256픽셀 SSIM과 비교 사진을 생성합니다. 모든 후보는 검토 대기로 남깁니다. "
        "클래스별 Train 24장의 변환 전후 분포, 얼굴 검출 실패 사진, "
        "세 분야 특징 지표도 저장합니다. "
        "얼굴은 중앙 정사각형 입력을 모든 비교 모델에 공통 적용하는 실험 후보이며, "
        "단색 테두리를 만들지 않습니다. 학습·최종 Test 성능 평가는 포함하지 않습니다.\n\n"
        "결과 ZIP 하나를 자동 다운로드하고 Drive에도 보관합니다. "
        "실패한 분야는 summary에 표시됩니다. "
        "실제 촬영 원본은 실행 ZIP에 포함되지 않습니다. "
        "큰 Source ZIP을 읽으므로 시간이 걸릴 수 있습니다."
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "preflight-guide",
                "metadata": {},
                "source": guide.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "id": "preflight-run",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
    }
    with (output / "Adaptation_Preflight_Colab.ipynb").open("x", encoding="utf-8") as stream:
        json.dump(notebook, stream, ensure_ascii=False, indent=2)
    (output / "Adaptation_Preflight_Colab.py").write_text(code, encoding="utf-8")
    (output / "README.md").write_text(guide, encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "output/domain_adaptation_20261108_final/mediflow_da_runtime.zip",
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=ROOT / "camera_capture/_review/domain_adaptation_colab_20261109",
    )
    args = parser.parse_args()
    build(args.output, args.bundle, args.previous)
