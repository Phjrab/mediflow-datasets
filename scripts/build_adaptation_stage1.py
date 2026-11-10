"""Export the common Drive bundle and its stage 1 Colab notebook."""

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import io
import json
import secrets
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_batch as batch
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets.candidate_reproduction import ROOT
from mediflow_datasets.models import get_model_spec

DOMAINS = ("skin", "web_skin", "hair")
ZIP_NAME = "MediFlow_Adaptation_Common_v3.zip"
NOTEBOOK_NAME = "MediFlow_Adaptation_Stage1_Saved_20261110.ipynb"
REFERENCES = ROOT / "camera_capture/_review/training_reference_inputs_20261110"
EXPANDED = ROOT / "camera_capture/_review/nine_experiments_20261109"


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def settings():
    old = data.read_json(ROOT / "configs/nine_experiments.json")
    config = {
        "schema": "mediflow_staged_adaptation_v1",
        "seed": 42,
        "epochs": 5,
        "initialization_policy": "saved_only",
        "validation_scope": "exploratory_historical_checkpoint_exposure_unverified",
        "learning_rate": 1e-5,
        "augmentation_probability": 0.5,
        "tasks": {},
    }
    for domain in DOMAINS:
        original = old["tasks"][domain]
        candidate, candidate_path, classes = data.candidate(domain)
        initial = get_model_spec(domain, "original")
        if initial.class_names() != classes:
            raise ValueError(f"Initial checkpoint class order differs: {domain}")
        basic = {
            "architecture": "b0",
            "size": initial.image_size[0],
            "label_smoothing": 0.0,
            "initialization": "saved original-only early checkpoint",
            "model": initial.model_path.relative_to(ROOT).as_posix(),
            "model_sha256": data.sha256(initial.model_path),
        }
        selected = {
            "architecture": {"skin": "b0", "web_skin": "pmg", "hair": "b1"}[domain],
            "size": candidate["input_size"],
            "label_smoothing": 0.05 if domain == "hair" else 0.0,
            "reference_candidate_id": candidate["candidate_id"],
            "initialization": "saved selected checkpoint",
            "model": candidate_path.relative_to(ROOT).as_posix(),
            "model_sha256": candidate["model_sha256"],
        }
        if domain == "hair":
            run_config = data.read_json(
                ROOT / "results" / candidate["extracted_directory"] / "run_config.json"
            )["settings"]
            if run_config["data_sha256"] != original["source_sha256"] or (
                run_config["classes"] != classes
            ):
                raise ValueError("Hair checkpoint training data differs")
            selected["training_source_sha256"] = run_config["data_sha256"]
        config["tasks"][domain] = {
            "domain": domain,
            "classes": classes,
            "source_name": original["source_name"],
            "source_sha256": original["source_sha256"],
            "input_policy": original["input_policy"],
            "batch_size": 16 if domain == "skin" else 8,
            "models": {"basic": basic, "selected": selected},
        }
    return config


def encrypt_and_verify(key, payload, name, scratch):
    """Use Windows .NET AES-GCM and verify decryption before writing the bundle."""
    script = scratch / "encrypt.ps1"
    if not script.exists():
        script.write_text(
            """param([string]$inputPath,[string]$outputPath,[string]$nonceText,[string]$member)
$ErrorActionPreference = 'Stop'
$keyBytes = [Convert]::FromBase64String([Console]::ReadLine())
$nonce = [Convert]::FromBase64String($nonceText)
$aad = [Text.Encoding]::UTF8.GetBytes($member)
$plain = [IO.File]::ReadAllBytes($inputPath)
$cipher = [byte[]]::new($plain.Length)
$tag = [byte[]]::new(16)
$aes = [Security.Cryptography.AesGcm]::new($keyBytes,16)
try {
    $aes.Encrypt($nonce,$plain,$cipher,$tag,$aad)
    $check = [byte[]]::new($plain.Length)
    $aes.Decrypt($nonce,$cipher,$tag,$check,$aad)
    if ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($plain)) -ne
        [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($check))) {
        throw 'AES-GCM round trip failed'
    }
    $stream = [IO.File]::Open($outputPath,[IO.FileMode]::Create)
    try { $stream.Write($nonce); $stream.Write($cipher); $stream.Write($tag) }
    finally { $stream.Dispose() }
} finally { $aes.Dispose() }
""",
            encoding="utf-8",
        )
    input_path, output_path = scratch / "private.npz", scratch / "encrypted.bin"
    input_path.write_bytes(payload)
    nonce = base64.b64encode(secrets.token_bytes(12)).decode("ascii")
    subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(script),
            str(input_path),
            str(output_path),
            nonce,
            name,
        ],
        input=base64.b64encode(key).decode("ascii") + "\n",
        text=True,
        check=True,
    )
    ciphertext = output_path.read_bytes()
    if len(ciphertext) != len(payload) + 28:
        raise ValueError("Encrypted entry size is inconsistent")
    return ciphertext


def calibration_entries(domain, task, key, scratch):
    path = REFERENCES / domain
    packet = data.read_json(path / "calibration.json")
    rows = [data.ImageRow(**row) for row in packet["rows"]]
    if runner.target_fingerprint(ROOT / "camera_capture", rows) != packet["fingerprint"]:
        raise ValueError(f"Capture fingerprint changed: {domain}")
    if len({row.person for row in rows}) != 4 or any(row.label is not None for row in rows):
        raise ValueError("Target inputs must come from the four unlabeled participants")
    expanded = data.read_json(EXPANDED / domain / "extended_profile.json")
    if data.read_json(path / "lighting_profile.json") != expanded["lighting"]:
        raise ValueError("Historical lighting profile changed")
    reference = {
        "extended_profile": expanded,
        "pixel_hashes": packet["pixel_hashes"],
        "capture_fingerprint": packet["fingerprint"],
        "target_count": len(rows),
    }
    if domain != "hair":
        reference["quarantine"] = data.read_json(path / "quarantine_plan.json")
    yield f"references/{domain}.json", encode(reference)
    for size in sorted({spec["size"] for spec in task["models"].values()}):
        previous = EXPANDED / domain / "target_inputs.npz"
        previous_meta = data.read_json(previous.with_suffix(".json"))
        if (
            previous_meta["input_size"] == size
            and previous_meta["input_policy"] == task["input_policy"]
            and previous_meta["rows"] == packet["rows"]
            and data.sha256(previous) == previous_meta["array_sha256"]
        ):
            payload = previous.read_bytes()
        else:
            images = np.stack(
                [
                    exp.image_tensor(
                        data.local_path(ROOT / "camera_capture", row.path).read_bytes(),
                        size,
                        task["input_policy"],
                    )
                    for row in rows
                ]
            )
            output = io.BytesIO()
            np.savez_compressed(output, images=images)
            payload = output.getvalue()
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            images = arrays["images"]
            if images.shape != (len(rows), size, size, 3) or images.dtype != np.float32:
                raise ValueError("Invalid Target input contract")
            if not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
                raise ValueError("Target pixels are outside the saved model contract")
        member = f"target_encrypted/{domain}_{size}.bin"
        yield member, encrypt_and_verify(key, payload, member, scratch)
        yield (
            f"target_encrypted/{domain}_{size}.json",
            encode(
                {
                    "cipher": "AES-256-GCM",
                    "layout": "12-byte nonce, ciphertext, 16-byte tag",
                    "aad_utf8": member,
                    "plaintext_sha256": hashlib.sha256(payload).hexdigest(),
                    "rows": [
                        {
                            "id": f"{domain}_{index:03}",
                            "person": r.person,
                            "split": "calibration",
                            "label": None,
                            "condition": r.condition,
                        }
                        for index, r in enumerate(rows, 1)
                    ],
                    "input_policy": task["input_policy"],
                    "input_size": size,
                }
            ),
        )
        print("Encrypted Target:", domain, size, len(rows), flush=True)


def make_notebook(sha, config):
    bootstrap = """import hashlib
import json
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
"""
    bootstrap += "\n\n" + inspect.getsource(batch.digest)
    bootstrap += "\n\n" + inspect.getsource(batch.unpack_bundle).replace(
        "mediflow_da_runtime.zip", ZIP_NAME
    )
    code = (
        bootstrap
        + f'''
from google.colab import drive

drive.mount("/content/drive")
MY_DRIVE = Path("/content/drive/MyDrive")
PROJECT = MY_DRIVE / "mediflow_Project"
BUNDLE = PROJECT / "{ZIP_NAME}"
if not BUNDLE.is_file():
    raise FileNotFoundError(f"공통 ZIP을 {{PROJECT}} 바로 아래에 올리세요: {{BUNDLE.name}}")
print("Drive 공통 ZIP 확인 중:", BUNDLE, flush=True)
EXPECTED = "{sha}"
with tempfile.TemporaryDirectory(prefix="mediflow_stage1_", dir="/content") as temporary:
    work = Path(temporary)
    local_bundle = work / BUNDLE.name
    shutil.copyfile(BUNDLE, local_bundle)
    runtime = unpack_bundle(local_bundle, work / "runtime", EXPECTED)
    sys.path.insert(0, str(runtime / "src"))
    try:
        from mediflow_datasets.adaptation_stage1_colab import launch
        STAGE1_RESULTS = launch(runtime, work, PROJECT, EXPECTED, MY_DRIVE)
    finally:
        sys.path.remove(str(runtime / "src"))
'''
    )
    compile(code, NOTEBOOK_NAME, "exec")
    guide = f"""# MediFlow 1차 모델 비교 · 6개 학습

공통 ZIP `{ZIP_NAME}`을 **내 드라이브 → mediflow_Project** 바로 아래에 한 번 올립니다.
이 노트북에서 **GPU 선택 → 모두 실행 → Drive 연결**을 합니다.

피부·얼굴·두피의 초기 원본 학습 모델과 기존 선정 모델을 불러와 같은 조명 변화를 적용합니다.
저장 모델 여섯 개에서 바로 조명 적응을 시작합니다.
조명 적응은 모델당 {config["epochs"]} epoch입니다. 화면에 `Epoch 1/{config["epochs"]}`, 배치 수,
loss, 검증 점수와 저장 단계가 표시됩니다. 같은 ZIP을 다시 실행하면 완료한 epoch까지 복원합니다.
한 번에 한 Colab 세션에서 실행하세요.

결과는 내 드라이브의 `MediFlow_adaptation_runs/{sha[:16]}/stage1`에 저장합니다.
1차 후보는 원본·조명 변화 Validation으로 고릅니다. 과거 학습 자료와의 겹침 여부가
확인되지 않은 상태에서 비교한 점수이므로 탐색 결과로 기록합니다.
2·3차에서 사용할 선택 모델의 1차 이전 가중치도 `initial.keras`로 별도 저장합니다.

공통 ZIP의 실제 촬영 입력은 암호화되어 있습니다. 1차는 복호화 키 없이 실행합니다.
3차에서 사용할 별도 키 파일은 로컬에 보관하세요.
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
                "source": guide.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "id": "run-stage1",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
    }


def build(output, notebook, key_file, *, reuse_key=False):
    if output.exists() or notebook.exists():
        raise FileExistsError("Choose new artifact paths")
    if reuse_key:
        key = base64.b64decode(key_file.read_text(encoding="ascii").strip(), validate=True)
        if len(key) != 32:
            raise ValueError("Invalid existing AES-256 key length")
    else:
        if key_file.exists():
            raise FileExistsError("Choose a new key file or specify --reuse-key")
        key = secrets.token_bytes(32)
    config = settings()
    output.mkdir(parents=True)
    bundle = output / ZIP_NAME
    manifest = {}
    with (
        tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as folder,
        zipfile.ZipFile(bundle, "x", zipfile.ZIP_DEFLATED) as archive,
    ):

        def add(name, body):
            if name in manifest:
                raise ValueError(f"Duplicate ZIP entry: {name}")
            manifest[name] = hashlib.sha256(body).hexdigest()
            archive.writestr(name, body)

        for path in sorted((ROOT / "src/mediflow_datasets").glob("*.py")):
            add(path.relative_to(ROOT).as_posix(), path.read_bytes())
        add("configs/adaptation_stage1.json", encode(config))
        add(
            "bundle_provenance.json",
            encode(
                {
                    "repository_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "code_state": "exact bundled source files and member SHA-256 manifest",
                    "planned_stage_counts": [6, 3, 3],
                    "implemented_stages": [1],
                    "encrypted_target_data_reserved_for_stage3": True,
                }
            ),
        )
        for domain, task in config["tasks"].items():
            for spec in task["models"].values():
                model_path = spec["model"]
                add(model_path, (ROOT / model_path).read_bytes())
                if manifest[model_path] != spec["model_sha256"]:
                    raise ValueError(f"Checkpoint bytes changed: {model_path}")
            for name, body in calibration_entries(domain, task, key, Path(folder)):
                add(name, body)
        archive.writestr("bundle_manifest.json", encode(manifest))
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None or any(
            hashlib.sha256(archive.read(name)).hexdigest() != expected
            for name, expected in manifest.items()
        ):
            raise ValueError("Bundle readback verification failed")
    sha = data.sha256(bundle)
    if not reuse_key:
        with key_file.open("x", encoding="ascii") as stream:
            stream.write(base64.b64encode(key).decode("ascii") + "\n")
    with notebook.open("xb") as stream:
        stream.write(encode(make_notebook(sha, config)))
    with (output / "package_info.json").open("xb") as stream:
        stream.write(
            encode(
                {
                    "bundle": ZIP_NAME,
                    "sha256": sha,
                    "bytes": bundle.stat().st_size,
                    "members_verified": len(manifest),
                    "notebook": notebook.name,
                    "implemented_stages": [1],
                    "encrypted_target": True,
                    "existing_target_key_reused": reuse_key,
                    "starting_checkpoints": {
                        domain: {
                            role: spec["model_sha256"] for role, spec in task["models"].items()
                        }
                        for domain, task in config["tasks"].items()
                    },
                }
            )
        )
    print("ZIP and notebook ready:", output, notebook, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--reuse-key", action="store_true")
    args = parser.parse_args()
    build(args.output, args.notebook, args.key_file, reuse_key=args.reuse_key)
