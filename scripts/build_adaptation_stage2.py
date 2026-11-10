"""Build a Colab notebook that reads the existing, immutable common v3 bundle."""

import argparse
import inspect
import json
from pathlib import Path

from mediflow_datasets import domain_adaptation_batch as batch
from mediflow_datasets.candidate_reproduction import ROOT

ZIP_NAME = "MediFlow_Adaptation_Common_v3.zip"
MODULES = ("adaptation_stage1.py", "adaptation_stage1_colab.py", "adaptation_stage2.py")


def make_notebook(bundle_sha, *, stage=2, with_curves=False):
    module_dir = ROOT / "src/mediflow_datasets"
    source = {name: (module_dir / name).read_text(encoding="utf-8") for name in MODULES}
    if stage == 3:
        source["adaptation_stage3.py"] = (module_dir / "adaptation_stage3.py").read_text(
            encoding="utf-8"
        )
    input_code = (
        ""
        if stage == 2
        else """    from getpass import getpass
    key = getpass("3차 복호화 키 한 줄 입력: ").strip()
    if not key:
        raise ValueError("로컬 복호화 키 파일의 한 줄을 입력하세요.")
"""
    )
    pipe = "" if stage == 2 else "stdin=subprocess.PIPE, "
    send_key = (
        ""
        if stage == 2
        else """            try:
                with process.stdin:
                    process.stdin.write(key + "\\n")
            finally:
                del key
"""
    )
    bootstrap = """import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from google.colab import drive
"""
    bootstrap += "\n\n" + inspect.getsource(batch.digest)
    bootstrap += "\n\n" + inspect.getsource(batch.unpack_bundle).replace(
        "mediflow_da_runtime.zip", ZIP_NAME
    )
    code = (
        bootstrap
        + f'''

drive.mount("/content/drive")
MY_DRIVE = Path("/content/drive/MyDrive")
PROJECT = MY_DRIVE / "mediflow_Project"
BUNDLE = PROJECT / "{ZIP_NAME}"
if not BUNDLE.is_file():
    raise FileNotFoundError(f"공통 ZIP v3 위치를 확인하세요. {{BUNDLE}}")
EXPECTED = "{bundle_sha}"
OVERLAY = {source!r}

print("Drive 공통 ZIP v3 확인 중", BUNDLE, flush=True)
with tempfile.TemporaryDirectory(prefix="mediflow_stage{stage}_", dir="/content") as temporary:
    work = Path(temporary)
    local_bundle = work / BUNDLE.name
    shutil.copyfile(BUNDLE, local_bundle)
    runtime = unpack_bundle(local_bundle, work / "runtime", EXPECTED)
    local_bundle.unlink()
    for name, module in OVERLAY.items():
        (runtime / "src/mediflow_datasets" / name).write_text(module, encoding="utf-8")
    env = {{**os.environ, "PYTHONPATH": str(runtime / "src"), "PYTHONIOENCODING": "utf-8",
           "PYTHONDONTWRITEBYTECODE": "1"}}
{input_code}    command = [sys.executable, "-u", "-m", "mediflow_datasets.adaptation_stage2",
               "--stage", "{stage}", "--launch",
               str(runtime), str(work), str(PROJECT), EXPECTED, str(MY_DRIVE)]
    with subprocess.Popen(command, env=env, {pipe}stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace",
                          start_new_session=True) as process:
        try:
{send_key}            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError("{stage}차가 완료되지 않았습니다. 위 로그를 확인하세요.")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
'''
    )
    compile(code, f"stage{stage}-colab", "exec")
    guide = f"""# MediFlow 2차 확장 적응

**GPU 선택 → 모두 실행 → Drive 연결**을 합니다. 기존 `mediflow_Project/{ZIP_NAME}`와
1차 선택 기록을 읽습니다. GPU 런타임 하나에서 실행하세요.

피부 기본·얼굴 선정·두피 선정 모델의 **1차 학습 전 저장본**에서 각각 5회 학습합니다.
1차와 같은 학습 사진과 조명 범위에 측정한 색·미세 대비·약한 흐림을 추가합니다.
검증은 원본·조명 변환·확장 변환으로 나뉩니다. 완료하면 학습 전·1차·2차 모델의
점수를 세 검증 조건에서 비교합니다. 결과는 탐색 비교로 기록합니다.

결과는 `내 드라이브/MediFlow_adaptation_runs/{bundle_sha[:16]}/stage2`에 저장합니다.
화면에 현재 분야, `Epoch 1/5`, 학습 배치, 검증과 저장 단계가 표시됩니다.
끊기면 같은 노트북을 다시 실행해 마지막 완료 epoch부터 이어갑니다.
"""
    if stage == 3:
        guide = f"""# MediFlow 3차 MMD·CORAL 특징 정렬

**GPU 선택 → 모두 실행 → Drive 연결 → 비공개 키 입력**을 합니다.
기존 공통 ZIP v3와 같은 실행 폴더의 1·2차 완료 기록을 읽습니다.
입력란이 뜨면 로컬 `MediFlow_Stage3_Target_Key_20261110.txt`의 한 줄을 붙여 넣으세요.

피부 기본·얼굴 선정·두피 선정의 **1차 학습 전 저장본**에서 각각 5 epoch 실행합니다.
2차와 같은 사진 변환과 학습량으로, 일부 배치에 MMD·CORAL 손실을 함께 더합니다.
사진은 네 참여자에서 고르게 뽑고, 학습에 사용한 라벨 없는 촬영본은 정렬 진단에 사용합니다.

결과는 `내 드라이브/MediFlow_adaptation_runs/{bundle_sha[:16]}/stage3`에 저장합니다.
원본·조명·확장 검증 점수와 MMD·CORAL, 학습곡선을 제공합니다.
중단됐으면 같은 노트북에서 마지막으로 저장된 회차부터 이어갑니다.
"""
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
                "id": "guide",
                "metadata": {},
                "source": guide.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "id": f"run-stage{stage}",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": code.splitlines(keepends=True),
            },
        ],
    }
    if with_curves:
        notebook["cells"][0]["source"].append(
            "\n마지막 셀은 저장한 기록으로 학습곡선을 만들고 Drive에 PNG로 저장합니다. "
            f"{stage}차를 이미 완료했다면 마지막 셀만 실행하세요.\n"
        )
        plot_source = (module_dir / "adaptation_curves.py").read_text(encoding="utf-8")
        plot_code = f'''
from pathlib import Path

from google.colab import drive
from IPython.display import Image, display

my_drive = Path("/content/drive/MyDrive")
if not my_drive.is_dir():
    drive.mount("/content/drive")
results = my_drive / "MediFlow_adaptation_runs" / "{bundle_sha[:16]}" / "stage{stage}"
if not all((results / domain / "complete.json").is_file()
           for domain in ("skin", "web_skin", "hair")):
    raise FileNotFoundError("세 분야의 {stage}차 학습을 완료한 뒤 이 셀을 실행하세요.")
plot_module = {{"__name__": "mediflow_adaptation_curves"}}
exec(compile({plot_source!r}, "adaptation_curves.py", "exec"), plot_module)
image_path = plot_module["save_curves"](
    results, {stage}, results / "stage{stage}_learning_curves.png"
)
print("학습곡선 저장", image_path)
display(Image(filename=str(image_path)))
'''
        compile(plot_code, "stage2-learning-curves", "exec")
        notebook["cells"].append(
            {
                "cell_type": "code",
                "id": f"plot-stage{stage}",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": plot_code.splitlines(keepends=True),
            }
        )
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--with-curves", action="store_true")
    parser.add_argument("--stage", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    info = json.loads(args.bundle.with_name("package_info.json").read_text(encoding="utf-8"))
    digest = batch.digest(args.bundle)
    if args.bundle.name != ZIP_NAME or info["bundle"] != ZIP_NAME or digest != info["sha256"]:
        raise ValueError("Expected the original, verified v3 bundle")
    with args.notebook.open("x", encoding="utf-8") as stream:
        json.dump(
            make_notebook(digest, stage=args.stage, with_curves=args.with_curves),
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
