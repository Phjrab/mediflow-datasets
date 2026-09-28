"""Create a self-contained Colab notebook for the fixed six-class Hair package."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/mediflow_datasets/hair_six_class_package.py"
OUTPUT = ROOT / "notebooks/18_hair_six_class_package_colab.ipynb"


def cell(content: str, *, code: bool = False) -> dict:
    result = {
        "cell_type": "code" if code else "markdown",
        "metadata": {},
        "source": content.splitlines(keepends=True),
    }
    if code:
        result.update(execution_count=None, outputs=[])
    return result


def build() -> Path:
    source = SOURCE.read_text(encoding="utf-8")
    embedded = (
        "import sys, types\n"
        f"SOURCE = {source!r}\n"
        "module = types.ModuleType('hair_six_class_package')\n"
        "sys.modules[module.__name__] = module\n"
        "exec(compile(SOURCE, 'hair_six_class_package.py', 'exec'), module.__dict__)\n"
        "from hair_six_class_package import package, verify_records\n"
    )
    cells = [
        cell(
            "# Hair 6클래스 선정 모델 패키징\n\n"
            "이미 완료한 4개 실험의 Validation 선정 결과와 고정 Test 기록을 확인하고, "
            "선정된 **B1·384 원본 학습 모델**을 팀 전달용 ZIP으로 묶습니다. "
            "학습과 Test 평가는 다시 실행하지 않습니다. 기존 Hair 5클래스 v1/v2도 변경하지 않습니다.\n"
        ),
        cell(
            "## 1. 경로\n\nDrive의 기존 실험 결과 폴더를 그대로 사용합니다. "
            "모델은 해당 폴더 안의 `b1_384_original/.../stage2_best.keras`에서 읽습니다.\n"
        ),
        cell(
            "PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'\n"
            "PARENT_RUN_DIR = (\n"
            "    PROJECT_ROOT + '/2_results/hair/'\n"
            "    'hair_six_class_four_20260928_011824_ad21e8ae'\n"
            ")\n",
            code=True,
        ),
        cell(
            "## 2. Drive 연결 및 모델 확인 환경\n\nGPU가 없어도 실행할 수 있습니다. 패키징 전에 모델을 실제로 열어 입출력을 검사합니다.\n"
        ),
        cell(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "%pip -q install tensorflow==2.20.0 keras==3.13.2\n"
            "import tensorflow as tf\n"
            "import keras\n"
            "if tf.__version__ != '2.20.0' or keras.__version__ != '3.13.2':\n"
            "    raise RuntimeError('런타임을 다시 시작한 뒤 처음부터 실행하세요.')\n",
            code=True,
        ),
        cell(
            "## 3. 고정 패키징 코드\n\n이 셀에는 현재 저장된 재현 코드가 포함되어 있습니다. 수정하지 마세요.\n"
        ),
        cell(embedded, code=True),
        cell(
            "## 4. 선정 기록 사전 확인\n\n모델 파일, 해시, 클래스 순서, Validation 선정 기록과 Test 기록이 맞는지 확인합니다.\n"
        ),
        cell(
            "from pathlib import Path\n"
            "records = verify_records(Path(PARENT_RUN_DIR))\n"
            "print('선정 모델:', records['model_path'])\n"
            "print('모델 SHA-256:', records['model_sha256'])\n"
            "print('클래스 순서:', records['classes'])\n"
            "print('기존 Test Accuracy:', records['test']['accuracy'])\n"
            "print('기존 Test Macro F1:', records['test']['macro_f1'])\n",
            code=True,
        ),
        cell(
            "## 5. 새 후보 ZIP 생성\n\n`2_results/hair/selected_models`에 새 6클래스 폴더·모델 ZIP·SHA-256 파일을 만듭니다. 기존 후보는 유지됩니다.\n"
        ),
        cell(
            "output, model_zip = package(PARENT_RUN_DIR, PROJECT_ROOT)\n"
            "print('패키지 폴더:', output)\n"
            "print('팀 전달용 모델 ZIP:', model_zip)\n"
            "print('ZIP SHA-256:', model_zip.with_suffix('.zip.sha256'))\n"
            "print('모델 크기(MiB):', round((output / 'hair_model.keras').stat().st_size / 1024**2, 1))\n",
            code=True,
        ),
    ]
    for index, item in enumerate(cells):
        item["id"] = f"cell-{index:02d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": OUTPUT.name},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
