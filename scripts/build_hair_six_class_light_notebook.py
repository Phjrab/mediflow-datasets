"""Build a self-contained Colab notebook for the fixed B0 lightweight alternate."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/19_hair_six_class_light_package_colab.ipynb"


def cell(content: str, *, code: bool = False) -> dict:
    item = {
        "cell_type": "code" if code else "markdown",
        "metadata": {},
        "source": content.splitlines(keepends=True),
    }
    if code:
        item.update(execution_count=None, outputs=[])
    return item


def build() -> Path:
    names = ("common_engine", "hair_six_class_light")
    sources = {
        name: (ROOT / f"src/mediflow_datasets/{name}.py").read_text(encoding="utf-8")
        for name in names
    }
    embedded = (
        "import sys, types\n"
        f"SOURCES = {sources!r}\n"
        "package_module = types.ModuleType('mediflow_datasets')\n"
        "package_module.__path__ = []\n"
        "sys.modules['mediflow_datasets'] = package_module\n"
        "for name, source in SOURCES.items():\n"
        "    module = types.ModuleType('mediflow_datasets.' + name)\n"
        "    sys.modules[module.__name__] = module\n"
        "    exec(compile(source, name + '.py', 'exec'), module.__dict__)\n"
        "from mediflow_datasets.hair_six_class_light import package_light, verify_candidate\n"
    )
    cells = [
        cell(
            "# Hair 6클래스 경량 모델: B0·256·증강\n\n"
            "이미 완료된 네 실험 중 두 B0·256 모델의 **Validation Macro F1**을 비교해 "
            "증강 모델을 경량 후보로 고정했습니다. 이 노트북은 새 학습 없이 고정 Test에서 "
            "한 번 평가하고 별도 모델 ZIP을 만듭니다. B1·384 성능 우선 ZIP은 유지합니다.\n\n"
            "이 노트북은 경량 후보의 Test 결과를 본 뒤 모델 조건을 다시 고르는 용도가 아닙니다. "
            "실제 장비 속도와 메모리 측정은 별도 단계입니다.\n"
        ),
        cell("## 1. Drive 경로\n\n기존 실험 결과와 새 6클래스 데이터 ZIP을 그대로 읽습니다.\n"),
        cell(
            "PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'\n"
            "PARENT_RUN_DIR = (\n"
            "    PROJECT_ROOT + '/2_results/hair/'\n"
            "    'hair_six_class_four_20260928_011824_ad21e8ae'\n"
            ")\n"
            "BATCH_SIZE = 32\n",
            code=True,
        ),
        cell("## 2. 실행 환경\n\nGPU를 사용할 수 있으면 평가가 더 빠릅니다.\n"),
        cell(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "%pip -q install tensorflow==2.20.0 keras==3.13.2\n"
            "import tensorflow as tf\n"
            "import keras\n"
            "if tf.__version__ != '2.20.0' or keras.__version__ != '3.13.2':\n"
            "    raise RuntimeError('런타임을 다시 시작하고 처음부터 실행하세요.')\n",
            code=True,
        ),
        cell("## 3. 고정 코드\n\n원본 실행 코드를 노트북에 포함했습니다. 수정하지 마세요.\n"),
        cell(embedded, code=True),
        cell("## 4. 경량 후보 확인\n\n모델 해시와 Validation 선정 근거를 먼저 확인합니다.\n"),
        cell(
            "from pathlib import Path\n"
            "candidate = verify_candidate(Path(PARENT_RUN_DIR))\n"
            "print('모델:', candidate['model_path'])\n"
            "print('Validation Accuracy:', candidate['candidate']['validation']['accuracy'])\n"
            "print('Validation Macro F1:', candidate['candidate']['validation']['macro_f1'])\n",
            code=True,
        ),
        cell(
            "## 5. Test 1회 평가와 패키징\n\n"
            "Test 사진은 실행 중 임시 공간에만 풀고, 평가 뒤 제거합니다. "
            "결과는 Drive의 `2_results/hair/selected_models`에 남습니다. "
            "중단 후 다시 실행하면 완료된 평가의 해시를 확인하고 재사용합니다.\n"
        ),
        cell(
            "output, model_zip, metrics = package_light(\n"
            "    PARENT_RUN_DIR, PROJECT_ROOT, batch_size=BATCH_SIZE\n"
            ")\n"
            "print('경량 후보 ZIP:', model_zip)\n"
            "print('ZIP SHA-256 파일:', model_zip.with_suffix('.zip.sha256'))\n"
            "print('Test Accuracy:', metrics['accuracy'])\n"
            "print('Test Macro F1:', metrics['macro_f1'])\n"
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
