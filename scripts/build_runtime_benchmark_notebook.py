"""Build the self-contained Colab runtime benchmark notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "mediflow_datasets" / "runtime_benchmark.py"
TARGET = ROOT / "notebooks" / "20_four_cohort_runtime_benchmark_colab.ipynb"


def cell(kind: str, source: str) -> dict:
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def build() -> None:
    notebook = {
        "cells": [
            cell(
                "markdown",
                "# MediFlow 저장 모델 실행 비용 비교\n\n"
                "Skin 4개, Web Skin 12개, Hair 5클래스 14개, Hair 6클래스 4개, "
                "총 **34개 기존 학습 조건**을 분리해 측정합니다. 재학습하지 않습니다. "
                "저장된 Validation 정확도·"
                "학습 시간은 기존 실행 기록에서 읽고, 이 Colab GPU에서 사진 한 장의 "
                "추론 시간을 새로 측정합니다. 서로 다른 도메인의 정확도를 직접 "
                "순위화하지 않습니다.\n\n"
                "MedSigLIP은 저장된 `.pt` 분류층과 `google/medsiglip-448` 기본 모델을 "
                "함께 불러옵니다. 이미지 전처리·기본 모델·분류층을 모두 포함한 전체 시간과 "
                "모델 계산 시간만 별도로 기록합니다. `HF_TOKEN`은 Colab Secrets에서만 읽으며 "
                "출력에 표시하지 않습니다.\n\n"
                "**준비:** GPU 런타임 선택 → Google Drive 연결 → 필요한 경우 Colab "
                "Secrets에서 `HF_TOKEN` 노트북 접근 허용. 결과는 Drive의 기존 "
                "`2_results` 아래 새 benchmark 폴더에 저장됩니다. "
                "기존 모델과 결과는 변경하지 않습니다. 초기 ZIP의 모델 6개는 "
                "기존 로컬 결과의 SHA-256과 대조하여 Colab 임시 폴더에만 풉니다. "
                "ZIP 안에서 같은 모델을 찾지 못하면 "
                "측정을 시작하지 않고 오류를 표시합니다.\n",
            ),
            cell(
                "code",
                "%pip -q install tensorflow==2.20.0 tensorflow-text==2.20.1 "
                "keras==3.15.1 keras-hub==0.31.1 "
                "matplotlib pillow 'transformers==4.53.2' "
                "'huggingface_hub>=0.33,<1' 'accelerate>=1.8,<2' safetensors\n",
            ),
            cell(
                "code",
                "from google.colab import drive\n"
                "drive.mount('/content/drive')\n"
                "import tensorflow as tf\n"
                "for gpu in tf.config.list_physical_devices('GPU'):\n"
                "    tf.config.experimental.set_memory_growth(gpu, True)\n"
                "import keras_hub  # Keras 저장 모델의 DINOv2 계층 등록\n"
                "print('GPU:', tf.config.list_physical_devices('GPU'))\n",
            ),
            cell(
                "code",
                "# 학습은 하지 않으며, 실행 기록과 저장 체크포인트만 읽습니다.\n"
                "PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'\n"
                "WARMUP = 10\n"
                "REPEATS = 30\n"
                "RESUME_DIR = ''  # 중단된 경우 화면에 출력된 저장 폴더를 입력\n"
                "# Hugging Face 이용 동의가 완료된 계정의 토큰을 Secrets에서 읽습니다.\n"
                "from google.colab import userdata\n"
                "try:\n"
                "    HF_TOKEN = userdata.get('HF_TOKEN')\n"
                "except Exception:\n"
                "    HF_TOKEN = None\n"
                "print('MedSigLIP 접근:', '설정됨' if HF_TOKEN else '미설정')\n",
            ),
            cell(
                "markdown",
                "## 측정 코드\n아래 셀은 이 노트북 자체에 포함된 코드입니다. "
                "별도 `.py` 업로드가 필요하지 않습니다.\n",
            ),
            cell("code", SOURCE.read_text(encoding="utf-8")),
            cell(
                "code",
                "# 오래 걸리는 측정 전에 34개 모델을 전부 찾았는지 확인합니다.\n"
                "from collections import Counter\n"
                "from pathlib import Path\n"
                "planned = discover(Path(PROJECT_ROOT))\n"
                "counts = dict(Counter(row['cohort'] for row in planned))\n"
                "missing = [(row['cohort'], row['experiment'], row['status'], "
                "row.get('error', '')) "
                "for row in planned if row['status'] != 'ready']\n"
                "print('발견한 실험:', len(planned), counts)\n"
                "print('누락:', missing)\n"
                "if counts != EXPECTED_COUNTS or missing:\n"
                "    raise RuntimeError('34개 모델 사전 점검 실패: 위 누락 목록을 확인하세요.')\n"
                "if not HF_TOKEN:\n"
                "    raise RuntimeError('MedSigLIP 2개 측정에 HF_TOKEN이 필요합니다.')\n"
                "print('34개 사전 점검 통과')\n",
            ),
            cell(
                "code",
                "from datetime import datetime\n"
                "from pathlib import Path\n"
                "project = Path(PROJECT_ROOT)\n"
                "destination = (Path(RESUME_DIR) if RESUME_DIR else "
                "project / '2_results' / "
                "('runtime_benchmark_' + datetime.now().strftime('%Y%m%d_%H%M%S')))\n"
                "records = run(project, destination, HF_TOKEN, warmup=WARMUP, "
                "repeats=REPEATS, planned_rows=planned)\n"
                "print('저장:', destination)\n"
                "print('상태:', dict(Counter(row['status'] for row in records)))\n"
                "print('도메인:', dict(Counter(row['cohort'] for row in records)))\n"
                "print('측정 완료:', sum(row['status'] == 'measured' for row in records), '/ 34')\n"
                "print('실패/누락 상세: benchmark.csv의 status와 error 열 확인')\n",
            ),
            cell(
                "markdown",
                "## 결과 읽는 법\n"
                "- `*_dashboard.png`: 도메인별 Validation 정확도, 전체 추론 "
                "p50·p95, 기록된 학습 시간.\n"
                "- `*_tradeoff.png`: 같은 도메인 안에서 정확도와 지연 시간 비교.\n"
                "- `*_metrics_overview.png`: 지표 히트맵과 모델 간 지연 시간 분포.\n"
                "- `benchmark.csv/json`: 체크포인트 경로, 기록 출처, 상태, 모든 수치.\n"
                "- `environment.json`: Colab 장치와 측정 횟수.\n\n"
                "학습 시간은 과거 Colab 실행에서 기록된 값이므로 이번 측정 GPU의 학습 속도를 "
                "뜻하지 않습니다. `checkpoint_missing`과 `failed`는 그래프에서 빼고 CSV에 이유를 "
                "남깁니다. 이 수치는 Colab GPU의 실행 비용이며 Jetson 실측값이 아닙니다.\n",
            ),
        ],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
