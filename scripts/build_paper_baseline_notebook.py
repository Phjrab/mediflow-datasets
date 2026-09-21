"""Build the standalone Colab notebook for the three-seed paper baseline."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(value, code=False):
    result = {
        "cell_type": "code" if code else "markdown",
        "metadata": {},
        "source": value.splitlines(keepends=True),
    }
    if code:
        result.update(execution_count=None, outputs=[])
    return result


def candidate_classes():
    candidate_root = ROOT / "results"
    paths = {
        "hair": candidate_root
        / "hair/candidates/public_candidate_v1_b1_256_ls005_20260907_120834/class_names.json",
        "web_skin": candidate_root
        / (
            "web_skin/candidates/public_candidate_v1_b0_256_ce_"
            "20260908_081603_3f1ce76e/class_names.json"
        ),
        "skin": candidate_root
        / (
            "skin/candidates/public_candidate_v1_b0_224_ce_augmented_"
            "20260909_075056/class_names.json"
        ),
    }
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}


def build():
    source_names = ("common_engine", "common_audit", "common_workflow", "paper_baseline")
    sources = {
        name: (ROOT / f"src/mediflow_datasets/{name}.py").read_text(encoding="utf-8")
        for name in source_names
    }
    profiles = candidate_classes()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    embedded = f"""import sys, types
SOURCES = {sources!r}
BUILD_COMMIT = {commit!r}
PROFILES = {profiles!r}
package = types.ModuleType('mediflow_datasets')
package.__path__ = []
sys.modules['mediflow_datasets'] = package
for name, source in SOURCES.items():
    module = types.ModuleType('mediflow_datasets.' + name)
    sys.modules[module.__name__] = module
    exec(compile(source, name + '.py', 'exec'), module.__dict__)
from mediflow_datasets.common_workflow import prepare
from mediflow_datasets.paper_baseline import run as run_baseline
from mediflow_datasets.paper_baseline import summarize
"""
    cells = [
        cell(
            """# MediFlow 논문 기반 강화 ① — Hair 3-seed 기준선

새 기법을 적용하기 전에 현재 Hair 최선 설정을 seed 42, 43, 44에서 반복합니다.
이 단계의 목적은 성능을 높이는 것이 아니라 **Validation 성능의 평균과 변동 범위**를
확보하는 것입니다.

- 고정: Clean 데이터 ZIP, 분할, 클래스, 증강본, B1, 256, Label Smoothing 0.05
- 반복: seed 42, 43, 44
- 선택 지표: Validation Macro F1
- 금지: Test 평가, 기존 후보 덮어쓰기, 새로운 증강 또는 새 기법 추가

노트북 하나에 실행 코드가 포함되어 있어 별도 저장소 연결은 필요 없습니다.
Colab GPU를 선택하고 설정 셀을 수정한 뒤 위에서부터 실행하세요.
"""
        ),
        cell(
            """## 1. 사용자 설정

`DATA_ZIP`에는 기존 Hair Clean ZIP의 전체 경로를 입력하세요.
`AUDIT_DIR`에는 이 ZIP을 검사했던 공통 검증 결과 폴더를 입력합니다.
검증을 생략하기로 한 동일 ZIP이면 `EXPECTED_DATA_SHA256`에 확인된 해시를 넣을 수 있습니다.
둘 중 하나는 반드시 필요합니다.

실행이 끊기면 출력된 폴더를 `RESUME_DIR`에 입력합니다. 완료된 seed 결과는 해시를 확인해
재사용하고, 중단된 seed는 새 attempt에서 다시 시작합니다.
"""
        ),
        cell(
            """DOMAIN = 'hair'
PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'
DATA_ZIP = ''
AUDIT_DIR = ''
EXPECTED_DATA_SHA256 = '2ac7260663cf69835ba50edb6ae8c7e7ac13be9c73b9f7ea24f60e0342e7e156'
RESUME_DIR = ''
MODE = 'baseline3'
SEED = 42
SEEDS = [42, 43, 44]
BATCH_SIZE = 32
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 15
EXTENSION_EPOCHS = 1  # 이 노트북에서는 사용하지 않음
TRAIN_VARIANT = 'augmented'
""",
            True,
        ),
        cell(
            """## 2. Drive 연결과 실행 환경

TensorFlow와 Keras 버전을 고정합니다. 설치 직후 버전 오류가 나면 Colab 세션을 다시 시작하고
처음부터 실행하세요. GPU가 없으면 학습을 시작하지 않습니다.
"""
        ),
        cell(
            """from google.colab import drive
drive.mount('/content/drive')
%pip -q install tensorflow==2.20.0 keras==3.13.2 pandas matplotlib pillow tqdm
import tensorflow as tf
import keras
if tf.__version__ != '2.20.0' or keras.__version__ != '3.13.2':
    raise RuntimeError('Colab 세션을 다시 시작한 뒤 처음부터 실행하세요.')
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('런타임 유형에서 GPU를 선택하세요.')
""",
            True,
        ),
        cell(
            """## 3. 실행 코드

아래 셀은 직접 수정하지 않습니다. 실행 코드와 생성 당시 commit은 결과 폴더에 함께 저장됩니다.
"""
        ),
        cell(embedded, True),
        cell(
            """## 4. 데이터 준비

Drive의 ZIP을 Colab 임시 디스크로 복사하고 SHA-256, 클래스 폴더와 평가 분할을 확인합니다.
세 seed는 같은 Train·Validation 사진을 사용하며 shuffle과 모델 초기화 seed만 달라집니다.
"""
        ),
        cell(
            """config = dict(
    domain=DOMAIN, project_root=PROJECT_ROOT, data_zip=DATA_ZIP,
    audit_dir=AUDIT_DIR, expected_data_sha256=EXPECTED_DATA_SHA256,
    resume_dir=RESUME_DIR, mode=MODE, seed=SEED, seeds=SEEDS,
    batch_size=BATCH_SIZE, epochs1=STAGE1_EPOCHS, epochs2=STAGE2_EPOCHS,
    extension_epochs=EXTENSION_EPOCHS, train_variant=TRAIN_VARIANT,
)
context = prepare(config, PROFILES, SOURCES, BUILD_COMMIT)
print('결과 폴더 / 중단 시 RESUME_DIR:', context['output'])
print('데이터 SHA-256:', context['data_hash'])
print('클래스 순서:', context['classes'])
""",
            True,
        ),
        cell(
            """## 5. Hair 기준선 3회 학습

각 seed에서 ImageNet EfficientNet-B1을 새로 초기화합니다. Stage 1은 분류층 15 epoch,
Stage 2는 후반부 일부를 낮은 학습률로 15 epoch 학습합니다. 모든 모델 선택은 Validation으로만
수행하며 Test 폴더는 읽지 않습니다.
"""
        ),
        cell("records = run_baseline(context)\n", True),
        cell(
            """## 6. 평균·표준편차와 그래프 저장

Validation Accuracy, Macro F1, 클래스별 F1의 3-seed 평균과 모집단 표준편차를 저장합니다.
학습곡선과 비교 대시보드도 생성하며 모델을 제외한 보고서 ZIP을 Drive에 만듭니다.
이 결과가 다음 Supervised Contrastive Learning 실험의 비교 기준입니다.
"""
        ),
        cell("summary, report_zip = summarize(context, records)\nsummary\n", True),
    ]
    for index, item in enumerate(cells):
        item["id"] = f"cell-{index:02d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": "04_hair_three_seed_baseline_colab.ipynb"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = ROOT / "notebooks/04_hair_three_seed_baseline_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return output


if __name__ == "__main__":
    build()
