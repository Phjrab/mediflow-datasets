"""Build the standalone Colab notebook for Hair paper experiments."""

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


def hair_classes():
    path = ROOT / (
        "results/hair/candidates/public_candidate_v1_b1_256_ls005_"
        "20260907_120834/class_names.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def build():
    source_names = (
        "common_engine",
        "common_audit",
        "common_workflow",
        "paper_baseline",
        "paper_suite",
    )
    sources = {
        name: (ROOT / f"src/mediflow_datasets/{name}.py").read_text(encoding="utf-8")
        for name in source_names
    }
    profiles = {"hair": hair_classes()}
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
from mediflow_datasets.paper_suite import DEFAULT_EXPERIMENTS
from mediflow_datasets.paper_suite import run as run_paper_suite
from mediflow_datasets.paper_suite import summarize as summarize_paper_suite
"""
    cells = [
        cell(
            """# MediFlow 논문 기반 강화 ② — Hair 다중 실험 자동 실행

한 번의 실행으로 논문 근거가 있는 Hair 분류 실험을 각각 독립적으로 수행합니다.
각 조건은 seed 42·43·44에서 반복하며 모델 선택에는 Validation만 사용합니다.

1. 현재 기준선: EfficientNet-B1 · 256 · Label Smoothing
2. Supervised Contrastive Learning: 유사 클래스의 특징 공간 분리
3. DINOv2 Small linear probe: 자기지도 사전학습 특징 비교
4. EfficientNetV2-S: Backbone 교체 비교
5. Multi-resolution: B1의 입력만 384로 변경

각 실험은 하나의 연구 가설이며 결과를 서로 섞지 않습니다. Test 폴더는 읽지 않습니다.
전체 15회 학습이므로 Colab이 끊길 수 있습니다. 각 완료 결과는 Drive에 즉시 저장되며,
다시 실행할 때 `RESUME_DIR`을 지정하면 완료 항목을 검증한 뒤 건너뜁니다.
"""
        ),
        cell(
            """## 1. 사용자 설정

기존 Hair Clean ZIP을 그대로 사용합니다. 기본 해시는 기존 후보 패키지와 학습 설정 양쪽에서
확인된 값입니다. 일부 실험만 실행하고 싶으면 `RUN_EXPERIMENTS`에서 이름을 빼면 됩니다.
동일 결과 폴더를 이어서 실행할 때는 실험 목록과 순서를 바꾸지 마세요.
"""
        ),
        cell(
            """DOMAIN = 'hair'
PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'
DATA_ZIP = '/content/drive/MyDrive/mediflow_datasets/hair_clean_v1_20260907_053616.zip'
AUDIT_DIR = ''
EXPECTED_DATA_SHA256 = '2ac7260663cf69835ba50edb6ae8c7e7ac13be9c73b9f7ea24f60e0342e7e156'
RESUME_DIR = ''
MODE = 'paper_suite'
SEED = 42
SEEDS = [42, 43, 44]
BATCH_SIZE = 32
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 15
EXTENSION_EPOCHS = 1  # 이 suite에서는 사용하지 않음
TRAIN_VARIANT = 'augmented'
RUN_EXPERIMENTS = [
    'baseline_b1_256',
    'supcon_b1_256',
    'dinov2_small_224',
    'efficientnetv2s_256',
    'multires_b1_384',
]
""",
            True,
        ),
        cell(
            """## 2. Drive와 실행 환경

GPU 런타임에서 실행하세요. DINOv2는 KerasHub 공식 preset과 해당 preset의 전처리기를
사용하므로 기존 EfficientNet의 내부 전처리를 억지로 재사용하지 않습니다.
"""
        ),
        cell(
            """from google.colab import drive
drive.mount('/content/drive')
%pip -q install tensorflow==2.20.0 keras==3.13.2 keras-hub==0.31.1 pandas matplotlib pillow tqdm
import tensorflow as tf
import keras
import keras_hub
if tf.__version__ != '2.20.0' or keras.__version__ != '3.13.2':
    raise RuntimeError('Colab 세션을 다시 시작한 뒤 처음부터 실행하세요.')
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('런타임 유형에서 GPU를 선택하세요.')
print('TensorFlow:', tf.__version__, 'Keras:', keras.__version__, 'KerasHub:', keras_hub.__version__)
""",
            True,
        ),
        cell(
            """## 3. 재현 코드 불러오기

아래 셀에는 실행 코드 전체가 포함되어 있습니다. 생성 당시 Git commit과 정확한 코드 사본이
Drive 결과 폴더에도 저장됩니다.
"""
        ),
        cell(embedded, True),
        cell(
            """## 4. 데이터 준비와 실행 폴더 생성

ZIP을 Colab 임시 디스크에 복사하고 SHA-256과 클래스 구조를 확인합니다. Train은 기존
augmented 폴더를, Validation은 모든 실험에서 동일한 original 폴더를 사용합니다.
"""
        ),
        cell(
            """config = dict(
    domain=DOMAIN, project_root=PROJECT_ROOT, data_zip=DATA_ZIP,
    audit_dir=AUDIT_DIR, expected_data_sha256=EXPECTED_DATA_SHA256,
    resume_dir=RESUME_DIR, mode=MODE, seed=SEED, seeds=SEEDS,
    batch_size=BATCH_SIZE, epochs1=STAGE1_EPOCHS, epochs2=STAGE2_EPOCHS,
    extension_epochs=EXTENSION_EPOCHS, train_variant=TRAIN_VARIANT,
    experiments=RUN_EXPERIMENTS,
)
context = prepare(config, PROFILES, SOURCES, BUILD_COMMIT)
print('결과 폴더 / 중단 시 RESUME_DIR:', context['output'])
print('데이터 SHA-256:', context['data_hash'])
print('클래스 순서:', context['classes'])
print('실행 실험:', RUN_EXPERIMENTS)
""",
            True,
        ),
        cell(
            """## 5. 전체 실험 자동 실행

위 목록 순서대로 각 실험의 seed 42·43·44를 실행합니다. 완료 모델과 기록은 매번 Drive에
저장됩니다. 오류가 발생하면 결과 폴더의 failure JSON과 마지막 출력 내용을 보관하세요.
"""
        ),
        cell("records = run_paper_suite(context)\n", True),
        cell(
            """## 6. 전체 비교표와 그래프 생성

실험별 Validation Accuracy·Macro F1·클래스별 F1의 3-seed 평균과 표준편차를 만듭니다.
모델 파일을 제외한 보고서 ZIP도 생성합니다. 여기서 가장 좋은 조건은 잠정 Validation 선두이며,
아직 최종 Test 결과나 공개 후보 모델이 아닙니다.
"""
        ),
        cell(
            "summary, report_zip = summarize_paper_suite(context, records)\nsummary\n",
            True,
        ),
    ]
    for index, item in enumerate(cells):
        item["id"] = f"cell-{index:02d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": "05_hair_paper_experiment_suite_colab.ipynb"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = ROOT / "notebooks/05_hair_paper_experiment_suite_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return output


if __name__ == "__main__":
    build()
