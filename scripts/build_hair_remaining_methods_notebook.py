"""Build the Hair single-seed screening notebook for the remaining methods."""

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
    source_names = ("common_engine", "common_audit", "common_workflow", "paper_suite")
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
from mediflow_datasets.paper_suite import run as run_paper_screen
from mediflow_datasets.paper_suite import summarize as summarize_paper_screen
"""
    cells = [
        cell(
            """# Hair 논문 기반 강화 — 남은 방법 3종 선별

이미 완료한 기준선과 SupCon seed 42는 다시 학습하지 않습니다. 남은 독립 가설 세 가지를
seed 42에서 한 번씩 실행한 뒤 Validation 결과로 다음 대상을 고릅니다.

1. DINOv2 Small: 자기지도 사전학습 특징의 전이 가능성
2. EfficientNetV2-S · 256: 최신 EfficientNet 계열 Backbone 비교
3. EfficientNet-B1 · 384: 미세한 두피 질감 보존을 위한 고해상도 비교

Test, SAM, ensemble, seed 43·44와 모델 패키징은 실행하지 않습니다.
"""
        ),
        cell(
            """## 1. 실험 계약

공통 고정 조건은 Hair Clean ZIP, 같은 Train·Validation 분할, augmented Train,
original Validation, 클래스 순서, Label Smoothing 0.05와 seed 42입니다.

DINOv2는 알고리즘에 필요한 전처리와 고정 encoder를 사용합니다. EfficientNetV2-S는
Backbone만 바꾸고, B1·384는 입력 해상도만 바꿉니다. 세 결과를 기존 기준선·SupCon의
저장된 seed 42 결과와 나중에 함께 비교합니다.
"""
        ),
        cell(
            """DOMAIN = 'hair'
PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'
DATA_ZIP = '/content/drive/MyDrive/mediflow_Project/datasets/hair_datasets.zip'
AUDIT_DIR = ''
EXPECTED_DATA_SHA256 = '2ac7260663cf69835ba50edb6ae8c7e7ac13be9c73b9f7ea24f60e0342e7e156'
RESUME_DIR = ''
MODE = 'paper_screen'
SEED = 42
SEEDS = [42]
BATCH_SIZE = 32
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 15
EXTENSION_EPOCHS = 1  # 사용하지 않음
TRAIN_VARIANT = 'augmented'
RUN_EXPERIMENTS = [
    'dinov2_small_224',
    'efficientnetv2s_256',
    'multires_b1_384',
]
""",
            True,
        ),
        cell(
            """## 2. Drive와 실행 환경

Colab GPU 런타임에서 실행합니다. DINOv2는 KerasHub 공식 `dinov2_small` preset과
ImageConverter를 사용합니다. 최초 실행에서는 사전학습 가중치를 내려받으므로 인터넷 연결이
필요합니다.
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
print(
    'TensorFlow:', tf.__version__,
    'Keras:', keras.__version__,
    'KerasHub:', keras_hub.__version__,
)
""",
            True,
        ),
        cell(
            """## 3. 재현 코드

아래 셀은 수정하지 않습니다. 실행에 사용한 정확한 코드가 결과 폴더에도 저장됩니다.
"""
        ),
        cell(embedded, True),
        cell(
            """## 4. 데이터 확인과 결과 폴더 생성

Hair ZIP을 Colab 임시 디스크로 복사하고 기존 SHA-256과 같은지 확인합니다. 지정 경로가
실제로 달라졌을 때만 `DATA_ZIP`을 수정하세요.
"""
        ),
        cell(
            """from pathlib import Path
import zipfile

configured_zip = Path(DATA_ZIP) if DATA_ZIP else None
if not configured_zip or not configured_zip.is_file() or not zipfile.is_zipfile(configured_zip):
    search_roots = [
        Path(PROJECT_ROOT) / 'datasets',
        Path('/content/drive/MyDrive/mediflow_datasets'),
    ]
    candidates = sorted({
        path
        for root in search_roots if root.is_dir()
        for path in root.rglob('*')
        if path.is_file() and 'hair' in path.name.lower() and zipfile.is_zipfile(path)
    })
    print('자동 검색된 Hair ZIP:', [str(path) for path in candidates])
    if len(candidates) != 1:
        raise ValueError('Hair 데이터 ZIP 하나를 DATA_ZIP에 정확한 전체 경로로 입력하세요.')
    DATA_ZIP = str(candidates[0])

config = dict(
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
            """## 5. 남은 방법 3종 자동 학습

DINOv2→EfficientNetV2-S→B1·384 순서로 총 3개 모델을 학습합니다. 각 실험이 끝날 때마다
모델과 기록을 Drive에 저장합니다. 중단되면 출력된 폴더를 `RESUME_DIR`에 넣고 같은 설정으로
다시 실행하세요.
"""
        ),
        cell("records = run_paper_screen(context)\n", True),
        cell(
            """## 6. Validation 선별 보고서

세 방법의 Accuracy, Macro F1, 클래스별 F1, 학습 시간과 학습곡선을 저장합니다. 결과 ZIP을
받은 뒤 기존 기준선·SupCon seed 42와 한 표로 합쳐 다음 실험을 결정합니다.
"""
        ),
        cell("summary, report_zip = summarize_paper_screen(context, records)\nsummary\n", True),
    ]
    for index, item in enumerate(cells):
        item["id"] = f"cell-{index:02d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": "08_hair_remaining_methods_screen_colab.ipynb"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = ROOT / "notebooks/08_hair_remaining_methods_screen_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return output


if __name__ == "__main__":
    build()
