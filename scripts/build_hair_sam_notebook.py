"""Build the standalone Hair B1/384 Adam versus SAM screening notebook."""

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
    source_names = ("common_engine", "common_audit", "common_workflow", "hair_sam")
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
from mediflow_datasets.hair_sam import run as run_sam_screen
from mediflow_datasets.hair_sam import summarize as summarize_sam_screen
"""
    cells = [
        cell(
            """# Hair 최종 선별 — B1·384 Adam 대 SAM

현재 seed 42 선두인 B1·384에 Sharpness-Aware Minimization(SAM)을 적용합니다.

논문 근거: Foret et al., *Sharpness-Aware Minimization for Efficiently Improving
Generalization*, ICLR 2021.
https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/

기존 B1·384의 정확한 Stage 1 체크포인트에서 시작하여, 기존 Adam 부분 미세조정과 SAM 부분
미세조정의 차이만 확인합니다. Test, ensemble, seed 반복과 모델 패키징은 수행하지 않습니다.
"""
        ),
        cell(
            """## 1. 비교 조건

고정: 데이터, Train·Validation 분할, augmented Train, original Validation, B1 Backbone,
384 입력, Label Smoothing 0.05, Stage 1 체크포인트, 후반 30개 레이어, 학습률 1e-5,
미세조정 15 epoch와 seed 42.

변경: 기존 Adam update 대신 rho 0.05의 SAM update를 사용합니다. SAM은 각 batch에서 주변
가중치 방향을 확인하기 위해 forward/backward 계산을 두 번 수행하므로 학습 시간이 늘어납니다.
"""
        ),
        cell(
            """DOMAIN = 'hair'
PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'
DATA_ZIP = '/content/drive/MyDrive/mediflow_Project/datasets/hair_datasets.zip'
AUDIT_DIR = ''
EXPECTED_DATA_SHA256 = '2ac7260663cf69835ba50edb6ae8c7e7ac13be9c73b9f7ea24f60e0342e7e156'
RESUME_DIR = ''
MODE = 'sam_screen'
SEED = 42
SEEDS = [42]
BATCH_SIZE = 32
STAGE1_EPOCHS = 15  # 새로 실행하지 않음
STAGE2_EPOCHS = 15
EXTENSION_EPOCHS = 1  # 사용하지 않음
TRAIN_VARIANT = 'augmented'
RUN_EXPERIMENTS = ['sam_b1_384']
SAM_RHO = 0.05
PARENT_RUN_DIR = (
    '/content/drive/MyDrive/mediflow_Project/2_results/hair/'
    'paper_screen_20260917_141641_874f03f1'
)
PARENT_STAGE1_SHA256 = '0e064976688f8c0f08869d2e6a31b4ebff3992720e21c8bb838bf5ba921e622e'
""",
            True,
        ),
        cell(
            """## 2. Drive와 실행 환경

Colab GPU 런타임에서 실행합니다. `PARENT_RUN_DIR`의 B1·384 Stage 1 모델이 그대로 남아
있어야 하며, 노트북은 SHA-256을 검사한 뒤에만 SAM 학습을 시작합니다.
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
            """## 3. 재현 코드

이 셀은 수정하지 않습니다. 정확한 실행 코드가 새 결과 폴더에 함께 저장됩니다.
"""
        ),
        cell(embedded, True),
        cell(
            """## 4. 데이터·부모 체크포인트 확인

Hair ZIP과 B1·384 Stage 1 모델의 SHA-256, 클래스 순서와 seed를 검사합니다. 하나라도 다르면
학습을 시작하지 않습니다.
"""
        ),
        cell(
            """config = dict(
    domain=DOMAIN, project_root=PROJECT_ROOT, data_zip=DATA_ZIP,
    audit_dir=AUDIT_DIR, expected_data_sha256=EXPECTED_DATA_SHA256,
    resume_dir=RESUME_DIR, mode=MODE, seed=SEED, seeds=SEEDS,
    batch_size=BATCH_SIZE, epochs1=STAGE1_EPOCHS, epochs2=STAGE2_EPOCHS,
    extension_epochs=EXTENSION_EPOCHS, train_variant=TRAIN_VARIANT,
    experiments=RUN_EXPERIMENTS, sam_rho=SAM_RHO,
    parent_run_dir=PARENT_RUN_DIR,
    parent_stage1_sha256=PARENT_STAGE1_SHA256,
)
context = prepare(config, PROFILES, SOURCES, BUILD_COMMIT)
print('결과 폴더 / 중단 시 RESUME_DIR:', context['output'])
print('데이터 SHA-256:', context['data_hash'])
print('부모 실행:', PARENT_RUN_DIR)
""",
            True,
        ),
        cell(
            """## 5. SAM 부분 미세조정

Stage 1은 다시 학습하지 않습니다. 기존 Stage 1 최적 모델에서 후반 30개 레이어를 열고
SAM으로 15 epoch만 학습합니다. BatchNormalization 레이어는 고정합니다.
"""
        ),
        cell("baseline_record, sam_record = run_sam_screen(context)\n", True),
        cell(
            """## 6. Adam 대 SAM Validation 보고서

동일한 Stage 1에서 출발한 기존 Adam 결과와 SAM 결과를 비교합니다. 보고서 ZIP에는 모델을
제외한 설정, Validation 지표, 예측 CSV와 그래프가 들어갑니다.
"""
        ),
        cell(
            "summary, report_zip = summarize_sam_screen(\n"
            "    context, baseline_record, sam_record\n"
            ")\nsummary\n",
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
            "colab": {"name": "09_hair_b1_384_sam_screen_colab.ipynb"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = ROOT / "notebooks/09_hair_b1_384_sam_screen_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return output


if __name__ == "__main__":
    build()
