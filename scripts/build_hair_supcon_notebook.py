"""Build the standalone Hair baseline versus SupCon Colab notebook."""

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
        "paper_suite",
        "hair_supcon",
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
from mediflow_datasets.hair_supcon import run as run_supcon_comparison
from mediflow_datasets.hair_supcon import summarize as summarize_supcon_comparison
"""
    cells = [
        cell(
            """# Hair 실험 1 — Supervised Contrastive Learning 비교

이번 노트북은 **한 가지 연구 가설만** 확인합니다.

> 같은 질환의 특징을 가깝게 모으고 다른 질환의 특징을 멀리 학습하면, 미세각질과 비듬처럼
> 시각적으로 비슷한 Hair 클래스를 기존 분류 학습보다 잘 구분할 수 있는가?

논문 근거: Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020.
https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html

이번 실행에는 DINOv2, EfficientNetV2, 384 해상도, SAM, ensemble을 포함하지 않습니다.
Test도 읽지 않으며 동일한 Validation에서 기준선과 SupCon만 비교합니다.
"""
        ),
        cell(
            """## 1. 무엇을 고정하고 무엇을 바꾸는가

고정 조건은 Hair Clean ZIP, Train·Validation 분할, 클래스 순서, augmented Train,
EfficientNet-B1, 입력 256, seed 42입니다.

기준선은 기존 방식인 head-only 15 epoch와 partial fine-tuning 15 epoch입니다.
SupCon은 15 epoch 동안 특징 공간을 학습한 뒤, 같은 Label Smoothing 분류기를 15 epoch
학습합니다. SupCon에서만 한 이미지의 약한 두 view를 만드는 것은 positive pair가 필요한
알고리즘의 필수 변경입니다.
"""
        ),
        cell(
            """DOMAIN = 'hair'
PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'
DATA_ZIP = '/content/drive/MyDrive/mediflow_datasets/hair_clean_v1_20260907_053616.zip'
AUDIT_DIR = ''
EXPECTED_DATA_SHA256 = '2ac7260663cf69835ba50edb6ae8c7e7ac13be9c73b9f7ea24f60e0342e7e156'
RESUME_DIR = ''
MODE = 'supcon_compare'
SEED = 42
SEEDS = [42]
BATCH_SIZE = 32
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 15
EXTENSION_EPOCHS = 1  # 사용하지 않음
TRAIN_VARIANT = 'augmented'
RUN_EXPERIMENTS = ['baseline_b1_256', 'supcon_b1_256']
""",
            True,
        ),
        cell(
            """## 2. Drive와 실행 환경

Colab 런타임을 GPU로 설정한 뒤 실행합니다. 설치 후 버전 오류가 나오면 세션을 다시 시작하고
처음부터 실행하세요.
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

이 셀은 수정하지 않습니다. 실행 코드와 생성 당시 commit은 결과 폴더에 함께 저장됩니다.
"""
        ),
        cell(embedded, True),
        cell(
            """## 4. 데이터 확인과 결과 폴더 생성

Drive의 Hair Clean ZIP을 복사하고 기존 SHA-256과 같은지 확인합니다. 평가에는 모든 조건에서
동일한 original Validation 사진을 사용합니다.
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
        raise ValueError('Hair 데이터 ZIP이 하나가 되도록 DATA_ZIP에 정확한 경로를 입력하세요.')
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
""",
            True,
        ),
        cell(
            """## 5. 기준선과 SupCon 학습

기준선 1회와 SupCon 1회만 실행합니다. 각 조건이 끝나면 모델과 기록을 Drive에 즉시 저장합니다.
Colab이 끊기면 위의 결과 폴더를 `RESUME_DIR`에 입력하고 같은 설정으로 다시 실행하세요.
"""
        ),
        cell("records = run_supcon_comparison(context)\n", True),
        cell(
            """## 6. Validation 비교 자료 저장

Accuracy, Macro F1, 클래스별 F1, 혼동행렬과 학습곡선을 저장합니다. 결과가 개선 가능성을
보일 때만 다음 단계에서 seed 43·44를 추가합니다. 이 결과만으로 최종 모델을 확정하지 않습니다.
"""
        ),
        cell(
            "summary, report_zip = summarize_supcon_comparison(context, records)\nsummary\n",
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
            "colab": {"name": "06_hair_supcon_comparison_colab.ipynb"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = ROOT / "notebooks/06_hair_supcon_comparison_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return output


if __name__ == "__main__":
    build()
