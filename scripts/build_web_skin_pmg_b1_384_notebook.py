"""Build the standalone Web Skin PMG B1/384 validation notebook."""

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


def build():
    source_names = (
        "common_engine",
        "common_workflow",
        "web_skin_wsdan",
        "web_skin_paper_suite",
        "web_skin_pmg_b1_384",
    )
    sources = {
        name: (ROOT / f"src/mediflow_datasets/{name}.py").read_text(encoding="utf-8")
        for name in source_names
    }
    classes = json.loads(
        (
            ROOT
            / "results/web_skin/candidates/"
            "public_candidate_v1_b0_256_ce_20260908_081603_3f1ce76e/"
            "class_names.json"
        ).read_text(encoding="utf-8")
    )
    profiles = {"web_skin": classes}
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
from mediflow_datasets.web_skin_pmg_b1_384 import run, summarize
"""
    cells = [
        cell(
            """# Web Skin 마지막 Validation 실험: PMG · EfficientNet-B1 · 384

현재 Validation 선두인 `PMG·B0·256·CE`를 다시 학습하지 않고 저장된 기준값으로 사용합니다.
새로 학습하는 모델은 **PMG·B1·384·CE 하나뿐**입니다. 동일한 데이터와 클래스 순서를
사용하며 최종 후보를 고르기 전이므로 Test는 열지 않습니다.

- PMG: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650154.pdf
- EfficientNet: https://proceedings.mlr.press/v97/tan19a.html
"""
        ),
        cell(
            """## 1. 이번 실험의 한 가지 가설

PMG의 `8×8 → 4×4 → 2×2 → 원본` 학습 방식, CE Loss, seed와 학습 횟수는 유지합니다.
Hair에서 가장 좋았던 B1·384 규모를 적용했을 때 얼굴 피부의 작은 차이를 더 잘 구분하는지
검증합니다.

기존 PMG는 batch 32였지만 B1·384의 GPU 메모리를 위해 이번 실행은 batch 16을 사용합니다.
이는 필요한 동반 변경으로 결과에 기록됩니다. 실제 사용 시에는 384×384 얼굴 사진 한 장을
입력하고, 세 branch와 fusion logit을 합산합니다.
"""
        ),
        cell(
            """DOMAIN = 'web_skin'
PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'
DATA_ZIP = '/content/drive/MyDrive/mediflow_Project/datasets/web_skin_datasets.zip'
AUDIT_DIR = ''
EXPECTED_DATA_SHA256 = 'f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d'
RESUME_DIR = ''
MODE = 'web_skin_pmg_b1_384'
SEED = 42
SEEDS = [42]
BATCH_SIZE = 16
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 10
EXTENSION_EPOCHS = 1  # 사용하지 않음
TRAIN_VARIANT = 'augmented'
RUN_EXPERIMENTS = ['pmg_b1_384_ce_seed_42']
PMG_JIGSAW_GRIDS = [8, 4, 2]
""",
            True,
        ),
        cell(
            """## 2. Drive·GPU·데이터 확인

Drive를 새로 연결하고 `web_skin_datasets.zip`이 실제 ZIP인지 확인합니다. 384 입력은 계산량이
크므로 Colab 런타임에서 GPU를 선택해야 합니다.
"""
        ),
        cell(
            """from google.colab import drive
from pathlib import Path
import zipfile

drive.mount('/content/drive', force_remount=True)

data_path = Path(DATA_ZIP)
dataset_dir = Path(PROJECT_ROOT) / 'datasets'
if not data_path.is_file():
    available = sorted(path.name for path in dataset_dir.iterdir()) if dataset_dir.is_dir() else []
    raise FileNotFoundError(
        f'Web Skin 데이터 파일을 찾을 수 없습니다: {data_path}\\n'
        f'datasets 폴더에서 확인된 항목: {available}'
    )
if not zipfile.is_zipfile(data_path):
    raise ValueError(f'ZIP 형식으로 열 수 없습니다: {data_path}')
print('데이터 파일 확인:', data_path)
print('데이터 파일 크기:', round(data_path.stat().st_size / 1024**3, 2), 'GiB')

%pip -q install tensorflow==2.20.0 keras==3.13.2 pandas matplotlib pillow tqdm
import tensorflow as tf
import keras
if tf.__version__ != '2.20.0' or keras.__version__ != '3.13.2':
    raise RuntimeError('Colab 세션을 다시 시작한 뒤 처음부터 실행하세요.')
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('런타임 유형에서 GPU를 선택하세요.')
print('GPU:', tf.config.list_physical_devices('GPU'))
""",
            True,
        ),
        cell(
            """## 3. 내장 재현 코드

별도 Python 파일은 필요 없습니다. 실행 코드와 해시는 결과 폴더에 자동 저장됩니다. 이 셀은
수정하지 않습니다.
"""
        ),
        cell(embedded, True),
        cell(
            """## 4. 데이터 준비와 실행 폴더 생성

데이터 SHA-256과 클래스 순서를 확인합니다. 중단 후 다시 실행할 때 출력된 결과 폴더를
`RESUME_DIR`에 입력합니다. 완료 모델은 검증 후 재사용하고, 완료되지 않은 attempt는 보존한
채 새 attempt에서 다시 시작합니다.
"""
        ),
        cell(
            """config = dict(
    domain=DOMAIN, project_root=PROJECT_ROOT, data_zip=DATA_ZIP,
    audit_dir=AUDIT_DIR, expected_data_sha256=EXPECTED_DATA_SHA256,
    resume_dir=RESUME_DIR, mode=MODE, seed=SEED, seeds=SEEDS,
    batch_size=BATCH_SIZE, epochs1=STAGE1_EPOCHS, epochs2=STAGE2_EPOCHS,
    extension_epochs=EXTENSION_EPOCHS, train_variant=TRAIN_VARIANT,
    experiments=RUN_EXPERIMENTS, pmg_jigsaw_grids=PMG_JIGSAW_GRIDS,
)
context = prepare(config, PROFILES, SOURCES, BUILD_COMMIT)
print('결과 폴더 / 중단 시 RESUME_DIR:', context['output'])
print('데이터 SHA-256:', context['data_hash'])
print('클래스 순서:', context['classes'])
""",
            True,
        ),
        cell(
            """## 5. PMG·B1·384 한 모델 학습

Stage 1은 분류부 중심으로 15 epoch, Stage 2는 EfficientNet-B1 후반부를 부분 미세조정하여
10 epoch 학습합니다. 모델은 Validation Accuracy가 가장 높았던 checkpoint를 저장합니다.
"""
        ),
        cell("record = run(context)\nrecord['validation']\n", True),
        cell(
            """## 6. 기존 PMG와 Validation 비교 자료 생성

기존 PMG·B0·256과 새 PMG·B1·384의 Accuracy, Macro F1, 클래스별 F1, 혼동행렬과 학습곡선을
저장합니다. Macro F1이 더 높은 모델을 최종 Test 후보로 표시하지만 Test 자체는 실행하지
않습니다. `.keras` 모델은 Drive 실행 폴더에 남고 결과 ZIP에서는 제외됩니다.
"""
        ),
        cell("summary, report_zip = summarize(context, record)\nsummary\n", True),
    ]
    for index, item in enumerate(cells):
        item["id"] = f"cell-{index:02d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": "12_web_skin_pmg_b1_384_colab.ipynb"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = ROOT / "notebooks/12_web_skin_pmg_b1_384_colab.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    build()
