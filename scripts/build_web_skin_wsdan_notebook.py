"""Build the standalone three-method Web Skin paper experiment notebook."""

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
from mediflow_datasets.web_skin_paper_suite import run, summarize
"""
    cells = [
        cell(
            """# Web Skin 논문 기반 3개 방법 통합 실험

기존 `B0·256·CE`는 다시 학습하지 않고 저장된 Validation 기준값을 사용합니다. 동일한 Web
Skin 분할과 seed 42에서 **WS-DAN, PMG, MixStyle**을 각각 독립적으로 학습합니다. 세 방법은
서로 결합하지 않으며 Test는 후보 선정 전까지 사용하지 않습니다.

- WS-DAN: https://arxiv.org/abs/1901.09891
- PMG: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650154.pdf
- MixStyle: https://openreview.net/forum?id=6xHJ37MVxxp
"""
        ),
        cell(
            """## 1. 비교 설계

공통 조건은 augmented Train, original Validation 500장, seed 42와 기존 클래스 순서입니다.
세 방법 모두 EfficientNet-B0와 256 입력을 사용하여 같은 조건에서 비교합니다.

PMG는 학습 때만 8×8, 4×4, 2×2 jigsaw를 사용하고 실제 추론에는 원본 한 장을 사용합니다.
MixStyle도 학습 중에만 작동합니다. WS-DAN은 추론에서 원본과 Attention crop 예측을 평균합니다.
"""
        ),
        cell(
            """DOMAIN = 'web_skin'
PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'
DATA_ZIP = '/content/drive/MyDrive/mediflow_Project/datasets/web_skin_datasets.zip'
AUDIT_DIR = ''
EXPECTED_DATA_SHA256 = 'f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d'
RESUME_DIR = ''
MODE = 'web_skin_paper_suite'
SEED = 42
SEEDS = [42]
BATCH_SIZE = 32
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 10
EXTENSION_EPOCHS = 1  # 사용하지 않음
TRAIN_VARIANT = 'augmented'
RUN_EXPERIMENTS = [
    'wsdan_b0_256_ce_seed_42',
    'pmg_b0_256_ce_seed_42',
    'mixstyle_b0_256_ce_seed_42',
]
ATTENTION_MAPS = 8
CROP_THRESHOLD = 0.5
DROP_THRESHOLD = 0.7
PMG_JIGSAW_GRIDS = [8, 4, 2]
MIXSTYLE_ALPHA = 0.1
MIXSTYLE_PROBABILITY = 0.5
""",
            True,
        ),
        cell(
            """## 2. Drive·GPU 확인

현재 Colab 마운트 기준 데이터 위치는
`내 드라이브/mediflow_Project/datasets/web_skin_datasets.zip`입니다. Google Drive 화면에서
확장자가 생략되어 보일 수 있지만 Colab 경로에는 `.zip`을 포함해야 합니다. 아래 셀은 Drive를
새로 연결하고 이 파일을 실제 ZIP으로 열 수 있는지 먼저 확인합니다.
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
    raise ValueError(
        f'파일은 존재하지만 ZIP 형식으로 열 수 없습니다: {data_path}\\n'
        'Drive 연결을 새로 고친 뒤에도 같으면 파일 업로드 상태를 확인하세요.'
    )
print('데이터 파일 확인:', data_path)
print('데이터 파일 크기:', round(data_path.stat().st_size / 1024**3, 2), 'GiB')

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
            """## 3. 내장 재현 코드

별도 Python 파일은 필요 없습니다. 정확한 실행 소스와 해시가 Drive 결과 폴더에 보존됩니다.
이 셀은 수정하지 않습니다.
"""
        ),
        cell(embedded, True),
        cell(
            """## 4. 데이터 준비

확인된 Web Skin ZIP의 SHA-256, 폴더 구조와 클래스 순서를 검사합니다. 다른 ZIP이면 학습을
시작하지 않습니다. 중단 후 이어갈 때 출력된 경로를 `RESUME_DIR`에 그대로 입력하면 완료된
실험은 해시를 확인한 뒤 건너뜁니다.
"""
        ),
        cell(
            """config = dict(
    domain=DOMAIN, project_root=PROJECT_ROOT, data_zip=DATA_ZIP,
    audit_dir=AUDIT_DIR, expected_data_sha256=EXPECTED_DATA_SHA256,
    resume_dir=RESUME_DIR, mode=MODE, seed=SEED, seeds=SEEDS,
    batch_size=BATCH_SIZE, epochs1=STAGE1_EPOCHS, epochs2=STAGE2_EPOCHS,
    extension_epochs=EXTENSION_EPOCHS, train_variant=TRAIN_VARIANT,
    experiments=RUN_EXPERIMENTS, attention_maps=ATTENTION_MAPS,
    crop_threshold=CROP_THRESHOLD, drop_threshold=DROP_THRESHOLD,
    pmg_jigsaw_grids=PMG_JIGSAW_GRIDS, mixstyle_alpha=MIXSTYLE_ALPHA,
    mixstyle_probability=MIXSTYLE_PROBABILITY,
)
context = prepare(config, PROFILES, SOURCES, BUILD_COMMIT)
print('결과 폴더 / 중단 시 RESUME_DIR:', context['output'])
print('데이터 SHA-256:', context['data_hash'])
print('클래스 순서:', context['classes'])
""",
            True,
        ),
        cell(
            """## 5. 세 실험 순차 실행

WS-DAN → PMG → MixStyle 순서입니다. 각 방법은 별도 모델과 폴더를
사용합니다. 한 방법이 끝날 때마다 완료 기록을 저장하므로 Colab이 끊겨도 앞선 결과는 남습니다.
"""
        ),
        cell("records = run(context)\n", True),
        cell(
            """## 6. 동일 형식 비교 자료 생성

기준선과 세 방법의 Validation Accuracy·Macro F1, 클래스별 F1, 혼동행렬, 학습곡선을 한꺼번에
저장합니다. 선택 전이므로 Test는 평가하지 않습니다. 모델은 Drive에 남고 다운로드용 결과 ZIP은
용량을 줄이기 위해 `.keras`를 포함하지 않습니다.
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
            "colab": {"name": "11_web_skin_three_paper_methods_colab.ipynb"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = ROOT / "notebooks/11_web_skin_wsdan_attention_colab.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    build()
