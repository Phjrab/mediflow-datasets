"""Build the Web Skin PMG B0/256 final Test and packaging Colab notebook."""

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


def web_skin_classes():
    path = ROOT / (
        "results/web_skin/candidates/"
        "public_candidate_v1_b0_256_ce_20260908_081603_3f1ce76e/class_names.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def build():
    source_names = ("common_engine", "common_workflow", "web_skin_pmg_final")
    sources = {
        name: (ROOT / f"src/mediflow_datasets/{name}.py").read_text(encoding="utf-8")
        for name in source_names
    }
    profiles = {"web_skin": web_skin_classes()}
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
from mediflow_datasets.web_skin_pmg_final import finalize
"""
    cells = [
        cell(
            """# Web Skin 최종 후보 — PMG·B0·256 Test 1회 평가와 패키징

Validation 비교와 실행 비용을 함께 고려해 고정한 PMG·EfficientNet-B0·256 후보를 다시
학습하지 않고 Test에서 최종 평가합니다. 선택 모델의 SHA-256을 먼저 확인하며 Test 결과로
설정을 변경하지 않습니다. 평가 후 모델, 클래스 순서, 전처리 계약과 결과를 ZIP으로 묶습니다.
"""
        ),
        cell(
            """## 1. 고정된 후보

- 방법: PMG(Progressive Multi-Granularity) adaptation
- Backbone: ImageNet EfficientNet-B0
- 입력: 256×256 RGB float32, 픽셀 0–255
- Loss / Optimizer: Cross Entropy / Adam
- 학습: Stage 1 15 epoch + Stage 2 10 epoch — 이미 완료
- 선택 체크포인트: `pmg_b0_256_ce_seed_42/.../stage2_best.keras`
- 고정 Validation: Accuracy 0.85, Macro F1 0.8473279632397033
- Test 평가: 이 노트북에서 처음 1회

PMG 모델은 네 branch의 logit을 출력합니다. 후보 ZIP의 `inference.py`가 네 출력을 합산한 뒤
softmax를 적용합니다. 중단 후에는 처음 출력된 결과 폴더를 `RESUME_DIR`에 넣어야 저장된
Test 결과를 재사용합니다.
"""
        ),
        cell(
            """DOMAIN = 'web_skin'
PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'
DATA_ZIP = '/content/drive/MyDrive/mediflow_Project/datasets/web_skin_datasets.zip'
AUDIT_DIR = ''
EXPECTED_DATA_SHA256 = 'f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d'
RESUME_DIR = ''
MODE = 'web_skin_pmg_final'
SEED = 42
SEEDS = [42]
BATCH_SIZE = 32
STAGE1_EPOCHS = 15  # 학습하지 않음
STAGE2_EPOCHS = 10  # 학습하지 않음
EXTENSION_EPOCHS = 1  # 사용하지 않음
TRAIN_VARIANT = 'augmented'
PARENT_RUN_DIR = (
    '/content/drive/MyDrive/mediflow_Project/2_results/web_skin/'
    'web_skin_paper_suite_20260922_124758_717b465e'
)
PARENT_MODEL_SHA256 = '83e659dd09a9355135ee0de0197037e81ece962777f59dedae8d9a37b49a3fc4'
""",
            True,
        ),
        cell(
            """## 2. Drive와 실행 환경

`PARENT_RUN_DIR`의 PMG B0·256 모델과 데이터 ZIP이 남아 있어야 합니다. 이 노트북은
학습하지 않으며 Test 추론과 패키징만 수행합니다. GPU를 선택하면 평가가 더 빠릅니다.
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
""",
            True,
        ),
        cell(
            """## 3. 재현 코드

수정하지 않습니다. 실행 코드는 결과와 최종 후보 패키지에 함께 저장됩니다.
"""
        ),
        cell(embedded, True),
        cell(
            """## 4. 데이터와 선택 모델 확인

데이터, 클래스 순서, 부모 실행 기록과 `stage2_best.keras` 해시가 모두 일치해야 결과 폴더가
준비됩니다. 설정이 다르면 Test를 읽기 전에 중단합니다.
"""
        ),
        cell(
            """config = dict(
    domain=DOMAIN, project_root=PROJECT_ROOT, data_zip=DATA_ZIP,
    audit_dir=AUDIT_DIR, expected_data_sha256=EXPECTED_DATA_SHA256,
    resume_dir=RESUME_DIR, mode=MODE, seed=SEED, seeds=SEEDS,
    batch_size=BATCH_SIZE, epochs1=STAGE1_EPOCHS, epochs2=STAGE2_EPOCHS,
    extension_epochs=EXTENSION_EPOCHS, train_variant=TRAIN_VARIANT,
    parent_run_dir=PARENT_RUN_DIR, parent_model_sha256=PARENT_MODEL_SHA256,
)
context = prepare(config, PROFILES, SOURCES, BUILD_COMMIT)
print('결과 폴더 / 중단 시 RESUME_DIR:', context['output'])
print('데이터 SHA-256:', context['data_hash'])
print('선택 모델 SHA-256:', PARENT_MODEL_SHA256)
""",
            True,
        ),
        cell(
            """## 5. 최종 Test와 패키징

이 셀 하나가 선택 기록 고정, Test 평가, 혼동행렬·오류 이미지 생성, 모델 계약 검사와 후보
ZIP 생성을 순서대로 수행합니다. 완료되면 `selected_models`의 모델 ZIP과 현재 결과 폴더의
보고서 ZIP 경로를 출력합니다. 같은 결과 폴더로 재개하면 저장된 Test 결과를 검증해 재사용합니다.
"""
        ),
        cell(
            "metrics, package, model_zip, report_zip = finalize(context)\nmetrics\n",
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
            "colab": {
                "name": "13_web_skin_pmg_b0_256_final_test_package_colab.ipynb"
            },
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = ROOT / "notebooks/13_web_skin_pmg_b0_256_final_test_package_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return output


if __name__ == "__main__":
    build()
