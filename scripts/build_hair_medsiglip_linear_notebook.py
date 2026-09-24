"""Build the standalone Hair MedSigLIP-448 linear-probe notebook."""

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
    source_names = ("common_engine", "common_workflow", "hair_medsiglip_linear")
    sources = {
        name: (ROOT / f"src/mediflow_datasets/{name}.py").read_text(encoding="utf-8")
        for name in source_names
    }
    classes = json.loads(
        (
            ROOT
            / "results/hair/candidates/"
            "public_candidate_v2_b1_384_ls005_adam_20260921_155906_82311da4/"
            "class_names.json"
        ).read_text(encoding="utf-8")
    )
    profiles = {"hair": classes}
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
from mediflow_datasets.hair_medsiglip_linear import run, summarize
"""
    cells = [
        cell(
            """# Hair 논문 기반 실험: MedSigLIP-448 Frozen Linear Probe

현재 배포 후보 `EfficientNet-B1·384·Label Smoothing 0.05·Adam v2`는 그대로 보존합니다.
이번 노트북은 Google의 의료 이미지·텍스트 사전학습 모델 **MedSigLIP-448**이 Hair의 두피
5개 클래스를 더 잘 분리하는지 Validation에서만 확인합니다. Test는 후보가 정해진 뒤 별도
노트북에서 한 번만 평가합니다.

- 공식 모델 카드: https://huggingface.co/google/medsiglip-448
- 공식 시작 안내: https://developers.google.com/health-ai-developer-foundations/medsiglip/get-started
"""
        ),
        cell(
            """## 1. 한 가지 연구 가설과 고정 조건

**가설:** 일반 ImageNet 특징보다 의료 영상으로 사전학습된 MedSigLIP 특징이 두피의 홍반,
각질, 비듬, 탈모와 피지 차이를 더 잘 분리한다.

고정 조건은 동일 clean ZIP, 클래스 순서, Augmented Train 15,047장, Original Validation
1,252장, seed 42입니다. 변경 변수는 특징 추출기뿐입니다. MedSigLIP 본체는 학습하지 않고
고정한 뒤 분류용 Linear 층 하나만 50 epoch 학습합니다. 이 방식은 전체 미세조정보다 계산량과
과적합 위험이 작아 첫 선별 실험에 적합합니다. MedSigLIP의 피부과 사전학습이 두피 현미경
영상에도 유효한지는 알려져 있지 않으므로 이 Validation 결과로 따로 판단합니다.
"""
        ),
        cell(
            """DOMAIN = 'hair'
PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'
DATA_ZIP = '/content/drive/MyDrive/mediflow_Project/datasets/hair_datasets.zip'
AUDIT_DIR = ''
EXPECTED_DATA_SHA256 = '2ac7260663cf69835ba50edb6ae8c7e7ac13be9c73b9f7ea24f60e0342e7e156'
RESUME_DIR = ''
MODE = 'hair_medsiglip_linear'
SEED = 42
SEEDS = [42]
BATCH_SIZE = 8
LINEAR_EPOCHS = 50
LINEAR_BATCH_SIZE = 256
LINEAR_LEARNING_RATE = 1e-3
LINEAR_WEIGHT_DECAY = 1e-4
EMBEDDING_SHARD_SIZE = 128
MODEL_ID = 'google/medsiglip-448'
TRAIN_VARIANT = 'augmented'
RUN_EXPERIMENTS = ['hair_medsiglip_448_frozen_linear_seed_42']
""",
            True,
        ),
        cell(
            """## 2. Drive·GPU·패키지 확인

Colab 런타임은 **A100 GPU 권장**, L4도 사용할 수 있습니다. 128장 단위 embedding 캐시는
Drive 결과 폴더에 저장되므로 연결이 끊기면 같은 폴더를 `RESUME_DIR`에 넣어 이어갑니다.
모델 이용약관 동의와 Colab Secrets의 `HF_TOKEN` 읽기 권한이 필요합니다.
"""
        ),
        cell(
            """from google.colab import drive, userdata
from pathlib import Path
import zipfile

drive.mount('/content/drive', force_remount=True)
data_path = Path(DATA_ZIP)
if not data_path.is_file():
    available = sorted(path.name for path in (Path(PROJECT_ROOT) / 'datasets').iterdir())
    raise FileNotFoundError(f'데이터 ZIP을 찾을 수 없습니다: {data_path}\\n확인된 항목: {available}')
if not zipfile.is_zipfile(data_path):
    raise ValueError(f'ZIP 형식이 아닙니다: {data_path}')

HF_TOKEN = userdata.get('HF_TOKEN')
if not HF_TOKEN or not HF_TOKEN.startswith('hf_'):
    raise ValueError('Colab 왼쪽 열쇠(Secrets)에 HF_TOKEN을 저장하고 노트북 접근을 켜세요.')

%pip -q install tensorflow==2.20.0 keras==3.13.2 pandas matplotlib pillow tqdm \
  "transformers==4.53.2" "huggingface_hub>=0.33,<1" "accelerate>=1.8,<2" safetensors

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # 이 노트북의 GPU는 PyTorch MedSigLIP 전용

import torch
if not torch.cuda.is_available():
    raise RuntimeError('런타임 유형에서 GPU를 선택하세요.')
print('데이터:', data_path)
print('GPU:', torch.cuda.get_device_name(0))
""",
            True,
        ),
        cell(
            """## 3. 내장 재현 코드

별도 저장소 연결은 필요 없습니다. 실행 코드와 생성 당시 commit, 설정, 환경 버전이 결과 폴더에
저장됩니다. 이 셀은 수정하지 않습니다.
"""
        ),
        cell(embedded, True),
        cell(
            """## 4. 데이터 준비와 실행 폴더 생성

ZIP 해시와 클래스 순서를 확인하고 로컬 런타임에 압축을 풉니다. 처음 실행한 뒤 출력되는 결과
폴더를 메모해 두세요. 중단 시 그 전체 경로를 위 `RESUME_DIR`에 입력합니다.
"""
        ),
        cell(
            """config = dict(
    domain=DOMAIN, project_root=PROJECT_ROOT, data_zip=DATA_ZIP,
    audit_dir=AUDIT_DIR, expected_data_sha256=EXPECTED_DATA_SHA256,
    resume_dir=RESUME_DIR, mode=MODE, seed=SEED, seeds=SEEDS,
    batch_size=BATCH_SIZE, epochs1=LINEAR_EPOCHS, epochs2=1,
    extension_epochs=1, train_variant=TRAIN_VARIANT,
    experiments=RUN_EXPERIMENTS, model_id=MODEL_ID,
    embedding_shard_size=EMBEDDING_SHARD_SIZE,
    linear_batch_size=LINEAR_BATCH_SIZE,
    linear_learning_rate=LINEAR_LEARNING_RATE,
    linear_weight_decay=LINEAR_WEIGHT_DECAY,
)
context = prepare(config, PROFILES, SOURCES, BUILD_COMMIT)
print('결과 폴더 / 중단 시 RESUME_DIR:', context['output'])
print('데이터 SHA-256:', context['data_hash'])
print('클래스 순서:', context['classes'])
""",
            True,
        ),
        cell(
            """## 5. MedSigLIP 특징 추출과 Linear 분류기 학습

처음에는 MedSigLIP 모델을 내려받고 16,299장의 특징을 추출하므로 Web Skin보다 오래 걸립니다.
`train embedding: 128 / 15047` 같은 출력은 완료된 사진 수입니다. 캐시가 있으면 다음 실행에서
`embedding cache`로 표시됩니다. 토큰은 모델 다운로드에만 사용하며 결과에 저장하지 않습니다.
"""
        ),
        cell("record = run(context, HF_TOKEN)\nrecord['validation']\n", True),
        cell(
            """## 6. 기존 v2와 Validation 비교 자료 생성

학습곡선, 혼동행렬, 기존 Hair B1·384 v2와의 Accuracy·Macro F1 비교 그림과 보고서 ZIP을 만듭니다.
선정 규칙은 Validation Macro F1 우선, 정확히 같으면 Accuracy입니다. 이 실행만으로 현재 v2를
교체하거나 Test를 평가하지 않습니다. embedding 캐시는 재개용이므로 보고서 ZIP에서 제외됩니다.
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
            "colab": {"name": "15_hair_medsiglip_linear_probe_colab.ipynb"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = ROOT / "notebooks/15_hair_medsiglip_linear_probe_colab.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    build()
