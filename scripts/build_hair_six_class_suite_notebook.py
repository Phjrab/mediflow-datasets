"""Generate a self-contained Colab notebook for the Hair six-class four-way suite."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "17_hair_six_class_four_experiments_colab.ipynb"
SOURCE_NAMES = ("common_engine", "hair_six_class_suite")


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


sources = {
    name: (ROOT / "src" / "mediflow_datasets" / (name + ".py")).read_text(encoding="utf-8")
    for name in SOURCE_NAMES
}
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
embedded = f"""import sys, types, hashlib
SOURCES = {sources!r}
BUILD_COMMIT = {commit!r}
SOURCE_HASHES = {{
    name: hashlib.sha256(code.encode()).hexdigest() for name, code in SOURCES.items()
}}
package = types.ModuleType('mediflow_datasets')
package.__path__ = []
sys.modules['mediflow_datasets'] = package
for name, code in SOURCES.items():
    module = types.ModuleType('mediflow_datasets.' + name)
    sys.modules[module.__name__] = module
    exec(compile(code, name + '.py', 'exec'), module.__dict__)
from mediflow_datasets import hair_six_class_suite as suite
print('내장 코드 해시:', SOURCE_HASHES)
"""

intro = """# Hair 6클래스: 원본·증강 × B0·256·B1·384 — 4개 실험

이 노트북은 기존 Hair 5클래스 ZIP과 새 `양호` ZIP을 **읽기 전용**으로 사용해
별도 Hair 6클래스 데이터 ZIP을 만듭니다. 기존 사진은 정제된 Train/Validation/Test
분할을 유지합니다. 새 `양호` 사진은 파일명 첫 숫자 ID로 사람을 묶어 약 80:10:10으로
나눕니다. 기존 Hair 파일명은 사람 ID가 없어 사람·세션 간 누수는 확인되지 않았습니다.

Train 증강은 기존 질환 클래스에서 각 클래스 원본의 약 50%를 추가 생성하고,
적은 `양호` Train에는 원본당 5개의 약한 변형을 추가합니다. 밝기·대비·색감·작은 회전·
좌우 반전·아주 약한 블러만 사용하며 Validation/Test에는 증강을 적용하지 않습니다.
증강 수량은 원본 사람 수가 증가했다는 의미가 아닙니다.

| 실험 | 모델 | 입력 | Loss | Train |
|---|---|---:|---|---|
| 1 | EfficientNet-B0 | 256 | CE | 원본 |
| 2 | EfficientNet-B0 | 256 | CE | 원본 + 증강 |
| 3 | EfficientNet-B1 | 384 | Label Smoothing 0.05 | 원본 |
| 4 | EfficientNet-B1 | 384 | Label Smoothing 0.05 | 원본 + 증강 |

각 실험은 ImageNet 가중치에서 새 6클래스 출력층으로 시작하며, head-only 15 epoch와
후반부 30개 층 미세조정 15 epoch를 실시합니다. **같은 모델의 원본·증강 두 실험은
Train 이미지와 그 수량에 따른 클래스 가중치만 다릅니다.** B0와 B1은
구조·해상도·Loss가 함께 달라 실용적인
모델 구성 비교이지 단일 변수의 인과 비교가 아닙니다. 네 실험을 같은 Validation의
Macro F1로 비교하고, 승자 하나만 고정 Test에서 평가합니다.

입력 ZIP이나 저장 모델은 덮어쓰지 않습니다. 데이터 ZIP은 Drive의 `datasets`, 전체
모델·로그·그래프는 `2_results/hair`에 저장합니다. 모델 없는 작은 보고서 ZIP도
같은 결과 폴더 옆에 만듭니다. 런타임이 끊기면 아래 `RESUME_DIR`에 출력된 결과 폴더를
넣고 처음부터 실행하면 완료 실험을 검증해 건너뜁니다.
"""

config = """from pathlib import Path

PROJECT_ROOT = Path('/content/drive/MyDrive/mediflow_Project')
RESUME_DIR = ''  # 중단 뒤에는 출력된 /content/drive/.../2_results/hair/... 폴더를 입력
SEED = 42
BATCH_SIZE = 32
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 15

# 원본 데이터 SHA-256: 코드에 검증된 값으로 고정되어 있습니다.
# 새 ZIP은 양호 사진의 6개 증상값이 모두 0이라는 사용자 확인을 라벨 근거로 기록합니다.
"""

setup = """from google.colab import drive
drive.mount('/content/drive')
%pip -q install tensorflow==2.20.0 keras==3.13.2 pillow matplotlib

import hashlib
import json
import platform
import shutil
import uuid
import zipfile
from datetime import datetime, timezone

import keras
import tensorflow as tf

if tf.__version__ != '2.20.0' or keras.__version__ != '3.13.2':
    raise RuntimeError('설치 버전이 다릅니다. 런타임을 다시 시작하고 처음부터 실행하세요.')
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('Colab 런타임에서 GPU를 선택하세요.')
dataset_dir = PROJECT_ROOT / 'datasets'
candidates = [dataset_dir / 'hair_datasets.zip', dataset_dir / 'hair_datasets']
valid = [p for p in candidates if p.is_file() and zipfile.is_zipfile(p)]
if len(valid) != 1:
    available = sorted(p.name for p in dataset_dir.iterdir()) if dataset_dir.is_dir() else []
    raise FileNotFoundError(f'기존 Hair ZIP 한 개를 확인하세요: {available}')
OLD_ZIP = valid[0]
GOOD_ZIP = dataset_dir / 'hair_good_raw_v1.zip'
if not GOOD_ZIP.is_file() or not zipfile.is_zipfile(GOOD_ZIP):
    raise FileNotFoundError(GOOD_ZIP)
print('기존 ZIP:', OLD_ZIP)
print('양호 ZIP:', GOOD_ZIP)
print('GPU:', tf.config.list_physical_devices('GPU'))
"""

prepare = """from mediflow_datasets import hair_six_class_suite as suite

source_hashes = {
    'old': suite.OLD_SHA256,
    'good': suite.GOOD_SHA256,
    'builder_code': SOURCE_HASHES['hair_six_class_suite'],
    'seed': SEED,
}
name_digest = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()[:12]
DATASET_ZIP = dataset_dir / ('hair_clean_v2_6class_' + name_digest + '.zip')
SHA_FILE = DATASET_ZIP.with_name(DATASET_ZIP.name + '.sha256')
free = shutil.disk_usage('/content').free
if free < 16 * 1024**3:
    raise RuntimeError('Colab 작업 공간 16GiB 이상이 필요합니다. 새 런타임을 사용하세요.')
work = Path('/content') / ('hair_six_class_' + uuid.uuid4().hex[:8])
work.mkdir()
local_dataset = work / 'dataset'
if DATASET_ZIP.exists():
    if not SHA_FILE.is_file():
        raise ValueError(
            '기존 v2 ZIP의 SHA-256 확인 파일이 없습니다. 파일을 수정하지 말고 점검하세요.'
        )
    expected = SHA_FILE.read_text().strip().split()[0]
    local_zip = work / DATASET_ZIP.name
    shutil.copyfile(DATASET_ZIP, local_zip)
    if suite.file_sha256(local_zip) != expected:
        raise ValueError('기존 v2 ZIP SHA-256이 다릅니다.')
    with zipfile.ZipFile(local_zip) as archive:
        for info in archive.infolist():
            path = Path(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\\\' in info.filename:
                raise ValueError('안전하지 않은 v2 ZIP 항목: ' + info.filename)
        archive.extractall(work)
    data_hash = expected
else:
    local_old = work / 'hair_datasets.zip'
    local_good = work / 'hair_good_raw_v1.zip'
    shutil.copyfile(OLD_ZIP, local_old)
    shutil.copyfile(GOOD_ZIP, local_good)
    contract = suite.build_dataset(local_old, local_good, local_dataset, seed=SEED)
    local_zip = work / DATASET_ZIP.name
    data_hash = suite.archive_dataset(local_dataset, local_zip)
    shutil.copyfile(local_zip, DATASET_ZIP)
    if suite.file_sha256(DATASET_ZIP) != data_hash:
        raise IOError('Drive에 저장한 v2 ZIP 해시가 다릅니다.')
    SHA_FILE.write_text(data_hash + '  ' + DATASET_ZIP.name + '\\n')
print('새 6클래스 데이터 ZIP:', DATASET_ZIP)
print('새 데이터 SHA-256:', data_hash)
print(json.dumps(json.loads((local_dataset / 'dataset_contract.json').read_text()),
                 ensure_ascii=False, indent=2))
"""

train = """from mediflow_datasets import hair_six_class_suite as suite

parent = PROJECT_ROOT / '2_results' / 'hair'
parent.mkdir(parents=True, exist_ok=True)
if RESUME_DIR:
    OUTPUT_DIR = Path(RESUME_DIR)
    if not OUTPUT_DIR.is_dir() or OUTPUT_DIR.parent != parent:
        raise ValueError('RESUME_DIR은 이 프로젝트의 Hair 결과 폴더여야 합니다.')
else:
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    OUTPUT_DIR = parent / ('hair_six_class_four_' + run_id)
    OUTPUT_DIR.mkdir(exist_ok=False)
for name, code in SOURCES.items():
    target = OUTPUT_DIR / (name + '.py')
    if target.exists() and target.read_text(encoding='utf-8') != code:
        raise ValueError('저장된 코드와 현재 노트북 코드가 다릅니다: ' + name)
    if not target.exists():
        target.write_text(code, encoding='utf-8')
print('결과 폴더 / 중단 시 RESUME_DIR:', OUTPUT_DIR)
result = suite.run_four(
    local_dataset, OUTPUT_DIR, data_sha256=data_hash,
    batch_size=BATCH_SIZE, stage1_epochs=STAGE1_EPOCHS,
    stage2_epochs=STAGE2_EPOCHS, seed=SEED,
    code_hashes=SOURCE_HASHES, code_commit=BUILD_COMMIT,
)
print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
print('모델 없는 보고서 ZIP:', result['report_zip'])
print('선정 모델 파일:', result['selected_model'])
"""

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"name": OUTPUT.name},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": [
        cell("markdown", intro),
        cell("code", config),
        cell("code", setup),
        cell("markdown", "## 재현 코드\n\n필요한 코드를 노트북 안에 포함했습니다.\n"),
        cell("code", embedded),
        cell("markdown", "## 6클래스 데이터 구성\n\n기존 ZIP은 유지하고 새 ZIP을 생성합니다.\n"),
        cell("code", prepare),
        cell("markdown", "## 4개 실험·그래프·선정 모델 Test\n\n끊기면 RESUME_DIR을 설정하세요.\n"),
        cell("code", train),
    ],
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
