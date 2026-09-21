"""Build standalone Colab notebooks from versioned runtime source."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_sources():
    """One-time migration; never alter historical source notebooks."""
    target = ROOT / "src/mediflow_datasets/common_engine.py"
    if not target.exists():
        source = (ROOT / "src/mediflow_datasets/experiment_suite.py").read_text(encoding="utf-8")
        source = source.replace(
            "Sequential Web Skin experiments", "Sequential domain-configured experiments"
        )
        source = source.replace("web_skin_sweep_v1", "mediflow_common_v1")
        source = source.replace('CLASSES = ["건선", "아토피", "여드름", "정상", "주사"]\n', "")
        source = source.replace("range(5)", "range(probabilities.shape[1])")
        source = source.replace("Dense(5,", 'Dense(spec["class_count"],')
        source = source.replace(
            "classification_metrics(truth, probabilities)\n",
            "classification_metrics(truth, probabilities, probabilities.shape[1])\n",
        )
        old = """    trainable = configure_partial(model, spec["loss"])
    h2, best2 = fit_stage(model, train, val, directory, "stage2", epochs2)"""
        new = """    trainable = []
    h2 = {key: [] for key in h1}
    best2 = best1
    if epochs2:
        trainable = configure_partial(model, spec["loss"])
        h2, best2 = fit_stage(model, train, val, directory, "stage2", epochs2)"""
        source = source.replace(old, new).replace(
            'max(h2["val_accuracy"])', 'max(h2["val_accuracy"], default=-1.0)'
        )
        target.write_text(source, encoding="utf-8")

    target = ROOT / "src/mediflow_datasets/common_audit.py"
    if not target.exists():
        nb = json.loads(
            (ROOT / "notebooks/web_skin_dataset_audit_colab.ipynb").read_text(encoding="utf-8")
        )
        body = "\n".join("".join(nb["cells"][i]["source"]) for i in (8, 10, 12))
        body = body.replace("'domain': 'web_skin'", "'domain': domain")
        body = body.replace("'Web Skin ' + kind", "domain + ' ' + kind")
        body = body.replace("실제 웹캠 데이터 검증은 별도", "실제 대상 장비 데이터 검증은 별도")
        body = body.replace(
            "    'next_step':", "    'protocol': 'common_audit_v1',\n    'next_step':"
        )
        body += "\nreturn summary\n"
        header = '''"""Original Web Skin mechanical audit generalized to three class contracts.

Byte/pixel hashes do not establish person, lesion, session or augmentation lineage.
"""
import hashlib
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from PIL import Image
from tqdm.auto import tqdm

from mediflow_datasets.common_engine import file_hash, write_json


def audit_dataset(extract_root, report_dir, domain, classes, data_hash):
    EXTRACT_ROOT, REPORT_DIR = Path(extract_root), Path(report_dir)
    CLASS_NAMES = classes
    CLASS_CODES = {name: f'C{i}' for i, name in enumerate(classes)}
    DATA_SHA256 = data_hash
    SPLITS = ('train', 'val', 'test')
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tif', '.tiff'}
    TRAIN_LOADER_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    sha256 = file_hash

    def save_json(name, value):
        write_json(REPORT_DIR / name, value)

    def save_csv(name, rows, columns):
        pd.DataFrame(rows, columns=columns).to_csv(
            REPORT_DIR / name, index=False, encoding='utf-8-sig')

'''
        target.write_text(
            header + "\n".join("    " + line if line else "" for line in body.splitlines()) + "\n",
            encoding="utf-8",
        )


def cell(text, code=False):
    result = {
        "cell_type": "code" if code else "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }
    if code:
        result.update(execution_count=None, outputs=[])
    return result


def build():
    profiles = {}
    for domain in ("hair", "web_skin", "skin"):
        base = ROOT / "results" / domain / "1_training" / "original"
        name = "class_names.json" if domain == "hair" else "results.json"
        data = json.loads((base / name).read_text(encoding="utf-8"))
        profiles[domain] = (
            [data[str(i)] for i in range(len(data))] if domain == "hair" else data["classes"]
        )
    sources = {
        name: (ROOT / f"src/mediflow_datasets/{name}.py").read_text(encoding="utf-8")
        for name in ("common_engine", "common_audit", "common_workflow")
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
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
from mediflow_datasets.common_workflow import prepare, run, finish, audit_run
"""
    for mode, filename, title in (
        ("audit", "01_common_dataset_audit_colab.ipynb", "데이터셋 검증"),
        ("comparison", "02_common_original_vs_augmented_colab.ipynb", "원본·증강본 비교"),
        ("suite", "03_common_six_experiments_colab.ipynb", "6개 실험과 선정 모델 패키징"),
    ):
        cells = [
            cell(f"""# MediFlow 공통 {title}

대상만 선택하여 Hair(두피 현미경), Web Skin(얼굴 웹캠), Skin(피부 현미경)에 사용합니다.
각 대상은 별도 모델이며 클래스 순서는 기존 결과 JSON에서 가져왔습니다.
이 노트북 하나에 실행 코드가 포함되어 있어 별도 Python 파일 업로드가 필요 없습니다.

**사용 순서:** 설정 수정 → Colab GPU 선택(학습 시) → 위에서부터 모두 실행.
검증은 CPU로 가능합니다. 최초 설치 후 버전 확인에서 멈추면 세션을 다시 시작하고 처음부터 실행하세요.
이 코드는 새 실험용입니다. 과거 성능을 재현했다고 간주하지 않습니다.
기존 결과는 그대로 두고 실행마다 새 폴더에 기록합니다.
"""),
            cell("""## 1. 사용자 설정 — 보통 여기만 수정합니다

`DOMAIN`: `hair`, `web_skin`, `skin` 중 하나입니다.
`PROJECT_ROOT`: Drive 프로젝트 폴더입니다. 이름을 바꿨다면 이 경로를 수정하세요.
`DATA_ZIP`: 비우면 datasets 아래에서 대상 이름으로 시작하는 ZIP을 찾습니다.
확장자가 없는 ZIP도 찾으며, 여러 개면 목록을 보여주고 멈춥니다. 그때 전체 경로를 입력하세요.
`AUDIT_DIR`: ① 검증이 완료된 결과 폴더입니다.
`EXPECTED_DATA_SHA256`: 별도 검증을 생략한 경우에만 확인된 ZIP 해시를 입력합니다.
학습 시 `AUDIT_DIR`과 `EXPECTED_DATA_SHA256` 중 하나가 필요합니다.
`RESUME_DIR`: 중단된 이번 공통 노트북 결과 폴더. 비우면 새 실행입니다.
완료 실험은 코드·설정·파일 일치 확인 후 재사용하며, 중단된 실험은 처음부터 다시 합니다.
과거 개별 노트북의 결과 폴더는 재개 대상으로 사용할 수 없습니다.
"""),
            cell(
                f"""DOMAIN = 'web_skin'
PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'
DATA_ZIP = ''
AUDIT_DIR = ''
EXPECTED_DATA_SHA256 = ''
RESUME_DIR = ''
MODE = {mode!r}
SEED = 42
BATCH_SIZE = 32
# 두 비교 실험은 모두 15회 head-only 학습. suite는 15회 + 미세조정 10회.
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 10
EXTENSION_EPOCHS = 5
# suite의 학습 데이터 종류. 원본·증강 비교에서는 자동으로 두 종류를 사용합니다.
TRAIN_VARIANT = 'augmented'
""",
                True,
            ),
            cell("""## 2. Drive 연결과 실행 환경

학습에는 GPU가 필요합니다. 메모리 부족 시 배치 크기를 16으로 낮추되 새 실험으로 시작하세요.
배치 크기 변경도 결과에 영향을 줄 수 있어 기록됩니다.
"""),
            cell(
                """from google.colab import drive
drive.mount('/content/drive')
%pip -q install tensorflow==2.20.0 keras==3.13.2 pandas matplotlib pillow tqdm
import tensorflow as tf
import keras
if tf.__version__ != '2.20.0' or keras.__version__ != '3.13.2':
    raise RuntimeError('Colab 세션을 다시 시작한 뒤 처음부터 실행하세요.')
if MODE != 'audit' and not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('런타임 유형에서 GPU를 선택하세요.')
""",
                True,
            ),
            cell("""## 3. 공통 실행 코드 — 직접 수정할 필요 없습니다

결과에 이 소스와 생성 당시 Git commit을 함께 보관합니다. 미커밋 변경을 포함한 실제
실행 코드는 소스 사본과 SHA-256(파일 내용 식별값)으로 남깁니다.
"""),
            cell(embedded, True),
            cell("""## 4. 데이터 준비와 실험 기록

Drive ZIP을 Colab 임시 디스크로 복사하고 해시와 안전한 압축 경로, 여유 공간을 확인합니다.
ZIP 내부는 `original/train/클래스`, `original/val/클래스`, `original/test/클래스`와
동일한 `augmented/...` 구조여야 합니다. Original/Augmented의 대소문자는 허용합니다.
예상 클래스와 다르면 자동으로 이름을 추측하지 않고 중단합니다.

학습 노트북은 공통 ① 보고서 또는 사용자가 입력한 SHA-256과 ZIP이 같은지 확인합니다.
SHA-256 방식은 파일이 바뀌지 않았다는 것만 확인하며 독립 데이터 감사를 대신하지 않습니다.
기계적 검사를 통과해도 사람·병변·촬영 세션·변형된 증강 파생본 누수가 없다는 뜻은 아닙니다.
현재 출처 대응 정보가 없어 해당 항목은 미검증으로 기록합니다.
"""),
            cell(
                """config = dict(domain=DOMAIN, project_root=PROJECT_ROOT, data_zip=DATA_ZIP,
              audit_dir=AUDIT_DIR, expected_data_sha256=EXPECTED_DATA_SHA256,
              resume_dir=RESUME_DIR, mode=MODE, seed=SEED,
              batch_size=BATCH_SIZE, epochs1=STAGE1_EPOCHS, epochs2=STAGE2_EPOCHS,
              extension_epochs=EXTENSION_EPOCHS, train_variant=TRAIN_VARIANT)
context = prepare(config, PROFILES, SOURCES, BUILD_COMMIT)
print('결과 폴더 / 중단 시 RESUME_DIR:', context['output'])
print('클래스 순서:', context['classes'])
""",
                True,
            ),
        ]
        if mode == "audit":
            cells += [
                cell("""## 5. 이미지 검사와 보고서 저장

손상·지원하지 않는 형식·클래스 개수·빈 클래스·파일/픽셀 중복·다른 라벨의 동일 사진을 검사합니다.
원본과 증강본의 검증/테스트가 같은지도 확인합니다. 중복 예시 그림과 개수 그래프를 저장합니다.
문제가 있으면 학습은 차단됩니다. 자동으로 사진을 삭제하거나 분할을 변경하지 않습니다.
`audit_summary.json`의 status와 limitations를 읽고, 출력된 폴더를 ②/③의 AUDIT_DIR에 넣으세요.
"""),
                cell("audit_run(context)\n", True),
            ]
        else:
            protocol = """두 실험 모두 B0 / 224 / CE / 분류층만 15회 학습입니다.
변경 변수는 학습 데이터 Original ↔ Augmented입니다. 검증·테스트는 Original을 공통 사용합니다.
이미 저장된 증강본을 사용하며 추가 온라인 증강은 적용하지 않습니다.
동일 epoch라도 증강본의 이미지 수가 많으면 총 학습 횟수(배치 수)가 증가합니다.
따라서 결과는 '같은 epoch 예산의 데이터 구성 비교'이며 동일 업데이트 수 비교는 아닙니다.
예전 노트북을 바이트 단위로 재현하는 기능은 아닙니다."""
            if mode == "suite":
                protocol = """|실험|비교 기준|변경 변수 / 이유|
|---|---|---|
|B0 224 CE|새 기준 실험|분류층 15회 + 후반부 미세조정 10회|
|B0 256 CE|B0 224 CE|입력 해상도만 변경: 작은 특징 보존 여부|
|B0 256 LS 0.05|B0 256 CE|Loss만 변경: 과도한 확신 완화 여부|
|B0 256 Focal 1.5|B0 256 CE|Loss만 변경: 어려운 사례에 집중하는 효과|
|B1 256 LS 0.05|B0 256 LS 0.05|Backbone만 변경: 모델 용량 효과|
|B1 +5회|B1 256 LS 0.05|미세조정 시간 연장 효과|

고정: 분할, 학습 데이터 종류, seed, 배치, 학습률(1e-4/1e-5), Dropout 0.3,
후반 30개 레이어 범위(BatchNormalization 제외), 추가 온라인 증강 없음.
B1은 구조가 달라 같은 30개 범위라도 학습 파라미터 수가 달라집니다.
레이어 목록과 파라미터 수를 기록합니다.
5회 추가 실험은 B1의 10회째 마지막 모델과 optimizer(학습 진행 상태)를 복원합니다.
연장 후 나빠지면 이전 우수 체크포인트를 유지합니다. 6개 모두가 독립 초기화 실험인 것은 아닙니다."""
            cells += [
                cell(
                    "## 5. 학습 실행 — 조건과 변경 이유\n\n"
                    + protocol
                    + """

EfficientNet의 ImageNet 사전학습 가중치에서 시작합니다. 외부 `/255`는 적용하지 않습니다.
각 epoch의 로그·최고 모델·단계 마지막 모델을 Drive에 저장합니다.
Colab 연결이 끊기면 동일 설정과 출력된 RESUME_DIR로 재실행하세요.
"""
                ),
                cell("records = run(context)\n", True),
                cell("""## 6. 비교 그림·최종 평가·패키징

전체 실험 Accuracy/Loss를 한 장에 나란히 배치합니다. 검증 Accuracy·Macro F1(클래스별 F1 평균),
혼동행렬(실제와 예측의 교차표), 오분류 사진, CSV도 저장합니다.
CE/LS/Focal은 Loss의 정의가 다르므로 손실 숫자를 실험 간 성능 순위로 비교하지 마세요.

검증 Accuracy로 후보를 고정하고 동점이면 먼저 실행한 실험/단계를 유지합니다.
Test는 선정 모델만 평가합니다. 두 원본/증강 모델의 비교도 검증 지표로 제공합니다.
이미 완료한 Test 평가는 해시 검증 후 재사용합니다.

`2_results/대상/selected_models/실행명/`에 모델·클래스 순서·전처리·평가·소스가
묶인 패키지를 저장합니다. 결과 ZIP은 로컬 보관용이며 후보 ZIP은 모델을 포함합니다.
모든 중간 모델은 `2_results/대상/실행명/`에 남습니다.
새 패키지는 공개 데이터 후보이며 실제 장비 성능은 미검증입니다.
Hair/Skin은 정상 클래스가 없고 Web Skin만 정상 클래스가 있습니다.
범위 밖 입력 거부 기능은 없으며 출력 점수는 정확할 확률로 보정되지 않았습니다.
"""),
                cell("finish(context, records)\n", True),
            ]
        nb = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "name": "python3"},
                "language_info": {"name": "python"},
                "colab": {"name": filename},
                "accelerator": "GPU",
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        for i, c in enumerate(cells):
            c["id"] = f"cell-{i:02d}"
        (ROOT / "notebooks" / filename).write_text(
            json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    bundle = ROOT / 'notebooks/common_colab_notebooks_v1.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / 'notebooks').glob('0*_common_*.ipynb')):
            archive.write(path, path.name)
        guide = ROOT / 'docs/research/COMMON_COLAB_NOTEBOOKS_GUIDE.md'
        if guide.exists():
            archive.write(guide, '사용안내.md')
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip():
            raise RuntimeError('Notebook bundle ZIP verification failed')

    legacy = ROOT / 'notebooks/legacy_notebooks_20260909.zip'
    if legacy.exists():
        with zipfile.ZipFile(legacy) as archive:
            if archive.testzip():
                raise RuntimeError('Existing legacy notebook ZIP is damaged')
        return
    legacy_files = sorted(
        path
        for path in (ROOT / 'notebooks').glob('*.ipynb')
        if not path.name.startswith(('01_common_', '02_common_', '03_common_'))
    )
    manifest = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in legacy_files
    }
    with zipfile.ZipFile(legacy, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in legacy_files:
            archive.write(path, 'legacy_notebooks/' + path.name)
        archive.writestr(
            'legacy_notebooks/MANIFEST_SHA256.json',
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    with zipfile.ZipFile(legacy) as archive:
        if archive.testzip():
            raise RuntimeError('Legacy notebook ZIP verification failed')
        for name, digest in manifest.items():
            content = archive.read('legacy_notebooks/' + name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise RuntimeError('Legacy notebook hash mismatch: ' + name)


if __name__ == "__main__":
    bootstrap_sources()
    build()
