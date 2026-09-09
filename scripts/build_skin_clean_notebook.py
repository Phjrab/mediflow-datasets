"""Generate the one-time standalone Skin clean dataset Colab notebook."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/04_one_time_skin_clean_builder_colab.ipynb"


def cell(source, code=False):
    value = {
        "cell_type": "code" if code else "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if code:
        value.update(execution_count=None, outputs=[])
    return value


def main():
    source = (ROOT / "src/mediflow_datasets/skin_clean_builder.py").read_text(encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    classes = [
        "광선각화증",
        "기저세포암",
        "보웬병",
        "사마귀",
        "지루각화증",
        "편평세포암",
        "표피낭종",
        "피부섬유종",
        "혈관종",
        "흑색점",
    ]
    cells = [
        cell("""# Skin clean v1 데이터셋 생성 — 1회용

검사에서 확인된 원본 분할 간 완전 동일 사진을 제거하고 새 Skin 데이터셋을 만듭니다.
기존 ZIP과 기존 모델은 변경하거나 삭제하지 않습니다.
생성이 끝나면 이 노트북은 archive로 옮겨도 됩니다.

처리 원칙:

1. Original의 모든 Train·Validation·Test를 다시 모읍니다.
2. RGB 픽셀이 같은 사진은 파일명이 달라도 한 장만 남깁니다.
3. 클래스별 고유 사진을 seed 42의 SHA-256 순서로 다시 나눕니다.
4. 클래스마다 Test 70장, Validation 100장, 나머지를 Train으로 둡니다.
5. 새 Train 원본마다 기존 Skin과 같은 증강 종류 중 하나를 적용합니다.
6. 분할·삭제 중복·증강 출처를 CSV로 기록하고 ZIP과 보고서를 Drive에 저장합니다.

사람·병변·촬영 세션 정보가 없으므로 그 단위의 누수와 유사 장면은 제거했다고 말할 수 없습니다.
새 분할이므로 기존 99% Validation/Test 결과는 새 데이터의 성능으로 사용할 수 없습니다.
"""),
        cell("""## 1. 사용자 설정

현재 감사 결과와 데이터 위치를 기본값으로 넣었습니다.
Drive 폴더를 바꾸지 않았다면 수정할 값이 없습니다.
`SOURCE_ZIP`을 비우면 `datasets` 아래에서 `skin_processed*` ZIP 하나를 찾습니다.
`AUDIT_DIR`은 방금 실행한 공통 ①의 결과 폴더입니다.
"""),
        cell(
            """PROJECT_ROOT = '/content/drive/MyDrive/mediflow_Project'
SOURCE_ZIP = ''
AUDIT_DIR = '/content/drive/MyDrive/mediflow_Project/reports/skin/audit_20260909_043652_a8c3efe9'
SEED = 42
VALIDATION_COUNT_PER_CLASS = 100
TEST_COUNT_PER_CLASS = 70
""",
            True,
        ),
        cell("""## 2. Drive 연결과 필요한 라이브러리

학습하지 않으므로 GPU는 필요하지 않습니다.
약 10GB ZIP을 복사·압축하므로 실행 시간이 걸릴 수 있습니다.
Colab 디스크 공간이 부족하면 더 큰 디스크 런타임이 필요합니다.
"""),
        cell(
            """from google.colab import drive
drive.mount('/content/drive')
%pip -q install "opencv-python-headless>=4.10,<5" "pillow>=10,<13" pandas matplotlib

import csv
import hashlib
import json
import platform
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
""",
            True,
        ),
        cell("""## 3. 내장된 정제 코드

별도 Python 파일을 업로드할 필요가 없습니다. 아래 소스와 생성 당시 commit을 결과에 함께 저장합니다.
"""),
        cell(
            f"""BUILDER_SOURCE = {source!r}
BUILD_COMMIT = {commit!r}
CLASS_NAMES = {classes!r}
namespace = {{}}
exec(compile(BUILDER_SOURCE, 'skin_clean_builder.py', 'exec'), namespace)
file_hash = namespace['file_hash']
write_json = namespace['write_json']
extract_original = namespace['extract_original']
find_original_root = namespace['find_original_root']
build_dataset = namespace['build_dataset']
make_zip = namespace['make_zip']
""",
            True,
        ),
        cell("""## 4. 입력 ZIP과 감사 보고서 고정

감사 보고서가 변경되지 않았는지 확인하고, 감사에 사용한 ZIP과 현재 ZIP의 SHA-256을 비교합니다.
현재 허용되는 감사 문제는 `cross_split_duplicates`뿐입니다. 손상 이미지·라벨 충돌·폴더 문제 등이
있으면 자동으로 중단합니다. 입력 ZIP은 Colab으로 복사한 뒤 다시 해시를 확인합니다.
"""),
        cell(
            """PROJECT = Path(PROJECT_ROOT)
AUDIT = Path(AUDIT_DIR)
if not PROJECT.is_dir() or not AUDIT.is_dir():
    raise FileNotFoundError('PROJECT_ROOT 또는 AUDIT_DIR을 확인하세요.')
if SOURCE_ZIP:
    candidates = [Path(SOURCE_ZIP)]
else:
    candidates = sorted(
        p for p in (PROJECT / 'datasets').rglob('skin_processed*')
        if p.is_file() and zipfile.is_zipfile(p)
    )
if len(candidates) != 1:
    raise ValueError(f'SOURCE_ZIP으로 기존 Skin ZIP 하나를 지정하세요: {candidates}')
source_zip = candidates[0]
audit_manifest = json.loads((AUDIT / 'audit_manifest.json').read_text(encoding='utf-8'))
for name, digest in audit_manifest.items():
    if file_hash(AUDIT / name) != digest:
        raise ValueError('감사 보고서가 변경됐습니다: ' + name)
audit_summary = json.loads((AUDIT / 'audit_summary.json').read_text(encoding='utf-8'))
if (audit_summary['domain'] != 'skin' or audit_summary['class_names'] != CLASS_NAMES
        or audit_summary['status'] != 'issues_found'):
    raise ValueError('현재 Skin 감사 결과가 아닙니다.')
with (AUDIT / 'audit_issues.csv').open(encoding='utf-8-sig') as stream:
    issue_types = {row['type'] for row in csv.DictReader(stream)}
if issue_types != {'cross_split_duplicates'}:
    raise ValueError(f'중복 외 문제가 있어 자동 정제를 중단합니다: {issue_types}')
with (AUDIT / 'label_conflicts.csv').open(encoding='utf-8-sig') as stream:
    if any(csv.DictReader(stream)):
        raise ValueError('서로 다른 클래스의 동일 사진이 있어 수동 검토가 필요합니다.')

run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
dataset_name = 'skin_clean_v1_' + run_id
local_root = Path('/content') / dataset_name
local_root.mkdir(parents=True, exist_ok=False)
local_zip = local_root / 'source.zip'
with source_zip.open('rb') as src, local_zip.open('xb') as dst:
    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
source_sha256 = file_hash(local_zip)
if source_sha256 != audit_summary['data_sha256'] or source_sha256 != file_hash(source_zip):
    raise ValueError('감사에 사용한 Skin ZIP과 현재 ZIP이 다릅니다.')
with zipfile.ZipFile(local_zip) as archive:
    original_bytes = sum(
        item.file_size for item in archive.infolist()
        if 'original' in [part.lower() for part in Path(item.filename).parts]
    )
required = source_zip.stat().st_size + original_bytes * 5 + 5 * 1024**3
free = shutil.disk_usage(local_root).free
if required > free:
    raise RuntimeError(f'Colab 공간 부족. 예상 필요 {required:,}, 여유 {free:,}')
print('입력 ZIP:', source_zip)
print('입력 SHA-256:', source_sha256)
""",
            True,
        ),
        cell("""## 5. 고유 이미지 재분할과 Train 증강

같은 픽셀 그룹에서 경로가 가장 앞선 파일을 대표로 보존합니다.
파일명이 아닌 이미지 내용으로 판단합니다.
새 분할은 클래스명·seed·픽셀 해시로 정렬하므로 같은 입력과 설정에서는 언제나 같습니다.
증강은 기존 Skin 코드의 11종과 강도를 유지하지만, 원본을 무작위 중복 선택하던 부분은 바꿨습니다.
모든 Train 원본을 정확히 한 번 사용하여 출처가 분명한 증강본 하나씩을 생성합니다.
"""),
        cell(
            """extracted = local_root / 'extracted_original'
extract_original(local_zip, extracted)
original_root = find_original_root(extracted)
dataset_root = local_root / dataset_name
manifest, counts = build_dataset(
    original_root, dataset_root, CLASS_NAMES, SEED,
    VALIDATION_COUNT_PER_CLASS, TEST_COUNT_PER_CLASS,
)
manifest.update({
    'source_zip': str(source_zip), 'source_zip_sha256': source_sha256,
    'source_audit_dir': str(AUDIT), 'source_audit_manifest': audit_manifest,
    'generated_utc': datetime.now(timezone.utc).isoformat(),
    'environment': {'python': platform.python_version(), 'opencv': cv2.__version__,
                    'numpy': np.__version__, 'pillow': PIL.__version__},
    'code_commit_at_generation': BUILD_COMMIT,
    'builder_source_sha256': hashlib.sha256(BUILDER_SOURCE.encode()).hexdigest(),
})
write_json(dataset_root / 'build_manifest.json', manifest)
display(pd.DataFrame(counts))
print(json.dumps(manifest, ensure_ascii=False, indent=2))
""",
            True,
        ),
        cell("""## 6. 보고서·그래프·새 ZIP을 Drive에 저장

새 ZIP은 기존 `skin_processed.zip`을 덮어쓰지 않습니다. 복사 후 ZIP CRC와 SHA-256을 다시 확인합니다.
분할 및 증강 출처 CSV, 클래스별 개수 그래프, 사용 코드와 원 감사 요약도 보고서 폴더에 저장합니다.
"""),
        cell(
            """report_dir = PROJECT / 'reports' / 'skin' / ('clean_builder_' + run_id)
report_dir.mkdir(parents=True, exist_ok=False)
(report_dir / 'skin_clean_builder.py').write_text(BUILDER_SOURCE, encoding='utf-8')
for name in ('build_manifest.json', 'split_manifest.csv', 'removed_exact_duplicates.csv',
             'augmentation_lineage.csv', 'dataset_counts.csv'):
    shutil.copyfile(dataset_root / name, report_dir / name)
shutil.copyfile(AUDIT / 'audit_summary.json', report_dir / 'source_audit_summary.json')

frame = pd.DataFrame(counts)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, kind in zip(axes, ('original', 'augmented')):
    frame[frame.kind == kind].pivot(
        index='class_name', columns='split', values='count').plot.bar(ax=ax)
    ax.set_title('Skin clean v1 / ' + kind)
    ax.set_xlabel('Class')
    ax.set_ylabel('Images')
fig.tight_layout()
fig.savefig(report_dir / 'dataset_counts.png', dpi=180)
plt.show()
plt.close(fig)

local_output_zip = local_root / (dataset_name + '.zip')
zip_sha256 = make_zip(dataset_root, local_output_zip)
drive_zip = PROJECT / 'datasets' / local_output_zip.name
with local_output_zip.open('rb') as src, drive_zip.open('xb') as dst:
    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
if file_hash(drive_zip) != zip_sha256:
    raise IOError('Drive ZIP 복사 검증 실패')
with zipfile.ZipFile(drive_zip) as archive:
    if archive.testzip():
        raise IOError('Drive ZIP CRC 검사 실패')
(drive_zip.with_suffix('.zip.sha256')).write_text(
    zip_sha256 + '  ' + drive_zip.name + '\\n', encoding='ascii')
write_json(report_dir / 'output.json', {
    'dataset_zip': str(drive_zip), 'sha256': zip_sha256,
    'bytes': drive_zip.stat().st_size, 'report_dir': str(report_dir),
    'next_step': 'Run common audit with DATA_ZIP set to this ZIP',
})
(report_dir / 'notebook_snapshot.py').write_text(
    '\\n\\n# ---- cell ----\\n\\n'.join(
        get_ipython().history_manager.input_hist_raw), encoding='utf-8')
write_json(report_dir / 'artifact_manifest.json', {
    p.name: {'sha256': file_hash(p), 'bytes': p.stat().st_size}
    for p in report_dir.iterdir() if p.is_file() and p.name != 'artifact_manifest.json'
})
report_zip = Path(shutil.make_archive(
    str(local_root / ('skin_clean_builder_reports_' + run_id)), 'zip',
    report_dir.parent, report_dir.name))
drive_report_zip = report_dir.parent / report_zip.name
with report_zip.open('rb') as src, drive_report_zip.open('xb') as dst:
    shutil.copyfileobj(src, dst)
if file_hash(report_zip) != file_hash(drive_report_zip):
    raise IOError('보고서 ZIP 복사 검증 실패')
print('새 데이터 ZIP:', drive_zip)
print('새 데이터 SHA-256:', zip_sha256)
print('검사 보고서:', report_dir)
print('공유할 보고서 ZIP:', drive_report_zip)
print('다음 공통 ① DATA_ZIP =', repr(str(drive_zip)))
""",
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
            "colab": {"name": OUTPUT.name},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
