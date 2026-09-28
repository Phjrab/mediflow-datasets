"""Build the standalone Colab notebook from the checked-in cross-audit module."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "mediflow_datasets" / "hair_good_cross_audit.py"
OUTPUT = ROOT / "notebooks" / "16_hair_good_cross_audit_colab.ipynb"


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


intro = """# Hair ‘양호’ 원본과 기존 Hair 데이터 교차 검사

이 노트북은 **학습하거나 데이터셋을 만들지 않습니다.**
Drive의 기존 Hair Clean ZIP과 새 양호 ZIP을 읽고, 동일 사진·동일 사람 ID가 있는지 확인합니다.
두 ZIP을 수정하지 않으며 결과는 `mediflow_Project/2_results/hair/` 아래에
새 폴더와 ZIP으로 저장합니다.

현재 양호 ZIP에는 JPG 534장이 있으며 라벨 JSON은 없습니다.
여섯 증상값이 모두 0이라는 것은 사용자가 원본에서 확인한 내용으로 기록하고,
이 노트북이 파일별 라벨을 독립 검증했다고 표현하지 않습니다.
기존 Hair v1 원본만 비교하며 저장 증강본은 다시 분할할 v2 원본이 아닙니다.

**실행:** Colab에서 이 노트북을 Drive의 `notebooks`에 업로드해 열고,
GPU 없이 셀을 순서대로 실행하세요. 기존 Hair ZIP 약 3GB를 Colab 작업 공간에
복사하므로 빈 공간과 시간이 필요합니다. 결과의 `audit_summary.json` 및
`good_cross_matches.csv`를 확인한 후 ZIP을 공유하면 다음 데이터셋 구성으로 이어갈 수 있습니다.
"""

setup = """from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import shutil
import zipfile
from datetime import datetime, timezone
from uuid import uuid4

PROJECT_ROOT = Path('/content/drive/MyDrive/mediflow_Project')
DATASET_DIR = PROJECT_ROOT / 'datasets'
hair_candidates = [
    DATASET_DIR / 'hair_datasets',
    DATASET_DIR / 'hair_datasets.zip',
]
valid_hair_zips = [p for p in hair_candidates if p.is_file() and zipfile.is_zipfile(p)]
if len(valid_hair_zips) != 1:
    available = sorted(p.name for p in DATASET_DIR.iterdir()) if DATASET_DIR.is_dir() else []
    raise FileNotFoundError(
        '기존 Hair ZIP을 하나로 확인할 수 없습니다. '
        f'확인한 후보: {[str(p) for p in hair_candidates]} / '
        f'datasets 폴더 항목: {available}'
    )
OLD_ZIP = valid_hair_zips[0]
GOOD_ZIP = DATASET_DIR / 'hair_good_raw_v1.zip'

for path in (OLD_ZIP, GOOD_ZIP):
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise FileNotFoundError(f'ZIP 파일을 확인하세요: {path}')
    print(path.name, f'{path.stat().st_size / 1024**2:.1f} MiB')

required = OLD_ZIP.stat().st_size + GOOD_ZIP.stat().st_size + 1024**3
if shutil.disk_usage('/content').free < required:
    raise RuntimeError('Colab 작업 공간이 부족합니다. 새 런타임을 사용하세요.')

work = Path('/content') / ('hair_good_audit_' + uuid4().hex[:8])
work.mkdir()
local_old = work / 'hair_datasets.zip'
local_good = work / 'hair_good_raw_v1.zip'
shutil.copyfile(OLD_ZIP, local_old)
shutil.copyfile(GOOD_ZIP, local_good)
print('Colab 임시 작업 공간으로 두 ZIP 복사 완료:', work)
"""

execute = """parent = PROJECT_ROOT / '2_results' / 'hair'
parent.mkdir(parents=True, exist_ok=True)
run_name = (
    'hair_good_cross_audit_'
    + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    + '_'
    + uuid4().hex[:8]
)
report_dir = parent / run_name
summary = audit(local_old, local_good, report_dir)
report_zip = Path(shutil.make_archive(str(report_dir), 'zip', root_dir=parent, base_dir=run_name))
print('검사 보고서:', report_dir)
print('공유할 결과 ZIP:', report_zip)
print('이 노트북은 Train/Validation/Test를 만들거나 학습하지 않았습니다.')
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
        cell("code", setup),
        cell(
            "markdown",
            "## 검사 함수\n\n아래 코드는 이 노트북에 포함돼 있습니다. "
            "별도의 Python 파일을 Drive에 올릴 필요가 없습니다.\n",
        ),
        cell("code", MODULE.read_text(encoding="utf-8")),
        cell(
            "markdown",
            "## 교차 검사 실행·결과 저장\n\n"
            "기존 ZIP은 읽기 전용이며 결과는 매번 새 이름으로 보존합니다.\n",
        ),
        cell("code", execute),
    ],
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
