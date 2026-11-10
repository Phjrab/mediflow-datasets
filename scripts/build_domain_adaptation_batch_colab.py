"""Create an analysis-only Hair/Skin Colab notebook using the existing runtime ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(output, bundle):
    source = ROOT / "src/mediflow_datasets/domain_adaptation_batch.py"
    body = source.read_text(encoding="utf-8")
    bundle_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
    code = body + f'\n\nBATCH_RESULTS = run_colab("{bundle_hash}")\n'
    compile(code, "Hair_Skin_Analysis_Colab.py", "exec")
    markdown = (
        "# MediFlow Hair · Skin 분석을 한 번에 실행\n\n"
        "GPU를 선택하고 **런타임 → 모두 실행**을 누르세요. "
        "파일 선택 창에서 기존 `mediflow_da_runtime.zip` 하나를 선택하고 Drive를 연결하면 됩니다. "
        "코드나 DOMAIN을 수정할 필요 없습니다.\n\n"
        "순서: **실행 ZIP 업로드 → 환경 확인 → Hair 5클래스 분석·미리보기 → "
        "Skin 10클래스 분석·미리보기 → 분야별 결과 ZIP 2개 다운로드**. "
        "기본 Colab 환경이 맞지 않으면 별도 분석 환경을 자동으로 설치합니다. "
        "Colab 기본 TensorFlow를 교체하지 않아 2.21/2.20 패키지 충돌을 피합니다. "
        "호환 환경에서는 설치를 반복하지 않습니다.\n\n"
        "학습·REVIEW_NOTE·최종 Test 평가 단계는 포함하지 않습니다. "
        "중복·분할 검사는 수행하고, 미해결 항목을 결과에 보존합니다. "
        "일반 참여자 사진의 정답과 Target 정확도를 만들지 않습니다.\n\n"
        "결과는 Drive 내 드라이브의 `mediflow_da_hair_skin_…/hair`, `…/skin`에 각각 보관합니다. "
        "다운로드가 차단되면 브라우저의 여러 파일 다운로드를 허용하거나 마지막 다운로드 버튼을 "
        "누르세요. 실패한 분야는 `mediflow_failed_…zip`으로 구분되며 로그를 포함합니다. "
        "실제 촬영 원본은 실행 ZIP에 포함되어 있지 않습니다."
    )
    book = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "colab": {"name": "Hair_Skin_Analysis_Colab.ipynb", "gpuType": "T4"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "hair-skin-guide",
                "metadata": {},
                "source": markdown.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "id": "hair-skin-run",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "Hair_Skin_Analysis_Colab.ipynb").open("x", encoding="utf-8") as stream:
        json.dump(book, stream, ensure_ascii=False, indent=2)
    (output / "Hair_Skin_Analysis_Colab.py").write_text(code, encoding="utf-8")
    (output / "README.md").write_text(markdown + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    build(args.output, args.bundle)
