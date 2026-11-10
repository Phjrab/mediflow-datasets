"""Package a separately named notebook from the completed all-domain calibration audit."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import build_adaptation_strength as core

from mediflow_datasets import domain_adaptation_batch as batch

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "presentation/report_assets/camera_profile_revision_20261110"


def notebook(study_json):
    book = core.notebook(study_json)
    study = json.loads(study_json)
    study_sha = hashlib.sha256(study_json.encode()).hexdigest()
    guide = """# MediFlow 세 분야 촬영 변환 재검토 수정본

**GPU 선택 → 모두 실행 → Google Drive 연결**로 실행합니다.
기존 공통 ZIP v3·원본 데이터 ZIP·1차와 2차 결과를 재사용합니다.
새 실행 ZIP 업로드나 Target 복호화 키 입력은 필요 없습니다.

피부 초기 B0·224, 얼굴 선정 PMG·B0·256, 두피 선정 B1·384의 1차 학습 전 상태에서
각 5 epoch, 총 세 분야를 순서대로 학습합니다. 이전 확대 후보는 학습하지 않았습니다.

이번 수정본은 세 분야 모두 조명 적용 후 남은 색 차이를 다시 점검했고,
흐림·미세 대비가 절대 경계와 대비로 보정한 경계 지표를 모두 악화시키지 않는 경우만 유지합니다.
아래는 실행에 쓰는 실제 설정입니다.
시작할 때 세 분야의 출발 모델·기존 결과·분할과 전체 입력 캐시 공간을 먼저 확인합니다.

| 분야 | Lab a 이동 | Lab b 이동 | 미세 대비 배율 | 흐림 sigma · ROI 64 기준 |
| --- | --- | --- | --- | --- |
"""
    for domain, value in study["domains"].items():
        p = value["profile"]
        guide += (
            f"| {domain} | `{p['color_a']}` | `{p['color_b']}` | "
            f"`{p['texture_gain']}` | `{p['blur_sigma_roi64']}` |\n"
        )
    guide += f"""
밝기·대비·좌우 조명, 데이터 분할·출발 모델·학습량·시드·학습률·변환 확률은 고정했습니다.
실제 사진과의 동일함이나 질환 분류 성능 향상은 검증되지 않았습니다.
이미 확인한 Source Test는 이 노트북에서 다시 추론하지 않습니다.
결과는 학습 전·기존 2차·이번 수정본을 같은 원본·기존 변환·수정 변환 Validation으로 비교합니다.
두피는 추가 색·흐림·미세 대비가 없어 수정 조명과 수정 확장 조건이 같습니다.

Drive 결과 폴더
`MediFlow_adaptation_runs/{study["bundle_sha256"][:16]}/measured_strength_{study_sha[:12]}`

화면에 epoch·배치·검증·저장을 표시하고 저장된 회차부터 이어 실행합니다.
완료하면 학습곡선과 비교표를 표시하고 `MediFlow_Revised_Camera_Results_20261111.zip`을
다운로드합니다. Drive의 위 결과 폴더에도 같은 이름으로 보관합니다.
"""
    book["cells"][0]["source"] = guide.splitlines(True)
    code = "".join(book["cells"][1]["source"])
    code = code.replace(
        "MediFlow_Stronger_Adaptation_Results_20261110.zip",
        "MediFlow_Revised_Camera_Results_20261111.zip",
    )
    code = code.replace(
        "실측 기반 후속 학습 · 세 분야 각각 5 epoch", "세 분야 촬영 변환 수정본 · 각 5 epoch"
    )
    code = code.replace(
        '    names.extend("code/"',
        '    names.append("startup_preflight.json")\n    names.extend("code/"',
    )
    compile(code, "revised-camera-colab", "exec")
    book["cells"][1]["source"] = code.splitlines(True)
    return book


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    study = core.read(AUDIT / "study.json")
    if (
        batch.digest(core.BUNDLE) != study["bundle_sha256"]
        or batch.digest(ROOT / "configs/camera_strength_20261110.json")
        != study["parent_study_sha256"]
        or batch.digest(ROOT / "src/mediflow_datasets/camera_revision.py")
        != study["profile_builder_sha256"]
        or core.read(AUDIT / "summary.json")["training_executed"]
    ):
        raise ValueError("Completed frozen calibration audit differs from its inputs")
    parent_runtime = dict(study["uncommitted_runtime_sha256"])
    for name, sha in parent_runtime.items():
        current = batch.digest(ROOT / "src/mediflow_datasets" / name)
        if current != sha and name != "adaptation_strength.py":
            raise ValueError("Training mathematics changed after calibration")
        study["uncommitted_runtime_sha256"][name] = current
    study["parent_runtime_sha256"] = parent_runtime
    study["pretraining_check_revision"] = "early_required_inputs_and_full_cache_space_v1"
    study["revision_decision_file_sha256"] = batch.digest(AUDIT / "decisions.json")
    study["revision_summary_file_sha256"] = batch.digest(AUDIT / "summary.json")
    study["notebook_builder_sha256"] = batch.digest(Path(__file__))
    study["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    text = json.dumps(study, ensure_ascii=False, indent=2) + "\n"
    artifacts = (
        (ROOT / "configs/camera_revised_20261111.json", text),
        (
            ROOT / "notebooks/camera_adaptation/MediFlow_Camera_Revised_Adaptation_20261111.ipynb",
            json.dumps(notebook(text), ensure_ascii=False, indent=1) + "\n",
        ),
    )
    for path, contents in artifacts:
        with path.open("w" if args.overwrite else "x", encoding="utf-8", newline="\n") as stream:
            stream.write(contents)
        print(path)


if __name__ == "__main__":
    main()
