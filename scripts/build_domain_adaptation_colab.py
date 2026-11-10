"""Package the current runtime, selected models and calibration measurements for Colab."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets.candidate_reproduction import ROOT


def cell(text, code=False):
    item = {
        "cell_type": "code" if code else "markdown",
        "metadata": {},
        "id": f"da-{cell.counter:02}",
        "source": text.splitlines(keepends=True),
    }
    cell.counter += 1
    if code:
        item.update(execution_count=None, outputs=[])
    return item


cell.counter = 0


def notebook(bundle_name):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "cells": [
            cell(
                "# MediFlow 실제 촬영 기반 적응 실험\n\n"
                "먼저 전체 Source Train과 측정된 Target을 비교합니다. "
                "Source-only 추가 학습, 밝기만 반영, 복합 증강을 같은 조건으로 비교합니다. "
                "Target은 라벨 없는 4명의 관찰 자료입니다. MMD/CORAL 학습은 보류합니다. "
                "GPU 런타임을 선택하고 처음부터 순서대로 실행하세요."
            ),
            cell(
                "%pip install -q tensorflow==2.20.0 keras==3.13.2 "
                "numpy pillow opencv-python-headless\n",
                True,
            ),
            cell(
                f"""from google.colab import files, drive
from pathlib import Path
import zipfile, sys, os, json, subprocess, uuid

uploaded = files.upload()  # PC에서 {bundle_name} 선택
if len(uploaded) != 1:
    raise ValueError("실행 ZIP 하나만 선택하세요.")
bundle = Path(next(iter(uploaded)))
runtime = Path("/content") / ("mediflow_da_" + uuid.uuid4().hex[:8])
with zipfile.ZipFile(bundle) as z:
    for name in z.namelist():
        if not (runtime / name).resolve().is_relative_to(runtime.resolve()):
            raise ValueError("Unsafe ZIP path")
    z.extractall(runtime)
sys.path.insert(0, str(runtime / "src"))
from mediflow_datasets.candidate_reproduction import sha256
manifest = json.loads((runtime / "bundle_manifest.json").read_text())
for name, digest in manifest.items():
    if sha256(runtime / name) != digest:
        raise ValueError("Bundle file changed: " + name)
drive.mount("/content/drive")
""",
                True,
            ),
            cell(
                """# 분야별로 따로 실행합니다. Hair는 기존 5클래스 후보입니다.
DOMAIN = "web_skin"  # web_skin / skin / hair
config = json.loads((runtime / "configs/domain_adaptation.json").read_text())
expected_name = config["tasks"][DOMAIN]["source_name"]
candidates = [p for p in Path("/content/drive/MyDrive").rglob(expected_name + "*")
              if p.is_file() and zipfile.is_zipfile(p)]
print("Source ZIP 후보", candidates)
if len(candidates) != 1:
    raise ValueError("위 목록에서 실제 ZIP 하나를 확인해 source_zip에 지정하세요.")
source_zip = candidates[0]
calibration = runtime / "calibration" / (DOMAIN + ".json")
# 결과 저장 위치는 이번 실행용 새 폴더입니다.
run_root = Path("/content/drive/MyDrive") / ("mediflow_da_" + DOMAIN + "_" + uuid.uuid4().hex[:8])
run_root.mkdir()
analysis_dir = run_root / "analysis"
train_dir = run_root / "training"
env = {**os.environ, "PYTHONPATH": str(runtime / "src"),
       "TF_NUM_INTEROP_THREADS": "2", "TF_NUM_INTRAOP_THREADS": "2"}
def execute(mode, *arguments):
    subprocess.run([sys.executable, "-m", "mediflow_datasets.domain_adaptation", mode,
                    "--domain", DOMAIN, "--config", str(runtime / "configs/domain_adaptation.json"),
                    "--source-zip", str(source_zip), "--calibration-file", str(calibration),
                    *map(str, arguments)], env=env, check=True)
print("결과 폴더", run_root)
""",
                True,
            ),
            cell(
                """execute("analyze", "--output", analysis_dir)
from IPython.display import display, SVG, Image
display(SVG(filename=str(analysis_dir / "metric_distributions.svg")))
display(SVG(filename=str(analysis_dir / "feature_pca.svg")))
print((analysis_dir / "source_audit.json").read_text())
""",
                True,
            ),
            cell(
                "## 변환 사진 검토\n\n각 클래스의 Train 표본에서 색과 경계를 확인합니다. "
                "인위적인 배경이 "
                "생기지 않는지 확인합니다. 파일 목록과 분할 간 유사 사진 후보도 확인합니다. "
                "다른 사례로 확인된 유사 사진은 paths, decision=distinct, reason을 적은 "
                "JSON으로 검토 근거를 남깁니다. "
                "동일 사진·사람·병변·세션·파생본 누수는 먼저 해결합니다."
            ),
            cell(
                """execute("preview", "--analysis", analysis_dir,
        "--output", run_root / "train_preview")
display(Image(filename=str(run_root / "train_preview/source_train_before_after.png")))
""",
                True,
            ),
            cell(
                """REVIEW_NOTE = ""  # 위 Train 변환을 검토한 결과와 판단 근거를 입력
# 확인할 수 없는 Source 인물·세션 정보의 한계를 이해한 뒤 설정
ACKNOWLEDGE_MISSING_IDENTITIES = False
NEAR_REVIEW_FILE = ""  # 근접 중복 후보가 있으면 검토 결과 JSON의 실제 경로
if not REVIEW_NOTE or not ACKNOWLEDGE_MISSING_IDENTITIES:
    raise ValueError("Train 표본과 누수 점검 결과를 먼저 확인하세요.")
profile_path = analysis_dir / "measured_profile.json"
profile = json.loads(profile_path.read_text())
profile.update(train_visual_review="approved", review_note=REVIEW_NOTE)
profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2))
extra = ["--near-review", NEAR_REVIEW_FILE] if NEAR_REVIEW_FILE else []
execute("train", "--analysis", analysis_dir, "--output", train_dir,
        "--acknowledge-incomplete-identities", *extra)
print((train_dir / "validation_selection.json").read_text())
""",
                True,
            ),
            cell(
                "## 최종 Test 평가\n\nValidation으로 저장본과 설정을 고정한 뒤 실행합니다. "
                "원본 Source Test와 복합 변환 Source Test를 별도로 평가합니다. "
                "참여자 원본 사진은 PC에 있습니다. 학습 이후 실제 사진의 예측은 PC에서 "
                "compare-target 명령으로 따로 계산합니다."
            ),
            cell(
                """execute("finalize", "--run", train_dir)
print((train_dir / "final_evaluation/comparison.json").read_text())
""",
                True,
            ),
            cell(
                """# Drive에 전체 실행 폴더를 보존하고 PC용 모델·결과 ZIP을 만듭니다.
destination = run_root / "trained_results.zip"
with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(train_dir.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(train_dir))
print("PC에서 compare-target을 실행할 결과 ZIP", destination)
""",
                True,
            ),
        ],
    }


def build(output, measurements):
    output.mkdir(parents=True, exist_ok=False)
    content = {}
    for path in (ROOT / "src/mediflow_datasets").glob("*.py"):
        content[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    content["configs/domain_adaptation.json"] = (
        ROOT / "configs/domain_adaptation.json"
    ).read_bytes()
    content["results/CANDIDATE_INDEX.json"] = (ROOT / "results/CANDIDATE_INDEX.json").read_bytes()
    for domain in ("web_skin", "skin", "hair"):
        spec, model, classes = data.candidate(domain)
        for path in model.parent.iterdir():
            if path.is_file() and path.suffix not in {".pyc"}:
                content[path.relative_to(ROOT).as_posix()] = path.read_bytes()
        folder = measurements / domain
        result = data.read_json(folder / "analysis.json")
        row_index = data.read_json(folder / "feature_diagnostics.json")["target_paths"]
        rows = [data.ImageRow(**row) for row in row_index]
        features = np.load(folder / "baseline_features.npz")
        pixel_hashes = [
            data.image_signatures(data.local_path(ROOT / "camera_capture", r.path).read_bytes())[1]
            for r in rows
        ]
        packet = {
            "domain": domain,
            "role": "calibration",
            "model_sha256": spec["model_sha256"],
            "people": sorted({row.person for row in rows}),
            "rows": [asdict(row) for row in rows],
            "measurements": [row for row in result["images"] if row["kind"] == "target"],
            "audit": data.read_json(folder / "target_audit.json"),
            "pixel_hashes": pixel_hashes,
            "fingerprint": runner.target_fingerprint(ROOT / "camera_capture", rows),
            "scores": features["target_scores"].tolist(),
            "features": features["target"].tolist(),
        }
        content[f"calibration/{domain}.json"] = json.dumps(packet, ensure_ascii=False).encode()
    manifest = {name: hashlib.sha256(body).hexdigest() for name, body in content.items()}
    with zipfile.ZipFile(
        output / "mediflow_da_runtime.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, body in content.items():
            archive.writestr(name, body)
        archive.writestr("bundle_manifest.json", json.dumps(manifest))
    book = notebook("mediflow_da_runtime.zip")
    for item in book["cells"]:
        if item["cell_type"] == "code" and not item["source"][0].startswith("%"):
            compile("".join(item["source"]), item["id"], "exec")
    with (output / "Domain_Adaptation_Colab.ipynb").open("x", encoding="utf-8") as stream:
        json.dump(book, stream, ensure_ascii=False, indent=2)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--measurements",
        type=Path,
        default=ROOT / "camera_capture/_review/domain_adaptation_run_20261108",
    )
    args = parser.parse_args()
    build(args.output, args.measurements)
