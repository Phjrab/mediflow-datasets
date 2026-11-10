"""Historical nine-condition exporter; the next training design is undecided.

The 2026-11-09 design was withdrawn on 2026-11-10. This helper remains for
reproducibility and reads preserved calibration inputs independently of output/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mediflow_datasets import camera_extended as camera
from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_batch as batch
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets import domain_adaptation_review as review_tools
from mediflow_datasets import domain_nine as nine
from mediflow_datasets.candidate_reproduction import ROOT
from mediflow_datasets.domain_adaptation_training import object_hash

REFERENCE_INPUTS = ROOT / "camera_capture/_review/training_reference_inputs_20261110"


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def config():
    value = data.read_json(ROOT / "configs/domain_adaptation.json")
    value.update(
        schema="mediflow_nine_v1",
        arms=list(nine.ARMS),
        batch_size=8,
        baseline_head_epochs=15,
        baseline_partial_epochs=10,
        alignment_fraction_per_loss=0.05,
        gradient_clip_norm=1.0,
        experiment_revision="nine_measured_camera_and_combined_alignment_v1",
    )
    for domain, task in value["tasks"].items():
        task.update(
            input_policy="center_square" if domain == "web_skin" else "legacy_resize",
            train_variant="augmented" if domain == "hair" else "original",
            geometry=False,
        )
    return value


def source_measurements(domain):
    path = (
        ROOT
        / "camera_capture/_review/domain_adaptation_colab_20261109"
        / domain
        / ("analysis/analysis.json")
    )
    return [r for r in data.read_json(path)["images"] if r["kind"] == "source"]


def qa(domain, profile, directory, excluded):
    previous = ROOT / "camera_capture/_review/preflight_v2_fbe37d44" / domain / "transform_qa"
    old = data.read_json(previous / "measurements.json")
    rows = []
    for index, row in enumerate(old):
        if row["path"] in excluded:
            continue
        if "/original/train/" not in row["path"]:
            raise ValueError("QA may use retained original Train only")
        with Image.open(previous / f"Q{index:04}_before.png") as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.float32)
        seed = exp.item_seed(data.ImageRow(row["path"], "train", row["label"]), 707)
        a, b = (camera.transform(image, profile, mode, seed) for mode in ("lighting", "expanded"))
        patch_a, _ = camera.focus_patch(a, {"metrics": row["original"]}, domain)
        patch_b, _ = camera.focus_patch(b, {"metrics": row["original"]}, domain)
        rows.append(
            {
                "id": f"Q{index:04}",
                "original_index": index,
                "path": row["path"],
                "split": "train",
                "label": row["label"],
                "lighting_clipping": review_tools.clipping(image, a, None),
                "expanded_clipping": review_tools.clipping(image, b, None),
                "expanded_delta_mean_abs": float(np.abs(b - a).mean()),
                "normalized_detail_ratio": camera.normalized_detail(patch_b)
                / max(camera.normalized_detail(patch_a), 1e-6),
            }
        )
    chosen = set()
    for label in sorted({r["label"] for r in rows}):
        indices = [i for i, row in enumerate(rows) if row["label"] == label]
        chosen.update(indices[:2])
    for key in (
        lambda r: r["expanded_clipping"]["new_near_endpoint_channels"],
        lambda r: r["expanded_delta_mean_abs"],
        lambda r: -r["normalized_detail_ratio"],
    ):
        chosen.add(max(range(len(rows)), key=lambda i, key=key: key(rows[i])))
    pages = []
    chosen = sorted(chosen)
    for offset in range(0, len(chosen), 4):
        sheet = Image.new("RGB", (768, 4 * 285 + 32), "#15212d")
        draw = ImageDraw.Draw(sheet)
        draw.text((5, 5), "Source Train: original / lighting / expanded", fill="white")
        for row_index, index in enumerate(chosen[offset : offset + 4]):
            row = rows[index]
            with Image.open(previous / f"{row['id']}_before.png") as opened:
                image = np.asarray(opened.convert("RGB"), dtype=np.float32)
            seed = exp.item_seed(data.ImageRow(row["path"], "train", row["label"]), 707)
            tiles = [
                image,
                camera.transform(image, profile, "lighting", seed),
                camera.transform(image, profile, "expanded", seed),
            ]
            for column, tile in enumerate(tiles):
                preview = Image.fromarray(np.rint(tile).clip(0, 255).astype(np.uint8))
                preview.thumbnail((252, 252))
                sheet.paste(preview, (column * 256, 32 + row_index * 285))
            draw.text(
                (4, 32 + row_index * 285 + 255),
                f"{row['id']} C{row['label']} / fixed seed 707",
                fill="white",
            )
        name = f"review_{offset // 4 + 1:03}.jpg"
        sheet.save(directory / name, quality=94)
        pages.append(name)
    summary = {
        "samples": len(rows),
        "by_class": dict(Counter(r["label"] for r in rows)),
        "max_new_endpoint": max(r["expanded_clipping"]["new_endpoint_channels"] for r in rows),
        "max_new_near_endpoint": max(
            r["expanded_clipping"]["new_near_endpoint_channels"] for r in rows
        ),
        "detail_ratio_min": min(r["normalized_detail_ratio"] for r in rows),
        "review_pages": pages,
        "reviewed_ids": [rows[i]["id"] for i in chosen],
        "all_previews_visually_reviewed": False,
        "note": "uint8 model-input previews; numerical all, visual selected per class/outlier",
    }
    runner.write_json(directory / "qa_measurements.json", rows)
    runner.write_json(directory / "qa_summary.json", summary)
    return summary


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    cfg, summary = config(), {}
    runner.write_json(output / "nine_experiments.json", cfg)
    for domain in nine.DOMAINS:
        spec, _, classes = runner.task_candidate(domain, cfg)
        directory = output / domain
        directory.mkdir()
        reference = REFERENCE_INPUTS / domain
        packet = data.read_json(reference / "calibration.json")
        lighting = data.read_json(reference / "lighting_profile.json")
        plan = None if domain == "hair" else data.read_json(reference / "quarantine_plan.json")
        rows = [data.ImageRow(**r) for r in packet["rows"]]
        if runner.target_fingerprint(ROOT / "camera_capture", rows) != packet["fingerprint"]:
            raise ValueError("Actual capture files changed: " + domain)
        measurements = {r["path"]: r for r in packet["measurements"]}
        target_rows = [measurements[r.path] for r in rows]
        images = np.stack(
            [
                exp.image_tensor(
                    data.local_path(ROOT / "camera_capture", r.path).read_bytes(),
                    spec["input_size"],
                    spec["input_policy"],
                )
                for r in rows
            ]
        ).astype(np.float32)
        np.savez_compressed(directory / "target_inputs.npz", images=images)
        meta = {
            "domain": domain,
            "input_policy": spec["input_policy"],
            "input_size": spec["input_size"],
            "rows": [asdict(r) for r in rows],
            "array_sha256": data.sha256(directory / "target_inputs.npz"),
            "pixel_hashes": packet["pixel_hashes"],
            "capture_fingerprint": packet["fingerprint"],
            "contains_identifiable_images": True,
            "target_accuracy": None,
        }
        runner.write_json(directory / "target_inputs.json", meta)
        excluded = set(plan["exclude_original"]["train"]) if plan else set()
        source = [r for r in source_measurements(domain) if r["path"] not in excluded]
        profile = camera.measured_extended_profile(domain, lighting, target_rows, images, source)
        runner.write_json(directory / "extended_profile.json", profile)
        runner.write_json(directory / "class_names.json", classes)
        if plan:
            runner.write_json(directory / "quarantine_plan.json", plan)
        proof = qa(domain, profile, directory, excluded)
        summary[domain] = {
            "photos": len(rows),
            "color_a": profile["color_a"],
            "color_b": profile["color_b"],
            "texture_gain": profile["texture_gain"],
            "blur_sigma_roi64": profile["blur_sigma_roi64"],
            "blur_input_scale": profile["blur_input_scale"],
            "qa": proof,
        }
    runner.write_json(output / "preparation_summary.json", summary)
    runner.write_json(
        output / "preparation_code.json",
        {
            "camera_code_sha256": data.sha256(ROOT / "src/mediflow_datasets/camera_extended.py"),
            "builder_code_sha256": data.sha256(Path(__file__)),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def package(output, preparation, decisions):
    output.mkdir(parents=True, exist_ok=False)
    proof = data.read_json(preparation / "preparation_code.json")
    if (
        data.sha256(ROOT / "src/mediflow_datasets/camera_extended.py")
        != proof["camera_code_sha256"]
    ):
        raise ValueError("Transform code changed after measured QA; regenerate preparation")
    decisions = data.read_json(decisions)
    cfg = data.read_json(preparation / "nine_experiments.json")
    content = {}
    for path in (ROOT / "src/mediflow_datasets").glob("*.py"):
        content[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    registry = data.read_json(ROOT / "results/CANDIDATE_INDEX.json")
    content["results/CANDIDATE_INDEX.json"] = encode(registry)
    content["configs/nine_experiments.json"] = encode(cfg)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for domain in nine.DOMAINS:
        directory = preparation / domain
        profile = data.read_json(directory / "extended_profile.json")
        qa_meta = data.read_json(directory / "qa_summary.json")
        decision = decisions[domain]
        if (
            decision["profile_sha256"] != object_hash(profile)
            or decision["decision"] != "approved_for_pilot"
            or set(decision["reviewed_pages"]) != set(qa_meta["review_pages"])
        ):
            raise ValueError("Missing current expanded-transform review: " + domain)
        evidence = {}
        for path in directory.iterdir():
            if path.is_file():
                body = path.read_bytes()
                content[f"review/{domain}/{path.name}"] = body
                if path.name != "extended_profile.json":
                    evidence[path.name] = hashlib.sha256(body).hexdigest()
        receipt = {
            "status": "reviewed_nine_pilot",
            "profile_sha256": object_hash(profile),
            "config_sha256": object_hash(cfg),
            "evidence_sha256": evidence,
            "visual_review": decision,
            "repository_commit": commit,
            "uncommitted_code_included": True,
            "identifiable_target_inputs_included": True,
            "target_role": "calibration and unlabeled alignment training",
            "independent_target_test": False,
            "target_accuracy": None,
        }
        content[f"review/{domain}/readiness.json"] = encode(receipt)
        model = ROOT / "results" / registry["candidates"][domain]["model"]
        for path in (model, model.with_name("class_names.json")):
            content[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    destination = output / "mediflow_nine_experiments_runtime.zip"
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, body in content.items():
            archive.writestr(name, body)
        archive.writestr(
            "bundle_manifest.json",
            encode({name: hashlib.sha256(body).hexdigest() for name, body in content.items()}),
        )
    sha = batch.digest(destination)
    bootstrap = (ROOT / "src/mediflow_datasets/domain_adaptation_batch.py").read_text(
        encoding="utf-8"
    )
    driver = f'''
from google.colab import files
print("새 mediflow_nine_experiments_runtime.zip 하나를 선택하세요.")
print("이 ZIP에는 얼굴을 포함한 실제 촬영 영상의 모델 입력 배열이 들어 있습니다.")
uploaded=files.upload()
if len(uploaded)!=1:
    raise ValueError("새 실행 ZIP 하나만 선택하세요.")
bundle=Path(next(iter(uploaded))).resolve()
del uploaded
if digest(bundle)!="{sha}":
    raise ValueError("함께 제공된 새 mediflow_nine_experiments_runtime.zip을 선택하세요.")
workspace=Path("/content")/("mediflow_nine_work_"+uuid.uuid4().hex[:8])
workspace.mkdir()
runtime=unpack_bundle(bundle,workspace/"runtime","{sha}")
sys.path.insert(0,str(runtime/"src"))
from mediflow_datasets.domain_nine_colab import run_prepared
NINE_RESULTS=run_prepared(runtime,workspace,"{sha}")
'''
    code = bootstrap + "\n" + driver
    compile(code, "MediFlow_Nine_Experiments.py", "exec")
    guide = (
        "# MediFlow 세 분야 × 세 조건, 총 9개 실험\n\n"
        "GPU를 선택하고 **모두 실행**하세요. 새 `mediflow_nine_experiments_runtime.zip` 하나를 "
        "선택하고 Drive를 연결하면 두피 → 얼굴 → 피부를 차례로 실행합니다. "
        "각 분야에서 조명 중심 / 색·질감·초점 확장 / 확장+MMD·CORAL 결합을 비교합니다.\n\n"
        "**실제 촬영 영상 124장의 모델 입력 배열이 ZIP에 포함됩니다. "
        "얼굴도 식별 가능한 영상입니다.** "
        "이 ZIP을 선택하면 Colab으로 전송됩니다. 원본 파일은 PC에 그대로 보존됩니다. "
        "질환 정답은 없고 정렬 조건의 Target 학습 자료로 사용합니다. 독립 Test가 아닙니다.\n\n"
        "피부·얼굴은 제외 목록을 적용해 ImageNet에서 새 공통 기준 모델을 한 번 준비합니다 "
        "(head 15회, 후반부 10회). 두피는 기존 B1·384 5클래스 모델을 사용합니다. "
        "그 후 같은 출발점에서 조건별 3회 추가 학습합니다. 9개는 GPU에서 자동 순차 실행합니다. "
        "MMD와 CORAL은 세 번째 조건에서 함께 쓰므로 개별 효과는 분리하지 않습니다.\n\n"
        "검증 자료로 선택을 고정한 뒤 원본 Test·조명 변화 Test·확장 변화 Test를 평가합니다. "
        "실제 사진은 예측 변화와 특징 거리만 기록하고 Target 정확도는 계산하지 않습니다. "
        "색·질감·초점 강도는 네 사람의 관측 변화로 제한한 파일럿 설정이며 "
        "장비 보정 상수가 아닙니다.\n\n"
        "결과는 Drive의 `mediflow_nine_…` 폴더와 결과 ZIP 하나에 보관합니다. 같은 ZIP으로 "
        "재실행하면 일치하는 기존 실행을 찾아 완료 체크포인트를 검증 후 재사용합니다. "
        "중단된 학습 조건은 새 시도 폴더에서 다시 시작하며 부분 결과를 지우지 않습니다. "
        "실패가 있으면 summary와 로그에 표시합니다. 모든 실패 가능성을 보장할 수는 없습니다.\n"
    )
    book = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "nine-guide",
                "metadata": {},
                "source": guide.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "id": "nine-run",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
    }
    notebook = output / "MediFlow_9_Experiments_Colab.ipynb"
    with notebook.open("x", encoding="utf-8") as stream:
        json.dump(book, stream, ensure_ascii=False, indent=2)
    (output / "MediFlow_9_Experiments_Colab.py").write_text(code, encoding="utf-8")
    (output / "README.md").write_text(guide, encoding="utf-8")
    runner.write_json(
        output / "package_info.json",
        {
            "sha256": sha,
            "bytes": destination.stat().st_size,
            "experiments": 9,
            "includes_identifiable_target_arrays": True,
            "actual_training_executed": False,
        },
    )
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "package"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    else:
        if args.preparation is None or args.decisions is None:
            parser.error("package needs --preparation and --decisions")
        package(args.output, args.preparation, args.decisions)
