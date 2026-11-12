"""Assemble recorded adaptation settings, device gaps and performance without new inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import zipfile
from pathlib import Path

LABELS = {
    "skin": "현미경 피부",
    "web_skin": "웹캠 얼굴",
    "hair": "현미경 두피",
    "phone": "스마트폰 얼굴",
}
METRICS = {
    "y_p50": ("밝기 중앙값", "휘도 0–255"),
    "contrast": ("대비 P90−P10", "휘도"),
    "lab_a": ("Lab a 색상", "Lab a"),
    "lab_b": ("Lab b 색상", "Lab b"),
    "detail": ("미세 경계", "회색조 잔차"),
    "normalized_detail": ("대비 보정 미세 경계", "잔차/대비"),
    "lighting_range": ("조명 불균일", "휘도"),
    "side_light": ("좌우 조명", "휘도"),
    "face_w": ("얼굴 폭 비율", "프레임 비율"),
    "face_h": ("얼굴 높이 비율", "프레임 비율"),
    "dark_ridge": ("모발 가림 대용 통계", "픽셀 비율"),
    "very_bright": ("밝은 끝점 비율", "픽셀 비율"),
    "very_dark": ("어두운 끝점 비율", "픽셀 비율"),
}


def gap_change(before, after):
    if any(v is not None and (not math.isfinite(v) or v < 0) for v in (before, after)):
        raise ValueError("Distances must be finite and nonnegative")
    if before is None or after is None:
        return {
            "difference": None,
            "ratio": None,
            "reduction_fraction": None,
            "direction": "미측정",
        }
    return {
        "difference": after - before,
        "ratio": after / before if before else None,
        "reduction_fraction": 1 - after / before if before else None,
        "direction": "가까워짐" if after < before else "멀어짐" if after > before else "동일",
    }


def profile_signature(profile):
    lighting = profile.get("lighting", profile)
    result = {
        k: lighting.get(k, default)
        for k, default in (
            ("brightness_offset", [0.0, 0.0]),
            ("contrast", [1.0, 1.0]),
            ("left_light", [0.0, 0.0]),
            ("scale_x", [1.0, 1.0]),
            ("scale_y", [1.0, 1.0]),
            ("photometric_operator", None),
        )
    }
    result.update(
        {
            k: profile.get(k, default)
            for k, default in (
                ("color_a", [0.0, 0.0]),
                ("color_b", [0.0, 0.0]),
                ("texture_gain", [1.0, 1.0]),
                ("blur_sigma_roi64", [0.0, 0.0]),
                ("blur_input_scale", 1.0),
            )
        }
    )
    return result


def require_same_profile(measured, trained):
    if profile_signature(measured) != profile_signature(trained):
        raise ValueError("Measured image profile differs from the actually trained profile")


def collect(root):
    root = Path(root).resolve()
    provenance = {}

    def load(relative):
        path = root / relative
        content = path.read_bytes()
        provenance[relative] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    ledger = load(
        "presentation/report_assets/adaptation_completion_20261112/COMPLETED_ADAPTATION_PARAMETERS_20261112.json"
    )
    for profile in ledger["profiles"]:
        source = profile["source_record"]
        actual = hashlib.sha256((root / source["path"]).read_bytes()).hexdigest()
        if actual != source["sha256"]:
            raise ValueError("Recorded completed-profile source has changed")
        provenance[source["path"]] = actual
    previous = load("presentation/report_assets/camera_similarity_check_20261110/summary.json")
    revised = load("presentation/report_assets/camera_profile_revision_20261110/summary.json")
    study = load("presentation/report_assets/camera_profile_revision_20261110/study.json")
    phone_review = load("presentation/report_assets/smartphone_joint_review_20261112/review.json")
    phone_distributions = load(
        "presentation/report_assets/smartphone_joint_review_20261112/distribution_summary.json"
    )
    phone_comparison = load("output/smartphone_web_skin_20261111/webcam_smartphone_comparison.json")
    bundle = (
        root / "camera_capture/_review/stage1_release_v3_20261110/MediFlow_Adaptation_Common_v3.zip"
    )
    references = None
    if bundle.is_file():
        with zipfile.ZipFile(bundle) as archive:
            references = {
                d: json.loads(archive.read(f"references/{d}.json"))
                for d in ("skin", "web_skin", "hair")
            }
    gaps, distributions, features, tests, cross_camera = [], [], [], [], []

    def add_gap(domain, stage, metric, before, after, cohort, source):
        name, unit = METRICS[metric]
        gaps.append(
            {
                "domain": domain,
                "domain_label": LABELS[domain],
                "stage": stage,
                "metric": metric,
                "metric_label": name,
                "unit": unit,
                "before": before,
                "after": after,
                **gap_change(before, after),
                "cohort": cohort,
                "source": source,
            }
        )

    for domain in ("skin", "web_skin", "hair"):
        actual2 = load(f"results/camera_adaptation_stage2_20261110/{domain}/run_configuration.json")
        actual4 = load(
            f"results/camera_adaptation_stage4_before_refit_20261111/{domain}/run_configuration.json"
        )
        if (
            actual2["identity"]["reference_sha256"]
            != study["domains"][domain]["previous_reference_sha256"]
        ):
            raise ValueError("Stage2 replay and training reference identities differ")
        if references is not None:
            require_same_profile(references[domain]["extended_profile"], actual2["profile"])
        require_same_profile(study["domains"][domain]["profile"], actual4["profile"])
        for metric in METRICS:
            if metric not in revised["gaps"][domain]:
                continue
            p = previous["domains"][domain]["distribution_gaps"].get(metric)
            current = revised["gaps"][domain][metric]
            if p:
                if p["original"] != current["original"]:
                    raise ValueError("Original replay reference changed")
                add_gap(
                    domain,
                    "2·3차 영상",
                    metric,
                    p["original"],
                    p["previous"],
                    "camera_uint8_fixed_roi",
                    "presentation/report_assets/camera_similarity_check_20261110/summary.json",
                )
            add_gap(
                domain,
                "4차 영상",
                metric,
                current["original"],
                current["revised"],
                "camera_uint8_fixed_roi",
                "presentation/report_assets/camera_profile_revision_20261110/summary.json",
            )
        for view in ("original", "revised", "target"):
            for metric, stats in revised["groups"][domain][view].items():
                if metric not in METRICS:
                    continue
                distributions.append(
                    {
                        "domain": domain,
                        "domain_label": LABELS[domain],
                        "view": view,
                        "metric": metric,
                        "metric_label": METRICS[metric][0],
                        "n": stats["n"],
                        "mean": stats["mean"],
                        "variance": stats["var"],
                        **{q: stats[q] for q in ("q05", "q25", "q50", "q75", "q95")},
                        "weighting": "저장된 이미지별 표본 통계; 거리 계산의 그룹 가중과 구분",
                        "source": (
                            "presentation/report_assets/camera_profile_revision_20261110/summary.json"
                        ),
                    }
                )
        path = f"results/camera_adaptation_stage3_20261110/{domain}/feature_comparison.json"
        diagnostic = load(path)
        for model, item in diagnostic["models"].items():
            features.append(
                {
                    "domain": domain,
                    "domain_label": LABELS[domain],
                    "model": model,
                    "source_count": item["source_validation_count"],
                    "target_count": item["target_training_count"],
                    "mmd": item["mmd"],
                    "coral": item["coral"],
                    "baseline_mmd": diagnostic["models"]["initial"]["mmd"],
                    "baseline_coral": diagnostic["models"]["initial"]["coral"],
                    "scope": "확장 Source Validation 표본 대 비라벨 Target; 동일 파일 내 비교",
                    "source": path,
                }
            )
        features.append(
            {
                "domain": domain,
                "domain_label": LABELS[domain],
                "model": "stage4_revised",
                "source_count": None,
                "target_count": None,
                "mmd": None,
                "coral": None,
                "baseline_mmd": None,
                "baseline_coral": None,
                "scope": "4차 모델 특징 진단 미측정",
                "source": path,
            }
        )

    replay_by_id = {item["id"]: item for item in phone_review["full_replays"]}
    phone_study = load("results/smartphone_joint_adaptation_20261112/reproduction/study.json")
    for stage, key in ((1, "stage1"), (2, "stage2"), (4, "stage4"), (5, None)):
        candidate_id = phone_review["controls"][key] if key else "f10926b662a4a43c"
        candidate = replay_by_id[candidate_id]
        profile = phone_study["profiles"][key] if key else phone_study["joint_profile"]
        require_same_profile(candidate["profile"], profile)
        for metric, after in candidate["assessment"]["gaps"].items():
            add_gap(
                "phone",
                "2·3차 영상" if stage == 2 else f"{stage}차 영상",
                metric,
                phone_review["baseline_gaps"][metric],
                after,
                "phone_float_fixed_roi",
                "presentation/report_assets/smartphone_joint_review_20261112/review.json",
            )
    for view in ("original", "stage1", "stage2", "stage4", "target"):
        for metric, stats in phone_distributions["conditions"][view].items():
            if metric not in METRICS:
                continue
            quantiles = stats["q05_q25_q50_q75_q95"]
            distributions.append(
                {
                    "domain": "phone",
                    "domain_label": LABELS["phone"],
                    "view": view,
                    "metric": metric,
                    "metric_label": METRICS[metric][0],
                    "n": stats["count"],
                    "mean": stats["mean"],
                    "variance": stats["variance"],
                    **dict(zip(("q05", "q25", "q50", "q75", "q95"), quantiles, strict=True)),
                    "weighting": "클래스/참여자 균등 가중",
                    "source": (
                        "presentation/report_assets/smartphone_joint_review_20261112/"
                        "distribution_summary.json"
                    ),
                }
            )
    phone_features = load(
        "results/smartphone_joint_adaptation_20261112/calibration_diagnostics.json"
    )
    for model, item in phone_features["models"].items():
        features.append(
            {
                "domain": "phone",
                "domain_label": LABELS["phone"],
                "model": model,
                "source_count": phone_features["source_count"],
                "target_count": phone_features["target_count"],
                "mmd": item["mmd"],
                "coral": item["coral"],
                "baseline_mmd": phone_features["models"]["initial"]["mmd"],
                "baseline_coral": phone_features["models"]["initial"]["coral"],
                "scope": "Source Train 40장 대 설정용 Target42장; 이전 Stage3 고정 기준",
                "source": (
                    "results/smartphone_joint_adaptation_20261112/calibration_diagnostics.json"
                ),
            }
        )
    for metric, item in phone_comparison["metrics"].items():
        cross_camera.append(
            {
                "metric": metric,
                "metric_label": METRICS[metric][0],
                "source_train": item["source_train"]["equal_group_median_of_medians"],
                "webcam": item["webcam"]["equal_group_median_of_medians"],
                "phone": item["smartphone"]["equal_group_median_of_medians"],
                "phone_minus_webcam": item["phone_minus_webcam"],
                "source": "output/smartphone_web_skin_20261111/webcam_smartphone_comparison.json",
            }
        )

    selected = {
        "skin": "stage4_revised",
        "web_skin": "stage4_revised",
        "hair": "stage2_expanded",
        "phone": "stage4",
    }
    result_sets = []
    for domain in ("skin", "web_skin", "hair"):
        old_path = f"results/camera_adaptation_final_test_20261110/{domain}.json"
        new_path = f"results/camera_adaptation_stage4_test_20261111/{domain}.json"
        old, new = load(old_path), load(new_path)
        result_sets.append(
            (
                domain,
                {**old["metrics"], **new["metrics"]},
                {
                    **dict.fromkeys(old["metrics"], old_path),
                    **dict.fromkeys(new["metrics"], new_path),
                },
            )
        )
    path = "results/smartphone_final_test_20261112/test_results.json"
    phone_test = load(path)
    result_sets.append(("phone", phone_test["metrics"], dict.fromkeys(phone_test["metrics"], path)))
    for domain, items, sources in result_sets:
        for model, conditions in items.items():
            for condition, metric in conditions.items():
                count = metric["labeled_count"]
                correct = sum(
                    metric["confusion_matrix"][i][i] for i in range(len(metric["confusion_matrix"]))
                )
                if correct / count != metric["accuracy"]:
                    raise ValueError("Recorded Accuracy differs from original confusion matrix")
                baseline = items["initial"].get(condition)
                base_correct = (
                    sum(
                        baseline["confusion_matrix"][i][i]
                        for i in range(len(baseline["confusion_matrix"]))
                    )
                    if baseline
                    else None
                )
                tests.append(
                    {
                        "domain": domain,
                        "domain_label": LABELS[domain],
                        "model": model,
                        "selected_before_test": model == selected[domain],
                        "condition": condition,
                        "count": count,
                        "correct": correct,
                        "accuracy": metric["accuracy"],
                        "macro_f1": metric["macro_f1"],
                        "baseline_correct": base_correct,
                        "correct_difference": correct - base_correct
                        if base_correct is not None
                        else None,
                        "source": sources[model],
                    }
                )
    if len(ledger["profiles"]) != 17 or len(ledger["runs"]) != 21 or len(tests) != 81:
        raise ValueError("Completed experiment or evaluation inventory differs")
    csv_path = (
        "presentation/report_assets/adaptation_completion_20261112/"
        "COMPLETED_ADAPTATION_PARAMETERS_20261112.csv"
    )
    with (root / csv_path).open(encoding="utf-8-sig", newline="") as stream:
        parameters = list(csv.DictReader(stream))
    provenance[csv_path] = hashlib.sha256((root / csv_path).read_bytes()).hexdigest()
    return {
        "schema": "mediflow_adaptation_numeric_review_v1",
        "as_of": "2026-11-12",
        "profiles": ledger["profiles"],
        "training_runs": ledger["runs"],
        "parameter_rows": parameters,
        "device_gaps": gaps,
        "distributions": distributions,
        "features": features,
        "source_test": tests,
        "webcam_phone": cross_camera,
        "missing_measurements": [
            "기존 장비 1차 단독 영상 W1 미측정",
            "기존 장비 4차 모델 MMD/CORAL 미측정",
            "추가5차 분포 요약 분위수는 이 표에 재계산하지 않음",
            "실제 장비 질환 정답 성능 미측정",
        ],
        "interpretation": "W1 reduction is not image similarity percentage or disease accuracy",
        "new_training_or_inference_executed": False,
        "source_files_sha256": provenance,
        "stage2_reference_identity_verified": True,
        "stage2_local_bundle_profile_checked": references is not None,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }


def write_report_data(root, output, *, overwrite=False):
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError("Use a new output folder; original results are preserved")
    result = collect(root)
    output.mkdir(parents=True, exist_ok=overwrite)
    mode = "w" if overwrite else "x"
    with (output / "numeric_review.json").open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    for name in ("device_gaps", "distributions", "features", "source_test", "webcam_phone"):
        rows = result[name]
        with (output / (name + ".csv")).open(mode, encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = write_report_data(args.root, args.output, overwrite=args.overwrite)
    print(
        {
            name: len(result[name])
            for name in ("profiles", "training_runs", "device_gaps", "features", "source_test")
        }
    )


if __name__ == "__main__":
    main()
