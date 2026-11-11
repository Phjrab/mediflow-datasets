"""Find a bounded common-or-nearer Stage2 profile using Train/calibration replay only."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from mediflow_datasets import camera_similarity as distance
from mediflow_datasets import smartphone_profile as phone
from mediflow_datasets.domain_adaptation_data import read_json, sha256

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "camera_capture/_review/smartphone_web_skin_20261111"
POLICY = {
    "source": "118 retained stratified original Train previews; 104 measured face ROIs",
    "target": "P01-P04 unlabeled calibration only; P05 holdout excluded",
    "changed_variable": "Stage2 additional appearance profile; Stage1 lighting fixed",
    "interpretation": "Nearest bounded approximation, not a common causal camera correction",
    "probe_per_class": 4,
    "probe_seed": 42,
    "full_replay_seeds": [42, 707, 1707],
    "minimum_improved_people": 3,
    "minimum_improved_omissions": 3,
    "max_person_relative_cost_increase": 0.05,
    "max_guard_relative_wasserstein_increase": 0.02,
    "spatial_correlation_q05_minimum": 0.95,
    "color_budget": "median per-person q90-q10, capped by the existing Lab limit",
    "texture_budget": "half of median within-person relative normalized-detail span, cap 0.15",
    "blur_grid_roi64": [0.05, 0.1, 0.2],
    "limits_meaning": "Declared engineering search guards, not validated clinical thresholds",
    "test_or_classifier_score_used_for_profile_selection": False,
}


def normalize_target(rows):
    return [{**r, "group": r["person"]} for r in rows]


def distances(source, target, keys):
    values = {key: distance.distribution_gap(source, target, key) for key in keys}
    if any(v is None or not np.isfinite(v) for v in values.values()):
        raise ValueError("Comparable finite image measurements are required")
    return values


def group_summary(rows, key, field):
    groups = phone.values_by_group(rows, key, field)
    medians = {str(group): float(np.median(values)) for group, values in groups.items()}
    return {
        "equal_group_median_of_medians": float(np.median(list(medians.values()))),
        "group_medians": medians,
        "pooled_equal_group_distribution": phone.distribution(rows, key, field),
    }


def camera_comparison(root, phone_packet):
    """Compare existing measured webcam frames with smartphone calibration, never P05."""
    path = root / "presentation/report_assets/camera_similarity_check_20261110/measurements.json"
    webcam = read_json(path)["web_skin"]["groups"]["target"]
    webcam = [
        {
            **row,
            "person": row["group"].split("/")[0],
            "metrics": phone.revision.add_normalized_detail(row["metrics"]),
        }
        for row in webcam
    ]
    keys = (
        "y_p50",
        "contrast",
        "lab_a",
        "lab_b",
        "detail",
        "normalized_detail",
        "side_light",
        "lighting_range",
        "face_w",
    )
    rows = {}
    for key in keys:
        source = group_summary(phone_packet["source"], key, "label")
        web = group_summary(webcam, key, "person")
        mobile = group_summary(phone_packet["calibration"], key, "person")
        rows[key] = {
            "source_train": source,
            "webcam": web,
            "smartphone": mobile,
            "phone_minus_webcam": mobile["equal_group_median_of_medians"]
            - web["equal_group_median_of_medians"],
        }
    return {
        "statistic": "Median per class/person, then median of equally weighted groups",
        "webcam_input": {
            "file": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "images": len(webcam),
            "people": 4,
        },
        "smartphone_input": {"images": 42, "people": 4, "holdout_excluded": 10},
        "scope": "Stored 256px center-square inputs, fixed cheek ROI metrics",
        "limitations": [
            "Phone EXIF/Display-P3 converted to upright sRGB; webcam input pipeline retained",
            "Person, capture, framing and device are confounded; not a paired camera test",
            "Detail is a cheek image statistic, not sensor resolution or disease accuracy",
        ],
        "metrics": rows,
    }


def assess(before, after, target, keys, correlation, guards):
    """Require aggregate and most-person/omission proximity, not a uniform median direction."""
    phone.validate_calibration(before, target)
    b, a = distances(before, target, keys), distances(after, target, keys)

    def cost(rows, subset):
        return float(
            np.mean([distance.distribution_gap(rows, subset, k) / max(b[k], 1e-8) for k in keys])
        )

    people = sorted({r["person"] for r in target})
    per_person, omissions = {}, {}
    for person in people:
        subset = [r for r in target if r["person"] == person]
        old, new = cost(before, subset), cost(after, subset)
        per_person[person] = {
            "before": old,
            "after": new,
            "relative_change": (new - old) / max(old, 1e-8),
        }
        subset = [r for r in target if r["person"] != person]
        omissions[person] = {"before": cost(before, subset), "after": cost(after, subset)}
    improvements = sum(v["after"] < v["before"] - 1e-8 for v in per_person.values())
    improved_omissions = sum(v["after"] < v["before"] - 1e-8 for v in omissions.values())
    guard_before = distances(before, target, guards)
    guard_after = distances(after, target, guards)
    guard_pass = all(guard_after[k] <= guard_before[k] * 1.02 + 1e-8 for k in guards)
    each_metric_ok = all(a[k] <= b[k] + 1e-8 for k in keys)
    ratio = float(np.mean([a[k] / max(b[k], 1e-8) for k in keys]))
    accepted = (
        ratio < 1 - 1e-6
        and each_metric_ok
        and guard_pass
        and correlation >= 0.95
        and improvements >= 3
        and improved_omissions >= 3
        and max(v["relative_change"] for v in per_person.values()) <= 0.05
    )
    return {
        "accepted": bool(accepted),
        "relative_cost": ratio,
        "wasserstein_before": b,
        "wasserstein_after": a,
        "person_improvements": improvements,
        "per_person": per_person,
        "omission_improvements": improved_omissions,
        "leave_one_person_out": omissions,
        "guard_before": guard_before,
        "guard_after": guard_after,
        "guard_passed": guard_pass,
        "spatial_correlation_q05": float(correlation),
    }


def candidate_ranges(target, key):
    if key in ("color_a", "color_b"):
        metric = "lab_a" if key == "color_a" else "lab_b"
        groups = phone.values_by_group(target, metric, "person")
        budget = min(
            phone.LIMITS["lab_offset"],
            float(np.median([np.quantile(v, 0.9) - np.quantile(v, 0.1) for v in groups.values()])),
        )
        candidates = [
            sorted([0.0, sign * budget * f]) for sign in (-1, 1) for f in (0.25, 0.5, 1.0)
        ]
        return candidates, {"measured_within_person_q90_q10_median": budget}
    if key == "texture_gain":
        groups = phone.values_by_group(target, "normalized_detail", "person")
        span = float(
            np.median(
                [
                    (np.quantile(v, 0.9) - np.quantile(v, 0.1)) / max(np.median(v), 1e-8)
                    for v in groups.values()
                ]
            )
        )
        budget = min(0.15, 0.5 * span)
        candidates = []
        for f in (0.25, 0.5, 1.0):
            delta = budget * f
            candidates.extend(([1 - delta, 1.0], [1.0, 1 + delta], [1 - delta, 1 + delta]))
        return candidates, {"measured_relative_detail_span": span, "gain_delta_cap": budget}
    if key == "blur_sigma_roi64":
        return [[0.0, v] for v in POLICY["blur_grid_roi64"]], {
            "range": "Small subgrid of the already examined focus limits; require both detail gaps"
        }
    raise ValueError("Unmeasured variable is not eligible")


def review(root, output):
    root, output = Path(root), Path(output)
    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=False)
    private = root / PRIVATE.relative_to(ROOT)
    packet = read_json(private / "measurements.json")
    old_profiles = read_json(private / "profiles.json")
    target = normalize_target(packet["calibration"])
    images, provenance = phone.source_previews(root)
    source_by_path = {r["path"]: r for r in packet["source"]}
    source = [source_by_path[r["path"]] for r in provenance]
    phone.validate_calibration(source, target)
    detector = phone.analysis.face_detector()
    chosen = deepcopy(old_profiles["stage1"])
    valid = [i for i, r in enumerate(source) if "y_p50" in r["metrics"]]
    probe = [i for label in range(5) for i in [j for j in valid if source[j]["label"] == label][:4]]
    probe_images, probe_source = [images[i] for i in probe], [source[i] for i in probe]
    records = []
    baseline, _ = phone.replay(images, source, chosen, detector)
    current = baseline
    for key in ("color_a", "color_b", "texture_gain", "blur_sigma_roi64"):
        print(f"[휴대폰 2차 재검토] {key}: 관찰 범위의 작은 후보 재생", flush=True)
        keys = (
            ("lab_a",)
            if key == "color_a"
            else ("lab_b",)
            if key == "color_b"
            else ("detail", "normalized_detail")
        )
        guards = tuple(k for k in ("y_p50", "contrast", "lab_a", "lab_b") if k not in keys)
        candidates, budget = candidate_ranges(target, key)
        before_probe, _ = phone.replay(probe_images, probe_source, chosen, detector, seeds=(42,))
        screened = []
        for bounds in candidates:
            candidate = deepcopy(chosen)
            candidate[key] = bounds
            after, corr = phone.replay(probe_images, probe_source, candidate, detector, seeds=(42,))
            screened.append(
                {
                    "range": bounds,
                    "assessment": assess(before_probe, after, target, keys, corr, guards),
                }
            )
        # Probe is a speed screen only. The final decision always uses every retained preview.
        ranked = sorted(screened, key=lambda r: r["assessment"]["relative_cost"])
        full = []
        best = None
        for item in ranked[:2]:
            candidate = deepcopy(chosen)
            candidate[key] = item["range"]
            after, corr = phone.replay(images, source, candidate, detector)
            check = assess(current, after, target, keys, corr, guards)
            full.append({"range": item["range"], "assessment": check})
            if check["accepted"] and (best is None or check["relative_cost"] < best[0]):
                best = (check["relative_cost"], candidate, after)
        if best:
            _, chosen, current = best
        print(f"[휴대폰 2차 재검토] {key}: 채택 범위 {chosen[key]}", flush=True)
        records.append(
            {
                "variable": key,
                "budget_evidence": budget,
                "probe": screened,
                "full_replays": full,
                "accepted_range": chosen[key],
            }
        )
    if chosen["lighting"] != old_profiles["stage1"]["lighting"]:
        raise ValueError("Stage1 lighting must remain fixed")
    profiles = {**deepcopy(old_profiles), "stage2": chosen}
    result = {
        "schema": "mediflow_phone_stage2_common_nearer_review_v1",
        "status": "measured",
        "policy": POLICY,
        "profile_source_sha256": sha256(private / "profiles.json"),
        "measurement_source_sha256": sha256(private / "measurements.json"),
        "source_manifest_sha256": sha256(
            root / "results/camera_adaptation_stage1_20261110/web_skin/source_manifest.json"
        ),
        "profiles": profiles,
        "selected_stage2_differs_from_stage1": chosen != old_profiles["stage1"],
        "decisions": records,
        "source_test_used": False,
        "target_holdout_used": False,
        "target_accuracy": None,
        "training_executed": False,
        "summary": {
            k: {
                "stage1_w1": distance.distribution_gap(baseline, target, k),
                "stage2_w1": distance.distribution_gap(current, target, k),
            }
            for k in phone.KEYS
        },
        "code_sha256": sha256(Path(__file__)),
    }
    phone.write_new(output / "review.json", result)
    phone.write_new(output / "stage2_measurements.json", current)
    phone.write_new(output / "camera_comparison.json", camera_comparison(root, packet))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review(ROOT, args.output)


if __name__ == "__main__":
    main()
