"""Review jointly measured smartphone appearance; never modify a training release."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from mediflow_datasets import camera_extended as camera
from mediflow_datasets import camera_similarity as distance
from mediflow_datasets import smartphone_profile as phone
from mediflow_datasets import smartphone_stage2_review as previous
from mediflow_datasets.domain_adaptation_data import ImageRow, read_json, sha256
from mediflow_datasets.domain_adaptation_experiment import item_seed

ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    "y_p50",
    "contrast",
    "lab_a",
    "lab_b",
    "detail",
    "normalized_detail",
    "side_light",
    "lighting_range",
)
POLICY = {
    "source": "118 retained original Train previews; 104 fixed face ROIs",
    "target": "P01-P04 calibration only, 42 images; no P05 or classifier results",
    "metrics": list(METRICS),
    "reference": "Untransformed Source Train, equal class/person weights",
    "lighting_fractions": [0.25, 0.5, 0.75, 1.0],
    "probe_per_class": 4,
    "probe_seed": 42,
    "full_replay_seeds": [42, 707, 1707],
    "beam_width": 4,
    "full_shortlist": 8,
    "screen_rank": "Mean per-metric relative gaps, then worst; not a similarity percentage",
    "acceptance": "All eight gaps nonincreasing vs raw Source, at least one decreasing",
    "per_seed_guard": "All eight gaps nonincreasing for each of three replay seeds",
    "minimum_improved_people": 3,
    "minimum_improved_omissions": 3,
    "max_person_relative_cost_increase": 0.05,
    "spatial_correlation_q05_minimum": 0.95,
    "new_float_endpoint_fraction_max": 0.0,
    "geometry": "Identity only; framing/person/disease are not camera calibration",
    "limits_meaning": "Declared engineering guards, not clinical preservation thresholds",
    "training_or_model_selection_performed": False,
}


def profile_id(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:16]


def gaps(rows, target):
    result = {k: distance.distribution_gap(rows, target, k) for k in METRICS}
    if any(v is None or not np.isfinite(v) for v in result.values()):
        raise ValueError("All joint metrics must have finite comparable observations")
    return result


def relative_gaps(reference, candidate):
    return {k: candidate[k] / max(reference[k], 1e-8) for k in METRICS}


def joint_gate(reference, candidate):
    return all(candidate[k] <= reference[k] + 1e-8 for k in METRICS) and any(
        candidate[k] < reference[k] - 1e-8 for k in METRICS
    )


def rank(reference, candidate):
    ratios = list(relative_gaps(reference, candidate).values())
    return float(np.mean(ratios)), float(max(ratios))


def lighting_candidates(study, source, target):
    phone.validate_calibration(source, target)
    identity = deepcopy(study["profiles"]["stage1"])
    identity["lighting"].update(
        brightness_offset=[0.0, 0.0], contrast=[1.0, 1.0], left_light=[0.0, 0.0]
    )
    result = [("identity", identity)]
    y = phone.anchor(source, "y_p50", "label")
    desired = np.quantile(
        [np.median(v) for v in phone.values_by_group(target, "y_p50", "person").values()],
        [0.25, 0.75],
    )
    for fraction in POLICY["lighting_fractions"]:
        light = phone.lighting_profile(source, target, fraction)
        concentrated = sorted(
            float(np.clip(phone.inverse_offset(y, v) * fraction, -96, 96)) for v in desired
        )
        for brightness, bounds in (
            ("zero_inclusive", light["brightness_offset"]),
            ("target_quartiles", concentrated),
        ):
            for contrast in ("interval", "measured_end"):
                p = deepcopy(identity)
                p["lighting"] = deepcopy(light)
                p["lighting"]["brightness_offset"] = list(bounds)
                if contrast == "measured_end":
                    value = max(light["contrast"], key=lambda v: abs(v - 1))
                    p["lighting"]["contrast"] = [value, value]
                result.append((f"light_{fraction}_{brightness}_{contrast}", p))
    return result


def replay(images, rows, profile, detector, seeds):
    by_seed = {seed: [] for seed in seeds}
    aggregate, correlations = [], []
    endpoint = 0.0
    for image, row in zip(images, rows, strict=True):
        metrics = []
        for seed in seeds:
            transformed = camera.transform(
                image,
                profile,
                "expanded",
                item_seed(ImageRow(row["path"], "train", row["label"]), seed),
            )
            measured = phone.revision.fast_measure(transformed, "web_skin", detector, row["box"])
            metrics.append(measured)
            by_seed[seed].append({**row, "metrics": measured})
            correlations.append(
                float(np.corrcoef(image.mean(-1).ravel(), transformed.mean(-1).ravel())[0, 1])
            )
            endpoint = max(
                endpoint,
                float(
                    np.mean(
                        ((transformed <= 0) | (transformed >= 255)) & (image > 0) & (image < 255)
                    )
                ),
            )
        aggregate.append({**row, "metrics": distance.median_repetitions(metrics)})
    return aggregate, by_seed, float(np.nanquantile(correlations, 0.05)), endpoint


def assess(source, after, target, per_seed, correlation, endpoint):
    phone.validate_calibration(source, target)
    before, changed = gaps(source, target), gaps(after, target)
    ratios = relative_gaps(before, changed)
    people, omissions = {}, {}

    def cost(rows, subset):
        return float(np.mean(list(relative_gaps(before, gaps(rows, subset)).values())))

    for person in sorted({r["person"] for r in target}):
        subset = [r for r in target if r["person"] == person]
        b, a = cost(source, subset), cost(after, subset)
        people[person] = {"before": b, "after": a, "relative_change": (a - b) / max(b, 1e-8)}
        subset = [r for r in target if r["person"] != person]
        omissions[person] = {"before": cost(source, subset), "after": cost(after, subset)}
    seed_gaps = {str(s): gaps(rows, target) for s, rows in per_seed.items()}
    seed_pass = all(all(g[k] <= before[k] + 1e-8 for k in METRICS) for g in seed_gaps.values())
    improved_people = sum(p["after"] < p["before"] - 1e-8 for p in people.values())
    improved_omissions = sum(p["after"] < p["before"] - 1e-8 for p in omissions.values())
    accepted = (
        joint_gate(before, changed)
        and seed_pass
        and correlation >= 0.95
        and endpoint == 0
        and improved_people >= 3
        and improved_omissions >= 3
        and max(p["relative_change"] for p in people.values()) <= 0.05
    )
    return {
        "accepted": bool(accepted),
        "gaps": changed,
        "relative_gaps": ratios,
        "mean_relative_gap": float(np.mean(list(ratios.values()))),
        "worst_relative_gap": float(max(ratios.values())),
        "nonworsening_metrics": [k for k in METRICS if changed[k] <= before[k] + 1e-8],
        "worsening_metrics": [k for k in METRICS if changed[k] > before[k] + 1e-8],
        "spatial_correlation_q05": correlation,
        "new_float_endpoint_max": endpoint,
        "per_seed_gaps": seed_gaps,
        "each_seed_all_metrics_nonworsening": bool(seed_pass),
        "improved_people": improved_people,
        "improved_omissions": improved_omissions,
        "people": people,
        "leave_one_person_out": omissions,
    }


def review(root, output):
    root, output = Path(root), Path(output)
    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=False)
    private = root / "camera_capture/_review/smartphone_web_skin_20261111"
    packet = read_json(private / "measurements.json")
    study_path = root / "configs/smartphone_web_skin_20261111.json"
    study = read_json(study_path)
    images, provenance = phone.source_previews(root)
    by_path = {r["path"]: r for r in packet["source"]}
    source = [by_path[r["path"]] for r in provenance]
    target = previous.normalize_target(packet["calibration"])
    phone.validate_calibration(source, target)
    detector = phone.analysis.face_detector()
    valid = [i for i, r in enumerate(source) if "y_p50" in r["metrics"]]
    indices = [
        i for label in range(5) for i in [j for j in valid if source[j]["label"] == label][:4]
    ]
    probe_images, probe_rows = [images[i] for i in indices], [source[i] for i in indices]
    probe_reference = gaps(probe_rows, target)
    candidates, screened, screen_cache = {}, [], {}

    def register(label, p):
        pid = profile_id(p)
        candidates.setdefault(pid, {"id": pid, "label": label, "profile": deepcopy(p)})
        if pid not in screen_cache:
            rows, _, corr, endpoint = replay(probe_images, probe_rows, p, detector, (42,))
            measured = gaps(rows, target)
            entry = {
                "id": pid,
                "label": label,
                "gaps": measured,
                "mean_relative_gap": rank(probe_reference, measured)[0],
                "worst_relative_gap": rank(probe_reference, measured)[1],
                "spatial_correlation_q05": corr,
                "new_float_endpoint_max": endpoint,
            }
            screen_cache[pid] = entry
            screened.append(entry)
        return pid

    light = [register(label, p) for label, p in lighting_candidates(study, source, target)]
    controls = {"original": light[0]}
    for stage in ("stage1", "stage2", "stage4"):
        controls[stage] = register(stage, study["profiles"][stage])
    near = read_json(root / "output/smartphone_web_skin_20261111/stage2_review.json")
    controls["prior_color_only"] = register("prior_color_only", near["profiles"]["stage2"])

    def beam(ids):
        unique = list(dict.fromkeys(ids))
        return sorted(
            unique,
            key=lambda i: (
                screen_cache[i]["mean_relative_gap"],
                screen_cache[i]["worst_relative_gap"],
            ),
        )[:4]

    print(f"[공동 검토] 조명 후보 {len(light)}개 점검 완료", flush=True)
    ac, ae = previous.candidate_ranges(target, "color_a")
    bc, be = previous.candidate_ranges(target, "color_b")

    # Both signed color directions were already examined in the prior measured review.
    def half_full(values):
        return [v for i, v in enumerate(values) if i % 3 in (1, 2)] + [[0.0, 0.0]]

    color_ids = []
    for pid in beam(light):
        for a in half_full(ac):
            for b in half_full(bc):
                p = deepcopy(candidates[pid]["profile"])
                p.update(color_a=a, color_b=b)
                color_ids.append(register(f"joint_color_{pid}", p))
    print(f"[공동 검토] 조명·색 조합 {len(set(color_ids))}개 점검 완료", flush=True)
    texture_options, te = previous.candidate_ranges(target, "texture_gain")
    texture_options = [[1.0, 1.0]] + [v for i, v in enumerate(texture_options) if i // 3 in (1, 2)]
    all_ids = [*light, *color_ids]
    for pid in beam(color_ids):
        for gain in texture_options:
            for sigma in (0.0, 0.2, 0.4):
                p = deepcopy(candidates[pid]["profile"])
                p.update(texture_gain=gain, blur_sigma_roi64=[0.0, sigma])
                all_ids.append(register(f"joint_detail_{pid}", p))
    print(f"[공동 검토] 총 {len(candidates)}개 고유 후보 점검 완료", flush=True)
    ranked = sorted(
        set(all_ids),
        key=lambda i: (screen_cache[i]["worst_relative_gap"], screen_cache[i]["mean_relative_gap"]),
    )
    by_mean = sorted(
        set(all_ids),
        key=lambda i: (screen_cache[i]["mean_relative_gap"], screen_cache[i]["worst_relative_gap"]),
    )
    shortlist = list(dict.fromkeys([*ranked[:4], *by_mean[:4], *controls.values()]))
    full, measurements = [], {}
    for pos, pid in enumerate(shortlist, 1):
        rows, seeds, corr, endpoint = replay(
            images, source, candidates[pid]["profile"], detector, (42, 707, 1707)
        )
        assessment = assess(source, rows, target, seeds, corr, endpoint)
        full.append({**candidates[pid], "assessment": assessment})
        measurements[pid] = {"median_replay_rows": rows, "per_seed": seeds}
        print(
            f"[공동 검토] 전체 표본 {pos}/{len(shortlist)} · "
            f"동시 비악화 {len(assessment['nonworsening_metrics'])}/8 · "
            f"채택 {assessment['accepted']}",
            flush=True,
        )
    accepted = [c for c in full if c["assessment"]["accepted"]]
    chosen = (
        min(
            accepted,
            key=lambda c: (
                c["assessment"]["worst_relative_gap"],
                c["assessment"]["mean_relative_gap"],
            ),
        )
        if accepted
        else None
    )
    balanced = min(
        (c for c in full if c["id"] != controls["original"]),
        key=lambda c: (c["assessment"]["worst_relative_gap"], c["assessment"]["mean_relative_gap"]),
    )
    result = {
        "schema": "mediflow_smartphone_joint_similarity_review_v1",
        "status": "measured_review_only",
        "policy": POLICY,
        "baseline_gaps": gaps(source, target),
        "unique_screened_candidates": len(candidates),
        "full_replayed_candidates": len(full),
        "source_count": len(source),
        "source_roi_count": len(valid),
        "calibration_count": len(target),
        "source_test_used": False,
        "source_validation_used": False,
        "target_holdout_used": False,
        "training_executed": False,
        "classifier_scores_used": False,
        "runtime_configuration_modified": False,
        "target_accuracy": None,
        "chosen": chosen,
        "best_balanced_in_review": balanced,
        "controls": controls,
        "full_replays": full,
        "screened": screened,
        "measured_appearance_budgets": {"color_a": ae, "color_b": be, "texture_gain": te},
        "input_provenance": {
            "study_sha256": sha256(study_path),
            "measurement_sha256": sha256(private / "measurements.json"),
            "source_manifest_sha256": study["source_manifest_sha256"],
            "preview_provenance": provenance,
            "analysis_code_sha256": sha256(Path(__file__)),
        },
        "limitations": [
            "Bounded beam search, not proof of a global optimum or impossibility",
            "Four people/phones confound device and person; unlabeled healthy/disease mix differs",
            "No geometry/content/noise/compression matching; no clinical preservation proof",
            "Three replay seeds are repeated transforms, not independent people or training seeds",
            "No refit on P05, Source Validation/Test or classifier accuracy",
        ],
    }
    phone.write_new(output / "review.json", result)
    phone.write_new(
        output / "measurements.json",
        {"source": source, "calibration": target, "candidates": measurements},
    )
    phone.write_new(output / "candidate_profiles.json", candidates)
    print(
        json.dumps(
            {
                "accepted": chosen["id"] if chosen else None,
                "balanced": balanced["id"],
                "candidates": len(candidates),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review(ROOT, args.output)


if __name__ == "__main__":
    main()
