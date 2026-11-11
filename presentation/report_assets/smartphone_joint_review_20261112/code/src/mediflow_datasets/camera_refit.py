"""Train-only, pixel-replayed photometric refit to unlabeled pilot camera captures."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mediflow_datasets import camera_extended as camera
from mediflow_datasets import camera_revision as revision
from mediflow_datasets import camera_similarity as audit
from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp

POLICY = {
    "brightness_offset_limit": 160.0,
    "contrast_limits": [0.25, 1.5],
    "side_light_limit": 64.0,
    "probe_per_class": 4,
    "full_replay_finalists": 3,
    "spatial_correlation_q05_minimum": 0.95,
    "chroma_limit_lab": 3.0,
    "color_budget": "Observed absolute within-person condition change q75; zero-inclusive",
    "hair_contrast": "Identity: scalp coverage and hair content confound global contrast",
    "selection": "Minimize worst remaining primary W1 fraction, then mean fraction",
    "limits_interpretation": "Engineering search limits, not clinically validated thresholds",
}
BASE = "presentation/report_assets/revised_pretraining_check_20261111/checked_inputs"
INPUT_HASH = "492ee01afb4579b6402fca29459c59d56618243589642fc022db18bbbe45a165"


def validate_rows(source, target):
    if not source or any(r.get("split") != "train" or r.get("label") is None for r in source):
        raise ValueError("Refit may only use labeled retained Source Train")
    if not target or any(
        r.get("split") != "calibration" or r.get("label") is not None for r in target
    ):
        raise ValueError("Refit Target must be unlabeled calibration, never Test")
    groups = {r["group"] for r in target}
    people = {g.split("/")[0] for g in groups}
    if len(people) != 4 or groups != {f"{p}/{c}" for p in people for c in ("ref", "alt")}:
        raise ValueError("Four participants with both capture conditions are required")


def anchor(rows, key, field):
    groups = defaultdict(list)
    for row in rows:
        if key in row["metrics"]:
            groups[row[field]].append(row["metrics"][key])
    if not groups:
        raise ValueError(f"Missing comparable metric: {key}")
    return np.array([np.median(values) for _, values in sorted(groups.items())])


def inverse_offset(source, desired):
    """Invert the existing smooth-boundary operator at a scalar luminance anchor."""
    if not 0 < source < 255 or not 0 < desired < 255:
        raise ValueError("Interior luminance anchors are required")
    if desired >= source:
        return float(-(255 - source) * np.log((255 - desired) / (255 - source)))
    return float(source * np.log(desired / source))


def lighting_candidates(profile, source, target):
    validate_rows(source, target)
    y = float(np.median(anchor(source, "y_p50", "label")))
    target_y = np.quantile(anchor(target, "y_p50", "group"), [0.25, 0.75])
    offset = np.clip(
        [inverse_offset(y, desired) for desired in target_y],
        -POLICY["brightness_offset_limit"],
        POLICY["brightness_offset_limit"],
    )
    # The grid brackets an observed-target estimate; it is not a global intensity multiplier.
    offsets = [profile["lighting"]["brightness_offset"]] + [
        (offset * fraction).clip(-160, 160).tolist() for fraction in (0.75, 1.0, 1.25)
    ]
    source_contrast = float(np.median(anchor(source, "contrast", "label")))
    if source_contrast <= 0:
        raise ValueError("Positive Source contrast anchor is required")
    ratio = np.quantile(anchor(target, "contrast", "group"), [0.25, 0.75]) / source_contrast
    contrasts = [profile["lighting"]["contrast"], [1.0, 1.0]]
    if profile["domain"] != "hair":
        contrasts += [np.clip(ratio * f, 0.25, 1.5).tolist() for f in (0.75, 1.0, 1.25)]
    lateral = [profile["lighting"]["left_light"]]
    if profile["domain"] == "web_skin":
        face_width = float(np.median(anchor(source, "face_w", "label")))
        observed = (
            np.quantile(anchor(target, "side_light", "group"), [0.25, 0.75])
            - np.median(anchor(source, "side_light", "label"))
        ) / max(0.42 * face_width, 0.1)
        lateral += [np.clip(observed, -64, 64).tolist()]
    candidates, keys = [], set()
    for b, c, side in itertools.product(offsets, contrasts, lateral):
        candidate = deepcopy(profile)
        candidate["lighting"].update(brightness_offset=b, contrast=c, left_light=side)
        key = json.dumps([b, c, side], sort_keys=True)
        if key not in keys:
            keys.add(key)
            candidates.append(candidate)
    return candidates, {
        "source_class_balanced_luminance_anchor": y,
        "target_equal_group_luminance_iqr": target_y.tolist(),
        "smooth_operator_inverse_offsets": offset.tolist(),
        "target_to_source_contrast_iqr_ratio": ratio.tolist(),
        "candidate_count": len(candidates),
    }


def spatial_correlation(original, transformed):
    a, b = (
        cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (64, 64)).ravel().astype(np.float64)
        for x in (original, transformed)
    )
    a, b = a - a.mean(), b - b.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / norm) if norm > 1e-12 else float(np.linalg.norm(a - b) < 1e-9)


def evaluate(images, rows, profile, domain, detector, seeds):
    per_seed = {seed: [] for seed in seeds}
    merged, correlations, endpoints = [], [], []
    for image, row in zip(images, rows, strict=True):
        box = audit.box_from_metrics(row["metrics"], len(image))
        repetitions = []
        for seed in seeds:
            item = data.ImageRow(row["path"], "train", row["label"])
            result = camera.transform(image, profile, "expanded", exp.item_seed(item, seed))
            metrics = revision.fast_measure(result, domain, detector, box)
            repetitions.append(metrics)
            per_seed[seed].append({**row, "metrics": metrics})
            correlations.append(spatial_correlation(image, result))
            endpoints.append(
                float(np.mean(((result <= 0) | (result >= 255)) & (image > 0) & (image < 255)))
            )
        merged.append({**row, "metrics": audit.median_repetitions(repetitions)})
    return {
        "rows": merged,
        "per_seed": per_seed,
        "spatial_correlation_q05": float(np.quantile(correlations, 0.05)),
        "new_float_endpoint_max": max(endpoints),
    }


def gap_score(candidate, baseline, target, keys):
    ratios = []
    gaps = {}
    for key in keys:
        before = audit.distribution_gap(baseline, target, key)
        after = audit.distribution_gap(candidate, target, key)
        if before is None or after is None:
            raise ValueError("Fixed evaluation metrics are missing")
        gaps[key] = {"before": before, "after": after}
        ratios.append(after / max(before, 1e-9))
    return (max(ratios), float(np.mean(ratios))), gaps


def primary_keys(domain):
    return (
        ("y_p50",)
        if domain == "hair"
        else (
            ("y_p50", "contrast", "side_light") if domain == "web_skin" else ("y_p50", "contrast")
        )
    )


def load_domain(root, domain, study, detector):
    old_audit = root / "presentation/report_assets/camera_similarity_check_20261110"
    previous = data.read_json(old_audit / "summary.json")["domains"][domain]
    entry = study["domains"][domain]
    manifest_path = (
        root / f"results/camera_adaptation_stage1_20261110/{domain}/source_manifest.json"
    )
    if data.sha256(manifest_path) != entry["source_manifest_sha256"]:
        raise ValueError("Retained Source split changed")
    retained = {r["path"]: r["label"] for r in data.read_json(manifest_path)["original"]["train"]}
    old_measurements = data.read_json(old_audit / "measurements.json")[domain]
    originals = old_measurements["groups"]["original"]
    paths = {r["path"]: r for r in previous["source_png_provenance"]}
    images, rows = [], []
    for row in originals:
        if retained.get(row["path"]) != row["label"]:
            raise ValueError("Source preview is not retained Train")
        path = root / paths[row["path"]]["preview"]
        if data.sha256(path) != paths[row["path"]]["sha256"]:
            raise ValueError("Original Train QA PNG changed")
        with Image.open(path) as photo:
            image = np.asarray(photo.convert("RGB"), dtype=np.float32)
        box = audit.box_from_metrics(row["metrics"], len(image))
        metrics = revision.fast_measure(image, domain, detector, box)
        images.append(image)
        rows.append({**row, "split": "train", "metrics": metrics})
    packet_path = root / entry["measurement_provenance"]["target"]
    if data.sha256(packet_path) != entry["measurement_provenance"]["target_sha256"]:
        raise ValueError("Target calibration packet changed")
    packet = data.read_json(packet_path)
    image_rows = [data.ImageRow(**r) for r in packet["rows"]]
    if runner.target_fingerprint(root / "camera_capture", image_rows) != packet["fingerprint"]:
        raise ValueError("Raw camera captures changed")
    cache = root / f"camera_capture/_review/nine_experiments_20261109/{domain}"
    meta = data.read_json(cache / "target_inputs.json")
    if (
        meta["rows"] != packet["rows"]
        or data.sha256(cache / "target_inputs.npz") != meta["array_sha256"]
    ):
        raise ValueError("Target model-input cache changed")
    with np.load(cache / "target_inputs.npz", allow_pickle=False) as archive:
        cached = archive["images"]
    targets, target_images = [], []
    for index, row in enumerate(image_rows):
        path = root / "camera_capture" / row.path
        original = exp.image_tensor(path.read_bytes(), meta["input_size"], meta["input_policy"])
        np.testing.assert_array_equal(original, cached[index])
        old = next(r for r in old_measurements["groups"]["target"] if r["path"] == row.path)
        box = audit.box_from_metrics(old["metrics"], len(original))
        metrics = revision.fast_measure(original, domain, detector, box)
        targets.append(
            {
                "path": row.path,
                "group": f"{row.person}/{row.condition}",
                "split": row.split,
                "label": row.label,
                "metrics": metrics,
            }
        )
        target_images.append(original)
    validate_rows(rows, targets)
    return images, rows, target_images, targets, previous


def refit_domain(images, source, target, profile, domain, detector):
    valid = [i for i, row in enumerate(source) if "y_p50" in row["metrics"]]
    probe = []
    for label in sorted({r["label"] for r in source}):
        choices = [i for i in valid if source[i]["label"] == label]
        choices.sort(key=lambda i: hashlib.sha256(source[i]["path"].encode()).hexdigest())
        if not choices:
            raise ValueError("Every class needs measured Train previews")
        probe += choices[: POLICY["probe_per_class"]]
    small_images, small_rows = [images[i] for i in probe], [source[i] for i in probe]
    keys = primary_keys(domain)
    baseline = evaluate(images, source, profile, domain, detector, audit.SEEDS)
    probe_base = evaluate(small_images, small_rows, profile, domain, detector, (42,))
    candidates, proposal = lighting_candidates(profile, source, target)
    scores = []
    for index, candidate in enumerate(candidates):
        measured = evaluate(small_images, small_rows, candidate, domain, detector, (42,))
        score, gaps = gap_score(measured["rows"], probe_base["rows"], target, keys)
        safe = measured["spatial_correlation_q05"] >= POLICY["spatial_correlation_q05_minimum"]
        scores.append({"index": index, "score": score, "primary_gaps": gaps, "safe": safe})
        if index % 5 == 0 or index + 1 == len(candidates):
            print(f"[{domain}] 조명 후보 {index + 1}/{len(candidates)} 대조", flush=True)
    finalists = sorted((r for r in scores if r["safe"]), key=lambda r: r["score"])[
        : POLICY["full_replay_finalists"]
    ]
    chosen_profile, chosen_result = deepcopy(profile), baseline
    best = (1.0, 1.0)
    full = []
    for item in finalists:
        candidate = candidates[item["index"]]
        measured = evaluate(images, source, candidate, domain, detector, audit.SEEDS)
        score, gaps = gap_score(measured["rows"], baseline["rows"], target, keys)
        stable = all(
            gap_score(measured["per_seed"][s], baseline["per_seed"][s], target, keys)[0][0]
            <= 1 + 1e-9
            for s in audit.SEEDS
        )
        safe = measured["spatial_correlation_q05"] >= POLICY["spatial_correlation_q05_minimum"]
        full.append(
            {**item, "full_score": score, "full_gaps": gaps, "stable": stable, "safe": safe}
        )
        if stable and safe and score < best:
            best, chosen_profile, chosen_result = score, deepcopy(candidate), measured

    # Re-evaluate residual color/focus after fitting light; do not inherit obsolete directions.
    neutral = deepcopy(chosen_profile)
    neutral.update(color_a=[0, 0], color_b=[0, 0], texture_gain=[1, 1], blur_sigma_roi64=[0, 0])
    light_result = evaluate(images, source, neutral, domain, detector, audit.SEEDS)
    color_base = deepcopy(neutral)
    direction = {}
    for axis in ("a", "b"):
        bounds, evidence = revision.color_direction(
            light_result["rows"],
            target,
            "lab_" + axis,
            profile["evidence"]["chroma"]["lab_" + axis]["participant_changes"],
        )
        color_base["color_" + axis] = bounds
        direction[axis] = evidence
    color_decisions = []
    color_profile, color_result = neutral, light_result
    for axes in (("a", "b"), ("b",), ("a",)):
        candidate = deepcopy(neutral)
        for axis in axes:
            candidate["color_" + axis] = color_base["color_" + axis]
        measured = evaluate(images, source, candidate, domain, detector, audit.SEEDS)
        accepted, evidence = revision.approve(measured, light_result, target, ("lab_a", "lab_b"))
        color_decisions.append({"axes": axes, "accepted": accepted, "evidence": evidence})
        if accepted and color_profile is neutral:
            color_profile, color_result = candidate, measured
    focus_keys = (
        ("detail", "normalized_detail", "dark_ridge")
        if domain == "hair"
        else ("detail", "normalized_detail")
    )
    texture = np.clip(
        np.quantile(profile["evidence"]["normalized_detail_ratios"], [0.25, 0.75]), 0.85, 1.15
    ).tolist()
    sigma = min(
        0.75,
        float(np.quantile([r["sigma_roi64"] for r in profile["evidence"]["focus_fits"]], 0.75)),
    )
    focus_decisions = []
    fitted_profile, fitted_result = color_profile, color_result
    for use_texture, use_blur in ((True, True), (True, False), (False, True)):
        candidate = deepcopy(color_profile)
        if use_texture:
            candidate["texture_gain"] = texture
        if use_blur:
            candidate["blur_sigma_roi64"] = [0, sigma]
        measured = evaluate(images, source, candidate, domain, detector, audit.SEEDS)
        accepted, evidence = revision.approve(measured, color_result, target, focus_keys)
        safe = measured["spatial_correlation_q05"] >= POLICY["spatial_correlation_q05_minimum"]
        focus_decisions.append(
            {
                "texture": use_texture,
                "blur": use_blur,
                "accepted": accepted and safe,
                "evidence": evidence,
            }
        )
        if accepted and safe and fitted_profile is color_profile:
            fitted_profile, fitted_result = candidate, measured
    score, _ = gap_score(fitted_result["rows"], baseline["rows"], target, keys)
    primary_stable = all(
        gap_score(fitted_result["per_seed"][s], baseline["per_seed"][s], target, keys)[0][0]
        <= 1 + 1e-9
        for s in audit.SEEDS
    )
    if (
        score[0] > 1 + 1e-9
        or not primary_stable
        or fitted_result["spatial_correlation_q05"] < 0.95
    ):
        fitted_profile, fitted_result = chosen_profile, chosen_result
        residual_choice = (
            "Keep fully replayed light winner: residual refit failed final primary guard"
        )
    else:
        residual_choice = "Recomputed residual chroma and focus accepted"
    fitted_profile["calibration_version"] = "pixel_replayed_photometric_refit_v1"
    fitted_profile["lighting"]["calibration_version"] = fitted_profile["calibration_version"]
    fitted_profile["lighting"]["bounds"] = {
        "max_brightness_offset": 160,
        "contrast": [0.25, 1.5],
        "max_left_light": 64,
    }
    fitted_profile["lighting"]["train_visual_review"] = (
        "numerical_refit_pending_contact_sheet_review"
    )
    fitted_profile["engineering_bounds"]["pixel_refit_policy"] = POLICY
    fitted_profile["limitations"] += [
        "Photometric proxy fit to four calibration participants; "
        "no physical equivalence established",
        "Spatial correlation is an engineering guard, not clinical lesion preservation validation",
        "Source previews, not the entire original Source dataset, enter pixel replay",
    ]
    return (
        fitted_profile,
        {
            "original": source,
            "previous_4": baseline["rows"],
            "fitted": fitted_result["rows"],
            "target": target,
        },
        {
            "proposal": proposal,
            "probe_count": len(probe),
            "probe_candidates": scores,
            "full_finalists": full,
            "color_direction": direction,
            "color_candidates": color_decisions,
            "focus_candidates": focus_decisions,
            "residual_choice": residual_choice,
            "new_float_endpoint_max": fitted_result["new_float_endpoint_max"],
            "spatial_correlation_q05": fitted_result["spatial_correlation_q05"],
        },
    )


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    source_config = root / BASE / "camera_revised_20261111.json"
    if data.sha256(source_config) != INPUT_HASH:
        raise ValueError("Frozen pre-refit configuration changed")
    study = data.read_json(source_config)
    detector = analysis.face_detector()
    measured, decisions, visuals, summaries = {}, {}, {}, {}
    for domain in audit.DOMAINS:
        images, source, target_images, target, prior = load_domain(root, domain, study, detector)
        print(
            f"[{domain}] 원본 Train {len(source)}장·실제 원본 {len(target)}장 재측정 완료",
            flush=True,
        )
        old = deepcopy(study["domains"][domain]["profile"])
        profile, groups, decision = refit_domain(images, source, target, old, domain, detector)
        study["domains"][domain]["profile"] = profile
        measured[domain], decisions[domain] = groups, decision
        summaries[domain] = {
            "source_count": len(source),
            "target_count": len(target),
            "groups": {name: analysis.statistics(rows) for name, rows in groups.items()},
            "gaps": {
                key: {
                    name: audit.distribution_gap(rows, target, key)
                    for name, rows in groups.items()
                    if name != "target"
                }
                for key in (*audit.METRICS, "normalized_detail")
            },
            "source_png_provenance": prior["source_png_provenance"],
            "visual_paths": prior["visual_paths"],
        }
        visuals[domain] = (images, source, target_images, target, old, profile)
        print(
            f"[{domain}] 재설정 밝기 {profile['lighting']['brightness_offset']} · "
            f"대비 {profile['lighting']['contrast']}",
            flush=True,
        )
    study.update(
        hypothesis="Pixel-replayed measured pilot photometric fit versus previous Stage 2",
        changed_condition="Measured camera photometric profile",
        accompanying_changes="Joint light-strength fit followed by residual color/focus re-audit",
        profile_builder_sha256=data.sha256(Path(__file__)),
        refit_schema="train_preview_pilot_pixel_replay_v1",
        refit_parent_study_sha256=INPUT_HASH,
        refit_policy=POLICY,
    )
    for name, value in (
        ("study.json", study),
        ("decisions.json", decisions),
        ("measurements.json", measured),
        (
            "summary.json",
            {
                "domains": summaries,
                "training_executed": False,
                "source_test_used": False,
                "target_accuracy": None,
            },
        ),
    ):
        revision.write_new(output / name, value)
    from mediflow_datasets.camera_refit_figures import figures

    figures(output, visuals, summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
