"""Bounded, measured Source-Train/Target photometric changes for a follow-up study."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from mediflow_datasets import domain_adaptation_analysis as analysis


def stronger_profile(previous, source_rows, target_rows, class_count):
    if not source_rows or any(row["split"] != "train" for row in source_rows):
        raise ValueError("Profile calibration requires Source Train only")
    if {row["label"] for row in source_rows} != set(range(class_count)):
        raise ValueError("Source class coverage differs from the classifier")
    if any(row["split"] != "calibration" or row.get("label") is not None for row in target_rows):
        raise ValueError("Target must be unlabeled calibration data")
    people = sorted({row["person"] for row in target_rows})
    if len(people) != 4:
        raise ValueError("Expected the four recorded participants")
    groups = analysis.grouped_target(target_rows)
    if set(groups) != {
        f"{person}/{condition}" for person in people for condition in ("ref", "alt")
    }:
        raise ValueError("Each participant needs both capture conditions")
    anchors, coverage, target_groups = {}, {}, {}
    for metric in ("y_p50", "contrast"):
        class_medians, counts = [], []
        for label in range(class_count):
            rows = [row for row in source_rows if row["label"] == label]
            values = [row["metrics"][metric] for row in rows if metric in row["metrics"]]
            if not values or not np.isfinite(values).all():
                raise ValueError("Every Source class needs comparable measurements")
            class_medians.append(float(np.median(values)))
            counts.append({"total": len(rows), "measured": len(values)})
        anchors[metric] = float(np.median(class_medians))
        coverage[metric] = {"class_medians": class_medians, "coverage_by_class": counts}
        target_groups[metric] = [group[metric]["q50"] for group in groups.values()]
        if not np.isfinite(target_groups[metric]).all():
            raise ValueError("Invalid Target measurements")
    if anchors["contrast"] <= 0:
        raise ValueError("Invalid Source contrast reference")

    result = deepcopy(previous)
    lighting = result["lighting"]
    result["previous_visual_review"] = {
        key: lighting.get(key) for key in ("profile_version", "train_visual_review", "review_note")
    }
    lighting["profile_version"] = "bounded_source_target_strength_v1"
    lighting["train_visual_review"] = "followup_examples_only"
    lighting["review_note"] = (
        "The earlier approval applies to the previous profile only. "
        "This follow-up has illustrative examples, not a full retained-Train corner review "
        "or clinical validation. Source/Target differences remain partly confounded."
    )
    brightness_raw = np.quantile(
        np.asarray(target_groups["y_p50"]) - anchors["y_p50"], [0.25, 0.75]
    )
    lighting["brightness_offset"] = np.clip(brightness_raw, -32, 32).tolist()
    contrast_raw = np.quantile(
        np.asarray(target_groups["contrast"]) / anchors["contrast"], [0.25, 0.75]
    )
    lighting["contrast"] = (
        [1.0, 1.0] if previous["domain"] == "hair" else np.clip(contrast_raw, 0.8, 1.25).tolist()
    )
    if previous["domain"] == "web_skin":
        lateral = previous["lighting"]["evidence"]["paired_left_minus_right_change"]
        lighting["left_light"] = np.clip(np.quantile(lateral, [0.25, 0.75]), -24, 24).tolist()
    evidence = previous["evidence"]
    for axis in ("a", "b"):
        changes = evidence["chroma"][f"lab_{axis}"]["participant_changes"]
        result[f"color_{axis}"] = np.clip(np.quantile(changes, [0.25, 0.75]), -3, 3).tolist()
        result["evidence"]["chroma"][f"lab_{axis}"]["range"] = result[f"color_{axis}"]
    result["texture_gain"] = np.clip(
        np.quantile(evidence["normalized_detail_ratios"], [0.25, 0.75]), 0.85, 1.15
    ).tolist()
    fits = evidence["focus_fits"]
    if len(fits) < 4 or any(row["fit_at_search_limit"] for row in fits):
        raise ValueError("Focus evidence is missing or at the search boundary")
    result["blur_sigma_roi64"] = [
        0.0,
        min(0.75, float(np.quantile([row["sigma_roi64"] for row in fits], 0.75))),
    ]
    result["calibration_version"] = "bounded_source_target_strength_v1"
    result["calibration_evidence"] = {
        "source_role": "retained_original_train",
        "target_role": "unlabeled_calibration",
        "source_class_balanced_anchors": anchors,
        "source_measurement_coverage": coverage,
        "target_equal_participant_condition_medians": target_groups,
        "raw_brightness_offsets": brightness_raw.tolist(),
        "raw_contrast_ratios": contrast_raw.tolist(),
        "brightness_limit": 32,
        "contrast_limits": [0.8, 1.25],
        "color_texture_focus_measured_fraction": 1.0,
        "hair_global_contrast": "identity; coverage and content remain confounded",
    }
    result["engineering_bounds"].update(
        measured_change_fraction=1.0, brightness_offset_limit=32, contrast_limits=[0.8, 1.25]
    )
    lighting["calibration_version"] = result["calibration_version"]
    lighting["bounds"] = {"max_brightness_offset": 32, "contrast": [0.8, 1.25]}
    lighting["evidence"] = {
        "method": "Source class-balanced median to equal participant/condition Target quartiles",
        "anchors": anchors,
        "coverage": coverage,
    }
    result["limitations"] = [
        *previous["limitations"],
        "Source/Target brightness and contrast also contain disease and composition differences",
        "Source face-ROI coverage varies by class; anchors describe successfully measured rows",
        "Bounds are engineering choices, not validated physical camera calibration",
        "This profile changes direction and calibration policy as well as amplitude",
        "The existing Source Test has already been inspected; select on Validation only",
    ]
    return result
