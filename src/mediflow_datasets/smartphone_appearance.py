"""Explicit weak/strong appearance experiment, separate from measured camera matching."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

SCHEMA = "mediflow_phone_user_appearance_strength_v1"
REFERENCE_SIZE = 256
SETTINGS = {
    "stage2": {"texture_gain": [0.95, 1.05], "blur_input_sigma_max": 0.5},
    "stage4": {"texture_gain": [0.85, 1.15], "blur_input_sigma_max": 1.0},
}


def requested_profiles(study):
    """Keep the measured ROI conversion; specify exploratory blur in input pixels."""
    profiles = deepcopy(study["profiles"])
    for stage, settings in SETTINGS.items():
        p = profiles[stage]
        scale = p["blur_input_scale"]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Positive finite ROI-to-input blur scale is required")
        p["texture_gain"] = list(settings["texture_gain"])
        p["blur_sigma_roi64"] = [0.0, settings["blur_input_sigma_max"] / scale]
    return profiles


def build_plan(study, study_sha256):
    profiles = requested_profiles(study)
    return {
        "schema": SCHEMA,
        "status": "planned_user_requested_exploration",
        "source_study_sha256": study_sha256,
        "source_manifest_sha256": study["source_manifest_sha256"],
        "source_test_used": False,
        "target_holdout_used": False,
        "training_executed": False,
        "distribution_improvement_established": False,
        "authorization": "User requested weak Stage2 and stronger Stage4 texture and blur",
        "hypothesis": "Added weak/strong appearance variation may improve Source robustness",
        "changed_variable": "Appearance variation bundle: microcontrast and Gaussian blur",
        "fixed": [
            "All Stage1 settings; all stages' lighting, color, geometry and input conversion",
            "Source split and class order; P01-P04 calibration and P05 holdout",
            "Original starting checkpoints; epochs, learning rate, seed and probability",
            "Stage3 uses Stage2 image profile and adds MMD/CORAL, independently initialized",
        ],
        "reference_input_size": REFERENCE_SIZE,
        "settings": deepcopy(SETTINGS),
        "profiles": profiles,
        "blur_sigma_roi64_exploratory_cap": max(
            p["blur_sigma_roi64"][1] for p in profiles.values()
        ),
        "range_rationale": {
            "texture": "Symmetric +/-5% and +/-15% engineering residual gain, not camera estimates",
            "blur": "0-0.5px and 0-1px at 256px; scale linearly to selected input size",
            "prior_measured_texture_budget": "Previous review examined gain changes up to 0.15",
            "limits": "ROI blur search cap expanded only for this requested exploration",
        },
        "limitations": [
            "Prior distribution-only review did not select texture or blur",
            "Added ranges are requested experimental settings, not measured optical focus",
            "Stage4 retains different lighting/color from Stage2, not an isolated strength test",
            "Texture and blur changed together; separate effects cannot be attributed",
            "No Target disease labels; no Target accuracy or clinical improvement claim",
        ],
    }


def validate_plan(study, study_sha256, plan):
    if (
        plan.get("schema") != SCHEMA
        or plan.get("status") != "planned_user_requested_exploration"
        or plan.get("source_study_sha256") != study_sha256
        or plan.get("source_manifest_sha256") != study["source_manifest_sha256"]
        or any(
            plan.get(key) is not False
            for key in (
                "source_test_used",
                "target_holdout_used",
                "training_executed",
                "distribution_improvement_established",
            )
        )
        or plan.get("profiles") != requested_profiles(study)
        or plan.get("settings") != SETTINGS
        or plan.get("reference_input_size") != REFERENCE_SIZE
        or plan.get("blur_sigma_roi64_exploratory_cap")
        != max(p["blur_sigma_roi64"][1] for p in plan["profiles"].values())
    ):
        raise ValueError("Appearance plan differs from requested strength/frozen split scope")
    return plan
