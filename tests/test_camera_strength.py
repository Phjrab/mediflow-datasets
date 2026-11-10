from copy import deepcopy

import numpy as np
import pytest

from mediflow_datasets.camera_strength import stronger_profile


def example(domain="web_skin"):
    previous = {
        "schema": "measured_camera_extended_v1",
        "domain": domain,
        "lighting": {
            "brightness_offset": [2, 3],
            "contrast": [1.1, 1.15],
            "left_light": [0, 2],
            "evidence": {"paired_left_minus_right_change": [3, 4, 5, 6]},
        },
        "color_a": [-0.5, 0.5],
        "color_b": [-0.5, 0.5],
        "texture_gain": [0.95, 1.05],
        "blur_sigma_roi64": [0, 0.3],
        "blur_input_scale": 2,
        "evidence": {
            "chroma": {
                axis: {"participant_changes": [-2, -1, 1, 2]} for axis in ("lab_a", "lab_b")
            },
            "normalized_detail_ratios": [0.8, 0.9, 1, 1.1],
            "focus_fits": [{"sigma_roi64": 0.6, "fit_at_search_limit": False}] * 8,
        },
        "engineering_bounds": {"measured_change_fraction": 0.5},
        "limitations": [],
    }
    source = [
        {"split": "train", "label": label, "metrics": {"y_p50": 160, "contrast": 40}}
        for label in range(2)
        for _ in range(3)
    ]
    target = [
        {
            "split": "calibration",
            "label": None,
            "person": person,
            "condition": condition,
            "metrics": {"y_p50": 110, "contrast": 20},
        }
        for person in ("P01", "P02", "P03", "P04")
        for condition in ("ref", "alt")
    ]
    return previous, source, target


def test_strength_uses_train_target_direction_and_full_measured_variation():
    previous, source, target = example()
    previous["lighting"]["train_visual_review"] = "approved"
    previous["lighting"]["review_note"] = "Earlier profile review"
    before = deepcopy(previous)
    profile = stronger_profile(previous, source, target, 2)
    assert previous == before
    assert profile["previous_visual_review"]["train_visual_review"] == "approved"
    assert profile["lighting"]["train_visual_review"] == "followup_examples_only"
    assert "previous profile only" in profile["lighting"]["review_note"]
    assert profile["lighting"]["brightness_offset"] == [-32, -32]
    assert profile["lighting"]["contrast"] == [0.8, 0.8]
    assert profile["color_a"] == [-1.25, 1.25]
    assert profile["blur_sigma_roi64"] == [0, 0.6]
    assert profile["blur_input_scale"] == previous["blur_input_scale"]
    np.testing.assert_allclose(profile["texture_gain"], [0.875, 1.025])
    previous["domain"] = "hair"
    assert stronger_profile(previous, source, target, 2)["lighting"]["contrast"] == [1, 1]


@pytest.mark.parametrize("role", ["source_test", "target_test", "target_labeled"])
def test_profile_rejects_test_and_unverified_labels(role):
    previous, source, target = example()
    if role == "source_test":
        source[0]["split"] = "test"
    elif role == "target_test":
        target[0]["split"] = "test"
    else:
        target[0]["label"] = 0
    with pytest.raises(ValueError):
        stronger_profile(previous, source, target, 2)
