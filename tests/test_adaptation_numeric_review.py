from copy import deepcopy
from pathlib import Path

import pytest

from mediflow_datasets.adaptation_numeric_review import (
    collect,
    gap_change,
    require_same_profile,
    write_report_data,
)


def test_distance_change_is_a_distance_reduction_not_a_similarity_score():
    actual = gap_change(68.00235648972958, 34.054261613813495)
    assert actual["direction"] == "가까워짐"
    assert actual["difference"] == 34.054261613813495 - 68.00235648972958
    assert actual["reduction_fraction"] == 1 - 34.054261613813495 / 68.00235648972958
    assert gap_change(2, 3)["reduction_fraction"] == -0.5
    assert gap_change(2, 3)["direction"] == "멀어짐"


def test_missing_and_zero_distance_are_kept_distinct():
    assert gap_change(0, 0)["direction"] == "동일"
    assert gap_change(0, 0)["reduction_fraction"] is None
    assert gap_change(None, 0)["direction"] == "미측정"
    assert gap_change(2, None)["ratio"] is None


@pytest.mark.parametrize("before,after", [(-1, 2), (1, float("nan")), (float("inf"), 1)])
def test_invalid_recorded_distances_are_not_silently_used(before, after):
    with pytest.raises(ValueError, match="finite"):
        gap_change(before, after)


def test_replay_profile_must_match_training_operator_and_strength():
    profile = {
        "lighting": {
            "brightness_offset": [-5, 0],
            "contrast": [1, 1],
            "left_light": [0, 0],
            "photometric_operator": "bounded_additive_v3",
        },
        "color_a": [0, 1],
        "blur_sigma_roi64": [0, 0.1],
        "blur_input_scale": 1.6,
    }
    require_same_profile(profile, deepcopy(profile))
    for field, value in [("color_a", [0, 0]), ("blur_input_scale", 1.0)]:
        changed = deepcopy(profile)
        changed[field] = value
        with pytest.raises(ValueError, match="actually trained"):
            require_same_profile(profile, changed)


def test_report_build_never_overwrites_existing_folder_by_default(tmp_path):
    original = tmp_path / "original.json"
    original.write_bytes(b'{"keep":"unaltered"}')
    with pytest.raises(FileExistsError):
        write_report_data(tmp_path, tmp_path)
    assert original.read_bytes() == b'{"keep":"unaltered"}'


def test_actual_record_inventory_distinguishes_missing_features_and_camera_profiles():
    report = collect(Path(__file__).resolve().parents[1])
    assert len(report["profiles"]) == 17 and len(report["training_runs"]) == 21
    assert len(report["parameter_rows"]) == 221 and len(report["source_test"]) == 81
    assert len(report["device_gaps"]) == 107
    missing = [r for r in report["features"] if r["mmd"] is None]
    assert len(missing) == 3 and all(r["model"] == "stage4_revised" for r in missing)
    assert report["stage2_reference_identity_verified"]
    phone_joint = [
        r for r in report["device_gaps"] if r["domain"] == "phone" and r["stage"] == "5차 영상"
    ]
    assert sum(r["direction"] == "가까워짐" for r in phone_joint) == 7
    assert next(r for r in phone_joint if r["metric"] == "lighting_range")["direction"] == "멀어짐"
