"""Keep bounded nearest-profile selection separate from median direction and Test tuning."""

from copy import deepcopy

import pytest

from mediflow_datasets.smartphone_stage2_review import assess, candidate_ranges, group_summary


def measured_rows():
    source = [
        {"split": "train", "label": label, "metrics": {"lab_a": value, "y_p50": 150.0}}
        for label in range(5)
        for value in (10.0, 11.0)
    ]
    target = [
        {
            "split": "calibration",
            "person": person,
            "group": person,
            "label": None,
            "metrics": {"lab_a": value, "y_p50": 150.0},
        }
        for person in ("P01", "P02", "P03", "P04")
        for value in (11.0, 12.0)
    ]
    return source, target


def test_nearer_distribution_can_pass_without_claiming_a_causal_camera_correction():
    before, target = measured_rows()
    after = deepcopy(before)
    for row in after:
        row["metrics"]["lab_a"] += 0.5
    result = assess(before, after, target, ("lab_a",), 0.999, ("y_p50",))
    assert result["accepted"] and result["person_improvements"] == 4
    assert result["omission_improvements"] == 4 and result["relative_cost"] < 1


def test_nearer_color_rejected_when_lighting_or_spatial_guard_deteriorates():
    before, target = measured_rows()
    after = deepcopy(before)
    for row in after:
        row["metrics"].update(lab_a=row["metrics"]["lab_a"] + 0.5, y_p50=130)
    result = assess(before, after, target, ("lab_a",), 0.999, ("y_p50",))
    assert not result["accepted"] and not result["guard_passed"]
    for row in after:
        row["metrics"]["y_p50"] = 150
    assert not assess(before, after, target, ("lab_a",), 0.5, ("y_p50",))["accepted"]


def test_holdout_is_rejected_before_nearest_profile_scoring():
    before, target = measured_rows()
    target[0].update(person="P05", split="holdout")
    with pytest.raises(ValueError, match="holdout"):
        assess(before, before, target, ("lab_a",), 1, ("y_p50",))


def test_color_grid_is_derived_from_observed_within_person_span_and_contains_zero():
    _, target = measured_rows()
    candidates, evidence = candidate_ranges(target, "color_a")
    budget = evidence["measured_within_person_q90_q10_median"]
    assert budget == pytest.approx(0.8)
    assert candidates and all(low <= 0 <= high for low, high in candidates)
    assert max(max(abs(low), abs(high)) for low, high in candidates) <= budget


def test_camera_comparison_keeps_equal_person_weight_despite_extra_captures():
    rows = [
        {"person": "P01", "metrics": {"y_p50": 100}},
        {"person": "P02", "metrics": {"y_p50": 200}},
    ]
    result = group_summary(rows + [deepcopy(rows[0]) for _ in range(10)], "y_p50", "person")
    assert result["equal_group_median_of_medians"] == 150
    assert result["pooled_equal_group_distribution"]["mean"] == pytest.approx(150)


def test_profile_update_rejects_changed_lighting_and_holdout_based_review(tmp_path):
    from mediflow_datasets.domain_adaptation_data import sha256
    from mediflow_datasets.smartphone_prepare import accepted_review
    from mediflow_datasets.smartphone_profile import empty_profile, write_new

    private = tmp_path / "private"
    private.mkdir()
    original = empty_profile({"brightness_offset": [-10, 0]})
    profiles = {name: deepcopy(original) for name in ("stage1", "stage2", "stage4")}
    write_new(private / "profiles.json", profiles)
    write_new(private / "measurements.json", {"source": [], "calibration": []})
    study = {"profiles": profiles, "source_manifest_sha256": "fixed_train_split"}
    changed = deepcopy(profiles)
    changed["stage2"]["color_a"] = [0, 1]
    report = {
        "schema": "mediflow_phone_stage2_common_nearer_review_v1",
        "status": "measured",
        "source_test_used": False,
        "target_holdout_used": False,
        "training_executed": False,
        "profile_source_sha256": sha256(private / "profiles.json"),
        "measurement_source_sha256": sha256(private / "measurements.json"),
        "source_manifest_sha256": "fixed_train_split",
        "profiles": changed,
    }
    path = tmp_path / "review.json"
    write_new(path, report)
    assert accepted_review(tmp_path, private, study, path)["profiles"]["stage2"]["color_a"] == [
        0,
        1,
    ]
    import json

    report["profiles"]["stage2"]["lighting"]["brightness_offset"] = [-50, 0]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Train-only"):
        accepted_review(tmp_path, private, study, path)
    report["profiles"] = changed
    report["target_holdout_used"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Train-only"):
        accepted_review(tmp_path, private, study, path)
