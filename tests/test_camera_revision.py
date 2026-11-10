from copy import deepcopy

import pytest

from mediflow_datasets.camera_revision import color_direction, does_not_worsen


def direction_rows():
    source = [{"split": "train", "label": 0, "metrics": {"lab_b": 0}}] * 9 + [
        {"split": "train", "label": 1, "metrics": {"lab_b": 2}}
    ]
    target = [
        {"label": None, "group": f"P{person:02}/{condition}", "metrics": {"lab_b": 10}}
        for person in range(1, 5)
        for condition in ("ref", "alt")
    ]
    return source, target


def test_color_uses_residual_direction_within_person_budget_and_keeps_inputs():
    source, target = direction_rows()
    original = deepcopy((source, target))
    bounds, evidence = color_direction(source, target, "lab_b", [-1, -2, 1, 2])
    assert (source, target) == original
    assert bounds == [0, 2]
    assert evidence["source_after_lighting_class_balanced_anchor"] == 1
    assert evidence["direction_stable"]
    assert set(evidence["leave_one_person_out_directions"].values()) == {1}
    source[0] = {**source[0], "split": "test"}
    with pytest.raises(ValueError):
        color_direction(source, target, "lab_b", [-1, -2, 1, 2])


def test_color_is_neutral_when_omitting_one_person_changes_direction_support():
    source, target = direction_rows()
    source = [{**row, "metrics": {"lab_b": 0}} for row in source]
    target = [
        {**row, "metrics": {"lab_b": -2 if row["group"].startswith("P01") else 1}} for row in target
    ]
    bounds, evidence = color_direction(source, target, "lab_b", [-1, -2, 1, 2])
    assert evidence["full_direction"] == 1
    assert not evidence["direction_stable"]
    assert bounds == [0, 0]


def test_focus_gate_rejects_better_raw_detail_with_worse_normalized_detail():
    baseline = [{"label": 0, "metrics": {"detail": 1, "normalized_detail": 1}}]
    target = [{"group": "P01/ref", "metrics": {"detail": 3, "normalized_detail": 0.5}}]
    wrong = [{"label": 0, "metrics": {"detail": 2, "normalized_detail": 1.5}}]
    correct = [{"label": 0, "metrics": {"detail": 2, "normalized_detail": 0.75}}]
    assert not does_not_worsen(wrong, baseline, target, ("detail", "normalized_detail"))[0]
    assert does_not_worsen(correct, baseline, target, ("detail", "normalized_detail"))[0]
