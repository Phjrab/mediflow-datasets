from copy import deepcopy

import pytest

from mediflow_datasets.smartphone_joint_review import (
    METRICS,
    assess,
    joint_gate,
    lighting_candidates,
    rank,
)
from mediflow_datasets.smartphone_profile import empty_profile


def rows():
    source = [
        {"split": "train", "label": i, "metrics": dict.fromkeys(METRICS, 10.0)}
        for i in range(5)
        for _ in range(2)
    ]
    target = [
        {
            "split": "calibration",
            "person": p,
            "group": p,
            "label": None,
            "metrics": dict.fromkeys(METRICS, 12.0),
        }
        for p in ("P01", "P02", "P03", "P04")
        for _ in range(2)
    ]
    after = deepcopy(source)
    for r in after:
        r["metrics"] = dict.fromkeys(METRICS, 11.5)
    return source, target, after


def test_average_improvement_cannot_hide_one_worsening_metric():
    reference = dict.fromkeys(METRICS, 1.0)
    changed = dict.fromkeys(METRICS, 0.5)
    changed["normalized_detail"] = 1.1
    assert rank(reference, changed)[0] < 1
    assert not joint_gate(reference, changed)
    changed["normalized_detail"] = 1.0
    assert joint_gate(reference, changed)
    assert not joint_gate(reference, reference)


def test_all_metrics_people_and_replay_seeds_pass_before_acceptance():
    source, target, after = rows()
    result = assess(source, after, target, {42: after, 707: after, 1707: after}, 0.999, 0)
    assert result["accepted"] and result["improved_people"] == 4
    assert result["improved_omissions"] == 4 and not result["worsening_metrics"]
    assert result["each_seed_all_metrics_nonworsening"]


def test_good_median_cannot_hide_one_bad_replay_seed():
    source, target, after = rows()
    bad = deepcopy(after)
    for r in bad:
        r["metrics"]["detail"] = 8
    result = assess(source, after, target, {42: after, 707: bad, 1707: after}, 0.999, 0)
    assert not result["accepted"] and not result["each_seed_all_metrics_nonworsening"]
    assert not result["worsening_metrics"]


@pytest.mark.parametrize("correlation,endpoint", [(0.94, 0), (0.999, 0.01)])
def test_spatial_and_float_clipping_guards_remain_required(correlation, endpoint):
    source, target, after = rows()
    assert not assess(source, after, target, {42: after}, correlation, endpoint)["accepted"]


def test_holdout_and_validation_cannot_enter_joint_review():
    source, target, after = rows()
    target[0].update(person="P05", split="holdout", group="P05")
    with pytest.raises(ValueError, match="holdout"):
        assess(source, after, target, {42: after}, 0.999, 0)
    source, target, after = rows()
    source[0]["split"] = "val"
    with pytest.raises(ValueError, match="Train"):
        assess(source, after, target, {42: after}, 0.999, 0)


def test_lighting_candidates_are_measured_bounded_and_do_not_warp_or_mutate_inputs():
    source, target, _ = rows()
    p = empty_profile({"brightness_offset": [0, 0], "scale_x": [1, 1], "scale_y": [1, 1]})
    p["blur_input_scale"] = 0.41125
    study = {"profiles": {"stage1": p}}
    original = deepcopy((study, source, target))
    candidates = lighting_candidates(study, source, target)
    assert len(candidates) == 17 and (study, source, target) == original
    for _, candidate in candidates:
        assert candidate["lighting"]["scale_x"] == [1, 1]
        assert candidate["lighting"]["scale_y"] == [1, 1]
        assert all(-96 <= v <= 96 for v in candidate["lighting"]["brightness_offset"])
        assert candidate["color_a"] == [0, 0] and candidate["blur_sigma_roi64"] == [0, 0]
    assert any("target_quartiles" in label for label, _ in candidates)
