import pytest

from mediflow_datasets.camera_similarity import distribution_gap, retained_previews


def test_replay_accepts_only_retained_train_with_matching_labels():
    manifest = {"original": {"train": [{"path": "retained", "label": 0}]}}
    rows = [
        {"path": "retained", "split": "train", "label": 0},
        {"path": "quarantined", "split": "train", "label": 1},
    ]
    assert retained_previews(rows, manifest) == [(0, rows[0])]
    with pytest.raises(ValueError):
        retained_previews([{**rows[0], "split": "test"}], manifest)
    with pytest.raises(ValueError):
        retained_previews([{**rows[0], "label": 1}], manifest)


def test_distance_equal_weights_classes_and_target_groups_despite_extra_images():
    source = [{"label": 0, "metrics": {"v": 0}}] * 9 + [{"label": 1, "metrics": {"v": 10}}]
    target = [{"group": "P01/ref", "metrics": {"v": 0}}] + [
        {"group": "P02/ref", "metrics": {"v": 10}}
    ] * 6
    assert distribution_gap(source, target, "v") == pytest.approx(0)
    shifted = [{**r, "metrics": {"v": r["metrics"]["v"] + 3}} for r in target]
    assert distribution_gap(source, shifted, "v") == pytest.approx(3)
    assert distribution_gap(source, target, "missing") is None
