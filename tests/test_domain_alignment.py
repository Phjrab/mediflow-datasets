import numpy as np
import pytest

from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets import domain_alignment as alignment


def test_alignment_matches_independent_covariance_and_kernel_metrics_and_has_gradients():
    import tensorflow as tf

    rng = np.random.default_rng(64)
    source = rng.normal(size=(9, 5)).astype(np.float32)
    target = (rng.normal(size=(7, 5)) * 1.3 + 0.5).astype(np.float32)
    reference = exp.feature_reference(source)
    reference["bandwidth_multipliers"] = [0.25, 1, 4]
    expected = [
        exp.feature_distance(
            source, target, {**reference, "bandwidth_sq": reference["bandwidth_sq"] * multiplier}
        )
        for multiplier in reference["bandwidth_multipliers"]
    ]
    a, b = tf.Variable(source), tf.Variable(target)
    with tf.GradientTape() as tape:
        mmd, coral = alignment.losses(a, b, reference)
        total = mmd + coral
    assert float(mmd) == pytest.approx(np.mean([v["mmd2_biased_rbf"] for v in expected]), abs=1e-5)
    assert float(coral) == pytest.approx(expected[0]["coral_covariance"], abs=1e-5)
    for gradient in tape.gradient(total, [a, b]):
        assert gradient is not None and np.isfinite(gradient.numpy()).all()
        assert np.linalg.norm(gradient.numpy()) > 0
    same = alignment.losses(source, source, reference)
    assert max(map(float, same)) < 1e-5
    with pytest.raises(tf.errors.InvalidArgumentError):
        alignment.losses(source[:1], target, reference)


def test_target_sampling_is_balanced_has_no_repeat_and_rejects_target_test():
    rows = [
        {"path": f"P{person}/{i}", "person": f"P{person}", "split": "calibration", "label": None}
        for person in range(4)
        for i in range(3)
    ]
    schedule = alignment.balanced_schedule(rows, [0, 2, 4, 6, 8, 10], 42, 0)
    assert len(schedule) == 3
    indices = []
    for batch in schedule.values():
        assert len({rows[i]["person"] for i in batch}) == 4
        indices.extend(batch)
    assert len(indices) == len(set(indices)) == 12
    assert schedule == alignment.balanced_schedule(rows, [0, 2, 4, 6, 8, 10], 42, 0)
    with pytest.raises(ValueError, match="unlabeled calibration"):
        alignment.balanced_schedule([{**rows[0], "split": "test"}, *rows[1:]], [0, 1], 42, 0)
