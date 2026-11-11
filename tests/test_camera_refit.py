from copy import deepcopy

import numpy as np
import pytest

from mediflow_datasets.camera_refit import (
    inverse_offset,
    lighting_candidates,
    spatial_correlation,
    validate_rows,
)
from mediflow_datasets.domain_adaptation_experiment import bounded_photometric_update


def rows():
    source = [
        {"split": "train", "label": i, "metrics": {"y_p50": 90, "contrast": 60}} for i in range(2)
    ]
    target = [
        {
            "split": "calibration",
            "label": None,
            "group": f"P{p:02}/{c}",
            "metrics": {"y_p50": 155, "contrast": 25},
        }
        for p in range(1, 5)
        for c in ("ref", "alt")
    ]
    return source, target


def test_photometric_inverse_reaches_measured_bright_and_dark_anchors():
    for source, target in ((90.0, 155.0), (170.0, 100.0)):
        image = np.full((2, 2, 3), source, np.float32)
        result = bounded_photometric_update(image, image + inverse_offset(source, target))
        np.testing.assert_allclose(result, target, atol=2e-5)
    with pytest.raises(ValueError):
        inverse_offset(0, 100)


def test_refit_rejects_test_source_and_verified_target_labels():
    source, target = rows()
    validate_rows(source, target)
    source[0]["split"] = "test"
    with pytest.raises(ValueError, match="Train"):
        validate_rows(source, target)
    source, target = rows()
    target[0]["label"] = 0
    with pytest.raises(ValueError, match="unlabeled"):
        validate_rows(source, target)


def test_hair_contrast_is_not_fitted_to_different_hair_coverage():
    source, target = rows()
    for t in target:
        t["metrics"]["contrast"] = 170
    profile = {
        "domain": "hair",
        "lighting": {
            "brightness_offset": [15, 26],
            "contrast": [1.0, 1.0],
            "left_light": [0.0, 0.0],
        },
    }
    saved = deepcopy((profile, source, target))
    candidates, evidence = lighting_candidates(profile, source, target)
    assert (profile, source, target) == saved
    assert evidence["target_to_source_contrast_iqr_ratio"][0] > 2
    assert all(p["lighting"]["contrast"] == [1.0, 1.0] for p in candidates)
    assert any(p["lighting"]["brightness_offset"][1] > 32 for p in candidates)


def test_spatial_guard_rejects_permuted_content_with_identical_histogram():
    image = np.random.default_rng(42).uniform(20, 210, (64, 64, 3)).astype(np.float32)
    assert spatial_correlation(image, image * 0.7 + 30) > 0.999
    shuffled = image.reshape(-1, 3)[np.random.default_rng(12).permutation(4096)].reshape(
        image.shape
    )
    assert spatial_correlation(image, shuffled) < 0.1
