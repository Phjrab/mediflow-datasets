import io

import cv2
import numpy as np
import pytest
from PIL import Image

from mediflow_datasets import camera_extended as camera
from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_experiment as exp


def lighting():
    return {
        "brightness_offset": [-3, 4],
        "contrast": [0.97, 1.04],
        "left_light": [0, 2],
        "scale_x": [1, 1],
        "scale_y": [1, 1],
        "photometric_operator": "bounded_additive_v3",
    }


def test_expanded_and_alignment_arms_share_exact_image_transform_and_preserve_rgb_contract():
    image = np.random.default_rng(6).uniform(10, 240, size=(64, 64, 3)).astype(np.float32)
    profile = {
        "lighting": lighting(),
        "color_a": [-1, 1],
        "color_b": [-0.5, 1.5],
        "texture_gain": [0.9, 1.1],
        "blur_sigma_roi64": [0, 0.5],
        "blur_input_scale": 1,
    }
    original = image.copy()
    a = camera.transform(image, profile, "lighting", 42)
    b = camera.transform(image, profile, "expanded", 42)
    c = camera.transform(image, profile, "aligned", 42)
    assert np.array_equal(a, exp.transform_image(image, profile["lighting"], "target_guided", 42))
    assert np.array_equal(b, c) and not np.array_equal(a, b)
    assert np.array_equal(image, original)
    assert b.dtype == np.float32 and b.shape == image.shape
    assert np.isfinite(b).all() and b.min() >= 0 and b.max() <= 255 and b.max() > 1
    identity = {
        **profile,
        "color_a": [0, 0],
        "color_b": [0, 0],
        "texture_gain": [1, 1],
        "blur_sigma_roi64": [0, 0],
    }
    assert np.array_equal(camera.transform(image, identity, "expanded", 42), a)


def test_profile_uses_real_within_condition_images_and_rejects_test_and_unverified_labels():
    rng = np.random.default_rng(15)
    base = rng.uniform(50, 200, (64, 64, 3)).astype(np.float32)
    rows, images = [], []
    for person in range(4):
        for condition in ("ref", "alt"):
            for i, sigma in enumerate((0, 0.1, 0.3, 0.5, 0.8)):
                image = base.copy() if sigma == 0 else cv2.GaussianBlur(base, (0, 0), sigma)
                if condition == "alt":
                    image = (image + np.asarray([3, -2, 4], dtype=np.float32)).clip(0, 255)
                buffer = io.BytesIO()
                Image.fromarray(np.rint(image).astype(np.uint8)).save(buffer, format="PNG")
                metrics = analysis.measure(buffer.getvalue(), "skin", 64, None)
                rows.append(
                    {
                        "person": f"P{person}",
                        "condition": condition,
                        "path": f"P{person}/{condition}/{i}",
                        "split": "calibration",
                        "label": None,
                        "metrics": metrics,
                    }
                )
                images.append(image)
    source = [{"split": "train", "label": 0, "metrics": rows[0]["metrics"]}]
    profile = camera.measured_extended_profile("skin", lighting(), rows, images, source)
    assert len(profile["evidence"]["focus_fits"]) == 8
    assert 0 < profile["blur_sigma_roi64"][1] <= 0.75
    assert all(0.85 <= v <= 1.15 for v in profile["texture_gain"])
    assert all(abs(v) <= 3 for v in profile["color_a"] + profile["color_b"])
    with pytest.raises(ValueError, match="Train only"):
        camera.measured_extended_profile(
            "skin", lighting(), rows, images, [{**source[0], "split": "test"}]
        )
    with pytest.raises(ValueError, match="Unverified Target"):
        camera.measured_extended_profile(
            "skin", lighting(), [{**rows[0], "label": 0}, *rows[1:]], images, source
        )
