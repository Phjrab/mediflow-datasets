from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_experiment as exp


def profile():
    return {
        "brightness_offset": [-12, 12],
        "contrast": [1, 1],
        "scale_x": [1, 1],
        "scale_y": [1, 1],
        "left_light": [0, 0],
    }


def test_apply_decision_does_not_truncate_strength_distribution():
    image = np.full((4, 4, 3), 128, np.float32)
    selected = [seed for seed in range(4096) if exp.augmentation_applies(seed, 0.5)]
    offsets = [
        float(exp.transform_image(image, profile(), "brightness_only", s)[0, 0, 0] - 128)
        for s in selected
    ]
    assert 1800 < len(selected) < 2300
    assert min(offsets) < -11.5 and max(offsets) > 11.5
    assert abs(np.mean(offsets)) < 0.6
    assert not exp.augmentation_applies(1, 0)
    assert exp.augmentation_applies(1, 1)


def test_unsafe_geometry_rejected_and_identical_arms_merged():
    image = np.full((16, 16, 3), 128, np.float32)
    p = profile()
    p["scale_x"] = [0.6, 0.7]
    with pytest.raises(ValueError, match="geometry"):
        exp.transform_image(image, p, "target_guided", 42)
    arms, aliases = exp.arm_plan(profile())
    assert tuple(arms) == ("source_only", "brightness_only")
    assert aliases == {"target_guided": "brightness_only"}
    p = profile()
    p["contrast"] = [0.9, 0.95]
    assert len(exp.arm_plan(p)[0]) == 3


def test_center_crop_preserves_rgb_range_and_box_coordinates():
    image = np.zeros((40, 80, 3), np.uint8)
    image[:, :20, 0] = 255
    image[:, 20:60, 1] = 200
    image[:, 60:, 2] = 255
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    output = exp.image_tensor(buffer.getvalue(), 32, "center_square")
    assert output.shape == (32, 32, 3)
    assert np.allclose(output[:, :, 1], 200)
    assert output[:, :, 0].max() == output[:, :, 2].max() == 0
    metrics = analysis.measure(
        buffer.getvalue(),
        "web_skin",
        32,
        None,
        face_box=(25, 5, 55, 35),
        input_policy="center_square",
    )
    assert metrics["width"] == metrics["height"] == 40
    assert metrics["native_width"] == 80 and metrics["face_w"] == 0.75
    with pytest.raises(ValueError, match="outside"):
        analysis.measure(
            buffer.getvalue(),
            "web_skin",
            32,
            None,
            face_box=(0, 5, 30, 35),
            input_policy="center_square",
        )


def test_class_balanced_anchor_and_no_synthetic_padding():
    source = [{"split": "train", "label": 0, "metrics": {"y_p50": 100.0}}] * 10
    source += [{"split": "train", "label": 1, "metrics": {"y_p50": 200.0}}]
    target = [
        {
            "split": "calibration",
            "person": f"P{p}",
            "condition": c,
            "metrics": {"y_p50": 140.0, "contrast": 20.0},
        }
        for p in range(4)
        for c in ("ref", "alt")
    ]
    p = analysis.calibrate("web_skin", source, target)
    assert p["evidence"]["source_brightness_anchor"] == 150
    assert p["brightness_offset"] == [-5, -5]
    assert p["scale_x"] == p["scale_y"] == [1, 1]
    assert p["evidence"]["source_roi_by_class"]["0"]["rows"] == 10


def test_packet_preprocessing_mismatch_is_rejected(tmp_path):
    from mediflow_datasets.domain_adaptation import load_calibration_packet

    p = tmp_path / "packet.json"
    p.write_text(json.dumps({"domain": "web_skin", "model_sha256": "model"}), encoding="utf-8")
    with pytest.raises(ValueError, match="preprocessing"):
        load_calibration_packet(p, "web_skin", "model", "center_square")


def test_hair_contrast_requires_common_coverage_sensitivity_agreement():
    source = [{"split": "train", "metrics": {"y_p50": 150.0}}]
    target = []
    for person in range(4):
        reference = [(0.1, 100), (0.2, 100), (0.4, 300)]
        alternative = (
            [(0.15, 130), (0.2, 130), (0.05, 10), (0.01, 10), (0.02, 10)]
            if person == 0
            else [(0.15, 80), (0.2, 80), (0.05, 60)]
        )
        for condition, values in (("ref", reference), ("alt", alternative)):
            for ridge, contrast in values:
                target.append(
                    {
                        "split": "calibration",
                        "person": f"P{person}",
                        "condition": condition,
                        "metrics": {"y_p50": 160.0, "dark_ridge": ridge, "contrast": contrast},
                    }
                )
    p = analysis.calibrate("hair", source, target)
    assert p["contrast"] == [1, 1]
    evidence = p["evidence"]["coverage_matched_contrast"]
    assert evidence["people"]["P0"]["contrast_ratio"] > 1


def test_bounded_photometry_reduces_hard_clipping_without_normalizing_input():
    ramp = np.linspace(0, 255, 256, dtype=np.float32)[:, None, None]
    image = np.repeat(np.repeat(ramp, 4, axis=1), 3, axis=2)
    for delta in (-48, 48):
        output = exp.bounded_photometric_update(image, image + delta)
        assert output.dtype == np.float32 and output.shape == image.shape
        assert np.isfinite(output).all() and output.min() >= 0 and output.max() <= 255
        assert np.all(np.diff(output[:, 0, 0]) >= 0)
        assert ((output[1:-1] > 0) & (output[1:-1] < 255)).all()
    assert np.array_equal(exp.bounded_photometric_update(image, image), image)
    p = profile()
    p["photometric_operator"] = "bounded_additive_v3"
    assert np.max(exp.transform_image(image, p, "brightness_only", 42)) > 200
