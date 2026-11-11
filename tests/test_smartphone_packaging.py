from copy import deepcopy

import numpy as np
import pytest
from PIL import Image, ImageCms

from mediflow_datasets import domain_adaptation_experiment as experiment
from mediflow_datasets import smartphone_package_inference as inference
from mediflow_datasets.smartphone_packaging import preprocessing, verify_selection
from mediflow_datasets.smartphone_profile import canonical_input


def test_phone_exif_and_icc_match_calibration_but_source_matches_frozen_test(tmp_path):
    pixels = np.random.default_rng(42).integers(0, 256, (47, 69, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[274] = 6
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = tmp_path / "phone.jpg"
    Image.fromarray(pixels).save(path, exif=exif, icc_profile=icc)
    metadata = preprocessing()
    phone = inference.load_image(path, metadata).numpy()[0]
    source = inference.load_image(path, metadata, "source").numpy()[0]
    np.testing.assert_array_equal(phone, canonical_input(path, 256))
    np.testing.assert_array_equal(
        source, experiment.image_tensor(path.read_bytes(), 256, "center_square")
    )
    assert phone.dtype == np.float32 and phone.max() > 1
    assert not np.array_equal(phone, source)


def test_phone_corrupt_icc_invalid_image_and_changed_contract_fail_explicitly(tmp_path):
    path = tmp_path / "broken.png"
    Image.new("RGB", (20, 30), (110, 130, 150)).save(path, icc_profile=b"not a profile")
    with pytest.raises(ValueError, match="ICC"):
        inference.load_image(path, preprocessing())
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="decode"):
        inference.load_image(path, preprocessing())
    with pytest.raises(ValueError, match="contract"):
        inference.load_image(path, {**preprocessing(), "external_division_by_255": True})
    with pytest.raises(FileNotFoundError):
        inference.load_image(tmp_path / "missing.jpg", preprocessing())


def test_test_cannot_reselect_or_supply_another_checkpoint_metrics():
    plan = {
        "models": {"stage4": {"sha256": "fixed"}},
        "selected_on_validation": "stage4",
        "test_used_for_selection": False,
        "task": {"classes": ["normal", "other"]},
        "test_count": 400,
    }
    metrics = {
        c: {"accuracy": 0.5}
        for c in ("clean", "phone_lighting", "phone_expanded", "phone_refit", "phone_joint")
    }
    result = {
        "selected_on_validation": "stage4",
        "test_used_for_selection": False,
        "classes": ["normal", "other"],
        "test_count": 400,
        "metrics": {"stage4": metrics},
    }
    selection = {"model": "stage4", "sha256": "fixed"}
    assert verify_selection({"plan": plan}, result, selection, "fixed") == metrics
    for altered in ("other-checkpoint", "stage5"):
        with pytest.raises(ValueError, match="후보"):
            verify_selection({"plan": plan}, result, selection, altered)
    changed = deepcopy(result)
    changed["test_used_for_selection"] = True
    with pytest.raises(ValueError, match="후보"):
        verify_selection({"plan": plan}, changed, selection, "fixed")
