import io
import json
import zipfile
from dataclasses import asdict

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from mediflow_datasets import adaptation_final_test as final


@pytest.fixture
def source(tmp_path):
    buffer = io.BytesIO()
    Image.fromarray(np.arange(36 * 48 * 3, dtype=np.uint8).reshape(36, 48, 3)).save(
        buffer, format="PNG"
    )
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "x") as writer:
        for variant in ("original", "augmented"):
            for split in ("train", "val", "test"):
                for name in ("C0/a.png", "C1/a.png", "C1/b.png"):
                    writer.writestr(f"dataset/{variant}/{split}/{name}", buffer.getvalue())
    task = {
        "domain": "skin",
        "classes": ["C1", "C0"],
        "source_sha256": final.digest(archive),
        "batch_size": 2,
        "input_policy": "center_square",
    }
    index, _ = final.data.source_index(archive, task["classes"], task["source_sha256"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"original": {"test": [asdict(r) for r in index["original", "test"]]}}),
        encoding="utf-8",
    )
    lock = {"manifest_sha256": final.digest(manifest), "test_count": 3}
    return archive, task, lock, manifest


def test_final_test_requires_original_paths_labels_count_and_pinned_archive(source):
    archive, task, lock, manifest = source
    rows = final.prepare_test(archive, task, lock, manifest)
    assert [r.label for r in rows] == [1, 0, 0]
    assert all(r.split == "test" and r.variant == "original" for r in rows)
    with pytest.raises(ValueError, match="SHA-256"):
        final.prepare_test(archive, {**task, "source_sha256": "0" * 64}, lock, manifest)
    with pytest.raises(ValueError, match="Test rows"):
        final.prepare_test(archive, task, {**lock, "test_count": 4}, manifest)
    declared = json.loads(manifest.read_bytes())
    declared["original"]["test"][0]["label"] = 0
    manifest.write_text(json.dumps(declared), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        final.prepare_test(archive, task, lock, manifest)
    with pytest.raises(ValueError, match="Test rows"):
        final.prepare_test(
            archive, task, {**lock, "manifest_sha256": final.digest(manifest)}, manifest
        )


def test_test_arms_receive_identical_rgb_inputs_and_deterministic_short_batches(source):
    archive, task, lock, manifest = source
    rows = final.prepare_test(archive, task, lock, manifest)
    profile = {
        "lighting": {
            "brightness_offset": [3, 5],
            "contrast": [0.97, 1.04],
            "left_light": [0, 2],
            "photometric_operator": "bounded_additive_v3",
        },
        "color_a": [-1, 1],
        "color_b": [-1, 1],
        "texture_gain": [0.9, 1.1],
        "blur_sigma_roi64": [0, 0.5],
        "blur_input_scale": 1,
    }
    calls = {"initial": [], "adapted": []}

    def predictor(name):
        def predict(images):
            calls[name].append(images.numpy().copy())
            return tf.convert_to_tensor(np.tile([0.25, 0.75], (len(images), 1)), tf.float32)

        return predict

    predictors = {name: predictor(name) for name in calls}
    reference = {"extended_profile": profile}
    before = final.digest(archive)
    first = final.evaluate_zip(archive, rows, task, {"size": 16}, reference, predictors, 1707)
    assert len(calls["initial"]) == 6
    assert [len(b) for b in calls["initial"]] == [2, 2, 2, 1, 1, 1]
    for a, b in zip(calls["initial"], calls["adapted"], strict=True):
        assert np.array_equal(a, b)
        assert a.dtype == np.float32 and a.shape[1:] == (16, 16, 3)
        assert 0 <= a.min() < a.max() <= 255 and a.max() > 1
    with zipfile.ZipFile(archive) as reader:
        expected = final.experiment.image_tensor(reader.read(rows[0].path), 16, "center_square")
    assert np.array_equal(calls["initial"][0][0], expected)
    assert not np.array_equal(calls["initial"][0], calls["initial"][1])
    second = final.evaluate_zip(archive, rows, task, {"size": 16}, reference, predictors, 1707)
    assert first == second and final.digest(archive) == before
    assert all(
        np.array_equal(a, b)
        for a, b in zip(calls["initial"][:6], calls["initial"][6:], strict=True)
    )
    for metrics in first["initial"].values():
        assert metrics["labeled_count"] == 3
        assert metrics["support"] == [2, 1]
        assert metrics["confusion_matrix"] == [[0, 2], [0, 1]]
