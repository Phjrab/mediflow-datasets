"""Protect preprocessing, PMG aggregation and checkpoint/metric identity at delivery."""

import copy
import json
import stat
import zipfile
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from mediflow_datasets import adaptation_package_inference as inference
from mediflow_datasets import adaptation_packaging as packaging
from mediflow_datasets.domain_adaptation_experiment import image_tensor


def preprocessing(policy="legacy_resize"):
    return {
        "input_size": 8,
        "input_policy": policy,
        "dtype": "float32",
        "color": "RGB",
        "layout": "NHWC",
        "pixel_range": [0, 255],
        "resize_method": "bilinear",
        "antialias": False,
        "exif_rotation_applied": False,
        "external_division_by_255": False,
    }


@pytest.mark.parametrize("policy", ["legacy_resize", "center_square"])
@pytest.mark.parametrize(
    "mode,shape", [("RGB", (13, 19, 3)), ("L", (19, 13)), ("RGBA", (13, 19, 4))]
)
def test_image_decoding_matches_frozen_test_with_odd_crop(tmp_path, policy, mode, shape):
    pixels = np.random.default_rng(42).integers(0, 256, shape, dtype=np.uint8)
    path = tmp_path / "입력.png"
    Image.fromarray(pixels).save(path)
    assert Image.open(path).mode == mode
    actual = inference.load_image(path, preprocessing(policy)).numpy()
    expected = image_tensor(path.read_bytes(), 8, policy)
    np.testing.assert_array_equal(actual[0], expected)
    assert actual.dtype == np.float32 and actual.max() > 1


def test_exif_orientation_is_not_silently_added(tmp_path):
    pixels = np.random.default_rng(7).integers(0, 256, (13, 19, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)
    np.testing.assert_array_equal(
        inference.load_image(path, preprocessing("center_square"))[0].numpy(),
        image_tensor(path.read_bytes(), 8, "center_square"),
    )


def tiny_package(folder, pmg=False):
    folder.mkdir()
    inputs = keras.Input((8, 8, 3))
    features = keras.layers.GlobalAveragePooling2D()(keras.layers.Rescaling(1 / 255)(inputs))
    outputs = []
    for index, bias in enumerate(([0, 1], [2, 0], [0.2, 0.7], [-0.1, 0.3])):
        output = keras.layers.Dense(
            2,
            activation=None if pmg else "softmax",
            kernel_initializer=keras.initializers.Constant(0.1),
            bias_initializer=keras.initializers.Constant(bias),
            name=f"head_{index}",
        )(features)
        outputs.append(output)
        if not pmg:
            break
    model = keras.Model(inputs, outputs if pmg else outputs[0])
    model.save(folder / "model.keras")
    card = {
        "package_id": "test_package",
        "domain": "web_skin" if pmg else "skin",
        "adaptation_stage": 4,
        "model_file": "model.keras",
        "model_sha256": inference.file_sha256(folder / "model.keras"),
        "class_names": ["클래스 가", "클래스 나"],
        "class_count": 2,
        "preprocessing": preprocessing(),
        "output_adapter": "sum_logits_softmax" if pmg else "direct_softmax",
        "normal_class_included": False,
    }
    packaging.write_json(folder / "model_card.json", card)
    packaging.write_json(folder / "class_names.json", card["class_names"])
    packaging.write_json(folder / "preprocessing.json", card["preprocessing"])
    packaging.manifest_for(folder, card["package_id"])
    return model


@pytest.mark.parametrize("pmg", [False, True])
def test_saved_package_scores_and_class_order_round_trip(tmp_path, pmg):
    folder = tmp_path / "package"
    original = tiny_package(folder, pmg)
    path = tmp_path / "white.png"
    Image.fromarray(np.full((9, 13, 3), 255, dtype=np.uint8)).save(path)
    inputs = inference.load_image(path, preprocessing())
    outputs = original(inputs, training=False)
    expected = tf.nn.softmax(tf.add_n(outputs)) if pmg else outputs
    loaded = inference.load_predictor(folder)
    prediction = loaded.predict_file(path)
    actual = [row["score"] for row in prediction["scores"]]
    np.testing.assert_allclose(actual, expected.numpy()[0], atol=1e-7)
    if pmg:
        mean_probabilities = np.mean([tf.nn.softmax(o).numpy()[0] for o in outputs], axis=0)
        assert not np.allclose(actual, mean_probabilities, atol=1e-3)
    assert prediction["predicted_class"] == loaded.classes[prediction["predicted_index"]]
    assert prediction["clinical_diagnosis"] is False
    assert prediction["input_scope_checked"] is False
    keras.backend.clear_session()


def test_metadata_and_model_corruption_are_rejected(tmp_path):
    folder = tmp_path / "package"
    tiny_package(folder)
    class_path = folder / "class_names.json"
    class_path.write_text('["바뀜", "클래스 나"]', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        inference.verify_package(folder)
    packaging.manifest_for(folder, "test_package")
    with pytest.raises(ValueError, match="Class order"):
        inference.verify_package(folder)
    card = packaging.read(folder / "model_card.json")
    class_path.write_text(json.dumps(card["class_names"]), encoding="utf-8")
    packaging.manifest_for(folder, "test_package")
    (folder / "model.keras").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        inference.verify_package(folder)
    keras.backend.clear_session()


def test_metrics_cannot_be_taken_from_a_different_hair_checkpoint():
    record = {
        "plan_sha256": "frozen_plan",
        "plan": {
            "domains": {
                "hair": {
                    "models": {
                        "stage2_expanded": {"sha256": "hair_stage2"},
                        "stage4_revised": {"sha256": "hair_stage4"},
                    },
                    "test_count": 1252,
                }
            }
        },
    }
    result = {
        "domain": "hair",
        "classes": ["a", "b"],
        "test_count": 1252,
        "identity": {"plan_sha256": "frozen_plan"},
        "metrics": {"stage4_revised": {"clean": {"accuracy": 995 / 1252}}},
    }
    with pytest.raises(ValueError, match="checkpoint"):
        packaging.metrics_for_checkpoint(
            record, result, "hair", "stage4_revised", "hair_stage2", ["a", "b"]
        )
    valid = copy.deepcopy(result)
    valid["metrics"]["stage2_expanded"] = {"clean": {"accuracy": 995 / 1252}}
    assert (
        packaging.metrics_for_checkpoint(
            record, valid, "hair", "stage2_expanded", "hair_stage2", ["a", "b"]
        )
        == valid["metrics"]["stage2_expanded"]
    )


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/file", "..\\escape"])
def test_extraction_rejects_escaping_entries(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(name, "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        packaging.extract_zip(archive, tmp_path / "extracted")


def test_manifest_paths_and_zip_symlinks_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        inference.package_path(Path(tmp_path).resolve(), "../outside")
    archive = tmp_path / "bad.zip"
    entry = zipfile.ZipInfo("link")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(entry, "../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        packaging.extract_zip(archive, tmp_path / "extracted")
