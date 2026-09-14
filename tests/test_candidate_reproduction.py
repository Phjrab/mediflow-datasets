import json

import numpy as np
import pytest
from PIL import Image

from mediflow_datasets.candidate_reproduction import compare, preprocess, run


@pytest.mark.parametrize("suffix", ["png", "jpg"])
def test_preprocess_matches_training_loader(tmp_path, suffix):
    tf = pytest.importorskip("tensorflow")
    folder = tmp_path / "sample"
    folder.mkdir()
    path = folder / ("pattern." + suffix)
    pixels = np.random.default_rng(42).integers(0, 256, (31, 57, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    dataset = tf.keras.utils.image_dataset_from_directory(
        tmp_path, labels=None, shuffle=False, image_size=(224, 224), batch_size=1
    )
    expected = next(iter(dataset)).numpy()
    actual = preprocess(path, 224).numpy()
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert actual.max() > 1


def test_compare_rejects_wrong_labels_inputs_and_scores(tmp_path):
    fields = dict(
        domain="hair",
        image="sample.png",
        image_sha256="a",
        model_sha256="b",
        candidate_id="id",
        class_names=["a", "b"],
        normal_class_included=False,
        input_shape=[1, 2, 2, 3],
        predicted_index=0,
        tensor="input.npy",
        scores=[0.7, 0.3],
    )
    reference = {"protocol": "v1", "cases": {"hair_0": fields}}
    np.save(tmp_path / "input.npy", np.full((1, 2, 2, 3), 255, dtype=np.float32))
    assert not compare(reference, reference, tmp_path, tmp_path)
    actual = json.loads(json.dumps(reference))
    actual["cases"]["hair_0"]["class_names"].reverse()
    actual["cases"]["hair_0"]["scores"] = [float("nan"), 0.3]
    actual["cases"]["hair_0"]["tensor"] = "normalized.npy"
    np.save(tmp_path / "normalized.npy", np.ones((1, 2, 2, 3), dtype=np.float32))
    failures = compare(reference, actual, tmp_path, tmp_path)
    assert len(failures) == 3


def test_output_is_never_overwritten(tmp_path):
    pytest.importorskip("tensorflow")
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert marker.read_text() == "keep"
