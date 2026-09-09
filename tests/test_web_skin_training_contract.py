"""Check real backbone wiring and notebook input handling without remote data."""

import ast
import json
import zipfile
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from mediflow_datasets import experiment_suite as suite

ROOT = Path(__file__).resolve().parents[1]


def read_archived_notebook(filename):
    path = ROOT / "notebooks" / filename
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    archive = ROOT / "notebooks" / "legacy_notebooks_20260909.zip"
    with zipfile.ZipFile(archive) as bundle:
        return json.loads(bundle.read(f"legacy_notebooks/{filename}").decode("utf-8"))


@pytest.mark.parametrize("backbone,size,loss", [("B0", 224, "ce"), ("B1", 256, "ls005")])
def test_real_efficientnet_input_freezing_and_gradient(backbone, size, loss, monkeypatch):
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    name = "EfficientNet" + backbone
    actual_builder = getattr(keras.applications, name)

    def offline_builder(**kwargs):
        assert kwargs["weights"] == "imagenet"
        assert kwargs["input_shape"] == (size, size, 3)
        # Exercise the real architecture; pretrained weights need a download.
        return actual_builder(**{**kwargs, "weights": None})

    monkeypatch.setattr(keras.applications, name, offline_builder)
    model = suite.build_model({"backbone": backbone, "size": size, "loss": loss})
    base = next(layer for layer in model.layers if isinstance(layer, keras.Model))
    assert model.output_shape == (None, 5)
    assert not base.trainable_variables
    rescaling = next(layer for layer in base.layers if isinstance(layer, keras.layers.Rescaling))
    np.testing.assert_allclose(rescaling(np.array([255.0])).numpy(), [1.0])
    suite.configure_partial(model, loss)
    for index, layer in enumerate(base.layers):
        assert layer.trainable == (
            index >= len(base.layers) - 30
            and not isinstance(layer, keras.layers.BatchNormalization)
        )
    moving = [
        v
        for layer in base.layers
        if isinstance(layer, keras.layers.BatchNormalization)
        for v in (layer.moving_mean, layer.moving_variance)
    ]
    before_moving = [v.numpy().copy() for v in moving]
    before_trainable = [v.numpy().copy() for v in base.trainable_variables]
    batch = np.random.default_rng(42).uniform(0, 255, (1, size, size, 3)).astype("float32")
    result = model.train_on_batch(batch, np.eye(5, dtype="float32")[[0]])
    assert np.isfinite(result).all()
    for before, after in zip(before_moving, moving, strict=True):
        np.testing.assert_array_equal(before, after.numpy())
    assert any(
        not np.array_equal(before, after.numpy())
        for before, after in zip(before_trainable, base.trainable_variables, strict=True)
    )
    keras.backend.clear_session()


@pytest.mark.parametrize(
    "filename",
    [
        "web_skin_all_experiments_colab.ipynb",
        "web_skin_all_experiments_reviewed_colab.ipynb",
    ],
)
def test_notebook_loader_keeps_web_skin_labels_paths_and_pixel_scale(tmp_path, filename):
    notebook = read_archived_notebook(filename)
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "def dataset_factory(" in "".join(c["source"])
    )
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "dataset_factory"
    )
    for index, name in enumerate(suite.CLASSES):
        directory = tmp_path / "val" / name
        directory.mkdir(parents=True)
        Image.new("RGB", (17, 19), (index * 40 + 20,) * 3).save(directory / "sample.png")
    namespace = dict(
        DATA_ROOT=tmp_path,
        CLASSES=suite.CLASSES,
        BATCH_SIZE=2,
        SEED=42,
        keras=keras,
        tf=tf,
        Path=Path,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), "loader", "exec"), namespace)
    dataset, paths = namespace["dataset_factory"]("val", 224, False)
    labels, values = [], []
    for images, targets in dataset:
        assert images.dtype == tf.float32
        assert images.shape[1:] == (224, 224, 3)
        labels.extend(targets.numpy().argmax(axis=1).tolist())
        values.extend(images.numpy()[:, 0, 0, 0].tolist())
    assert labels == list(range(5))
    np.testing.assert_allclose(values, [20, 60, 100, 140, 180])
    assert [Path(p).parts[1] for p in paths] == suite.CLASSES
    reported = json.loads((ROOT / "results/web_skin/augmented/results.json").read_text("utf-8"))
    assert suite.CLASSES == reported["classes"]


def test_reviewed_notebook_syntax_and_engine_match():
    notebook = read_archived_notebook("web_skin_all_experiments_reviewed_colab.ipynb")
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(
            "\n".join(line for line in source.splitlines() if not line.startswith("%pip ")),
            "reviewed-cell",
            "exec",
        )
        if source.startswith("ENGINE_SOURCE = "):
            assert ast.literal_eval(ast.parse(source).body[0].value) == Path(
                suite.__file__
            ).read_text("utf-8")


@pytest.mark.parametrize("loss_name", ["ce", "ls005", "focal15"])
def test_loss_checkpoint_round_trip(loss_name, tmp_path):
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(5, activation="softmax")])
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss=suite.loss_function(loss_name))
    x, y = np.ones((1, 3), dtype="float32"), np.eye(5, dtype="float32")[[2]]
    model.train_on_batch(x, y)
    path = tmp_path / "loss.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    assert int(restored.optimizer.iterations.numpy()) == 1
    assert restored.loss.get_config() == model.loss.get_config()
    np.testing.assert_allclose(model(x).numpy(), restored(x).numpy())
