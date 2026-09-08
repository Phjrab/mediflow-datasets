"""Offline checks for selection, validation-only execution, and safe trial reuse."""

import json

import keras
import numpy as np
import pytest
import tensorflow as tf

from mediflow_datasets import experiment_suite as suite


def test_metrics_keep_missing_classes_and_reject_bad_arrays():
    truth = np.array([0, 0, 1])
    probabilities = np.eye(5)[[0, 1, 1]]
    metrics = suite.classification_metrics(truth, probabilities)
    assert metrics["accuracy"] == 2 / 3
    assert metrics["support"] == [2, 1, 0, 0, 0]
    assert metrics["class_f1"] == pytest.approx([2 / 3, 2 / 3, 0, 0, 0])
    assert metrics["macro_f1"] == pytest.approx(4 / 15)
    with pytest.raises(ValueError):
        suite.classification_metrics([], np.zeros((0, 5)))
    with pytest.raises(ValueError):
        suite.classification_metrics([0], np.full((1, 5), np.nan))


def test_selection_uses_validation_accuracy_and_keeps_ties():
    records = [
        {"id": "earlier", "validation": {"accuracy": 0.7, "macro_f1": 0.6}},
        {"id": "later", "validation": {"accuracy": 0.7, "macro_f1": 0.9}},
        {"id": "lower", "validation": {"accuracy": 0.6, "macro_f1": 1}},
    ]
    assert suite.select_winner(records)["id"] == "earlier"
    assert not suite.checkpoint_choice(0.7, 0.7)
    assert not suite.checkpoint_choice(0.7, 0.6)
    assert suite.checkpoint_choice(0.7, 0.8)


def tiny_model(spec):
    inputs = keras.Input((8, 8, 3))
    x = keras.layers.Rescaling(1 / 255)(inputs)
    x = keras.layers.Conv2D(2, 1)(x)
    x = keras.layers.BatchNormalization()(x)
    backbone = keras.Model(inputs, x, name="efficientnet_tiny")
    backbone.trainable = False
    outer = keras.Input((8, 8, 3))
    features = keras.layers.GlobalAveragePooling2D()(backbone(outer, training=False))
    outputs = keras.layers.Dense(5, activation="softmax")(features)
    model = keras.Model(outer, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=suite.loss_function(spec["loss"]),
        metrics=["accuracy"],
    )
    return model


def test_trial_extension_resume_and_artifact_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "build_model", tiny_model)
    calls = []

    def datasets(split, size, shuffle):
        assert split in {"train", "val"}, "Test must not be accessed while selecting trials"
        calls.append(split)
        images = np.full((5, 8, 8, 3), 100, dtype=np.float32)
        labels = np.eye(5, dtype=np.float32)
        return tf.data.Dataset.from_tensor_slices((images, labels)).batch(5), [
            f"{split}/image{i}.png" for i in range(5)
        ]

    spec = {"id": "tiny", "backbone": "B1", "size": 8, "loss": "ls005"}
    record = suite.run_trial(spec, datasets, tmp_path, "test", epochs1=1, epochs2=1)
    assert record["validation"]["count"] == 5
    assert len(record["history"]["accuracy"]) == 2
    selected = suite.selected_model_path(tmp_path, record)
    assert selected.is_file()
    prior_calls = len(calls)
    assert suite.run_trial(spec, datasets, tmp_path, "test") == record
    assert len(calls) == prior_calls
    with pytest.raises(ValueError, match="settings"):
        suite.cached_record(tmp_path, "tiny", "changed")
    extended = suite.extend_b1(record, datasets, tmp_path, "test", epochs=1)
    assert extended["optimizer_restored"]
    assert len(extended["history"]["accuracy"]) == 3
    assert suite.extend_b1(record, datasets, tmp_path, "test", epochs=1) == extended
    restored = keras.models.load_model(
        tmp_path / extended["id"] / extended["attempt"] / "extension_last.keras"
    )
    assert int(restored.optimizer.iterations.numpy()) == 2
    backbone = next(layer for layer in restored.layers if isinstance(layer, keras.Model))
    assert all(
        not layer.trainable
        for layer in backbone.layers
        if isinstance(layer, keras.layers.BatchNormalization)
    )
    # Corruption must be detected rather than silently treating the trial as complete.
    selected.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="corrupted"):
        suite.cached_record(tmp_path, "tiny", "test")


def test_interrupted_attempt_is_preserved(tmp_path, monkeypatch):
    directory = tmp_path / "tiny" / "attempt_interrupted"
    directory.mkdir(parents=True)
    file = directory / "stage1_log.csv"
    file.write_text("old incomplete log", encoding="utf-8")
    assert suite.cached_record(tmp_path, "tiny", "test") is None
    assert file.read_text(encoding="utf-8") == "old incomplete log"


def test_notebook_embeds_exact_engine_and_has_no_repeated_audit():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks/web_skin_all_experiments_colab.ipynb").read_text(encoding="utf-8")
    )
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for index, source in enumerate(sources):
        # Colab's pip magic is not Python syntax.
        code = "\n".join(line for line in source.splitlines() if not line.startswith("%pip "))
        compile(code, f"cell-{index}", "exec")
    embedded = next(source for source in sources if source.startswith("ENGINE_SOURCE = "))
    import ast

    assignment = ast.parse(embedded).body[0]
    assert ast.literal_eval(assignment.value) == Path(suite.__file__).read_text(encoding="utf-8")
    assert "pixel_sha256" not in "\n".join(sources)
    assert "f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d" in "\n".join(sources)
