import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from mediflow_datasets import web_skin_paper_suite, web_skin_wsdan

ROOT = Path(__file__).resolve().parents[1]


def context(tmp_path, mode="web_skin_paper_suite"):
    experiments = (
        web_skin_paper_suite.EXPERIMENTS
        if mode == "web_skin_paper_suite"
        else [web_skin_wsdan.TRIAL_ID]
    )
    return {
        "config": {
            "domain": "web_skin",
            "mode": mode,
            "train_variant": "augmented",
            "seed": 42,
            "seeds": [42],
            "experiments": experiments,
            "attention_maps": 8,
            "crop_threshold": 0.5,
            "drop_threshold": 0.7,
            "pmg_jigsaw_grids": [8, 4, 2],
            "mixstyle_alpha": 0.1,
            "mixstyle_probability": 0.5,
        },
        "classes": web_skin_wsdan.EXPECTED_CLASSES,
        "data_hash": web_skin_wsdan.BASELINE["data_sha256"],
        "output": tmp_path,
    }


def test_suite_contract_has_exact_three_methods_and_no_test_access(tmp_path):
    assert web_skin_paper_suite.validate(context(tmp_path)) is None
    assert web_skin_paper_suite.EXPERIMENTS == [
        "wsdan_b0_256_ce_seed_42",
        "pmg_b0_256_ce_seed_42",
        "mixstyle_b0_256_ce_seed_42",
    ]
    source = inspect.getsource(web_skin_paper_suite.run)
    assert '"test"' not in source
    assert web_skin_wsdan.BASELINE["validation_accuracy"] == 0.796
    assert web_skin_wsdan.BASELINE["validation_macro_f1"] == 0.7915394647566528


def test_suite_rejects_different_data_and_method_list(tmp_path):
    value = context(tmp_path)
    value["data_hash"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        web_skin_paper_suite.validate(value)
    value = context(tmp_path)
    value["config"]["experiments"] = value["config"]["experiments"][:-1]
    with pytest.raises(ValueError, match="세 조건"):
        web_skin_paper_suite.validate(value)


def test_wsdan_model_output_and_serialization(tmp_path):
    model, probe = web_skin_wsdan.build_model(
        5, size=64, attention_maps=4, weights=None
    )
    images = tf.zeros((2, 64, 64, 3), dtype=tf.float32)
    predictions, attention = probe(images, training=False)
    assert predictions.shape == (2, 5)
    assert attention.shape[-1] == 4
    assert tf.reduce_all(tf.math.is_finite(predictions))
    ensemble = web_skin_wsdan.AttentionEnsemblePredictor(model, 64, 0.5)
    assert ensemble(images).shape == (2, 5)
    path = tmp_path / "wsdan.keras"
    model.save(path)
    assert tf.keras.models.load_model(path, compile=False)(images).shape == (2, 5)


def test_pmg_contract_jigsaw_outputs_and_serialization(tmp_path):
    images = tf.reshape(tf.range(2 * 64 * 64 * 3, dtype=tf.float32), (2, 64, 64, 3))
    shuffled = web_skin_paper_suite._jigsaw(images, 8)
    assert shuffled.shape == images.shape
    assert np.allclose(
        np.sort(shuffled.numpy().reshape(2, -1), axis=1),
        np.sort(images.numpy().reshape(2, -1), axis=1),
    )
    model = web_skin_paper_suite.build_pmg_model(5, size=64, weights=None)
    outputs = model(tf.zeros((2, 64, 64, 3)))
    assert len(outputs) == 4 and all(value.shape == (2, 5) for value in outputs)
    predictor = web_skin_paper_suite._pmg_predictor(model)
    assert predictor(tf.zeros((2, 64, 64, 3))).shape == (2, 5)
    path = tmp_path / "pmg.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    assert len(restored(tf.zeros((1, 64, 64, 3)))) == 4


def test_mixstyle_changes_training_only_and_roundtrips(tmp_path):
    layer = web_skin_paper_suite.MixStyle(probability=1.0, alpha=0.1)
    images = tf.reshape(tf.range(4 * 8 * 8 * 3, dtype=tf.float32), (4, 8, 8, 3))
    assert np.array_equal(layer(images, training=False).numpy(), images.numpy())
    assert not np.array_equal(layer(images, training=True).numpy(), images.numpy())
    model = web_skin_paper_suite.build_mixstyle_model(5, size=64, weights=None)
    assert model.jit_compile is False
    path = tmp_path / "mixstyle.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    probabilities = restored(tf.zeros((2, 64, 64, 3)), training=False)
    assert probabilities.shape == (2, 5)
    assert tf.reduce_all(tf.abs(tf.reduce_sum(probabilities, axis=1) - 1) < 1e-5)


def test_official_method_components_are_recorded():
    source = inspect.getsource(web_skin_paper_suite)
    assert "[8, 4, 2]" in source
    assert "MixStyle" in source
    assert source.count("jit_compile=False") >= 2
    assert set(web_skin_paper_suite.PAPERS) == {"wsdan", "pmg", "mixstyle"}
    summary_source = inspect.getsource(web_skin_paper_suite.summarize)
    assert '"completed_experiments"' in summary_source
    assert '"omitted_experiments"' in summary_source
    assert "len(records)" in summary_source


def test_web_skin_notebook_embeds_current_sources():
    path = ROOT / "notebooks/11_web_skin_wsdan_attention_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 13
    all_source = "\n".join("".join(item["source"]) for item in notebook["cells"])
    assert "MODE = 'web_skin_paper_suite'" in all_source
    assert "PMG_JIGSAW_GRIDS = [8, 4, 2]" in all_source
    assert "MIXSTYLE_ALPHA = 0.1" in all_source
    assert "HF_TOKEN" not in all_source
    assert "DERM_MODEL_ID" not in all_source
    assert (
        "DATA_ZIP = '/content/drive/MyDrive/mediflow_Project/datasets/"
        "web_skin_datasets.zip'"
    ) in all_source
    assert "자율설계2/web_skin_processed.zip" not in all_source
    assert "force_remount=True" in all_source
    assert "zipfile.is_zipfile(data_path)" in all_source
    for item in notebook["cells"]:
        if item["cell_type"] != "code":
            continue
        assert item["outputs"] == []
        source = "".join(item["source"])
        compile(
            "\n".join(line for line in source.splitlines() if not line.startswith("%pip ")),
            path.name,
            "exec",
        )
        if "SOURCES = " not in source:
            continue
        assignments = {
            node.targets[0].id: node.value
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        embedded = ast.literal_eval(assignments["SOURCES"])
        for name, code in embedded.items():
            expected = (ROOT / f"src/mediflow_datasets/{name}.py").read_text(
                encoding="utf-8"
            )
            assert code == expected
