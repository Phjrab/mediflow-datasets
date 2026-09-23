import ast
import inspect
import json
from pathlib import Path

import pytest
import tensorflow as tf

from mediflow_datasets import (
    web_skin_paper_suite,
    web_skin_pmg_b1_384,
    web_skin_wsdan,
)

ROOT = Path(__file__).resolve().parents[1]


def context(tmp_path):
    return {
        "config": {
            "domain": "web_skin",
            "mode": "web_skin_pmg_b1_384",
            "train_variant": "augmented",
            "seed": 42,
            "seeds": [42],
            "experiments": web_skin_pmg_b1_384.EXPERIMENTS,
            "pmg_jigsaw_grids": [8, 4, 2],
            "batch_size": 16,
            "epochs1": 15,
            "epochs2": 10,
        },
        "classes": web_skin_wsdan.EXPECTED_CLASSES,
        "data_hash": web_skin_pmg_b1_384.BASELINE["data_sha256"],
        "output": tmp_path,
    }


def test_contract_is_single_validation_experiment(tmp_path):
    assert web_skin_pmg_b1_384.validate(context(tmp_path)) is None
    assert web_skin_pmg_b1_384.EXPERIMENTS == ["pmg_b1_384_ce_seed_42"]
    assert web_skin_pmg_b1_384.BASELINE["validation_macro_f1"] == pytest.approx(
        0.8473279632397033
    )
    assert '"test"' not in inspect.getsource(web_skin_pmg_b1_384.run)
    assert '"test"' not in inspect.getsource(web_skin_pmg_b1_384.summarize)


def test_contract_rejects_batch_epoch_and_data_changes(tmp_path):
    value = context(tmp_path)
    value["config"]["batch_size"] = 32
    with pytest.raises(ValueError, match="batch size"):
        web_skin_pmg_b1_384.validate(value)
    value = context(tmp_path)
    value["config"]["epochs2"] = 15
    with pytest.raises(ValueError, match="Stage 1=15"):
        web_skin_pmg_b1_384.validate(value)
    value = context(tmp_path)
    value["data_hash"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        web_skin_pmg_b1_384.validate(value)


def test_pmg_b1_model_contract():
    model = web_skin_paper_suite.build_pmg_model(
        5, size=64, weights=None, backbone_name="B1"
    )
    assert model.name == "web_skin_pmg_b1"
    images = tf.zeros((1, 64, 64, 3), dtype=tf.float32)
    outputs = model(images, training=False)
    assert len(outputs) == 4
    assert all(output.shape == (1, 5) for output in outputs)
    probabilities = web_skin_paper_suite._pmg_predictor(model)(images, training=False)
    assert probabilities.shape == (1, 5)
    assert tf.reduce_all(tf.math.is_finite(probabilities))


def test_notebook_embeds_current_sources_and_fixed_settings():
    path = ROOT / "notebooks/12_web_skin_pmg_b1_384_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 13
    all_source = "\n".join("".join(item["source"]) for item in notebook["cells"])
    for expected in (
        "MODE = 'web_skin_pmg_b1_384'",
        "BATCH_SIZE = 16",
        "STAGE1_EPOCHS = 15",
        "STAGE2_EPOCHS = 10",
        "RUN_EXPERIMENTS = ['pmg_b1_384_ce_seed_42']",
        "PMG_JIGSAW_GRIDS = [8, 4, 2]",
        "web_skin_datasets.zip",
    ):
        assert expected in all_source
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
