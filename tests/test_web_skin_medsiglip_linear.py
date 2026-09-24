import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from mediflow_datasets import web_skin_medsiglip_linear as module

ROOT = Path(__file__).resolve().parents[1]


def context(tmp_path):
    return {
        "config": {
            "domain": "web_skin",
            "mode": "web_skin_medsiglip_linear",
            "train_variant": "augmented",
            "seed": 42,
            "seeds": [42],
            "experiments": module.EXPERIMENTS,
            "model_id": module.MODEL_ID,
            "batch_size": 8,
            "epochs1": 50,
            "epochs2": 1,
            "extension_epochs": 1,
            "embedding_shard_size": 128,
            "linear_batch_size": 256,
            "linear_learning_rate": 1e-3,
            "linear_weight_decay": 1e-4,
        },
        "classes": module.EXPECTED_CLASSES,
        "data_hash": module.BASELINE["data_sha256"],
        "output": tmp_path,
    }


def test_contract_is_one_validation_only_experiment(tmp_path):
    assert module.validate(context(tmp_path)) is None
    assert module.EXPERIMENTS == ["medsiglip_448_frozen_linear_seed_42"]
    assert module.BASELINE["validation_accuracy"] == pytest.approx(0.85)
    assert module.BASELINE["validation_macro_f1"] == pytest.approx(0.8473279632397033)
    source = inspect.getsource(module.run)
    assert '_extract_split(context, "train"' in source
    assert '_extract_split(context, "val"' in source
    assert '_extract_split(context, "test"' not in source


def test_contract_rejects_changes(tmp_path):
    value = context(tmp_path)
    value["config"]["model_id"] = "another/model"
    with pytest.raises(ValueError, match="model_id"):
        module.validate(value)
    value = context(tmp_path)
    value["config"]["batch_size"] = 16
    with pytest.raises(ValueError, match="batch_size"):
        module.validate(value)
    value = context(tmp_path)
    value["data_hash"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        module.validate(value)


def test_shard_round_trip_and_mismatch(tmp_path):
    path = tmp_path / "shard.npz"
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    labels = np.asarray([0, 1, 2])
    paths = ["a.jpg", "b.jpg", "c.jpg"]
    module._save_shard(path, embeddings, labels, paths)
    loaded = module._read_shard(path, paths, labels)
    np.testing.assert_array_equal(loaded, embeddings)
    assert module._read_shard(path, ["changed.jpg"] * 3, labels) is None


def test_selection_rule_uses_macro_f1_then_accuracy():
    record = {"validation": {"accuracy": 0.80, "macro_f1": 0.848}}
    assert module._selected(record)
    record = {"validation": {"accuracy": 0.851, "macro_f1": module.BASELINE["validation_macro_f1"]}}
    assert module._selected(record)
    record = {"validation": {"accuracy": 0.90, "macro_f1": 0.84}}
    assert not module._selected(record)


def test_notebook_embeds_current_sources_and_fixed_settings():
    path = ROOT / "notebooks/14_web_skin_medsiglip_linear_probe_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 13
    all_source = "\n".join("".join(item["source"]) for item in notebook["cells"])
    for expected in (
        "MODE = 'web_skin_medsiglip_linear'",
        "BATCH_SIZE = 8",
        "LINEAR_EPOCHS = 50",
        "EMBEDDING_SHARD_SIZE = 128",
        "MODEL_ID = 'google/medsiglip-448'",
        "medsiglip_448_frozen_linear_seed_42",
        "web_skin_datasets.zip",
    ):
        assert expected in all_source
    for item in notebook["cells"]:
        if item["cell_type"] != "code":
            continue
        assert item["outputs"] == []
        source = "".join(item["source"])
        python_source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("%pip ")
        )
        compile(
            python_source,
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
            expected = (ROOT / f"src/mediflow_datasets/{name}.py").read_text(encoding="utf-8")
            assert code == expected
