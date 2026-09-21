import ast
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from mediflow_datasets import paper_suite

ROOT = Path(__file__).resolve().parents[1]


def context(tmp_path):
    return {
        "config": {
            "domain": "hair",
            "mode": "paper_suite",
            "train_variant": "augmented",
            "seeds": [42, 43, 44],
            "experiments": ["baseline_b1_256", "supcon_b1_256"],
        },
        "classes": ["a", "b", "c", "d", "e"],
        "output": tmp_path,
        "signature": "fixed",
    }


def test_suite_contract_and_order(tmp_path):
    value = context(tmp_path)
    assert paper_suite.validate_suite(value) == ["baseline_b1_256", "supcon_b1_256"]
    value["config"]["seeds"] = [42]
    with pytest.raises(ValueError, match="SEEDS"):
        paper_suite.validate_suite(value)


def test_suite_dispatches_every_experiment_and_seed(tmp_path, monkeypatch):
    calls = []

    def fake_standard(value, spec):
        calls.append((spec["experiment_id"], spec["seed"], "standard"))
        return {"id": spec["id"], "spec": spec}

    def fake_supcon(value, spec):
        calls.append((spec["experiment_id"], spec["seed"], "supcon"))
        return {"id": spec["id"], "spec": spec}

    monkeypatch.setattr(paper_suite, "_run_standard", fake_standard)
    monkeypatch.setattr(paper_suite, "_run_supcon", fake_supcon)
    records = paper_suite.run(context(tmp_path))
    assert len(records) == 6
    assert calls == [
        ("baseline_b1_256", 42, "standard"),
        ("baseline_b1_256", 43, "standard"),
        ("baseline_b1_256", 44, "standard"),
        ("supcon_b1_256", 42, "supcon"),
        ("supcon_b1_256", 43, "supcon"),
        ("supcon_b1_256", 44, "supcon"),
    ]


def test_supcon_loss_is_finite_and_rewards_same_class_alignment():
    labels = tf.one_hot([0, 0, 1, 1], 2)
    aligned = tf.constant([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    mixed = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    aligned_loss = float(paper_suite.supervised_contrastive_loss(labels, aligned))
    mixed_loss = float(paper_suite.supervised_contrastive_loss(labels, mixed))
    assert np.isfinite(aligned_loss)
    assert aligned_loss < mixed_loss


def test_paper_suite_notebook_embeds_current_sources():
    path = ROOT / "notebooks/05_hair_paper_experiment_suite_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 13
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
