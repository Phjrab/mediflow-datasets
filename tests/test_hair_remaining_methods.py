import ast
import inspect
import json
from pathlib import Path

from mediflow_datasets import paper_suite

ROOT = Path(__file__).resolve().parents[1]


def screen_context(tmp_path):
    return {
        "config": {
            "domain": "hair",
            "mode": "paper_screen",
            "train_variant": "augmented",
            "seed": 42,
            "seeds": [42],
            "experiments": [
                "dinov2_small_224",
                "efficientnetv2s_256",
                "multires_b1_384",
            ],
        },
        "classes": ["a", "b", "c", "d", "e"],
        "output": tmp_path,
        "signature": "fixed",
    }


def test_screen_contract_contains_only_three_remaining_methods(tmp_path):
    assert paper_suite.validate_suite(screen_context(tmp_path)) == [
        "dinov2_small_224",
        "efficientnetv2s_256",
        "multires_b1_384",
    ]


def test_screen_dispatches_each_method_once(tmp_path, monkeypatch):
    calls = []

    def fake_run(context, spec):
        calls.append((spec["experiment_id"], spec["seed"], spec["method"]))
        return {"id": spec["id"], "spec": spec}

    monkeypatch.setattr(paper_suite, "_run_dinov2", fake_run)
    monkeypatch.setattr(paper_suite, "_run_standard", fake_run)
    records = paper_suite.run(screen_context(tmp_path))
    assert len(records) == 3
    assert calls == [
        ("dinov2_small_224", 42, "dinov2"),
        ("efficientnetv2s_256", 42, "standard"),
        ("multires_b1_384", 42, "standard"),
    ]


def test_dinov2_uses_token_pooling_instead_of_2d_pooling():
    source = inspect.getsource(paper_suite._build_dinov2_classifier)
    assert "GlobalAveragePooling2D" not in source
    assert "GlobalAveragePooling1D" in source
    assert "keep_cls_token" in source
    assert "remove_cls_token" in source
    assert "jit_compile=False" in source


def test_remaining_methods_notebook_embeds_current_sources():
    path = ROOT / "notebooks/08_hair_remaining_methods_screen_colab.ipynb"
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
