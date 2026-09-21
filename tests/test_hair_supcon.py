import ast
import json
from pathlib import Path

import pytest

from mediflow_datasets import hair_supcon

ROOT = Path(__file__).resolve().parents[1]


def comparison_context(tmp_path):
    return {
        "config": {
            "domain": "hair",
            "mode": "supcon_compare",
            "train_variant": "augmented",
            "seed": 42,
            "seeds": [42],
            "experiments": ["baseline_b1_256", "supcon_b1_256"],
        },
        "classes": ["a", "b", "c", "d", "e"],
        "output": tmp_path,
        "signature": "fixed",
    }


def test_protocol_rejects_extra_seed_or_condition(tmp_path):
    context = comparison_context(tmp_path)
    hair_supcon.validate(context)
    context["config"]["seeds"] = [42, 43]
    with pytest.raises(ValueError, match="seed 42"):
        hair_supcon.validate(context)


def test_run_executes_only_baseline_and_supcon(tmp_path, monkeypatch):
    calls = []

    def fake_standard(context, spec):
        calls.append((spec["experiment_id"], spec["seed"], spec["protocol"]))
        return {"id": spec["id"], "spec": spec}

    def fake_supcon(context, spec):
        calls.append((spec["experiment_id"], spec["seed"], spec["protocol"]))
        return {"id": spec["id"], "spec": spec}

    monkeypatch.setattr(hair_supcon.paper_suite, "_run_standard", fake_standard)
    monkeypatch.setattr(hair_supcon.paper_suite, "_run_supcon", fake_supcon)
    records = hair_supcon.run(comparison_context(tmp_path))
    assert len(records) == 2
    assert calls == [
        ("baseline_b1_256", 42, hair_supcon.PROTOCOL),
        ("supcon_b1_256", 42, hair_supcon.PROTOCOL),
    ]
    assert all("dinov2" not in str(call).lower() for call in calls)


def test_supcon_notebook_embeds_current_sources():
    path = ROOT / "notebooks/06_hair_supcon_comparison_colab.ipynb"
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
