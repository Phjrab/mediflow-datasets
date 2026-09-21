import ast
import json
from pathlib import Path

import pytest

from mediflow_datasets import paper_baseline

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_contracts_match_research_plan():
    assert paper_baseline.BASELINES == {
        "hair": {
            "backbone": "B1",
            "size": 256,
            "loss": "ls005",
            "epochs1": 15,
            "epochs2": 15,
        },
        "web_skin": {
            "backbone": "B0",
            "size": 256,
            "loss": "ce",
            "epochs1": 15,
            "epochs2": 10,
        },
        "skin": {
            "backbone": "B0",
            "size": 224,
            "loss": "ce",
            "epochs1": 15,
            "epochs2": 0,
        },
    }


def test_hair_protocol_requires_exact_fixed_conditions():
    context = {
        "config": {
            "domain": "hair",
            "mode": "baseline3",
            "train_variant": "augmented",
            "epochs1": 15,
            "epochs2": 15,
        }
    }
    assert paper_baseline.validate_protocol(context)["backbone"] == "B1"
    context["config"]["epochs2"] = 10
    with pytest.raises(ValueError, match="STAGE2_EPOCHS"):
        paper_baseline.validate_protocol(context)


def test_run_uses_three_independent_seeds_and_validation_only(tmp_path, monkeypatch):
    calls = []

    def fake_factory(context, variant, seed=None):
        calls.append(("factory", variant, seed))
        return object()

    def fake_run_trial(spec, loader, root, signature, seed, epochs1, epochs2):
        calls.append(("trial", spec["id"], seed, epochs1, epochs2, loader))
        return {"id": spec["id"], "spec": spec}

    monkeypatch.setattr(paper_baseline.flow, "factory", fake_factory)
    monkeypatch.setattr(paper_baseline.engine, "run_trial", fake_run_trial)
    context = {
        "config": {
            "domain": "hair",
            "mode": "baseline3",
            "train_variant": "augmented",
            "epochs1": 15,
            "epochs2": 15,
            "seeds": [42, 43, 44],
        },
        "classes": ["a", "b", "c", "d", "e"],
        "output": tmp_path,
        "signature": "fixed",
    }
    records = paper_baseline.run(context)
    assert [record["spec"]["seed"] for record in records] == [42, 43, 44]
    assert [call[2] for call in calls if call[0] == "factory"] == [42, 43, 44]
    assert [call[2] for call in calls if call[0] == "trial"] == [42, 43, 44]
    assert not any("test" in str(call).lower() for call in calls)


def test_paper_baseline_notebook_embeds_current_sources():
    path = ROOT / "notebooks/04_hair_three_seed_baseline_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
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
        assert ast.literal_eval(assignments["PROFILES"])["hair"] == [
            "모낭사이홍반",
            "미세각질",
            "비듬",
            "탈모",
            "피지과다",
        ]
