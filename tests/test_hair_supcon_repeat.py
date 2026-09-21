import ast
import json
from pathlib import Path

import pytest

from mediflow_datasets import hair_supcon_repeat

ROOT = Path(__file__).resolve().parents[1]


def repeat_context(tmp_path):
    return {
        "config": {
            "domain": "hair",
            "mode": "supcon_repeat",
            "train_variant": "augmented",
            "seed": 43,
            "seeds": [43, 44],
            "experiments": ["baseline_b1_256", "supcon_b1_256"],
        },
        "classes": ["a", "b", "c", "d", "e"],
        "output": tmp_path,
        "signature": "fixed",
    }


def test_repeat_contract_rejects_wrong_seeds(tmp_path):
    context = repeat_context(tmp_path)
    hair_supcon_repeat.validate(context)
    context["config"]["seeds"] = [42, 43, 44]
    with pytest.raises(ValueError, match="43, 44"):
        hair_supcon_repeat.validate(context)


def test_repeat_runs_four_paired_trials(tmp_path, monkeypatch):
    calls = []

    def fake_run(context, spec):
        calls.append((spec["experiment_id"], spec["seed"], spec["protocol"]))
        return {"id": spec["id"], "spec": spec}

    monkeypatch.setattr(hair_supcon_repeat.paper_suite, "_run_standard", fake_run)
    monkeypatch.setattr(hair_supcon_repeat.paper_suite, "_run_supcon", fake_run)
    records = hair_supcon_repeat.run(repeat_context(tmp_path))
    assert len(records) == 4
    assert calls == [
        ("baseline_b1_256", 43, hair_supcon_repeat.PROTOCOL),
        ("supcon_b1_256", 43, hair_supcon_repeat.PROTOCOL),
        ("baseline_b1_256", 44, hair_supcon_repeat.PROTOCOL),
        ("supcon_b1_256", 44, hair_supcon_repeat.PROTOCOL),
    ]


def test_repeat_notebook_embeds_current_sources():
    path = ROOT / "notebooks/07_hair_supcon_repeat_seeds_colab.ipynb"
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
