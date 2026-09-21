import ast
import inspect
import json
from pathlib import Path

import pytest

from mediflow_datasets import hair_sam

ROOT = Path(__file__).resolve().parents[1]


def sam_context(tmp_path):
    return {
        "config": {
            "domain": "hair",
            "mode": "sam_screen",
            "train_variant": "augmented",
            "seed": 42,
            "seeds": [42],
            "experiments": ["sam_b1_384"],
            "sam_rho": 0.05,
            "parent_run_dir": "/content/drive/example",
            "parent_stage1_sha256": "a" * 64,
        },
        "output": tmp_path,
    }


def test_sam_contract_is_single_seed_and_single_condition(tmp_path):
    context = sam_context(tmp_path)
    hair_sam.validate(context)
    context["config"]["seeds"] = [42, 43]
    with pytest.raises(ValueError, match="42"):
        hair_sam.validate(context)


def test_sam_step_perturbs_restores_and_applies_second_gradients():
    source = inspect.getsource(hair_sam._fit_sam)
    assert source.count("tf.GradientTape()") == 2
    assert "variable.assign_add(perturbation)" in source
    assert "variable.assign_sub(perturbation)" in source
    assert "optimizer.apply_gradients" in source


def test_sam_contract_requires_exact_parent_checkpoint(tmp_path):
    context = sam_context(tmp_path)
    context["config"]["parent_stage1_sha256"] = ""
    with pytest.raises(ValueError, match="SHA-256"):
        hair_sam.validate(context)


def test_sam_notebook_embeds_current_sources():
    path = ROOT / "notebooks/09_hair_b1_384_sam_screen_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 13
    all_source = "\n".join("".join(item["source"]) for item in notebook["cells"])
    assert "MODE = 'sam_screen'" in all_source
    assert "SAM_RHO = 0.05" in all_source
    assert "STAGE2_EPOCHS = 15" in all_source
    assert "PARENT_STAGE1_SHA256 =" in all_source
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
