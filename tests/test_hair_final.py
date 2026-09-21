import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from mediflow_datasets import hair_final

ROOT = Path(__file__).resolve().parents[1]


def final_context(tmp_path):
    return {
        "config": {
            "domain": "hair",
            "mode": "final_candidate",
            "train_variant": "augmented",
            "seed": 42,
            "seeds": [42],
            "parent_run_dir": "/content/drive/example",
            "parent_model_sha256": "a" * 64,
        },
        "output": tmp_path,
    }


def test_final_contract_accepts_only_fixed_hair_candidate(tmp_path):
    context = final_context(tmp_path)
    hair_final.validate(context)
    context["config"]["domain"] = "web_skin"
    with pytest.raises(ValueError, match="Hair"):
        hair_final.validate(context)


def test_final_evaluation_freezes_selection_before_loading_test():
    source = inspect.getsource(hair_final._evaluate_once)
    assert source.index("selection_before_test.json") < source.index('("test", 384, False)')
    assert 'marker = output / "test_completed.json"' in source
    assert "if marker.exists()" in source


def test_parent_accepts_legacy_validation_record_without_test_flag(tmp_path):
    parent = tmp_path / "parent"
    trial = parent / hair_final.PARENT_TRIAL
    attempt = trial / "attempt_legacy"
    attempt.mkdir(parents=True)
    model = attempt / hair_final.MODEL_NAME
    model.write_bytes(b"fixed model bytes")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    record = {
        "id": hair_final.PARENT_TRIAL,
        "selected_stage": "stage2",
        "selected_model": hair_final.MODEL_NAME,
        "attempt": attempt.name,
        "artifact_hashes": {f"{attempt.name}/{hair_final.MODEL_NAME}": digest},
    }
    (trial / "completed.json").write_text(json.dumps(record), encoding="utf-8")
    run_config = {
        "settings": {
            "data_sha256": "data-hash",
            "classes": ["C0", "C1"],
            "seed": 42,
        }
    }
    (parent / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    context = final_context(tmp_path / "output")
    context["config"]["parent_run_dir"] = str(parent)
    context["config"]["parent_model_sha256"] = digest
    context["data_hash"] = "data-hash"
    context["classes"] = ["C0", "C1"]
    returned, returned_model, returned_hash = hair_final._parent(context)
    assert returned == record
    assert returned_model == model
    assert returned_hash == digest


def test_final_package_preserves_384_preprocessing_contract():
    source = inspect.getsource(hair_final._package)
    assert '"input_shape": [384, 384, 3]' in source
    assert '"input_pixel_range": [0, 255]' in source
    assert '"external_normalization": False' in source
    assert '"internal_rescaling": "1/255"' in source


def test_final_notebook_embeds_current_sources():
    path = ROOT / "notebooks/10_hair_b1_384_final_test_package_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 11
    all_source = "\n".join("".join(item["source"]) for item in notebook["cells"])
    assert "MODE = 'final_candidate'" in all_source
    assert "PARENT_MODEL_SHA256 =" in all_source
    assert "finalize(context)" in all_source
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
