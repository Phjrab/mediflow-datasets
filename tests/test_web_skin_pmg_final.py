import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from mediflow_datasets import web_skin_pmg_final

ROOT = Path(__file__).resolve().parents[1]


def final_context(tmp_path):
    return {
        "config": {
            "domain": "web_skin",
            "mode": "web_skin_pmg_final",
            "train_variant": "augmented",
            "seed": 42,
            "seeds": [42],
            "parent_run_dir": "/content/drive/example",
            "parent_model_sha256": web_skin_pmg_final.EXPECTED_MODEL_SHA256,
        },
        "output": tmp_path,
        "classes": web_skin_pmg_final.EXPECTED_CLASSES,
        "data_hash": web_skin_pmg_final.EXPECTED_DATA_SHA256,
    }


def test_final_contract_accepts_only_fixed_web_skin_candidate(tmp_path):
    context = final_context(tmp_path)
    web_skin_pmg_final.validate(context)
    context["config"]["domain"] = "hair"
    with pytest.raises(ValueError, match="Web Skin"):
        web_skin_pmg_final.validate(context)


def test_final_evaluation_freezes_selection_before_loading_test():
    source = inspect.getsource(web_skin_pmg_final._evaluate_once)
    assert source.index("selection_before_test.json") < source.index(
        '("test", 256, False)'
    )
    assert 'marker = output / "test_completed.json"' in source
    assert "if marker.exists()" in source


def test_parent_requires_exact_pmg_record(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    trial = parent / web_skin_pmg_final.PARENT_TRIAL
    attempt = trial / "attempt_fixed"
    attempt.mkdir(parents=True)
    model = attempt / web_skin_pmg_final.MODEL_NAME
    model.write_bytes(b"fixed pmg model bytes")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    record = {
        "id": web_skin_pmg_final.PARENT_TRIAL,
        "selected_stage": "stage2",
        "selected_model": web_skin_pmg_final.MODEL_NAME,
        "test_evaluated": False,
        "attempt": attempt.name,
        "spec": {
            "method": "efficientnet_b0_pmg_adaptation",
            "input_size": 256,
            "jigsaw_grids": [8, 4, 2],
        },
        "validation": web_skin_pmg_final.EXPECTED_VALIDATION,
        "artifact_hashes": {
            f"{attempt.name}/{web_skin_pmg_final.MODEL_NAME}": digest
        },
    }
    (trial / "completed.json").write_text(json.dumps(record), encoding="utf-8")
    run_config = {
        "settings": {
            "data_sha256": web_skin_pmg_final.EXPECTED_DATA_SHA256,
            "classes": web_skin_pmg_final.EXPECTED_CLASSES,
            "seed": 42,
        }
    }
    (parent / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    context = final_context(tmp_path / "output")
    context["config"]["parent_run_dir"] = str(parent)
    monkeypatch.setattr(web_skin_pmg_final, "EXPECTED_MODEL_SHA256", digest)
    returned, returned_model, returned_hash = web_skin_pmg_final._parent(context)
    assert returned == record
    assert returned_model == model
    assert returned_hash == digest


def test_final_package_preserves_pmg_256_preprocessing_contract():
    source = inspect.getsource(web_skin_pmg_final._package)
    inference = web_skin_pmg_final._inference_source()
    assert '"input_shape": [256, 256, 3]' in source
    assert '"input_pixel_range": [0, 255]' in source
    assert '"external_normalization": False' in source
    assert '"internal_rescaling": "1/255"' in source
    assert "keras.layers.Add" in inference
    assert 'Activation("softmax"' in inference


def test_final_notebook_embeds_current_sources():
    path = ROOT / "notebooks/13_web_skin_pmg_b0_256_final_test_package_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 11
    all_source = "\n".join("".join(item["source"]) for item in notebook["cells"])
    assert "MODE = 'web_skin_pmg_final'" in all_source
    assert web_skin_pmg_final.EXPECTED_MODEL_SHA256 in all_source
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
