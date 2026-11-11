import ast
import hashlib
import importlib.util
import io
import json
import sys
import types
import zipfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from mediflow_datasets import adaptation_stage4_test as final


def test_reusing_previous_scores_does_not_invent_new_condition_comparisons(tmp_path):
    old = {"labeled_count": 10, "accuracy": 0.6, "macro_f1": 0.5}
    new = {"labeled_count": 10, "accuracy": 0.7, "macro_f1": 0.6}
    prior = {
        "metrics": {
            name: {c: dict(old) for c in final.original.CONDITIONS}
            for name in final.original.MODEL_LABELS
        }
    }
    fresh = {
        "domain": "skin",
        "selected_on_validation": "stage4_revised",
        "metrics": {"stage4_revised": {c: dict(new) for c in final.CONDITIONS}},
    }
    unchanged = deepcopy((prior, fresh))
    combined = final.combine_results(fresh, prior)
    assert (prior, fresh) == unchanged
    assert all(
        "stage4_expanded" not in combined["metrics"][name] for name in final.original.MODEL_LABELS
    )
    final.write_comparison(tmp_path, [combined])
    import csv

    with (tmp_path / "test_comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 17
    new_rows = [r for r in rows if r["model"] == "stage4_revised"]
    assert all(r["measurement_source"] == "new_stage4_inference" for r in new_rows)
    assert all(
        r["delta_macro_f1_vs_initial"] == ""
        for r in new_rows
        if r["condition"].startswith("stage4_")
    )
    assert all(
        r["delta_macro_f1_vs_initial"] != ""
        for r in new_rows
        if not r["condition"].startswith("stage4_")
    )


def test_five_models_receive_identical_frozen_conditions_and_keep_original_test(tmp_path):
    buffer = io.BytesIO()
    image = np.arange(32 * 40 * 3, dtype=np.uint8).reshape(32, 40, 3)
    Image.fromarray(image).save(buffer, format="PNG")
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "x") as writer:
        for variant in ("original", "augmented"):
            for split in ("train", "val", "test"):
                for name in ("C0/a.png", "C1/a.png", "C1/b.png"):
                    writer.writestr(f"dataset/{variant}/{split}/{name}", buffer.getvalue())
    task = {
        "domain": "skin",
        "classes": ["C1", "C0"],
        "batch_size": 2,
        "source_sha256": final.digest(archive),
        "input_policy": "center_square",
    }
    index, _ = final.data.source_index(archive, task["classes"], task["source_sha256"])
    rows = index["original", "test"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"original": {"test": [asdict(r) for r in rows]}}))
    lock = {"manifest_sha256": final.digest(manifest), "test_count": 3}
    assert final.original.prepare_test(archive, task, lock, manifest) == rows
    profile = {
        "lighting": {
            "brightness_offset": [3, 5],
            "contrast": [0.9, 1.0],
            "left_light": [0, 2],
            "photometric_operator": "bounded_additive_v3",
        },
        "color_a": [-1, 1],
        "color_b": [-1, 1],
        "texture_gain": [0.9, 1.1],
        "blur_sigma_roi64": [0, 0.5],
        "blur_input_scale": 1,
    }
    revised = {**profile, "lighting": {**profile["lighting"], "brightness_offset": [24, 32]}}
    calls = {name: [] for name in final.MODEL_LABELS}

    def predictor(name):
        def predict(images):
            calls[name].append(images.numpy().copy())
            return tf.convert_to_tensor(np.tile([0.25, 0.75], (len(images), 1)), tf.float32)

        return predict

    predictors = {name: predictor(name) for name in calls}
    before = final.digest(archive)
    metrics, scores = final.evaluate_zip(
        archive, rows, task, {"size": 16}, profile, revised, predictors, 1707
    )
    for name in calls:
        assert [len(batch) for batch in calls[name]] == [2] * 5 + [1] * 5
        assert all(np.array_equal(a, b) for a, b in zip(calls["initial"], calls[name], strict=True))
        assert all(batch.dtype == np.float32 and batch.max() > 1 for batch in calls[name])
        assert set(metrics[name]) == set(final.CONDITIONS)
        assert all(m["confusion_matrix"] == [[0, 2], [0, 1]] for m in metrics[name].values())
        assert all(s.shape == (3, 2) for s in scores[name].values())
    assert not np.array_equal(calls["initial"][1], calls["initial"][3])
    old = final.original.evaluate_zip(
        archive, rows, task, {"size": 16}, {"extended_profile": profile}, predictors, 1707
    )
    assert all(
        old[name][condition] == metrics[name][condition]
        for name in predictors
        for condition in final.original.CONDITIONS
    )
    again, _ = final.evaluate_zip(
        archive, rows, task, {"size": 16}, profile, revised, predictors, 1707
    )
    assert again == metrics and final.digest(archive) == before
    with pytest.raises(ValueError, match="Test rows"):
        final.original.prepare_test(archive, task, {**lock, "test_count": 4}, manifest)


@pytest.mark.parametrize("failure", ["environment", "gpu"])
def test_four_stage_colab_bootstrap_reports_failures_and_keeps_logs(
    tmp_path, monkeypatch, capsys, failure
):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "stage4_builder", scripts / "build_adaptation_stage4_test.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    my_drive = tmp_path / "drive/MyDrive"
    project = my_drive / "mediflow_Project"
    project.mkdir(parents=True)
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    batch = """import os
import sys
def prepare_python(work, executor):
    assert os.environ["MPLBACKEND"] == "Agg"
    print("stage4 fixture environment", flush=True)
    (work / "environment_probe.log").write_text("stage4 diagnostics")
"""
    batch += (
        '    raise RuntimeError("stage4 fixture failed")\n'
        if failure == "environment"
        else '    return sys.executable, {"gpu": []}\n'
    )
    bundle = project / builder.previous.BUNDLE_NAME
    with zipfile.ZipFile(bundle, "x") as writer:
        writer.writestr("src/mediflow_datasets/__init__.py", "")
        writer.writestr("src/mediflow_datasets/domain_adaptation_batch.py", batch)
        writer.writestr(
            "src/mediflow_datasets/domain_adaptation_io.py", "def write_progress(*a): pass\n"
        )
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    stage4_name = "measured_strength_492ee01afb45"
    plan_json = json.dumps({"bundle_sha256": sha, "stage4_drive_folder": stage4_name})
    root = my_drive / "MediFlow_adaptation_runs" / sha[:16]
    for number in (1, 2, 3):
        folder = root / f"stage{number}"
        folder.mkdir(parents=True)
        (folder / "bundle.json").write_text("{}")
    (root / stage4_name).mkdir()
    (root / stage4_name / "study.json").write_text("{}")
    google, colab = types.ModuleType("google"), types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: None)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    notebook = builder.notebook(plan_json)
    source = "".join(notebook["cells"][1]["source"])
    assert "mediflow_datasets.adaptation_stage4_test" in source
    assert "MediFlow_Stage4_Test_Results_20261111.zip" in source
    assert "e6bc85bc0a7d" not in source

    class LocalContent(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value.startswith("/content"):
                return ast.copy_location(
                    ast.Constant(str(tmp_path) + node.value[len("/content") :]), node
                )
            return node

    code = compile(
        ast.fix_missing_locations(LocalContent().visit(ast.parse(source))), "stage4", "exec"
    )
    with pytest.raises(RuntimeError, match="종료 코드 1"):
        exec(code, {"__name__": "__main__"})
    assert "stage4 fixture environment" in capsys.readouterr().out
    logs = root / ("stage4_test_" + hashlib.sha256(plan_json.encode()).hexdigest()[:12]) / "logs"
    assert "stage4 diagnostics" == next(logs.glob("*_environment_probe.log")).read_text()
    assert next(logs.glob("run_*.log")).is_file()
    assert not list(tmp_path.glob("mediflow_final_test_*"))


def test_colab_stages_exact_runtime_bytes_before_starting_external_process(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "stage4_staging_builder", scripts / "build_adaptation_stage4_test.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    project = tmp_path / "drive/MyDrive/mediflow_Project"
    project.mkdir(parents=True)
    bundle = project / builder.previous.BUNDLE_NAME
    runtime_hashes = {
        name: builder.previous.sha256(builder.ROOT / builder.previous.MODULE_DIR / name)
        for name in builder.RUNTIME
    }
    with zipfile.ZipFile(bundle, "x") as writer:
        for name in builder.RUNTIME:
            writer.write(
                builder.ROOT / builder.previous.MODULE_DIR / name,
                "src/mediflow_datasets/" + name,
            )
    bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    stage4_name = "measured_strength_492ee01afb45"
    root = project.parent / "MediFlow_adaptation_runs" / bundle_sha[:16]
    (root / "stage1").mkdir(parents=True)
    (root / "stage1/bundle.json").write_text("{}")
    (root / stage4_name).mkdir()
    (root / stage4_name / "study.json").write_text("{}")
    plan = {
        "bundle_sha256": bundle_sha,
        "stage4_drive_folder": stage4_name,
        "runtime_sha256": runtime_hashes,
    }
    book = builder.notebook(json.dumps(plan))
    source = "".join(book["cells"][1]["source"])
    google, colab = types.ModuleType("google"), types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: None)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    class StopAfterStaging(Exception):
        pass

    def inspect_staged_runtime(command, **kwargs):
        runtime = Path(kwargs["env"]["PYTHONPATH"]).parent
        for name, expected in runtime_hashes.items():
            staged = runtime / "src/mediflow_datasets" / name
            assert hashlib.sha256(staged.read_bytes()).hexdigest() == expected, name
        assert (runtime / "src/mediflow_datasets/adaptation_final_test.py").read_bytes() == (
            builder.ROOT / "src/mediflow_datasets/adaptation_final_test.py"
        ).read_bytes()
        raise StopAfterStaging

    monkeypatch.setattr(builder.previous.subprocess, "Popen", inspect_staged_runtime)

    class LocalContent(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value.startswith("/content"):
                return ast.copy_location(
                    ast.Constant(str(tmp_path) + node.value[len("/content") :]), node
                )
            return node

    code = compile(
        ast.fix_missing_locations(LocalContent().visit(ast.parse(source))), "staging-check", "exec"
    )
    with pytest.raises(StopAfterStaging):
        exec(code, {"__name__": "__main__"})
