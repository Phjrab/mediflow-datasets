import ast
import hashlib
import io
import sys
import zipfile
from copy import deepcopy
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import keras
import numpy as np
import pytest
from PIL import Image

from mediflow_datasets import smartphone_final_test as final
from mediflow_datasets.domain_adaptation_data import read_json, sha256, source_index
from mediflow_datasets.smartphone_joint_release import json_bytes
from mediflow_datasets.smartphone_profile import empty_profile
from mediflow_datasets.smartphone_test_release import notebook


@pytest.fixture
def frozen(tmp_path):
    runtime, prior, joint = (tmp_path / name for name in ("runtime", "prior", "joint"))
    for directory in (runtime, prior, joint):
        directory.mkdir()
    source = tmp_path / "dataset.zip"
    with zipfile.ZipFile(source, "x") as archive:
        for variant in ("original", "augmented"):
            for split in ("train", "val", "test"):
                for i, label in enumerate(("C1", "C0", "C0")):
                    pixels = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
                    stream = io.BytesIO()
                    Image.fromarray(pixels + i).save(stream, format="PNG")
                    archive.writestr(f"data/{variant}/{split}/{label}/{i}.png", stream.getvalue())
    task = {
        "domain": "web_skin",
        "classes": ["C0", "C1"],
        "source_sha256": sha256(source),
        "input_policy": "center_square",
        "batch_size": 2,
    }
    indexed, _ = source_index(source, task["classes"], task["source_sha256"])
    rows = indexed["original", "test"]
    (runtime / "source_manifest.json").write_bytes(
        json_bytes({"original": {"test": [asdict(row) for row in rows]}})
    )
    inputs = keras.Input((8, 8, 3))
    features = keras.layers.GlobalAveragePooling2D()(keras.layers.Rescaling(1 / 255)(inputs))
    model = keras.Model(inputs, [keras.layers.Dense(2)(features) for _ in range(4)])
    model.save(runtime / "initial.keras")
    models = {}
    for name in final.MODELS:
        folder, group = (
            (runtime, "runtime")
            if name == "initial"
            else ((joint, "joint") if name == "stage5" else (prior, "prior"))
        )
        path = folder / (name + ".keras")
        if name != "initial":
            path.write_bytes((runtime / "initial.keras").read_bytes())
        models[name] = {"root": group, "file": path.name, "sha256": sha256(path)}
    (prior / "complete.json").write_bytes(b'{"status":"completed"}')
    profile = empty_profile(
        {
            "photometric_operator": "bounded_additive_v3",
            "brightness_offset": [-8, -3],
            "contrast": [1, 1.05],
            "left_light": [0, 0],
            "scale_x": [1, 1],
            "scale_y": [1, 1],
        }
    )
    plan = {
        "schema": "mediflow_phone_final_test_v1",
        "task": task,
        "model_spec": {"architecture": "pmg", "size": 8},
        "models": models,
        "conditions": list(final.CONDITIONS),
        "profiles": {c: deepcopy(profile) for c in final.CONDITIONS[1:]},
        "manifest_sha256": sha256(runtime / "source_manifest.json"),
        "test_count": 3,
        "test_support": [2, 1],
        "transform_seed": 1707,
        "selected_on_validation": "stage4",
        "test_used_for_selection": False,
        "target_photos_used": False,
        "input_records": [
            {"root": "prior", "file": "complete.json", "sha256": sha256(prior / "complete.json")}
        ],
    }
    envelope = {"plan": plan, "plan_sha256": final.object_hash(plan)}
    (runtime / "evaluation_plan.json").write_bytes(json_bytes(envelope))
    (runtime / "src/mediflow_datasets").mkdir(parents=True)
    (runtime / "src/mediflow_datasets/provenance.py").write_text("# synthetic fixture\n")
    manifest = {
        "files": {
            p.relative_to(runtime).as_posix(): sha256(p) for p in runtime.rglob("*") if p.is_file()
        }
    }
    (runtime / "runtime_manifest.json").write_bytes(json_bytes(manifest))
    return runtime, source, prior, joint, tmp_path / "output", tmp_path / "local", plan, rows


def test_frozen_runtime_split_and_model_corruption_are_rejected(frozen):
    runtime, source, prior, joint, output, local, plan, rows = frozen
    assert final.verify_runtime(runtime)["plan"] == plan
    paths = final.verify_models(plan, {"runtime": runtime, "prior": prior, "joint": joint})
    assert tuple(paths) == final.MODELS
    checked = final.original.prepare_test(
        source, plan["task"], plan, runtime / "source_manifest.json"
    )
    assert checked == rows
    (prior / "stage4.keras").write_bytes(b"changed")
    with pytest.raises(ValueError, match="모델 해시"):
        final.verify_models(plan, {"runtime": runtime, "prior": prior, "joint": joint})
    (runtime / "evaluation_plan.json").write_text("{}")
    with pytest.raises(ValueError, match="해시 불일치"):
        final.verify_runtime(runtime)


def test_same_test_pixels_no_external_normalization_and_profile_cache_guard(frozen):
    runtime, source, prior, joint, output, local, plan, rows = frozen
    first = final.input_views(source, rows, plan, local)
    with zipfile.ZipFile(source) as archive:
        clean = final.experiment.image_tensor(archive.read(rows[0].path), 8, "center_square")
    np.testing.assert_array_equal(first["clean"][0], clean)
    assert first["clean"].dtype == np.float32 and first["clean"].max() > 1
    assert not np.array_equal(first["clean"], first["phone_joint"])
    second = final.input_views(source, rows, plan, local)
    for condition in final.CONDITIONS:
        np.testing.assert_array_equal(first[condition], second[condition])
    changed = deepcopy(plan)
    changed["profiles"]["phone_joint"]["color_a"] = [0, 1]
    with pytest.raises(ValueError, match="캐시 설정"):
        final.input_views(source, rows, changed, local)
    with (local / "clean.npy").open("r+b") as stream:
        stream.seek(-4, 2)
        stream.write(b"oops")
    with pytest.raises(ValueError, match="캐시 손상"):
        final.input_views(source, rows, plan, local)


def test_real_synthetic_inference_resumes_all_30_pairs_and_exports_only_results(
    frozen, monkeypatch
):
    runtime, source, prior, joint, output, local, plan, rows = frozen
    original_files = {
        p: sha256(p) for folder in (runtime, prior, joint) for p in folder.rglob("*") if p.is_file()
    }
    before_source = sha256(source)
    archive = final.run(runtime, source, prior, joint, output, local, require_gpu=False)
    summary = read_json(output / "summary.json")
    assert summary["comparisons"] == 30 and summary["source_test_evaluated"]
    assert not summary["training_executed"] and not summary["target_photos_used"]
    assert not summary["test_used_for_selection"] and summary["selected_before_test"] == "stage4"
    assert summary["target_accuracy"] is None
    results = read_json(output / "test_results.json")
    for conditions in results["metrics"].values():
        assert tuple(conditions) == final.CONDITIONS
        assert all(v["labeled_count"] == 3 and v["support"] == [2, 1] for v in conditions.values())
    assert len(list((output / "plots").glob("*.png"))) == 7
    with zipfile.ZipFile(archive) as exported:
        assert exported.testzip() is None
        assert len([n for n in exported.namelist() if n.endswith(".npz")]) == 30
        assert not any(n.endswith(".keras") or "target_encrypted" in n for n in exported.namelist())

    def fail_load(*args, **kwargs):
        raise AssertionError("Completed Test must not infer again")

    monkeypatch.setattr(keras.models, "load_model", fail_load)
    final.run(runtime, source, prior, joint, output, local, require_gpu=False)
    assert {p: sha256(p) for p in original_files} == original_files
    assert sha256(source) == before_source
    score = output / "scores/initial__clean.npz"
    score.write_bytes(b"changed")
    with pytest.raises(ValueError, match="예측 파일이 다릅니다"):
        final.run(runtime, source, prior, joint, output, local, require_gpu=False)


def test_input_directories_cannot_be_used_as_output(frozen):
    runtime, source, prior, joint, output, local, plan, rows = frozen
    with pytest.raises(ValueError, match="분리"):
        final.run(runtime, source, prior, joint, prior, local, require_gpu=False)


@pytest.mark.parametrize(
    "values",
    [
        np.array([[np.nan, 1]], np.float32),
        np.array([[0.1, 0.2]], np.float32),
        np.array([[-0.1, 1.1]], np.float32),
    ],
)
def test_invalid_prediction_scores_are_rejected(values):
    with pytest.raises(ValueError, match="예측 점수"):
        final.checked_scores(values, 1, 2)


def test_notebook_is_key_free_and_assembles_only_allowed_source_runtime(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    base = project / "MediFlow_Smartphone_WebSkin_Runtime_20261111.zip"
    unchanged = b"starting weights"
    with zipfile.ZipFile(base, "w") as archive:
        archive.writestr("starting.keras", unchanged)
        archive.writestr("target_encrypted/holdout_256.bin", b"must not extract")
        archive.writestr("target_encrypted/calibration_256.bin", b"must not extract")
    patched = {"evaluation_plan.json": b'{"plan":{"task":{}}}'}
    manifest = {
        "files": {
            "starting.keras": hashlib.sha256(unchanged).hexdigest(),
            **{p: hashlib.sha256(b).hexdigest() for p, b in patched.items()},
        }
    }
    patched["runtime_manifest.json"] = json_bytes(manifest)
    book = notebook(
        sha256(base),
        patched,
        hashlib.sha256(patched["runtime_manifest.json"]).hexdigest(),
        "newrun",
        "oldrun",
        "joint_prior",
    )
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    code = "\n".join("".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code")
    assert "getpass" not in code and "files.upload" not in code and "stdin" not in code
    assert "smartphone_final_test" in code and "smartphone_joint_training" not in code
    assert "s['plan']['task']" in code and "files.download" in code
    setup = "".join(book["cells"][1]["source"])
    replacements = {
        'Path("/content/drive/MyDrive/mediflow_Project")': "PROJECT_TEST",
        'Path("/content/mediflow_phone_test_newrun")': "RUNTIME_TEST",
        'Path("/content/drive/MyDrive/MediFlow_smartphone_runs/oldrun")': "PREVIOUS_TEST",
        'Path("/content/drive/MyDrive/MediFlow_smartphone_runs/test_newrun")': "OUTPUT_TEST",
        'Path("/content/mediflow_phone_test_work_newrun")': "LOCAL_TEST",
        'Path("/content/drive/MyDrive/MediFlow_smartphone_runs/joint_prior")': "JOINT_TEST",
    }
    for old, new in replacements.items():
        assert old in setup
        setup = setup.replace(old, new)
    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=lambda _: None)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    previous, joint = tmp_path / "previous", tmp_path / "joint"
    previous.mkdir()
    joint.mkdir()
    environment = {
        "PROJECT_TEST": project,
        "RUNTIME_TEST": tmp_path / "runtime",
        "PREVIOUS_TEST": previous,
        "JOINT_TEST": joint,
        "OUTPUT_TEST": tmp_path / "output",
        "LOCAL_TEST": tmp_path / "local",
    }
    exec(compile(setup, "setup", "exec"), environment)
    runtime = environment["RUNTIME"]
    assert (runtime / "starting.keras").read_bytes() == unchanged
    assert not (runtime / "target_encrypted").exists()
    exec(compile(setup, "setup", "exec"), environment)
    (runtime / "evaluation_plan.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="파일이 다릅니다"):
        exec(compile(setup, "setup", "exec"), environment)
