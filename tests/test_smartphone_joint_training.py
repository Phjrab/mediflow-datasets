import ast
import hashlib
import sys
import zipfile
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import keras
import numpy as np
import pytest
import tensorflow as tf

from mediflow_datasets import smartphone_joint_training as joint
from mediflow_datasets import smartphone_training as old
from mediflow_datasets.domain_adaptation_data import ImageRow, read_json, sha256
from mediflow_datasets.smartphone_joint_release import build_notebook, json_bytes
from mediflow_datasets.smartphone_profile import empty_profile


def test_embedded_notebook_reuses_base_and_starts_only_one_new_trial():
    files = {"study.json": b"{}"}
    manifest = {
        "schema": "mediflow_phone_runtime_manifest_v1",
        "files": {"study.json": hashlib.sha256(b"{}").hexdigest()},
    }
    files["runtime_manifest.json"] = json_bytes(manifest)
    book = build_notebook(
        "a" * 64,
        files,
        hashlib.sha256(files["runtime_manifest.json"]).hexdigest(),
        "newrun",
        "oldrun",
    )
    for c in book["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    source = "".join(book["cells"][3]["source"])
    assert "smartphone_joint_training" in source and '"--previous",str(PREVIOUS)' in source
    assert "key+" in source and "--key" not in source
    tree = ast.parse(source)
    writes = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "write"
        and n.args
        and isinstance(n.args[0], ast.BinOp)
    ]
    assert len(writes) == 1 and writes[0].args[0].right.value == "\n"
    setup = "".join(book["cells"][1]["source"])
    assert "MediFlow_Smartphone_WebSkin_Runtime_20261111.zip" in setup
    assert "MediFlow_smartphone_runs/oldrun" in setup and "joint_newrun" in setup
    assert "files.upload" not in setup


def test_previous_models_are_hash_locked_and_read_only(tmp_path):
    (tmp_path / "stage4").mkdir()
    (tmp_path / "stage4/best.keras").write_bytes(b"old-model")
    old.write_progress(tmp_path / "runtime_identity.json", {"study": "old"})
    old.write_progress(tmp_path / "summary.json", {"status": "completed"})
    recipe = {
        "previous_result_files": {
            n: sha256(tmp_path / n) for n in ("runtime_identity.json", "summary.json")
        },
        "previous_runtime_identity": {"study": "old"},
        "previous_models": {
            "stage4": {
                "file": "stage4/best.keras",
                "sha256": sha256(tmp_path / "stage4/best.keras"),
            }
        },
    }
    result = joint.verify_previous(tmp_path, {"joint_experiment": recipe})
    assert result["stage4"] == tmp_path / "stage4/best.keras"
    (tmp_path / "stage4/best.keras").write_bytes(b"changed")
    with pytest.raises(ValueError, match="모델 해시"):
        joint.verify_previous(tmp_path, {"joint_experiment": recipe})


def test_notebook_assembles_runtime_from_old_zip_and_embedded_patch(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    base = project / "MediFlow_Smartphone_WebSkin_Runtime_20261111.zip"
    unchanged = b"original starting weights"
    with zipfile.ZipFile(base, "w") as z:
        z.writestr("starting.keras", unchanged)
        z.writestr("target_encrypted/holdout_256.bin", b"must not extract")
    files = {"study.json": b'{"domain":"web_skin"}\n'}
    manifest = {
        "schema": "mediflow_phone_runtime_manifest_v1",
        "files": {
            "starting.keras": hashlib.sha256(unchanged).hexdigest(),
            "study.json": hashlib.sha256(files["study.json"]).hexdigest(),
        },
    }
    files["runtime_manifest.json"] = json_bytes(manifest)
    book = build_notebook(
        sha256(base),
        files,
        hashlib.sha256(files["runtime_manifest.json"]).hexdigest(),
        "newrun",
        "oldrun",
    )
    source = "".join(book["cells"][1]["source"])
    replacements = {
        'Path("/content/drive/MyDrive/mediflow_Project")': "PROJECT_TEST",
        'Path("/content/mediflow_phone_joint_newrun")': "RUNTIME_TEST",
        'Path("/content/drive/MyDrive/MediFlow_smartphone_runs/oldrun")': "PREVIOUS_TEST",
        'Path("/content/drive/MyDrive/MediFlow_smartphone_runs/joint_newrun")': "OUTPUT_TEST",
        'Path("/content/mediflow_phone_joint_work_newrun")': "LOCAL_TEST",
    }
    for old_text, new_text in replacements.items():
        assert old_text in source
        source = source.replace(old_text, new_text)
    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=lambda _: None)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    previous = tmp_path / "previous"
    previous.mkdir()
    environment = {
        "PROJECT_TEST": project,
        "RUNTIME_TEST": tmp_path / "runtime",
        "PREVIOUS_TEST": previous,
        "OUTPUT_TEST": tmp_path / "output",
        "LOCAL_TEST": tmp_path / "local",
    }
    exec(compile(source, "runtime_setup", "exec"), environment)
    runtime = environment["RUNTIME"]
    assert (runtime / "starting.keras").read_bytes() == unchanged
    assert not (runtime / "target_encrypted/holdout_256.bin").exists()
    assert (runtime / "study.json").read_bytes() == files["study.json"]
    exec(compile(source, "runtime_setup", "exec"), environment)
    (runtime / "study.json").write_bytes(b"modified")
    with pytest.raises(ValueError, match="런타임 파일이 다릅니다"):
        exec(compile(source, "runtime_setup", "exec"), environment)


def context(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    inner = keras.Input((8, 8, 3))
    x = keras.layers.Conv2D(4, 1, activation="relu")(keras.layers.Rescaling(1 / 255)(inner))
    backbone = keras.Model(inner, x)
    inputs = keras.Input((8, 8, 3))
    features = keras.layers.GlobalAveragePooling2D()(backbone(inputs))
    model = keras.Model(inputs, keras.layers.Dense(2, activation="softmax")(features))
    model.save(runtime / "start.keras")
    spec = {
        "architecture": "b0",
        "size": 8,
        "label_smoothing": 0.0,
        "model": "start.keras",
        "model_sha256": sha256(runtime / "start.keras"),
        "role": "selected",
    }
    p = empty_profile(
        {
            "brightness_offset": [-5, 0],
            "contrast": [1, 1.05],
            "left_light": [0, 0],
            "scale_x": [1, 1],
            "scale_y": [1, 1],
            "photometric_operator": "bounded_additive_v3",
        }
    )
    study = {
        "seed": 42,
        "epochs": 2,
        "learning_rate": 1e-5,
        "augmentation_probability": 0.5,
        "task": {
            "classes": ["a", "b"],
            "batch_size": 4,
            "input_policy": "center_square",
            "source_sha256": "source",
        },
        "profiles": {name: deepcopy(p) for name in ("stage1", "stage2", "stage4")},
        "joint_profile": p,
    }
    old.write_progress(runtime / "study.json", study)
    rows = [ImageRow(str(i), "train", i % 2) for i in range(8)]
    val = [ImageRow("v" + str(i), "val", i % 2) for i in range(4)]
    cache = np.random.default_rng(42).uniform(30, 190, (12, 8, 8, 3)).astype("float32")
    return runtime, study, spec, rows, val, cache


def test_joint_view_uses_fixed_val_seed_and_rejects_reused_different_profile(tmp_path):
    runtime, study, spec, rows, val, cache = context(tmp_path)
    view = joint.joint_view(cache, rows, val, study, 8, tmp_path / "joint_view")
    np.testing.assert_array_equal(
        view, joint.joint_view(cache, rows, val, study, 8, tmp_path / "joint_view")
    )
    assert view.shape == (4, 8, 8, 3) and view.dtype == np.float32 and np.isfinite(view).all()
    study["joint_profile"]["color_a"] = [0, 1]
    with pytest.raises(ValueError, match="캐시 설정"):
        joint.joint_view(cache, rows, val, study, 8, tmp_path / "joint_view")


def test_stage5_real_updates_resume_and_do_not_add_alignment(tmp_path, monkeypatch):
    runtime, study, spec, rows, val, cache = context(tmp_path)
    destination = tmp_path / "output"
    destination.mkdir()
    view = joint.joint_view(cache, rows, val, study, 8, tmp_path / "views")
    monkeypatch.setattr(old, "curves", lambda *args: None)
    save = old.train.save_epoch

    def interrupt(state, folder, local, entry):
        committed = save(state, folder, local, entry)
        if entry["epoch"] == 1:
            raise RuntimeError("interrupt after saved epoch")
        return committed

    monkeypatch.setattr(old.train, "save_epoch", interrupt)
    with pytest.raises(RuntimeError, match="saved epoch"):
        old.run_trial(
            runtime,
            destination,
            tmp_path / "saving",
            study,
            spec,
            5,
            rows,
            val,
            cache,
            {"phone_joint": view},
            None,
        )
    monkeypatch.setattr(old.train, "save_epoch", save)
    result = old.run_trial(
        runtime,
        destination,
        tmp_path / "saving",
        study,
        spec,
        5,
        rows,
        val,
        cache,
        {"phone_joint": view},
        None,
    )
    assert result["stage"] == 5 and result["epochs_completed"] == 2
    assert result["selection_condition"] == "phone_joint"
    history = read_json(destination / "stage5/history.json")
    assert all(e["alignment_steps"] == 0 and e["target_images_used"] == 0 for e in history)
    assert result["best_model"]["sha256"] != spec["model_sha256"]
    assert sha256(runtime / "start.keras") == spec["model_sha256"]
    assert (
        old.run_trial(
            runtime,
            destination,
            tmp_path / "saving",
            study,
            spec,
            5,
            rows,
            val,
            cache,
            {"phone_joint": view},
            None,
        )
        == result
    )
    keras.backend.clear_session()


def test_common_comparison_reuses_20_old_metrics_and_never_opens_holdout(tmp_path, monkeypatch):
    runtime, study, spec, rows, val, cache = context(tmp_path)
    destination = tmp_path / "output"
    (destination / "stage5").mkdir(parents=True)
    (destination / "stage5/best.keras").write_bytes((runtime / "start.keras").read_bytes())
    metric = {"macro_f1": 0.5, "accuracy": 0.5, "confusion_matrix": [[1, 1], [1, 1]]}
    prior_validation = {
        name: {c: deepcopy(metric) for c in old.CONDITIONS}
        for name in ("initial", "stage1", "stage2", "stage3", "stage4")
    }
    old.write_progress(runtime / "previous_validation.json", prior_validation)
    prior_diagnostics = {
        name: {
            "mmd": 0.2,
            "coral": 0.1,
            "target_accuracy": None,
            "predictions": [{"id": str(i), "predicted_index": 0, "score": 0.5} for i in range(4)],
        }
        for name in prior_validation
    }
    old.write_progress(runtime / "previous_calibration_diagnostics.json", prior_diagnostics)
    old.write_progress(runtime / "feature_reference.json", {"reference": "unit"})
    study["joint_experiment"] = {"candidate_id": "unit"}
    calls = []

    def evaluate(path, spec, cache, views, rows, val, task, label):
        calls.append(tuple(views))
        return {c: deepcopy(metric) for c in ("clean", *views)}

    monkeypatch.setattr(joint, "evaluate", evaluate)
    monkeypatch.setattr(joint.features, "sampled_indices", lambda rows, count: list(range(4)))
    monkeypatch.setattr(
        joint.features,
        "feature_outputs",
        lambda model, images, spec, batch: (
            np.tile([0.6, 0.4], (len(images), 1)),
            np.ones((len(images), 4), np.float32),
        ),
    )
    monkeypatch.setattr(
        joint.domain_alignment, "losses", lambda *args: (tf.constant(0.1), tf.constant(0.05))
    )
    monkeypatch.setattr(joint, "plots", lambda *args: None)
    target = {
        "rows": [
            {"id": str(i), "person": "P01", "split": "calibration", "label": None} for i in range(4)
        ],
        "images": cache[:4],
    }
    views = {c: cache[8:] for c in joint.CONDITIONS[1:]}
    paths = {name: runtime / "start.keras" for name in ("stage1", "stage2", "stage3", "stage4")}
    joint.final_comparison(
        runtime, destination, paths, study, spec, rows, val, cache, views, target
    )
    result = read_json(destination / "validation_comparison.json")
    for name, conditions in prior_validation.items():
        for c, value in conditions.items():
            assert result[name][c] == value
    assert calls == [("phone_joint",)] * 5 + [joint.CONDITIONS[1:]]
    assert read_json(destination / "summary.json")["total_validation_results"] == 30
    assert not read_json(destination / "selection.json")["target_holdout_used"]
    diagnostics = read_json(destination / "calibration_diagnostics.json")
    assert not diagnostics["target_holdout_used"] and diagnostics["target_accuracy"] is None
    for name, value in prior_diagnostics.items():
        assert diagnostics["models"][name] == value
    zipped = joint.results_zip(destination)
    import zipfile

    with zipfile.ZipFile(zipped) as z:
        assert len([n for n in z.namelist() if n.endswith(".keras")]) == 1
        assert not any("snapshot_" in n or "Key" in n for n in z.namelist())
    keras.backend.clear_session()
