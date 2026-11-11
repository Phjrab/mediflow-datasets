"""Phone input contracts, subject holdout and actual checkpoint-resume behaviour."""

import ast
import base64
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image, ImageCms

from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import smartphone_notebook as notebook
from mediflow_datasets import smartphone_profile as profile
from mediflow_datasets import smartphone_training as training


def test_exif_and_icc_canonicalization_keeps_float_rgb_255(tmp_path):
    raw = np.zeros((13, 19, 3), np.uint8)
    raw[:, :9] = [230, 80, 45]
    raw[:, 9:] = [40, 140, 220]
    picture = Image.fromarray(raw)
    exif = picture.getexif()
    exif[274] = 6
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = tmp_path / "phone.png"
    picture.save(path, exif=exif, icc_profile=icc)
    actual = profile.canonical_input(path, 8)
    upright = np.rot90(raw, -1)
    crop = upright[3:16, :, :]
    expected = tf.image.resize(crop, (8, 8), method="bilinear", antialias=False).numpy()
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32 and actual.max() > 1
    assert Image.open(path).getexif()[274] == 6  # original is untouched


def test_holdout_cannot_enter_profile_calibration():
    source = [{"split": "train", "label": 0}]
    targets = [
        {"person": p, "split": "calibration", "label": None} for p in ("P01", "P02", "P03", "P04")
    ]
    profile.validate_calibration(source, targets)
    with pytest.raises(ValueError, match="never holdout"):
        profile.validate_calibration(
            source, targets + [{"person": "P05", "split": "holdout", "label": None}]
        )
    with pytest.raises(ValueError, match="Train only"):
        profile.validate_calibration([{"split": "test", "label": 0}], targets)


def test_participant_weighting_ignores_unequal_photo_counts():
    rows = [{"person": "P01", "metrics": {"m": 10.0}}, {"person": "P02", "metrics": {"m": 30.0}}]
    first = profile.distribution(rows, "m", "person")
    second = profile.distribution(rows + [deepcopy(rows[0]) for _ in range(11)], "m", "person")
    assert first["mean"] == pytest.approx(second["mean"])
    assert second["mean"] == 20.0


def test_encrypted_roles_and_bad_key(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    from mediflow_datasets.smartphone_prepare import encrypt_inputs

    private = tmp_path / "private"
    private.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    rows = []
    for person in ("P01", "P02", "P03", "P04", "P05"):
        for i in range(2):
            rows.append(
                {
                    "id": person + str(i),
                    "person": person,
                    "phone": "test_phone",
                    "session": "S01",
                    "split": "holdout" if person == "P05" else "calibration",
                    "label": None,
                }
            )
    profile.write_new(private / "inventory.json", rows)
    for size in (224, 256):
        np.savez_compressed(
            private / f"target_inputs_{size}.npz",
            images=np.full((len(rows), size, size, 3), 150, np.float32),
        )
    key = AESGCM.generate_key(bit_length=256)
    text = base64.b64encode(key).decode()
    encrypt_inputs(private, runtime, key)
    calibration = training.decrypt_target(runtime, "calibration", 256, text)
    holdout = training.decrypt_target(runtime, "holdout", 256, text)
    assert {r["person"] for r in calibration["rows"]} == {"P01", "P02", "P03", "P04"}
    assert {r["person"] for r in holdout["rows"]} == {"P05"}
    with pytest.raises(ValueError, match="복호화"):
        training.decrypt_target(runtime, "calibration", 256, base64.b64encode(b"x" * 32).decode())
    meta = runtime / "target_encrypted/calibration_256.json"
    value = json.loads(meta.read_bytes())
    value["rows"][0]["label"] = 3
    meta.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        training.decrypt_target(runtime, "calibration", 256, text)


def test_notebook_compiles_and_sends_actual_newline_to_key_pipe():
    book = notebook.build_notebook("a" * 64)
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    source = "".join(next(c for c in book["cells"] if c["id"] == "run-five-trials")["source"])
    tree = ast.parse(source)
    key_write = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "write"
        and n.args
        and isinstance(n.args[0], ast.BinOp)
    ]
    assert len(key_write) == 1 and key_write[0].args[0].right.value == "\n"
    assert "getpass.getpass" in source and "--key" not in source


def test_colab_environment_recovers_missing_ensurepip_without_changing_kernel_tf(tmp_path):
    """Exercise the notebook setup with a Colab-like venv failure, without network calls."""
    source = "".join(
        next(
            c
            for c in notebook.build_notebook("a" * 64)["cells"]
            if c["id"] == "isolated-environment"
        )["source"]
    )
    source = source.replace('Path("/content/mediflow_phone_env")', "VENV_TEST_ROOT")
    dataset = tmp_path / "existing_dataset.zip"
    dataset.write_bytes(b"test")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="NVIDIA L4\n")
        if command[1:3] == ["-m", "venv"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="ensurepip missing")
        if "-c" in command:
            return SimpleNamespace(returncode=0, stdout=str(dataset) + "\n")
        return SimpleNamespace(returncode=0, stdout="")

    environment = {
        "Path": Path,
        "VENV_TEST_ROOT": tmp_path / "isolated",
        "RUNTIME": tmp_path,
        "PROJECT": tmp_path,
        "subprocess": SimpleNamespace(run=run),
        "sys": SimpleNamespace(executable="notebook-kernel-python"),
        "os": SimpleNamespace(environ={}),
    }
    exec(compile(source, "colab_setup", "exec"), environment)
    assert environment["SOURCE"] == dataset
    installs = [cmd for cmd in commands if cmd[1:4] == ["-m", "pip", "install"]]
    assert len(installs) == 2
    assert "virtualenv>=20,<21" in installs[0]
    assert installs[1][0] == str(tmp_path / "isolated/bin/python")
    assert "tensorflow[and-cuda]==2.20.0" in installs[1]


def tiny_context(tmp_path):
    keras.utils.set_random_seed(42)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    inputs = keras.Input((8, 8, 3))
    inner = keras.Input((8, 8, 3))
    hidden = keras.layers.Conv2D(4, 1, activation="relu")(keras.layers.Rescaling(1 / 255)(inner))
    base = keras.Model(inner, hidden)
    features = keras.layers.GlobalAveragePooling2D()(base(inputs))
    outputs = keras.layers.Dense(2, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    model.save(runtime / "start.keras")
    spec = {
        "role": "basic",
        "architecture": "b0",
        "size": 8,
        "label_smoothing": 0.0,
        "model": "start.keras",
        "model_sha256": data.sha256(runtime / "start.keras"),
    }
    treatment = profile.empty_profile(
        {
            "brightness_offset": [-5, 0],
            "contrast": [1, 1],
            "left_light": [0, 0],
            "scale_x": [1, 1],
            "scale_y": [1, 1],
        }
    )
    study = {
        "seed": 42,
        "epochs": 2,
        "learning_rate": 1e-5,
        "augmentation_probability": 0.5,
        "task": {"domain": "web_skin", "classes": ["a", "b"], "batch_size": 4},
        "profiles": {s: deepcopy(treatment) for s in ("stage1", "stage2", "stage4")},
    }
    profile.write_new(runtime / "study.json", study)
    rows = [data.ImageRow(str(i), "train", i % 2) for i in range(8)]
    val = [data.ImageRow("v" + str(i), "val", i % 2) for i in range(4)]
    rng = np.random.default_rng(42)
    cache = rng.uniform(30, 180, (12, 8, 8, 3)).astype("float32")
    views = {name: cache[8:].copy() for name in training.CONDITIONS[1:]}
    destination = tmp_path / "output"
    destination.mkdir()
    target = {
        "rows": [
            {"id": p, "person": p, "split": "calibration", "label": None}
            for p in ("P01", "P02", "P03", "P04")
        ],
        "images": rng.uniform(220, 250, (4, 8, 8, 3)).astype("float32"),
        "sha256": "test_target",
    }
    return runtime, destination, tmp_path / "saving", study, spec, rows, val, cache, views, target


def test_actual_training_resumes_after_committed_epoch_and_preserves_start(tmp_path, monkeypatch):
    context = tiny_context(tmp_path)
    original = training.train.save_epoch

    def interrupt(state, folder, local, entry):
        result = original(state, folder, local, entry)
        if entry["epoch"] == 1:
            raise RuntimeError("simulated interruption after epoch commit")
        return result

    monkeypatch.setattr(training.train, "save_epoch", interrupt)
    monkeypatch.setattr(training, "curves", lambda *args: None)
    with pytest.raises(RuntimeError, match="simulated"):
        training.run_trial(*context[:5], 1, *context[5:9], None)
    folder = context[1] / "stage1_basic"
    assert (folder / "epoch_001.json").exists() and not (folder / "complete.json").exists()
    monkeypatch.setattr(training.train, "save_epoch", original)
    result = training.run_trial(*context[:5], 1, *context[5:9], None)
    assert result["epochs_completed"] == 2
    assert data.sha256(context[0] / "start.keras") == context[4]["model_sha256"]
    assert len(list(folder.glob("snapshot_*"))) == 1
    assert training.run_trial(*context[:5], 1, *context[5:9], None) == result
    keras.backend.clear_session()


def test_alignment_runs_real_updates_and_never_uses_holdout(tmp_path, monkeypatch):
    context = list(tiny_context(tmp_path))
    context[3]["epochs"] = 1
    monkeypatch.setattr(training, "curves", lambda *args: None)
    result = training.run_trial(*context[:5], 3, *context[5:])
    history = json.loads((context[1] / "stage3/history.json").read_bytes())
    assert result["epochs_completed"] == 1 and history[0]["alignment_steps"] == 1
    assert history[0]["target_images_used"] == 4
    assert history[0]["alignment"]["mmd"] > 0 and history[0]["alignment"]["coral"] >= 0
    assert result["best_model"]["sha256"] != context[4]["model_sha256"]
    assert data.sha256(context[0] / "start.keras") == context[4]["model_sha256"]
    keras.backend.clear_session()


@pytest.mark.parametrize("reuse_stage2", [True, False])
def test_all_trials_freeze_selection_before_holdout(tmp_path, monkeypatch, reuse_stage2):
    import zipfile

    context = list(tiny_context(tmp_path))
    runtime, destination, _, study, spec, rows, val, cache, views, target = context
    study["epochs"] = 1
    study["reuse_identical_stage2"] = reuse_stage2
    if not reuse_stage2:
        study["profiles"]["stage2"]["color_a"] = [0.0, 0.5]
    study["task"]["models"] = {"basic": spec, "selected": {**spec, "role": "selected"}}
    source = tmp_path / "source.zip"
    source.write_bytes(b"unit-source-archive")
    study["task"]["source_sha256"] = data.sha256(source)
    frozen = {"original": {"train": [], "val": [], "test": []}}
    training.write_progress(runtime / "source_manifest.json", frozen)
    study["source_manifest_sha256"] = data.sha256(runtime / "source_manifest.json")
    training.write_progress(runtime / "runtime_manifest.json", {"unit_test": True})
    training.write_progress(runtime / "reference.json", {})

    def prepare(domain, archive, task, reference, out):
        assert domain == "web_skin"
        training.write_progress(out / "source_manifest.json", frozen)
        return rows, val, {}

    def targets(root, role, size, key):
        if role == "holdout":
            assert (destination / "selection.json").exists()
            return {
                **target,
                "rows": [
                    {"id": f"P05_{i:02}", "person": "P05", "split": "holdout", "label": None}
                    for i in (1, 2)
                ],
                "images": target["images"][:2],
            }
        return target

    monkeypatch.setattr(training, "verify_runtime", lambda p: study)
    monkeypatch.setattr(training.train, "prepare_source", prepare)
    monkeypatch.setattr(training, "input_cache", lambda *args: (cache, views))
    monkeypatch.setattr(training, "decrypt_target", targets)
    monkeypatch.setattr(training, "curves", lambda *args: None)
    archive = training.run(
        runtime, source, destination, tmp_path / "local", "test", require_gpu=False
    )
    assert len(list(destination.glob("stage*/complete.json"))) == 5
    result = data.read_json(destination / "summary.json")
    assert result["status"] == "completed" and result["source_test_evaluated"] is False
    assert result["training_trials"] == (4 if reuse_stage2 else 5)
    assert result["aliases"] == ({"stage2": "stage1"} if reuse_stage2 else {})
    assert (destination / "stage2/best.keras").exists() is not reuse_stage2
    assert (
        data.read_json(destination / "feature_and_prediction_diagnostics.json")["target_accuracy"]
        is None
    )
    with zipfile.ZipFile(archive) as zipped:
        assert len([p for p in zipped.namelist() if p.endswith("best.keras")]) == (
            4 if reuse_stage2 else 5
        )
        assert not any("snapshot_" in p or "Key" in p for p in zipped.namelist())
