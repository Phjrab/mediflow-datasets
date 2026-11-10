"""Regression coverage for compiled steps and Drive epoch recovery."""

import io
import zipfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from mediflow_datasets import adaptation_stage1 as stage
from mediflow_datasets import adaptation_stage2 as stage2
from mediflow_datasets.domain_adaptation_data import ImageRow


def small_model(pmg=False):
    inputs = keras.Input((16, 16, 3))
    x = keras.layers.GlobalAveragePooling2D()(keras.layers.Rescaling(1 / 255)(inputs))
    outputs = (
        [keras.layers.Dense(2)(x) for _ in range(4)]
        if pmg
        else keras.layers.Dense(2, activation="softmax")(x)
    )
    return keras.Model(inputs, outputs)


def test_pmg_compiled_step_retains_four_updates_and_accepts_final_short_batch():
    model = small_model(pmg=True)
    opt = keras.optimizers.Adam(1e-4)
    spec = {"architecture": "pmg", "size": 16, "label_smoothing": 0}
    train, infer = stage.steps(model, spec, opt, 2)
    for n in (3, 1):
        images = tf.ones((n, 16, 16, 3), dtype=tf.float32) * 120
        assert np.isfinite(float(train(images, tf.zeros((n,), dtype=tf.int32))))
        assert infer(images).shape == (n, 2)
    assert int(opt.iterations) == 8


def test_interrupted_epoch_resumes_weights_and_optimizer(tmp_path, monkeypatch, capsys):
    images = np.random.default_rng(11).uniform(30, 210, (6, 16, 16, 3)).astype(np.float32)
    train = [ImageRow(f"train/{i}.png", "train", i % 2) for i in range(4)]
    val = [ImageRow(f"val/{i}.png", "val", i) for i in range(2)]
    lighting = {
        "brightness_offset": [-1, 1],
        "contrast": [1, 1],
        "left_light": [0, 0],
        "photometric_operator": "bounded_additive_v3",
    }
    spec = {"architecture": "b0", "size": 16, "label_smoothing": 0, "role": "basic"}
    config = {"seed": 42, "augmentation_probability": 1.0}
    task = {"domain": "skin", "classes": ["a", "b"], "batch_size": 3}
    context = (config, task, train, val, lighting, images, images[4:].copy(), list(range(4)))
    folder, local = tmp_path / "lighting", tmp_path / "local"
    save = stage.save_epoch

    def interrupt_after_committing(*args):
        save(*args)
        raise KeyboardInterrupt

    monkeypatch.setattr(stage, "save_epoch", interrupt_after_committing)
    keras.utils.set_random_seed(42)
    model = small_model()
    initial_weights = model.get_weights()
    with pytest.raises(KeyboardInterrupt):
        stage.phase(model, context, folder, local, spec, "lighting", 2, 1e-4)
    first_epoch = (folder / "epoch_001.json").read_bytes()
    monkeypatch.setattr(stage, "save_epoch", save)
    resumed = small_model()
    resumed.set_weights(initial_weights)
    stage.phase(resumed, context, folder, local, spec, "lighting", 2, 1e-4)
    assert (folder / "epoch_001.json").read_bytes() == first_epoch
    assert [x["epoch"] for x in stage.epoch_records(folder, 2)] == [1, 2]
    opt = keras.optimizers.Adam(1e-4)
    opt.build(resumed.trainable_variables)
    checkpoint = tf.train.Checkpoint(model=resumed, optimizer=opt)
    stage.restore_epoch(checkpoint, folder, stage.epoch_records(folder, 2)[-1])
    assert int(opt.iterations) == 4
    output = capsys.readouterr().out
    assert "Epoch 1/2 학습 | 1/2" in output and "Epoch 2/2" in output
    assert "Drive에서 1/2 epoch 복원" in output


def write_source(archive):
    with zipfile.ZipFile(archive, "x") as z:
        for variant in ("original", "augmented"):
            for split_index, split in enumerate(stage.data.SPLITS):
                for label, name in enumerate(("a", "b")):
                    rng = np.random.default_rng(100 + 2 * split_index + label)
                    image = rng.integers(20, 235, (16, 16, 3), np.uint8)
                    buffer = io.BytesIO()
                    Image.fromarray(image).save(buffer, format="PNG")
                    z.writestr(f"source/{variant}/{split}/{name}/image.png", buffer.getvalue())


def test_saved_skin_runs_lighting_without_preparation_or_augmentation_lineage(
    tmp_path, monkeypatch
):
    archive = tmp_path / "source.zip"
    write_source(archive)
    original = tmp_path / "saved.keras"
    small_model().save(original)
    monkeypatch.setattr(stage, "ROOT", tmp_path)
    monkeypatch.setattr(stage.exp, "configure_partial", lambda model: None)
    spec = {
        "architecture": "b0",
        "size": 16,
        "label_smoothing": 0,
        "role": "basic",
        "model": original.name,
        "model_sha256": stage.data.sha256(original),
    }
    task = {
        "domain": "skin",
        "classes": ["a", "b"],
        "batch_size": 2,
        "input_policy": "legacy_resize",
        "source_sha256": stage.data.sha256(archive),
        "models": {"basic": spec, "selected": spec},
    }
    profile = {
        "brightness_offset": [-1, 1],
        "contrast": [1, 1],
        "left_light": [0, 0],
        "photometric_operator": "bounded_additive_v3",
    }
    train, val, descendants = stage.prepare_source(
        "skin", archive, task, {"pixel_hashes": []}, tmp_path
    )
    assert not descendants
    config = {
        "seed": 42,
        "augmentation_probability": 1.0,
        "epochs": 1,
        "learning_rate": 1e-5,
        "validation_scope": "exploratory_checkpoint",
    }
    cache = stage.image_cache(tmp_path, archive, train, val, descendants, task, 16, profile)
    result = stage.trial(
        (config, task, train, val, profile, *cache), tmp_path / "run", tmp_path / "scratch", spec, 1
    )
    assert result["validation_scope"] == "exploratory_checkpoint"
    assert (tmp_path / "run/initial.keras").read_bytes() == original.read_bytes()
    assert (tmp_path / "run/lighting/epoch_001.json").is_file()
    assert not list((tmp_path / "run").glob("prepare_*"))


def test_expanded_stage_starts_from_bundle_and_compares_on_common_validation(tmp_path, monkeypatch):
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        stage.write_progress(path, value)

    runtime, previous, destination = (tmp_path / name for name in ("runtime", "stage1", "stage2"))
    archive = tmp_path / "source.zip"
    write_source(archive)
    runtime.mkdir()
    original = runtime / "original.keras"
    model = small_model()
    model.save(original)
    start_weights = model.get_weights()
    model.set_weights([weights + 0.1 for weights in start_weights])
    earlier_best = previous / "skin/basic/best.keras"
    earlier_best.parent.mkdir(parents=True)
    model.save(earlier_best)
    spec = {
        "architecture": "b0",
        "size": 16,
        "label_smoothing": 0,
        "model": original.name,
        "model_sha256": stage.data.sha256(original),
    }
    task = {
        "domain": "skin",
        "classes": ["a", "b"],
        "batch_size": 2,
        "input_policy": "legacy_resize",
        "source_sha256": stage.data.sha256(archive),
        "models": {"basic": spec},
    }
    config = {
        "seed": 42,
        "augmentation_probability": 1.0,
        "epochs": 1,
        "learning_rate": 1e-5,
        "validation_scope": "exploratory_checkpoint",
        "tasks": {d: task for d in stage.DOMAINS},
    }
    profile = {
        "schema": "measured_camera_extended_v1",
        "domain": "skin",
        "lighting": {"brightness_offset": [-1, 1], "contrast": [1, 1], "left_light": [0, 0]},
        "color_a": [0.4, 0.4],
        "color_b": [0.2, 0.2],
        "texture_gain": [0.95, 0.95],
        "blur_sigma_roi64": [0.25, 0.25],
        "blur_input_scale": 2.0,
    }
    reference = {"pixel_hashes": [], "extended_profile": profile}
    write_json(runtime / "configs/adaptation_stage1.json", config)
    write_json(runtime / "references/skin.json", reference)
    write_json(runtime / "bundle_provenance.json", {})
    write_json(previous / "bundle.json", {"stage": 1, "sha256": "abc"})
    write_json(
        previous / "summary.json",
        {
            "completed_trials": [f"{d}/{r}" for d in stage.DOMAINS for r in stage.ARMS],
            "failures": {},
            "selection_written": True,
        },
    )
    choices = {}
    for domain in stage.DOMAINS:
        write_json(
            previous / domain / "basic/complete.json",
            {
                "initial": {"sha256": spec["model_sha256"]},
                "best_model": {"sha256": stage.data.sha256(earlier_best)},
                "model_config": {**spec, "role": "basic"},
            },
        )
        choices[domain] = {
            "role": "basic",
            "model_config": {**spec, "role": "basic"},
            "initial_checkpoint": {
                "path": f"{domain}/basic/initial.keras",
                "sha256": spec["model_sha256"],
            },
            "stage1_checkpoint": {
                "path": f"{domain}/basic/best.keras",
                "sha256": stage.data.sha256(earlier_best),
            },
        }
    write_json(
        previous / "selection.json",
        {
            "schema": "mediflow_adaptation_stage1_selection_v1",
            "selected": choices,
        },
    )
    stage.prepare_source("skin", archive, task, reference, previous / "skin")
    phase = stage.phase

    def check_initial_weights(model, *args, **kwargs):
        for actual, expected in zip(model.get_weights(), start_weights, strict=True):
            np.testing.assert_array_equal(actual, expected)
        return phase(model, *args, **kwargs)

    monkeypatch.setattr(stage.exp, "configure_partial", lambda model: None)
    monkeypatch.setattr(stage, "phase", check_initial_weights)
    local = tmp_path / "scratch"
    local.mkdir()
    output = destination / "skin"
    stage2.run_domain(runtime, previous, "abc", "skin", archive, output, local)
    record = stage2.read_json(output / "complete.json")
    assert record["best_epoch"] == 1 and not record["source_test_evaluated"]
    assert set(record["validation"]) == {"initial", "stage1_lighting", "stage2_expanded"}
    assert all(set(m) == {"clean", "lighting", "expanded"} for m in record["validation"].values())
    assert stage.data.sha256(output / "best.keras") == record["best_model"]["sha256"]
    assert not np.array_equal(
        np.load(local / "val_lighting_16.npy"), np.load(local / "val_expanded_16.npy")
    )
    stage2.run_domain(runtime, previous, "abc", "skin", tmp_path / "missing.zip", output, local)


def test_aligned_phase_keeps_source_updates_and_records_target_usage(tmp_path):
    rng = np.random.default_rng(21)
    images = rng.uniform(30, 220, (8, 16, 16, 3)).astype(np.float32)
    train = [ImageRow(f"train/{i}.png", "train", i % 2) for i in range(6)]
    val = [ImageRow(f"val/{i}.png", "val", i) for i in range(2)]
    model = small_model()
    spec = {"architecture": "b0", "size": 16, "label_smoothing": 0, "role": "basic"}
    config = {"seed": 42, "augmentation_probability": 0.5}
    task = {"domain": "skin", "classes": ["a", "b"], "batch_size": 2}
    profile = {
        "lighting": {"brightness_offset": [-1, 1], "contrast": [1, 1], "left_light": [0, 0]},
        "color_a": [0.4, 0.4],
        "color_b": [0.2, 0.2],
        "texture_gain": [0.95, 0.95],
        "blur_sigma_roi64": [0.25, 0.25],
        "blur_input_scale": 2.0,
    }
    alignment = {
        "rows": [{"person": f"P{i}", "split": "calibration", "label": None} for i in range(4)],
        "images": images[:4] * 0.5,
        "reference": {"mean": [0, 0, 0], "std": [1, 1, 1], "bandwidth_sq": 1.0},
        "weights": {"mmd": 0.05, "coral": 0.05},
    }
    context = (
        config,
        task,
        train,
        val,
        profile,
        images,
        {"lighting": images[6:], "expanded": images[6:] * 0.95},
        list(range(6)),
    )
    result = stage.phase(
        model,
        context,
        tmp_path / "aligned",
        tmp_path / "saving",
        spec,
        "aligned",
        1,
        1e-5,
        alignment=alignment,
    )
    assert result["optimizer_updates"] == 3
    assert result["alignment"]["updates"] == 1
    assert result["alignment"]["target_images_used"] == 4
    assert result["alignment"]["mmd_mean"] > 0 and result["alignment"]["coral_mean"] > 0
