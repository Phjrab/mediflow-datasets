import io
import json
import zipfile

import keras
import numpy as np
import pytest
from PIL import Image

from mediflow_datasets import adaptation_stage1 as train
from mediflow_datasets import adaptation_strength as study_runner


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_strength_starts_before_adaptation_and_compares_on_validation(tmp_path, monkeypatch):
    runtime, stage1, stage2 = (tmp_path / name for name in ("runtime", "stage1", "stage2"))
    runtime.mkdir()
    inputs = keras.Input((16, 16, 3))
    backbone = keras.Sequential([keras.Input((16, 16, 3)), keras.layers.Conv2D(2, 1)])
    features = keras.layers.GlobalAveragePooling2D()(backbone(inputs))
    model = keras.Model(inputs, keras.layers.Dense(2, activation="softmax")(features))
    initial_weights = [v.copy() for v in model.get_weights()]
    initial = runtime / "initial.keras"
    model.save(initial)
    original_hash = train.data.sha256(initial)
    model.set_weights([v + 0.1 for v in model.get_weights()])
    old = runtime / "old.keras"
    model.save(old)
    old_hash = train.data.sha256(old)
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "x") as target:
        for variant in ("original", "augmented"):
            for index, split in enumerate(("train", "val", "test")):
                for label in range(2):
                    image = np.random.default_rng(index * 10 + label).integers(
                        0, 256, (16, 16, 3), dtype=np.uint8
                    )
                    buffer = io.BytesIO()
                    Image.fromarray(image).save(buffer, format="PNG")
                    target.writestr(
                        f"fixture/{variant}/{split}/C{label}/image.png", buffer.getvalue()
                    )
    spec = {
        "architecture": "b0",
        "size": 16,
        "label_smoothing": 0,
        "model": "initial.keras",
        "model_sha256": original_hash,
    }
    task = {
        "domain": "skin",
        "classes": ["C0", "C1"],
        "batch_size": 2,
        "input_policy": "legacy_resize",
        "source_name": "source",
        "source_sha256": train.data.sha256(archive),
        "models": {"basic": spec, "selected": spec},
    }
    config = {
        "tasks": {d: {**task, "domain": d} for d in train.DOMAINS},
        "seed": 42,
        "augmentation_probability": 0.5,
        "epochs": 1,
        "learning_rate": 1e-5,
        "validation_scope": "fixture",
    }
    weak = {
        "schema": "measured_camera_extended_v1",
        "domain": "skin",
        "lighting": {"brightness_offset": [1, 1], "contrast": [1, 1], "left_light": [0, 0]},
        "color_a": [0, 0],
        "color_b": [0, 0],
        "texture_gain": [1, 1],
        "blur_sigma_roi64": [0, 0],
        "blur_input_scale": 1,
    }
    reference = {"pixel_hashes": [], "extended_profile": weak}
    write_json(runtime / "configs/adaptation_stage1.json", config)
    write_json(runtime / "references/skin.json", reference)
    choices = {}
    for domain in train.DOMAINS:
        chosen = {**spec, "role": "basic"}
        adapted = {"path": f"{domain}/basic/best.keras", "sha256": old_hash}
        choices[domain] = {
            "role": "basic",
            "model_config": chosen,
            "initial_checkpoint": {
                "path": f"{domain}/basic/initial.keras",
                "sha256": original_hash,
            },
            "stage1_checkpoint": adapted,
        }
        write_json(
            stage1 / domain / "basic/complete.json",
            {
                "initial": {"sha256": original_hash},
                "best_model": {"sha256": old_hash},
                "model_config": chosen,
            },
        )
        write_json(
            stage2 / domain / "complete.json",
            {
                "best_model": {"file": "best.keras", "sha256": old_hash},
                "model_config": chosen,
                "stage1_best": adapted,
            },
        )
        (stage2 / domain / "best.keras").write_bytes(old.read_bytes())
    write_json(stage1 / "bundle.json", {"stage": 1, "sha256": "fixture"})
    write_json(
        stage1 / "summary.json",
        {
            "completed_trials": [f"{d}/{a}" for d in train.DOMAINS for a in train.ARMS],
            "failures": {},
            "selection_written": True,
        },
    )
    write_json(
        stage1 / "selection.json",
        {"schema": "mediflow_adaptation_stage1_selection_v1", "selected": choices},
    )
    write_json(stage2 / "bundle.json", {"stage": 2, "sha256": "fixture"})
    write_json(
        stage2 / "summary.json",
        {"completed_domains": list(train.DOMAINS), "failures": {}, "comparison_written": True},
    )
    write_json(
        stage2 / "skin/run_configuration.json", {"profile": weak, "training_configuration": config}
    )
    train.prepare_source("skin", archive, task, reference, stage1 / "skin")
    (stage2 / "skin/source_manifest.json").write_bytes(
        (stage1 / "skin/source_manifest.json").read_bytes()
    )
    entry = {
        "profile": {**weak, "lighting": {**weak["lighting"], "brightness_offset": [8, 8]}},
        "previous_reference_sha256": train.object_hash(reference),
        "source_manifest_sha256": train.data.sha256(stage1 / "skin/source_manifest.json"),
        "previous_best_sha256": old_hash,
    }
    study = tmp_path / "study.json"
    write_json(
        study,
        {
            "bundle_sha256": "fixture",
            "configuration_sha256": train.object_hash(config),
            "domains": {"skin": entry},
        },
    )
    phase = train.phase

    def checked_phase(model, *args, **kwargs):
        for actual, expected in zip(model.get_weights(), initial_weights, strict=True):
            np.testing.assert_array_equal(actual, expected)
        return phase(model, *args, **kwargs)

    monkeypatch.setattr(train, "phase", checked_phase)
    local, output = tmp_path / "local", tmp_path / "output"
    local.mkdir()
    study_runner.run_domain(runtime, stage1, stage2, study, "skin", archive, output, local)
    result = json.loads((output / "complete.json").read_bytes())
    assert result["best_epoch"] == 1 and result["source_test_evaluated"] is False
    assert set(result["validation"]) == {"initial", "previous_expanded", "stronger_expanded"}
    assert all(
        set(metrics) == {"clean", "weak", "lighting", "expanded"}
        for metrics in result["validation"].values()
    )
    for metrics in result["validation"].values():
        assert all(value["labeled_count"] == 2 for value in metrics.values())
    before = (output / "complete.json").read_bytes()
    study_runner.run_domain(
        runtime, stage1, stage2, study, "skin", tmp_path / "missing.zip", output, local
    )
    assert (output / "complete.json").read_bytes() == before
    assert not np.array_equal(
        np.load(local / "previous_validation.npy"), np.load(local / "val_expanded_16.npy")
    )
    broken = json.loads(study.read_bytes())
    broken["domains"]["skin"]["profile"]["domain"] = "hair"
    with pytest.raises(ValueError, match="frozen inputs"):
        study_runner.frozen_inputs(runtime, stage1, stage2, broken, "skin")


def test_space_check_includes_previous_condition_validation_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace

    train_count, val_count, size = 2, 3, 16
    old_estimate = (train_count + 3 * val_count) * size * size * 3 * 4
    revised_estimate = (train_count + 4 * val_count) * size * size * 3 * 4
    reserve = 1024**3
    assert study_runner.cache_bytes(train_count, val_count, 0, size) == revised_estimate
    monkeypatch.setattr(
        study_runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=old_estimate + reserve)
    )
    with pytest.raises(RuntimeError, match="공간 부족"):
        study_runner.require_cache_space(tmp_path, train_count, val_count, 0, size)
    monkeypatch.setattr(
        study_runner.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=revised_estimate + reserve),
    )
    assert study_runner.require_cache_space(tmp_path, train_count, val_count, 0, size) == (
        revised_estimate + reserve
    )
