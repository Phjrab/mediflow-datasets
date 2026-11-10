from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_resolution as resolution
from mediflow_datasets import domain_nine as nine
from mediflow_datasets.domain_adaptation_training import object_hash


def tiny(domain):
    import keras

    inside = keras.Input((32, 32, 3))
    feature = keras.layers.Conv2D(4, 1)(keras.layers.Rescaling(1 / 255)(inside))
    backbone = keras.Model(inside, feature)
    backbone.trainable = False
    x = keras.Input((32, 32, 3))
    pooled = keras.layers.GlobalAveragePooling2D()(backbone(x))
    if domain == "web_skin":
        fusion = keras.layers.Dense(4, activation="elu", name="pmg_fusion")(pooled)
        outputs = [keras.layers.Dense(2)(fusion) for _ in range(4)]
    else:
        outputs = keras.layers.Dense(2, activation="softmax")(pooled)
    return keras.Model(x, outputs)


def archive_fixture(path, domain):
    with zipfile.ZipFile(path, "x") as z:
        for variant in ("original", "augmented"):
            for split_index, split in enumerate(data.SPLITS):
                for label in range(2):
                    for i in range(2 if split == "train" else 1):
                        seed = 100 * split_index + 10 * label + i
                        pixels = np.random.default_rng(seed).integers(
                            20, 240, (32, 32, 3), np.uint8
                        )
                        buffer = io.BytesIO()
                        Image.fromarray(pixels).save(buffer, format="PNG")
                        z.writestr(
                            f"{domain}/{variant}/{split}/C{label}/{i}.png", buffer.getvalue()
                        )
    return data.sha256(path)


def test_all_nine_paths_actual_gradients_resume_and_no_test_selection_leak(tmp_path, monkeypatch):
    config = {
        "schema": "mediflow_nine_v1",
        "arms": list(nine.ARMS),
        "seed": 42,
        "epochs": 1,
        "batch_size": 2,
        "baseline_head_epochs": 1,
        "baseline_partial_epochs": 1,
        "learning_rate": 1e-4,
        "augmentation_probability": 1,
        "gradient_clip_norm": 1,
        "alignment_fraction_per_loss": 0.05,
        "validation_clean_macro_f1_tolerance": 0.01,
        "validation_clean_recall_tolerance": 0.05,
        "validation_min_shift_gain": 0.005,
        "tasks": {},
    }
    candidates, archives, reviews = {}, {}, {}
    for domain in nine.DOMAINS:
        checkpoint = tmp_path / (domain + ".keras")
        tiny(domain).save(checkpoint)
        archives[domain] = tmp_path / (domain + "_datasets.zip")
        digest = archive_fixture(archives[domain], domain)
        spec = {
            "input_size": 32,
            "model_sha256": data.sha256(checkpoint),
            "output_adapter": "sum_logits_softmax" if domain == "web_skin" else "direct_softmax",
            "test_accuracy": 0.123,
        }
        candidates[domain] = spec, checkpoint, ["C0", "C1"]
        config["tasks"][domain] = {
            "source_sha256": digest,
            "input_policy": "legacy_resize",
            "train_variant": "augmented" if domain == "hair" else "original",
            "loss": "ls005" if domain == "hair" else "ce",
        }
    config_path = tmp_path / "config.json"
    runner.write_json(config_path, config)
    assert nine.load_config(config_path)["arms"] == list(nine.ARMS)
    monkeypatch.setattr(data, "candidate", lambda domain: candidates[domain])
    fresh_calls = []

    def fresh(domain, *args):
        fresh_calls.append(domain)
        return tiny(domain)

    monkeypatch.setattr(resolution, "fresh_model", fresh)
    for domain in nine.DOMAINS:
        review = tmp_path / (domain + "_review")
        review.mkdir()
        reviews[domain] = review
        target = [
            {"path": f"P{p}/{i}.png", "person": f"P{p}", "split": "calibration", "label": None}
            for p in range(4)
            for i in range(2)
        ]
        images = np.stack(
            [
                np.random.default_rng(i).uniform(
                    25 + 15 * (i // 2), 60 + 15 * (i // 2), (32, 32, 3)
                )
                for i in range(8)
            ]
        ).astype(np.float32)
        np.savez_compressed(review / "target_inputs.npz", images=images)
        runner.write_json(
            review / "target_inputs.json",
            {
                "domain": domain,
                "input_policy": "legacy_resize",
                "rows": target,
                "array_sha256": data.sha256(review / "target_inputs.npz"),
                "pixel_hashes": [],
            },
        )
        light = {
            "brightness_offset": [-2, 3],
            "contrast": [0.98, 1.03],
            "left_light": [0, 1],
            "scale_x": [1, 1],
            "scale_y": [1, 1],
            "photometric_operator": "bounded_additive_v3",
        }
        profile = {
            "lighting": light,
            "color_a": [-0.5, 0.5],
            "color_b": [-0.5, 0.5],
            "texture_gain": [0.95, 1.05],
            "blur_sigma_roi64": [0, 0.2],
            "blur_input_scale": 1,
        }
        runner.write_json(review / "extended_profile.json", profile)
        if domain != "hair":
            plan = resolution.quarantine_plan(
                domain,
                [],
                {},
                ["C0", "C1"],
                config["tasks"][domain]["source_sha256"],
                {"original/train": 4, "original/val": 2, "original/test": 2},
            )
            runner.write_json(review / "quarantine_plan.json", plan)
        (review / "evidence.txt").write_text("simulated review evidence", encoding="utf-8")
        runner.write_json(
            review / "readiness.json",
            {
                "status": "reviewed_nine_pilot",
                "config_sha256": object_hash(config),
                "profile_sha256": object_hash(profile),
                "evidence_sha256": {"evidence.txt": data.sha256(review / "evidence.txt")},
            },
        )
    original_predictions = nine.predictions
    test_calls = []

    def guarded(model, rows, reader, spec, *args, **kwargs):
        if rows[0].split == "test":
            assert (tmp_path / spec["domain"] / "frozen_selection.json").exists()
            test_calls.append(spec["domain"])
        return original_predictions(model, rows, reader, spec, *args, **kwargs)

    monkeypatch.setattr(nine, "predictions", guarded)
    original_arm = nine.run_arm
    inject_failure = True

    def arm(output, name, *args, **kwargs):
        if output.name == "hair" and name == "expanded" and inject_failure:
            raise RuntimeError("simulated interrupted arm")
        return original_arm(output, name, *args, **kwargs)

    monkeypatch.setattr(nine, "run_arm", arm)
    arguments = {}
    for domain in nine.DOMAINS:
        arguments[domain] = SimpleNamespace(
            domain=domain,
            config=config_path,
            source_zip=archives[domain],
            review=reviews[domain],
            output=tmp_path / domain,
        )
        if domain == "hair":
            with pytest.raises(RuntimeError, match="Some training arms failed"):
                nine.run_domain(arguments[domain])
            assert "hair" not in test_calls
            assert (tmp_path / "hair/lighting/complete.json").exists()
            assert (tmp_path / "hair/aligned/complete.json").exists()
        else:
            nine.run_domain(arguments[domain])
    inject_failure = False
    nine.run_domain(arguments["hair"])
    assert set(fresh_calls) == {"skin", "web_skin"} and len(fresh_calls) == 2
    hashes = {}
    for domain in nine.DOMAINS:
        root = tmp_path / domain
        assert data.read_json(root / "status.json")["status"] == "completed"
        frozen = data.read_json(root / "frozen_selection.json")
        assert set(frozen["arms"]) == set(nine.ARMS)
        steps = []
        for name in nine.ARMS:
            record = frozen["arms"][name]
            history = data.read_json(root / name / record["history_relative"])
            steps.append(history[0]["optimizer_updates"])
            if name == "aligned":
                assert history[0]["alignment_steps"] > 0 and history[0]["mmd_sum"] > 0
            else:
                assert history[0]["alignment_steps"] == 0
        assert len(set(steps)) == 1
        result = data.read_json(root / "final_evaluation/comparison.json")
        assert result["selected_before_test"] == frozen["selected"]
        assert result["results"]["aligned"]["target_descriptive"]["accuracy"] is None
        assert result["results"]["baseline"]["original_source_test"]["count"] == 2
        hashes[domain] = data.sha256(root / "final_evaluation/comparison.json")
    assert len(test_calls) == 36  # 3 domains x (baseline + 3 arms) x 3 Test conditions.
    calls_before_resume = len(test_calls)

    def never_train(*args, **kwargs):
        raise AssertionError("Completed experiments should not train again")

    monkeypatch.setattr(nine, "train_epoch", never_train)
    for domain in nine.DOMAINS:
        nine.run_domain(arguments[domain])
        assert data.sha256(tmp_path / domain / "final_evaluation/comparison.json") == hashes[domain]
    assert len(test_calls) == calls_before_resume


def test_target_inputs_reject_label_and_tensor_tampering(tmp_path):
    spec = {"input_policy": "legacy_resize", "input_size": 32}
    images = np.full((4, 32, 32, 3), 100, dtype=np.float32)
    np.savez_compressed(tmp_path / "target_inputs.npz", images=images)
    rows = [{"person": f"P{i}", "split": "calibration", "label": None} for i in range(4)]
    meta = {
        "domain": "skin",
        "input_policy": "legacy_resize",
        "rows": rows,
        "array_sha256": data.sha256(tmp_path / "target_inputs.npz"),
    }
    runner.write_json(tmp_path / "target_inputs.json", meta)
    assert nine.target_inputs(tmp_path, "skin", spec)[1].shape == images.shape
    changed = {**meta, "rows": [{**rows[0], "label": 0}, *rows[1:]]}
    (tmp_path / "target_inputs.json").write_text(
        __import__("json").dumps(changed), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unlabeled"):
        nine.target_inputs(tmp_path, "skin", spec)
    (tmp_path / "target_inputs.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="array changed"):
        nine.target_inputs(tmp_path, "skin", spec)
