from __future__ import annotations

from dataclasses import asdict

import pytest

from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_resolution as resolution
from mediflow_datasets.domain_adaptation_training import object_hash


def test_quarantine_components_preserve_test_and_do_not_clear_identity():
    paths = {s: f"data/original/{s}/C0/x.png" for s in data.SPLITS}
    # Train connects Validation to Test transitively, so both lower-priority splits quarantine it.
    groups = [{"paths": [paths["train"], paths[s]]} for s in ("val", "test")]
    manifest = {p: {"split": s, "label": 0} for s, p in paths.items()}
    plan = resolution.quarantine_plan(
        "skin", groups, manifest, ["C0", "C1"], "source",
        {f"original/{s}": 3 for s in data.SPLITS},
    )
    assert plan["exclude_original"] == {"train": [paths["train"]], "val": [paths["val"]]}
    assert plan["expected_original_counts"] == {"train": 2, "val": 2, "test": 3}
    assert plan["test_membership_changed"] is False
    assert plan["identity_independence_verified"] is False
    source = {(v, s): [data.ImageRow(p, s, label, variant=v)
              for label, p in enumerate([paths[s], f"data/{v}/{s}/C1/y.png"])]
              for v in ("original", "augmented") for s in data.SPLITS}
    # Emptying a class cannot pass, even with a plausible quarantine path.
    plan["expected_original_counts"] = {"train": 1, "val": 1, "test": 2}
    with pytest.raises(ValueError, match="class became empty"):
        resolution.filtered_source(source, plan, ["C0", "C1"], "source")


def test_within_participant_calibration_is_independent_of_missing_source_face_rois():
    source = [{"path": "a", "split": "train", "label": 0,
               "metrics": {"frame_y_mean": 240, "roi_status": "face_not_found"}}]
    target = [{"person": f"P{p}", "condition": condition, "split": "calibration",
               "metrics": {"y_p50": 80 + (20 if condition == "alt" else 0), "contrast": 20}}
              for p in range(4) for condition in ("ref", "alt")]
    profile = analysis.calibrate("web_skin", source, target, brightness_policy="within_participant")
    assert profile["brightness_offset"] == [10, 10]
    assert "source_brightness_anchor" not in profile["evidence"]
    changed = [{**source[0], "metrics": {"frame_y_mean": 1, "y_p50": 1}}]
    other = analysis.calibrate("web_skin", changed, target, brightness_policy="within_participant")
    assert other["brightness_offset"] == profile["brightness_offset"]
    with pytest.raises(ValueError, match="Train"):
        analysis.calibrate("web_skin", [{**source[0], "split": "test"}], target,
                           brightness_policy="within_participant")


def test_fresh_factories_never_load_mediflow_checkpoint_weights(monkeypatch):
    calls = []
    monkeypatch.setattr(resolution.engine, "build_model", lambda spec: calls.append(spec) or "new")
    assert resolution.fresh_model("skin", {"input_size": 224}, 10) == "new"
    assert calls == [{"backbone": "B0", "size": 224, "class_count": 10, "loss": "ce"}]
    with pytest.raises(ValueError, match="ImageNet"):
        resolution.fresh_model("skin", {"input_size": 224}, 10, weights="old_checkpoint")


def test_fresh_pipeline_uses_only_retained_originals_and_portable_new_baseline(
    tmp_path, monkeypatch,
):
    import io
    import shutil
    import zipfile
    from types import SimpleNamespace

    import keras
    import numpy as np
    from PIL import Image

    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "x") as archive:
        for v in ("original", "augmented"):
            for sindex, split in enumerate(data.SPLITS):
                for label in range(2):
                    for i in range(2 if split == "train" else 1):
                        values = np.random.default_rng(100 * sindex + 10 * label + i).integers(
                            0, 256, (32, 32, 3), dtype=np.uint8)
                        buffer = io.BytesIO()
                        Image.fromarray(values).save(buffer, format="PNG")
                        archive.writestr(f"d/{v}/{split}/C{label}/{i}.png", buffer.getvalue())
    classes = ["C0", "C1"]
    source, digest = data.source_index(archive_path, classes, data.sha256(archive_path))

    def tiny():
        inside = keras.Input((32, 32, 3))
        body = keras.layers.Conv2D(3, 1)(keras.layers.Rescaling(1 / 255)(inside))
        backbone = keras.Model(inside, body)
        backbone.trainable = False
        x = keras.Input((32, 32, 3))
        pooled = keras.layers.GlobalAveragePooling2D()(backbone(x))
        return keras.Model(x, keras.layers.Dense(2, activation="softmax")(pooled))

    old_model = tmp_path / "old.keras"
    tiny().save(old_model)
    old_sha = data.sha256(old_model)
    spec = {"model_sha256": old_sha, "input_size": 32, "output_adapter": "direct_softmax",
            "test_accuracy": 0.123, "validation_macro_f1": 0.123}
    monkeypatch.setattr(data, "candidate", lambda domain: (spec, old_model, classes))
    monkeypatch.setattr(resolution, "fresh_model", lambda *args: tiny())
    train, test = source["original", "train"][0], source["original", "test"][0]
    plan = resolution.quarantine_plan(
        "skin", [{"paths": [train.path, test.path]}],
        {r.path: {"split": r.split, "label": r.label} for rows in source.values() for r in rows},
        classes, digest, {f"{v}/{s}": len(rows) for (v, s), rows in source.items()},
    )
    config = {
        "schema": "mediflow_domain_adaptation_v1", "seed": 42, "epochs": 1, "batch_size": 2,
        "learning_rate": 1e-4, "augmentation_probability": 0.5, "train_variant": "original",
        "baseline_head_epochs": 1, "baseline_partial_epochs": 1,
        "validation_clean_macro_f1_tolerance": 0.01, "validation_clean_recall_tolerance": 0.05,
        "validation_min_shift_gain": 0.005,
        "tasks": {"skin": {"source_sha256": digest, "loss": "ce"}},
    }
    config_path = tmp_path / "config.json"
    runner.write_json(config_path, config)
    profile = {
        "reference_model_sha256": old_sha, "calibration_fingerprint": "fixture",
        "input_policy": "legacy_resize", "train_visual_review": "approved",
        "brightness_offset": [-2, 2], "contrast": [0.9, 0.95], "left_light": [0, 0],
        "photometric_operator": "bounded_additive_v3", "scale_x": [1, 1], "scale_y": [1, 1],
    }
    review = tmp_path / "review"
    review.mkdir()
    runner.write_json(review / "quarantine_plan.json", plan)
    runner.write_json(review / "profile_v4.json", profile)
    (review / "evidence.txt").write_text("simulated Train review", encoding="utf-8")
    runner.write_json(review / "readiness.json", {
        "domain": "skin", "status": "ready_for_fresh_baseline_pilot",
        "profile_sha256": object_hash(profile), "quarantine_plan_sha256": object_hash(plan),
        "config_sha256": object_hash(config),
        "evidence_sha256": {"evidence.txt": data.sha256(review / "evidence.txt")},
    })
    target = [data.ImageRow(f"P{i}/x.png", "calibration", person=f"P{i}") for i in range(4)]
    packet = tmp_path / "packet.json"
    runner.write_json(packet, {
        "domain": "skin", "role": "calibration", "model_sha256": old_sha,
        "people": [r.person for r in target], "rows": [asdict(r) for r in target],
        "fingerprint": "fixture", "pixel_hashes": [],
        "audit": {"blocking_leaks_or_conflicts": [], "near_duplicate_review": []},
    })
    args = SimpleNamespace(
        domain="skin", config=config_path, review=review, source_zip=archive_path,
        calibration_file=packet, output=tmp_path / "run",
    )
    resolution.run(args)
    contract = data.read_json(args.output / "run_contract.json")
    selection = data.read_json(args.output / "validation_selection.json")
    assert contract["model_sha256"] != old_sha and data.sha256(old_model) == old_sha
    assert "test_accuracy" not in contract["spec"]
    assert "validation_macro_f1" not in contract["spec"]
    assert {r["variant"] for r in selection["source_train_rows"]} == {"original"}
    assert train.path not in {r["path"] for r in selection["source_train_rows"]}
    assert len(selection["source_train_rows"]) == 3
    final = data.read_json(args.output / "final_evaluation/comparison.json")
    assert final["selected_before_test"] == selection["selected"]
    assert len(data.read_json(args.output / "final_evaluation/source_test_paths.json")) == 2
    assert final["results"]["baseline"]["target"]["accuracy"] is None
    moved = tmp_path / "relocated"
    shutil.copytree(args.output, moved)
    assert data.sha256(runner.baseline_checkpoint(moved, contract)) == contract["model_sha256"]
    (args.output / "source_subset.json").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="subset changed"):
        runner.finalize(SimpleNamespace(run=args.output, domain="skin"))
