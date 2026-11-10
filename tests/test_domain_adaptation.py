from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_data as data


def image_bytes(seed):
    pixels = np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def source_zip(path):
    with zipfile.ZipFile(path, "x") as archive:
        for variant in ("original", "augmented"):
            for split_index, split in enumerate(("train", "val", "test")):
                for label in (0, 1):
                    archive.writestr(
                        f"dataset/{variant}/{split}/C{label}/image.png",
                        image_bytes(10 * split_index + label),
                    )
    return data.sha256(path)


def test_source_pin_class_order_and_split_audit(tmp_path):
    archive = tmp_path / "dataset.zip"
    digest = source_zip(archive)
    source, _ = data.source_index(archive, ["C1", "C0"], digest)
    assert source["original", "train"][0].label == 1
    report, pixels = data.audit_source(archive, source)
    assert len(pixels) == 6
    assert not report["blocking_leaks_or_conflicts"]
    assert all(report["original_augmented_consistency"].values())
    assert report["identity_metadata"]["origin"]["complete"]
    assert not report["identity_metadata"]["person"]["complete"]
    with pytest.raises(ValueError, match="SHA-256"):
        data.source_index(archive, ["C1", "C0"], "0" * 64)


def test_leakage_finds_derivatives_and_unreviewed_near_duplicates():
    train = data.ImageRow("train/a.png", "train", 0, "P1", "S1", origin="a")
    derived = data.ImageRow("test/b.png", "test", 0, "P1", "S2", origin="a")
    entries = [(train, "a", "pixels_a", 0), (derived, "b", "pixels_b", 1)]
    report = data.audit_groups(entries)
    assert {"person", "origin"} <= {
        issue["kind"] for issue in report["blocking_leaks_or_conflicts"]
    }
    assert report["near_duplicate_review"]
    reviewed = data.audit_groups(entries, near_reviews=[(train.path, derived.path)])
    assert not reviewed["near_duplicate_review"]
    assert reviewed["blocking_leaks_or_conflicts"]


def test_calibration_rejects_test_and_preserves_unknown_color():
    train = [{"split": "train", "metrics": {"y_p50": 180.0, "contrast": 20.0}}]
    target = [
        {
            "split": "calibration",
            "person": f"P{p}",
            "condition": condition,
            "metrics": {
                "y_p50": 90.0 + (20 if condition == "alt" else 0),
                "contrast": 20.0 + (10 if condition == "alt" else 0),
            },
        }
        for p in range(4)
        for condition in ("ref", "alt")
    ]
    profile = analysis.calibrate("web_skin", train, target)
    assert profile["brightness_offset"] == [-32.0, -32.0]
    assert profile["contrast"] == [1.15, 1.15]
    assert "lab_a" not in profile
    with pytest.raises(ValueError, match="Train"):
        analysis.calibrate("web_skin", [{**train[0], "split": "test"}], target)


def test_unlabeled_accuracy_and_feature_metrics_are_separate():
    from mediflow_datasets.domain_adaptation_experiment import (
        classification,
        feature_distance,
        feature_reference,
    )

    source = np.random.default_rng(4).normal(size=(10, 5))
    ref = feature_reference(source)
    same = feature_distance(source, source, ref)
    shifted = feature_distance(source, source + 1, ref)
    assert same["mmd2_biased_rbf"] < 1e-6 and same["coral_covariance"] < 1e-6
    assert shifted["mmd2_biased_rbf"] > same["mmd2_biased_rbf"]
    record = classification([data.ImageRow("a", "calibration")], np.array([[0.9, 0.1]]), 2)
    assert record["accuracy"] is None and record["macro_f1"] is None


def test_training_uses_validation_without_reading_test(tmp_path, monkeypatch):
    keras = pytest.importorskip("keras")
    from mediflow_datasets.domain_adaptation_experiment import training_run

    archive = tmp_path / "source.zip"
    source, _ = data.source_index(archive, ["C0", "C1"], source_zip(archive))
    inner_input = keras.Input((32, 32, 3))
    inner_output = keras.layers.Conv2D(3, 1)(keras.layers.Rescaling(1 / 255)(inner_input))
    backbone = keras.Model(inner_input, inner_output, name="tiny_backbone")
    outer = keras.Input((32, 32, 3))
    pooled = keras.layers.GlobalAveragePooling2D()(backbone(outer))
    model = keras.Model(outer, keras.layers.Dense(2, activation="softmax")(pooled))
    checkpoint = tmp_path / "baseline.keras"
    model.save(checkpoint)
    output = tmp_path / "run"
    output.mkdir()
    config = {
        "seed": 42,
        "batch_size": 2,
        "epochs": 1,
        "learning_rate": 1e-5,
        "augmentation_probability": 1.0,
        "validation_clean_macro_f1_tolerance": 0.01,
        "validation_clean_recall_tolerance": 0.05,
        "validation_min_shift_gain": 0.005,
    }
    profile = {
        "brightness_offset": [-10, -5],
        "contrast": [1, 1],
        "scale_x": [1, 1],
        "scale_y": [1, 1],
        "left_light": [0, 0],
    }
    reads = []
    actual_read = zipfile.ZipFile.read

    def observe_read(archive, name, *args, **kwargs):
        reads.append(name)
        return actual_read(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", observe_read)
    result = training_run(
        {
            "source_zip": archive,
            "source": source,
            "output": output,
            "config": config,
            "spec": {"input_size": 32, "output_adapter": "direct_softmax", "loss": "ce"},
            "profile": profile,
            "classes": ["C0", "C1"],
            "model": checkpoint,
            "da_reason": "test",
        }
    )
    assert result["source_test_evaluated"] is False
    assert all("/test/" not in name for name in reads)
    assert result["target_accuracy"] is None
    assert set(result["arms"]) == {"source_only", "brightness_only"}
    assert result["equivalent_arms"] == {"target_guided": "brightness_only"}
    assert all(Path(record["model"]).is_file() for record in result["arms"].values())
    assert json.loads((output / "validation_selection.json").read_text(encoding="utf-8"))[
        "selected"
    ]
    # The final entry point opens Test only after the saved validation selection is frozen.
    from argparse import Namespace

    from mediflow_datasets import domain_adaptation as runner

    spec = {"input_size": 32, "output_adapter": "direct_softmax"}
    monkeypatch.setattr(data, "candidate", lambda domain: (spec, checkpoint, ["C0", "C1"]))
    runner.write_json(output / "frozen_profile.json", profile)
    runner.write_json(
        output / "run_contract.json",
        {
            "domain": "web_skin",
            "classes": ["C0", "C1"],
            "source_sha256": data.sha256(archive),
            "baseline_model": str(checkpoint),
            "model_sha256": data.sha256(checkpoint),
            "profile_sha256": data.sha256(output / "frozen_profile.json"),
            "target_calibration_fingerprint": "fixture",
            "spec": spec,
            "config": config,
        },
    )
    runner.write_json(
        output / "frozen_selection.json",
        {
            "selection_sha256": data.sha256(output / "validation_selection.json"),
            "contract_sha256": data.sha256(output / "run_contract.json"),
        },
    )
    packet = tmp_path / "calibration.json"
    runner.write_json(
        packet,
        {
            "domain": "web_skin",
            "role": "calibration",
            "people": ["P1", "P2", "P3", "P4"],
            "model_sha256": data.sha256(checkpoint),
            "rows": [
                {"path": f"{person}/a.png", "split": "calibration", "person": person}
                for person in ("P1", "P2", "P3", "P4")
            ],
            "fingerprint": "fixture",
            "audit": {"blocking_leaks_or_conflicts": [], "near_duplicate_review": []},
        },
    )
    args = Namespace(
        run=output,
        domain="web_skin",
        source_zip=archive,
        calibration_file=packet,
        target=tmp_path,
        target_manifest=None,
    )
    reads.clear()
    runner.finalize(args)
    assert any("/test/" in name for name in reads)
    final = data.read_json(output / "final_evaluation/comparison.json")
    assert final["target_test_count"] == 0
    assert final["results"]["baseline"]["original_source_test"]["count"] == 2
    assert final["results"]["baseline"]["target"]["accuracy"] is None
    with pytest.raises(FileExistsError):
        runner.finalize(args)
