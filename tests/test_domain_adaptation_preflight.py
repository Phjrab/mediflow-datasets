from __future__ import annotations

import io
import json
import zipfile

import cv2
import numpy as np
import pytest
from PIL import Image

from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_preflight as preflight


def image(seed):
    raw = np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    out = io.BytesIO()
    Image.fromarray(raw).save(out, format="PNG")
    return raw, out.getvalue()


def test_near_review_collapses_provenance_copies_without_auto_clearance(tmp_path):
    archive = tmp_path / "source.zip"
    a, body_a = image(1)
    b, body_b = image(2)
    rows = [
        data.ImageRow("original/train/a.png", "train", 0, origin="original/train/a.png"),
        data.ImageRow("augmented/train/a.png", "train", 0, origin="original/train/a.png"),
        data.ImageRow("original/test/b.png", "test", 1, origin="original/test/b.png"),
        data.ImageRow("augmented/test/b.png", "test", 1, origin="original/test/b.png"),
    ]
    with zipfile.ZipFile(archive, "x") as z:
        for i, row in enumerate(rows):
            z.writestr(row.path, body_a if i < 2 else body_b)
    before = data.sha256(archive)
    pairs = [[a.path, b.path] for a in rows[:2] for b in rows[2:]]
    result = preflight.review_near_pairs(
        archive, {"all": rows}, {"near_duplicate_review": pairs}, tmp_path / "review"
    )
    assert result == {"raw_pairs": 4, "groups": 1, "auto_cleared": 0}
    output = json.loads((tmp_path / "review/candidate_groups.json").read_text(encoding="utf-8"))
    assert len(output[0]["raw_pairs"]) == 4 and output[0]["decision"] == "pending"
    assert data.sha256(archive) == before


def test_similarity_matches_tensorflow_population_ssim():
    tf = pytest.importorskip("tensorflow")
    a, _ = image(3)
    b = np.clip(a.astype(float) * 0.9 + 8, 0, 255).astype(np.float32)
    x, y = [cv2.cvtColor(v.astype(np.float32), cv2.COLOR_RGB2GRAY)[:, :, None] for v in [a, b]]
    expected = float(tf.image.ssim(x, y, max_val=255))
    assert preflight.ssim_gray(a, b) == pytest.approx(expected, abs=1e-5)
    assert preflight.ssim_gray(a, a) == pytest.approx(1)


def test_transform_quality_check_reads_train_only(tmp_path, monkeypatch):
    archive = tmp_path / "source.zip"
    raw, body = image(7)
    row = data.ImageRow("original/train/a.png", "train", 0)
    test = data.ImageRow("original/test/a.png", "test", 0)
    with zipfile.ZipFile(archive, "x") as z:
        z.writestr(row.path, body)
        z.writestr(test.path, body)
    seen = []
    original = zipfile.ZipFile.read

    def track(z, name, *args, **kwargs):
        seen.append(name)
        return original(z, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", track)
    profile = {
        "domain": "skin",
        "brightness_offset": [-5, 5],
        "contrast": [1, 1],
        "scale_x": [1, 1],
        "scale_y": [1, 1],
        "left_light": [0, 0],
    }
    measured = [{"path": row.path, "metrics": {}}]
    spec = {"input_size": 32, "input_policy": "legacy_resize"}
    summary = preflight.quality_check(
        archive, [row], measured, [], profile, spec, tmp_path / "qa", 1
    )
    assert summary["sample_count"] == 1 and summary["visual_review"] == "pending"
    assert seen == [row.path]
    with pytest.raises(ValueError, match="Train only"):
        preflight.quality_check(archive, [test], measured, [], profile, spec, tmp_path / "bad", 1)


def test_preflight_collects_artifacts_without_training_or_test_evaluation(tmp_path, monkeypatch):
    from argparse import Namespace

    import keras

    from mediflow_datasets import domain_adaptation as runner

    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "x") as z:
        for variant in ("original", "augmented"):
            for n, split in enumerate(("train", "val", "test")):
                for label in range(2):
                    z.writestr(f"data/{variant}/{split}/C{label}/x.png", image(n * 10 + label)[1])
    source, _ = data.source_index(archive, ["C0", "C1"], data.sha256(archive))
    inputs = keras.Input((32, 32, 3))
    pooled = keras.layers.GlobalAveragePooling2D()(keras.layers.Rescaling(1 / 255)(inputs))
    model = keras.Model(inputs, keras.layers.Dense(2, activation="softmax")(pooled))
    checkpoint = tmp_path / "model.keras"
    model.save(checkpoint)
    spec = {
        "input_size": 32,
        "output_adapter": "direct_softmax",
        "model_sha256": data.sha256(checkpoint),
    }
    monkeypatch.setattr(data, "candidate", lambda domain: (spec, checkpoint, ["C0", "C1"]))
    config = {
        "schema": "mediflow_domain_adaptation_v1",
        "epochs": 1,
        "batch_size": 2,
        "seed": 42,
        "learning_rate": 1e-5,
        "augmentation_probability": 0.5,
        "tasks": {"skin": {"source_sha256": data.sha256(archive), "input_policy": "legacy_resize"}},
    }
    config_path = tmp_path / "config.json"
    runner.write_json(config_path, config)
    previous = tmp_path / "previous"
    previous.mkdir()
    runner.write_json(
        previous / "analysis.json",
        {
            "domain": "skin",
            "source_sha256": data.sha256(archive),
            "images": [
                {
                    "kind": "source",
                    "path": row.path,
                    "split": "train",
                    "label": row.label,
                    "metrics": {"width": 32, "height": 32, "y_p50": 128.0, "contrast": 40.0},
                }
                for row in source["original", "train"]
            ],
        },
    )
    runner.write_json(
        previous / "source_audit.json",
        {"near_duplicate_review": [], "identity_metadata": {"person": {"complete": False}}},
    )
    target = [
        data.ImageRow(f"P{p}/{c}.png", "calibration", person=f"P{p}", condition=c)
        for p in range(4)
        for c in ("ref", "alt")
    ]
    from dataclasses import asdict

    packet = {
        "domain": "skin",
        "model_sha256": spec["model_sha256"],
        "role": "calibration",
        "people": [f"P{p}" for p in range(4)],
        "rows": [asdict(r) for r in target],
        "fingerprint": "fixture",
        "scores": [[0.5, 0.5]] * 8,
        "features": [[128, 128, 128]] * 8,
        "measurements": [
            {
                **asdict(row),
                "kind": "target",
                "metrics": {
                    "y_p50": 140.0 + (4 if row.condition == "alt" else 0),
                    "contrast": 40.0,
                },
            }
            for row in target
        ],
    }
    calibration = tmp_path / "calibration.json"
    runner.write_json(calibration, packet)
    reads = []
    original = zipfile.ZipFile.read

    def track(z, name, *args, **kwargs):
        reads.append(name)
        return original(z, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", track)
    args = Namespace(
        config=config_path,
        domain="skin",
        previous=previous,
        source_zip=archive,
        calibration_file=calibration,
        output=tmp_path / "result",
        samples_per_class=1,
    )
    preflight.preflight(args)
    result = data.read_json(args.output / "preflight_status.json")
    assert result["status"] == "data_collected_review_pending"
    assert not result["training_executed"] and not result["source_test_accuracy_evaluated"]
    assert result["target_accuracy"] is None
    assert all("/test/" not in path and "/val/" not in path for path in reads)
    assert (args.output / "feature_diagnostics/feature_diagnostics.json").is_file()
