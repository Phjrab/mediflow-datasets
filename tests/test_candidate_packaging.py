"""Packaging is tested with random weights; it does not assert candidate performance."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import keras
import pytest

from mediflow_datasets import candidate_packaging as packaging
from mediflow_datasets import experiment_suite

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "results/web_skin/candidates/public_candidate_v1_b0_256_ce_20260908_081603_3f1ce76e"
)


def notebook_policy():
    return json.loads((PACKAGE / "packaging_policy.json").read_text("utf-8"))


def test_packaged_source_and_reviewed_report_fingerprints():
    packaged_source = (PACKAGE / "packaging_source.py").read_text("utf-8")
    compile(packaged_source, "packaging_source.py", "exec")
    assert packaged_source == Path(packaging.__file__).read_text("utf-8")
    source = ROOT / "results/web_skin/experiments/suite_20260908_014452_72768a42"
    policy = notebook_policy()
    for relative, digest in policy["report_hashes"].items():
        data = (source / relative).read_bytes()
        checksums = {hashlib.sha256(data).hexdigest()}
        if relative.endswith((".csv", ".json", ".py")):
            normalized = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            checksums.add(hashlib.sha256(normalized).hexdigest())
        assert digest in checksums, relative


def test_package_model_and_reports_round_trip_with_random_weights(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = ROOT / "results/web_skin/experiments/suite_20260908_014452_72768a42"
    policy = notebook_policy()
    for relative in policy["report_hashes"]:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / relative, target)
    # Use the exact real topology with offline random weights for IO/forward verification.
    keras.backend.clear_session()
    base = keras.applications.EfficientNetB0(
        weights=None, include_top=False, input_shape=(256, 256, 3)
    )
    inputs = keras.Input((256, 256, 3))
    features = keras.layers.GlobalAveragePooling2D()(base(inputs, training=False))
    outputs = keras.layers.Dense(5, activation="softmax")(keras.layers.Dropout(0.3)(features))
    model = keras.Model(inputs, outputs)
    target_model = source / policy["model_relative_path"]
    model.save(target_model)
    policy["model_sha256"] = packaging.sha256_file(target_model)
    for name in ("model_card.json", "selection_before_test.json"):
        path = source / name
        value = packaging.read_json(path)
        value["model_sha256"] = policy["model_sha256"]
        path.write_text(json.dumps(value, ensure_ascii=False), "utf-8")
    policy["report_hashes"] = {
        relative: packaging.sha256_file(source / relative) for relative in policy["report_hashes"]
    }
    package, archive = packaging.build_package(source, tmp_path / "out", policy)
    manifest = packaging.read_json(package / "package_manifest.json")
    assert not manifest["new_training_or_test_evaluation"]
    assert manifest["normal_class_included"]
    assert packaging.sha256_file(package / "web_skin_model.keras") == policy["model_sha256"]
    for relative, digest in manifest["artifact_hashes"].items():
        assert packaging.sha256_file(package / relative) == digest
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.endswith(".keras")]) == 1
    drive = tmp_path / "drive"
    copied, zipped, digest = packaging.publish_package(package, archive, drive)
    assert copied.exists() and digest == packaging.sha256_file(zipped)
    with pytest.raises(FileExistsError):
        packaging.publish_package(package, archive, drive)
    (source / "selection_before_test.json").write_text("changed", "utf-8")
    with pytest.raises(ValueError, match="report changed"):
        packaging.verify_source(source, policy)
    # Restore the report fingerprint for a separate wrong-model rejection check.
    shutil.copyfile(
        package / "source_reports/selection_before_test.json", source / "selection_before_test.json"
    )
    target_model.write_bytes(b"wrong model")
    with pytest.raises(ValueError, match="model hash"):
        packaging.verify_source(source, policy)
    keras.backend.clear_session()


def test_reject_wrong_model_shape_and_escaping_path(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        packaging.within(tmp_path, "../outside")
    model = keras.Sequential(
        [keras.Input((8, 8, 3)), keras.layers.Flatten(), keras.layers.Dense(5)]
    )
    path = tmp_path / "wrong.keras"
    model.save(path)
    with pytest.raises(ValueError, match="shape"):
        packaging.check_model_contract(path)
    assert packaging.CLASSES == experiment_suite.CLASSES
