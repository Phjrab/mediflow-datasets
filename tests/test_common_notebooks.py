"""Offline integration of audit gates, 5/10-class training, resume and packaging."""

import ast
import json
import shutil
import zipfile
from pathlib import Path

import keras
import numpy as np
import pytest
from PIL import Image

from mediflow_datasets import common_engine as engine
from mediflow_datasets import common_workflow as flow
from mediflow_datasets.models import MODEL_VARIANTS

ROOT = Path(__file__).resolve().parents[1]


def make_data(root, classes):
    for split_index, split in enumerate(("train", "val", "test")):
        for class_index, name in enumerate(classes):
            for kind in ("original", "augmented"):
                directory = root / kind / split / name
                directory.mkdir(parents=True)
                pixels = np.full((8, 8, 3), 20 + split_index * 50 + class_index, dtype="uint8")
                Image.fromarray(pixels).save(directory / "sample.png")
    return root


def tiny_model(spec):
    inputs = keras.Input((None, None, 3))
    x = keras.layers.Rescaling(1 / 255)(inputs)
    x = keras.layers.Conv2D(2, 1)(x)
    x = keras.layers.BatchNormalization()(x)
    base = keras.Model(inputs, x, name="efficientnet_tiny")
    base.trainable = False
    outer = keras.Input((None, None, 3))
    x = keras.layers.GlobalAveragePooling2D()(base(outer, training=False))
    outputs = keras.layers.Dense(spec["class_count"], activation="softmax")(x)
    model = keras.Model(outer, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=engine.loss_function(spec["loss"]),
        metrics=["accuracy"],
    )
    return model


@pytest.mark.parametrize("mode,count", [("comparison", 5), ("suite", 10)])
def test_end_to_end_modes_resume_and_final_package(tmp_path, monkeypatch, mode, count):
    matplotlib = pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("IPython")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(engine, "build_model", tiny_model)
    monkeypatch.setattr(plt, "show", lambda: None)
    project = tmp_path / "project"
    (project / "datasets").mkdir(parents=True)
    classes = [f"class_{i}" for i in range(count)]
    data = make_data(tmp_path / "source", classes)
    archive = shutil.make_archive(str(project / "datasets/hair_datasets"), "zip", data)
    sources = {
        name: (ROOT / f"src/mediflow_datasets/{name}.py").read_text(encoding="utf-8")
        for name in ("common_engine", "common_audit", "common_workflow")
    }
    config = dict(
        domain="hair",
        mode="audit",
        project_root=str(project),
        data_zip=archive,
        audit_dir="",
        resume_dir="",
        seed=42,
        batch_size=count,
        epochs1=1,
        epochs2=1,
        extension_epochs=1,
        train_variant="augmented",
    )
    context = flow.prepare(config, {"hair": classes}, sources, "test", tmp_path / "local")
    summary = flow.audit_run(context)
    assert summary["status"] == "mechanical_checks_passed_with_limitations"
    config.update(mode=mode, audit_dir=str(context["output"]))
    training = flow.prepare(config, {"hair": classes}, sources, "test", tmp_path / "local")
    # Loading for validation is exactly the same across Original/Augmented arms.
    load_a, load_b = flow.factory(training, "original"), flow.factory(training, "augmented")
    ds, paths_a = load_a("val", 224, False)
    _, paths_b = load_b("val", 224, False)
    assert paths_a == paths_b
    images, labels = next(iter(ds))
    assert images.numpy().max() > 1
    assert labels.shape[-1] == count
    records = flow.run(training)
    assert len(records) == (2 if mode == "comparison" else 6)
    assert not (training["output"] / "final_test_metrics.json").exists()
    predictions = pd.read_csv(
        training["output"] / records[0]["id"] / records[0]["attempt"] / "validation_predictions.csv"
    )
    assert len([c for c in predictions if c.startswith("prob_C")]) == count
    if mode == "comparison":
        assert all(len(r["history"]["accuracy"]) == 1 for r in records)
    else:
        assert records[-1]["optimizer_restored"]
        assert len(records[-1]["history"]["accuracy"]) == 3
    config["resume_dir"] = str(training["output"])
    resumed = flow.prepare(config, {"hair": classes}, sources, "test", tmp_path / "local")
    monkeypatch.setattr(engine, "build_model", lambda _: pytest.fail("Completed trial reran"))
    assert flow.run(resumed) == records
    card = flow.finish(resumed, records)
    assert card["class_names"] == classes
    package = next((project / "2_results/hair/selected_models").glob("*.zip"))
    with zipfile.ZipFile(package) as zipped:
        assert zipped.testzip() is None
        assert any(p.endswith("/model.keras") for p in zipped.namelist())
    # Once evaluated, finishing again must reuse test output without predictions.
    monkeypatch.setattr(engine, "evaluate_to_files", lambda *a: pytest.fail("Test evaluated twice"))
    assert flow.finish(resumed, records) == card
    config["seed"] += 1
    with pytest.raises(ValueError, match="코드/설정"):
        flow.prepare(config, {"hair": classes}, sources, "test", tmp_path / "local")


def test_audit_detects_cross_split_and_label_conflicts(tmp_path):
    pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("IPython")
    from mediflow_datasets.common_audit import audit_dataset

    data = make_data(tmp_path / "data", ["a", "b"])
    shutil.copyfile(data / "original/train/a/sample.png", data / "augmented/test/b/sample.png")
    report = tmp_path / "report"
    report.mkdir()
    summary = audit_dataset(data, report, "hair", ["a", "b"], "hash")
    assert summary["status"] == "issues_found"
    issues = pd.read_csv(report / "audit_issues.csv")
    assert {"cross_split_duplicates", "label_conflicts", "evaluation_mismatch"} <= set(issues.type)


def test_training_accepts_verified_zip_hash_without_audit_folder(tmp_path):
    project = tmp_path / "project"
    (project / "datasets").mkdir(parents=True)
    classes = [f"class_{i}" for i in range(5)]
    data = make_data(tmp_path / "source", classes)
    archive = Path(shutil.make_archive(str(project / "datasets/skin_datasets"), "zip", data))
    sources = {
        name: (ROOT / f"src/mediflow_datasets/{name}.py").read_text(encoding="utf-8")
        for name in ("common_engine", "common_audit", "common_workflow")
    }
    config = dict(
        domain="skin",
        mode="comparison",
        project_root=str(project),
        data_zip=str(archive),
        audit_dir="",
        expected_data_sha256=engine.file_hash(archive),
        resume_dir="",
        seed=42,
        batch_size=5,
        epochs1=1,
        epochs2=1,
        extension_epochs=1,
        train_variant="augmented",
    )
    context = flow.prepare(
        config,
        {"skin": classes},
        sources,
        "test",
        tmp_path / "local",
    )
    assert context["audit"]["protocol"] == "expected_sha256_v1"
    assert context["output"].parent == project / "2_results" / "skin"
    assert (context["output"] / "audit_summary.json").is_file()

    config["expected_data_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="확인된 SHA-256"):
        flow.prepare(
            config,
            {"skin": classes},
            sources,
            "test",
            tmp_path / "local-mismatch",
        )


@pytest.mark.parametrize("name", ["../escape.png", "/absolute.png", "a\\b.png", "C:/x.png"])
def test_zip_unsafe_paths_rejected_before_extraction(tmp_path, name):
    source = tmp_path / "bad.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(name, b"bad")
    if "\\" in name:
        # Windows ZipInfo normalizes separators when writing; emulate an external ZIP.
        source.write_bytes(source.read_bytes().replace(b"a/b.png", b"a\\b.png"))
    with pytest.raises(ValueError):
        flow.extract_zip(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_notebooks_are_standalone_and_class_order_matches_original_reports():
    notebooks = sorted((ROOT / "notebooks").glob("0*_common_*.ipynb"))
    assert len(notebooks) == 3
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell["outputs"] == []
            source = "".join(cell["source"])
            compile(
                "\n".join(s for s in source.splitlines() if not s.startswith("%pip ")),
                path.name,
                "exec",
            )
            if "SOURCES = " not in source:
                continue
            nodes = ast.parse(source).body
            assignments = {
                n.targets[0].id: n.value
                for n in nodes
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            }
            sources = ast.literal_eval(assignments["SOURCES"])
            for name, code in sources.items():
                assert code == (ROOT / f"src/mediflow_datasets/{name}.py").read_text(
                    encoding="utf-8"
                )
            profiles = ast.literal_eval(assignments["PROFILES"])
            for domain, classes in profiles.items():
                assert classes == MODEL_VARIANTS[domain]["original"].class_names()


@pytest.mark.parametrize("count,epochs2", [(5, 0), (10, 1)])
def test_common_engine_real_training_dynamic_classes_and_resume(
    tmp_path, monkeypatch, count, epochs2
):
    import csv

    import tensorflow as tf

    monkeypatch.setattr(engine, "build_model", tiny_model)
    calls = []

    def datasets(split, size, shuffle):
        assert split in ("train", "val"), "Test accessed during selection"
        calls.append(split)
        images = np.full((count, 8, 8, 3), 100, dtype="float32")
        labels = np.eye(count, dtype="float32")
        return tf.data.Dataset.from_tensor_slices((images, labels)).batch(count), [
            f"{split}/{i}.png" for i in range(count)
        ]

    spec = dict(id="trial", backbone="B1", size=8, loss="ls005", class_count=count)
    record = engine.run_trial(spec, datasets, tmp_path, "test", epochs1=1, epochs2=epochs2)
    assert record["validation"]["count"] == count
    assert len(record["validation"]["confusion_matrix"]) == count
    assert len(record["history"]["accuracy"]) == 1 + epochs2
    predictions = tmp_path / record["id"] / record["attempt"] / "validation_predictions.csv"
    with predictions.open(encoding="utf-8-sig") as f:
        assert len(next(csv.reader(f))) == 3 + count
    previous_calls = len(calls)
    assert engine.run_trial(spec, datasets, tmp_path, "test") == record
    assert len(calls) == previous_calls
    if epochs2:
        extended = engine.extend_b1(record, datasets, tmp_path, "test", epochs=1)
        assert extended["optimizer_restored"]
        restored = keras.models.load_model(
            tmp_path / extended["id"] / extended["attempt"] / "extension_last.keras"
        )
        assert int(restored.optimizer.iterations.numpy()) == 2
        assert restored.output_shape == (None, count)
    else:
        assert record["selected_stage"] == "stage1"
        assert not (tmp_path / record["id"] / record["attempt"] / "stage2_best.keras").exists()
    engine.selected_model_path(tmp_path, record).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="corrupted"):
        engine.cached_record(tmp_path, "trial", "test")


def test_actual_efficientnet_supports_ten_classes_without_external_scaling(monkeypatch):
    builder = keras.applications.EfficientNetB0

    def offline(**kwargs):
        assert kwargs["weights"] == "imagenet"
        return builder(**{**kwargs, "weights": None})

    monkeypatch.setattr(keras.applications, "EfficientNetB0", offline)
    model = engine.build_model(dict(backbone="B0", size=224, class_count=10, loss="ce"))
    assert model.output_shape == (None, 10)
    base = next(layer for layer in model.layers if isinstance(layer, keras.Model))
    rescale = next(layer for layer in base.layers if isinstance(layer, keras.layers.Rescaling))
    assert float(rescale.scale) == 1 / 255
    scores = model(np.full((1, 224, 224, 3), 127.5, dtype="float32"), training=False).numpy()
    assert scores.shape == (1, 10)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores.sum(1), [1], atol=1e-5)
    keras.backend.clear_session()


def test_audit_gate_rejects_changed_domain_hash_or_issue_status(tmp_path):
    summary = dict(
        domain="hair",
        class_names=["a"],
        data_sha256="abc",
        protocol="common_audit_v1",
        status="mechanical_checks_passed_with_limitations",
    )
    engine.write_json(tmp_path / "audit_summary.json", summary)
    for name in ("image_inventory.csv", "audit_issues.csv"):
        (tmp_path / name).write_text("header\n", encoding="utf-8")
    engine.write_json(
        tmp_path / "audit_manifest.json",
        {
            name: engine.file_hash(tmp_path / name)
            for name in ("audit_summary.json", "image_inventory.csv", "audit_issues.csv")
        },
    )
    assert flow.verify_audit(tmp_path, "hair", ["a"], "abc") == summary
    for domain, classes, digest in [
        ("skin", ["a"], "abc"),
        ("hair", ["b"], "abc"),
        ("hair", ["a"], "changed"),
    ]:
        with pytest.raises(ValueError):
            flow.verify_audit(tmp_path, domain, classes, digest)
    (tmp_path / "image_inventory.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="변경"):
        flow.verify_audit(tmp_path, "hair", ["a"], "abc")
