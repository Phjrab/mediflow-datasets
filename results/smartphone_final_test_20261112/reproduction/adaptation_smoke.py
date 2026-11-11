"""Check actual saved models with one disposable batch; never write research checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from PIL import Image

from mediflow_datasets import adaptation_stage1 as train
from mediflow_datasets import adaptation_strength as runner
from mediflow_datasets import camera_extended as camera
from mediflow_datasets import domain_adaptation_data as data


def layers_recursive(model):
    seen = set()
    stack = list(model.layers)
    while stack:
        layer = stack.pop()
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        yield layer
        stack.extend(getattr(layer, "layers", ()))


def check_domain(root, runtime, study_path, domain, scratch, output):
    study = data.read_json(study_path)
    frozen = runner.frozen_inputs(
        runtime,
        root / "results/camera_adaptation_stage1_20261110",
        root / "results/camera_adaptation_stage2_20261110",
        study,
        domain,
    )
    task, spec = frozen.task, frozen.spec
    manifest = data.read_json(frozen.manifest)
    retained = {r["path"]: r["label"] for r in manifest["original"]["train"]}
    old = data.read_json(
        root / "presentation/report_assets/camera_similarity_check_20261110/summary.json"
    )
    sources = old["domains"][domain]["source_png_provenance"]
    grouped = {label: [] for label in range(len(task["classes"]))}
    for row in sources:
        grouped[retained[row["path"]]].append(row)
    batch = task["batch_size"]
    chosen = [grouped[index % len(grouped)][index // len(grouped)] for index in range(batch)]
    images = []
    labels = []
    applied = 0
    for row in chosen:
        path = root / row["preview"]
        if data.sha256(path) != row["sha256"]:
            raise ValueError("Smoke input preview changed")
        with Image.open(path) as photo:
            image = np.asarray(photo.convert("RGB"), dtype=np.float32)
        label = retained[row["path"]]
        seed = train.exp.item_seed(data.ImageRow(row["path"], "train", label), 42, 0)
        if train.exp.augmentation_applies(seed, frozen.config["augmentation_probability"]):
            image = camera.transform(image, frozen.profile, "expanded", seed)
            applied += 1
        images.append(image)
        labels.append(label)
    images = np.stack(images)
    if images.shape != (batch, spec["size"], spec["size"], 3) or not np.isfinite(images).all():
        raise ValueError("Actual model input contract failed")
    if images.min() < 0 or images.max() > 255 or images.max() <= 1:
        raise ValueError("Expected unnormalized float32 RGB 0-255 inputs")
    keras.utils.set_random_seed(42)
    model = keras.models.load_model(frozen.initial, compile=False)
    print(f"[{domain}] 실제 저장 모델 로드 · {batch}장 · {spec['size']}px", flush=True)
    train.exp.configure_partial(model)
    layers = list(layers_recursive(model))
    assert any(
        isinstance(layer, keras.layers.Rescaling) and np.allclose(layer.scale, 1 / 255)
        for layer in layers
    )
    assert not any(
        isinstance(layer, keras.layers.BatchNormalization) and layer.trainable for layer in layers
    )
    optimizer = keras.optimizers.Adam(frozen.config["learning_rate"])
    step, predict = train.steps(model, spec, optimizer, len(task["classes"]))
    scores = predict(images).numpy()
    assert scores.shape == (batch, len(task["classes"]))
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores.sum(axis=1), 1, rtol=1e-5, atol=1e-6)
    weights = [v.numpy().copy() for v in model.trainable_variables]
    tensor_images, tensor_labels = next(
        iter(train.pipeline(images, list(range(batch)), labels, batch, spec["size"]))
    )
    print(f"[{domain}] 임시 모델 1배치 갱신 확인", flush=True)
    loss = float(step(tensor_images, tensor_labels))
    assert np.isfinite(loss)
    expected = 4 if spec["architecture"] == "pmg" else 1
    assert int(optimizer.iterations) == expected
    assert any(
        not np.array_equal(a, v.numpy())
        for a, v in zip(weights, model.trainable_variables, strict=True)
    )
    states = [v.numpy().copy() for v in model.trainable_variables]
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    snapshots = scratch / "snapshots"
    snapshots.mkdir(parents=True)
    record = train.save_epoch(checkpoint, snapshots, scratch / "checkpoint", {"epoch": 1})
    step(tensor_images, tensor_labels)
    train.restore_epoch(checkpoint, snapshots, record)
    assert int(optimizer.iterations) == expected
    for saved, variable in zip(states, model.trainable_variables, strict=True):
        np.testing.assert_array_equal(saved, variable.numpy())
    scores = predict(images).numpy()
    exported = train.export_model(model, scratch, "disposable_smoke", scratch / "export")
    loaded = keras.models.load_model(scratch / exported["file"], compile=False)
    reloaded = train.predictor(loaded, spec)(images).numpy()
    np.testing.assert_allclose(scores, reloaded, rtol=1e-5, atol=1e-6)
    assert data.sha256(frozen.initial) == spec["model_sha256"]
    report = {
        "domain": domain,
        "study_sha256": data.sha256(study_path),
        "smoke_code_sha256": data.sha256(Path(__file__)),
        "status": "passed",
        "full_training_executed": False,
        "disposable_batch_size": batch,
        "optimizer_updates_per_batch": expected,
        "additional_updates_before_restore": expected,
        "input_size": spec["size"],
        "classes": task["classes"],
        "model_sha256": spec["model_sha256"],
        "transformed_images_in_batch": applied,
        "input_range": [float(images.min()), float(images.max())],
        "checkpoint_restored": True,
        "keras_save_reload_verified": True,
        "original_model_unchanged": True,
        "source_test_used": False,
        "target_accuracy": None,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "gpu": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"[{domain}] 임시 배치·체크포인트 복원·저장/재로드 통과", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "runtime", "study", "scratch", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--domain", choices=train.DOMAINS, required=True)
    args = parser.parse_args()
    check_domain(args.root, args.runtime, args.study, args.domain, args.scratch, args.output)


if __name__ == "__main__":
    main()
