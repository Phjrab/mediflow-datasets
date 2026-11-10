"""Compare two eligible starting models per domain with the same lighting treatment."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import time
import uuid
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from mediflow_datasets import camera_extended, domain_alignment
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets import domain_adaptation_resolution as resolution
from mediflow_datasets.candidate_reproduction import ROOT
from mediflow_datasets.domain_adaptation_io import write_progress
from mediflow_datasets.domain_adaptation_training import object_hash

DOMAINS = ("skin", "web_skin", "hair")
ARMS = ("basic", "selected")
WORKERS = min(4, os.cpu_count() or 1)


def show(label, step, total, extra=""):
    print(f"{label} | {step}/{total} ({100 * step / total:.1f}%) {extra}", flush=True)


def prepare_source(domain, archive_path, task, reference, out):
    classes = task["classes"]
    print(f"[{domain}] 데이터 ZIP 해시·클래스 확인", flush=True)
    source, sha = data.source_index(archive_path, classes, task["source_sha256"])
    retained = (
        resolution.filtered_source(source, reference["quarantine"], classes, sha)
        if reference.get("quarantine")
        else {k: v for k, v in source.items() if k[0] == "original"}
    )
    # Skin's historical distinction is Original versus Augmented preparation. Use only
    # linked descendants of retained originals. Adaptation itself uses identical originals.
    train = retained["original", "train"]
    originals = {r.path: r for r in train}
    prepare_augmented = domain == "skin" and any(
        spec.get("skin_augmented_preparation") for spec in task.get("models", {}).values()
    )
    descendants = [
        r
        for r in source["augmented", "train"]
        if prepare_augmented
        and r.origin in originals
        and r.label == originals[r.origin].label
        and r.path.rsplit("/", 1)[-1] != r.origin.rsplit("/", 1)[-1]
    ]
    # For the saved Hair model, include its previous augmented source in the leakage scan.
    scanned = (
        [r for rows in source.values() for r in rows]
        if domain == "hair"
        else [r for rows in retained.values() for r in rows] + descendants
    )
    label = f"[{domain}] 중복·분할 확인 (사진)"
    entries = []
    with zipfile.ZipFile(archive_path) as archive, ThreadPoolExecutor(WORKERS) as pool:

        def signature(row):
            return row, *data.image_signatures(archive.read(row.path))

        for start in range(0, len(scanned), 128):
            entries.extend(pool.map(signature, scanned[start : start + 128]))
            done = len(entries)
            if start == 0 or done % 1024 == 0 or done == len(scanned):
                show(label, done, len(scanned))
    report = data.audit_groups(entries)
    dropped = set()
    if descendants:
        involved = {p for pair in report["near_duplicate_review"] for p in pair}
        involved.update(
            p for leak in report["blocking_leaks_or_conflicts"] for p in leak.get("paths", ())
        )
        dropped = {r.path for r in descendants if r.path in involved}
        entries = [entry for entry in entries if entry[0].path not in dropped]
        report = data.audit_groups(entries)
    report["excluded_augmented_descendants"] = sorted(dropped)
    report["source_target_pixel_overlap"] = len(
        {entry[2] for entry in entries} & set(reference["pixel_hashes"])
    )
    report["source_sha256"] = sha
    report["retained_counts"] = {split: len(retained["original", split]) for split in data.SPLITS}
    if domain == "hair":
        consistency = {}
        for split in data.SPLITS:
            original, augmented = (
                Counter(
                    (row.label, pixel)
                    for row, _, pixel, _ in entries
                    if row.split == split and row.variant == variant
                )
                for variant in ("original", "augmented")
            )
            consistency[split] = (
                not (original - augmented) if split == "train" else original == augmented
            )
        report["original_augmented_consistency"] = consistency
    write_progress(out / "data_audit.json", report)
    if (
        report["blocking_leaks_or_conflicts"]
        or report["near_duplicate_review"]
        or report["source_target_pixel_overlap"]
        or not all(report.get("original_augmented_consistency", {}).values())
    ):
        raise ValueError(f"{domain} 분할 점검 실패. data_audit.json을 확인하세요.")
    pixel_by_path = {row.path: pixel for row, _, pixel, _ in entries}
    paired = {}
    for row in descendants:
        if row.path not in dropped and pixel_by_path[row.path] != pixel_by_path[row.origin]:
            paired.setdefault(row.origin, row)
    if prepare_augmented and {r.label for r in paired.values()} != set(range(len(classes))):
        raise ValueError("피부 선정 방식 재현에 필요한 클래스별 파생본이 확인되지 않았습니다.")
    write_progress(
        out / "source_manifest.json",
        {
            "original": {
                split: [asdict(r) for r in retained["original", split]] for split in data.SPLITS
            },
            "skin_preparation_pairs": {parent: asdict(row) for parent, row in paired.items()},
        },
    )
    return train, retained["original", "val"], paired


def image_cache(folder, archive_path, train, val, pairs, task, size, profile, *, expanded=None):
    """Cache exact float32 model inputs on local SSD once for each input resolution."""
    rows = [*train, *val, *pairs.values()]
    conditions = ("lighting", "expanded") if expanded is not None else ("lighting",)
    size_bytes = (len(rows) + len(conditions) * len(val)) * size * size * 3 * 4
    if shutil.disk_usage(folder).free < size_bytes + 1024**3:
        raise RuntimeError(
            f"{task['domain']} 입력 캐시 공간 부족 (필요 {size_bytes / 1e9:.1f} GB)."
        )
    shape = (size, size, 3)
    cache = np.lib.format.open_memmap(
        folder / f"inputs_{size}.npy", mode="w+", dtype=np.float32, shape=(len(rows), *shape)
    )
    shifted = {
        name: np.lib.format.open_memmap(
            folder / f"val_{name}_{size}.npy", mode="w+", dtype=np.float32, shape=(len(val), *shape)
        )
        for name in conditions
    }
    with zipfile.ZipFile(archive_path) as archive, ThreadPoolExecutor(WORKERS) as pool:

        def prepare(item):
            i, row = item
            with tf.device("/CPU:0"):
                image = exp.image_tensor(archive.read(row.path), size, task["input_policy"])
            cache[i] = image
            if row.split == "val":
                seed = exp.item_seed(row, 1707)
                shifted["lighting"][i - len(train)] = exp.transform_image(
                    image, profile, "target_guided", seed
                )
                if expanded is not None:
                    shifted["expanded"][i - len(train)] = camera_extended.transform(
                        image, expanded, "expanded", seed
                    )

        for start in range(0, len(rows), 128):
            list(pool.map(prepare, enumerate(rows[start : start + 128], start)))
            end = min(start + 128, len(rows))
            if start == 0 or end % 512 == 0 or end == len(rows):
                show(f"[{task['domain']} · {size}px] 입력 캐시 준비 (사진)", end, len(rows))
    cache.flush()
    for images in shifted.values():
        images.flush()
    lookup = {row.path: i for i, row in enumerate(rows)}
    paired_indexes = [
        lookup[pairs[r.path].path] if r.path in pairs else i for i, r in enumerate(train)
    ]
    return cache, shifted if expanded is not None else shifted["lighting"], paired_indexes


def pipeline(images, indexes, labels, batch, size, transform=None):
    def read(index):
        image = images[int(index)]
        return image if transform is None else transform(int(index), image)

    def get(index, label):
        image = tf.numpy_function(read, [index], tf.float32)
        return tf.ensure_shape(image, (size, size, 3)), label

    ds = tf.data.Dataset.from_tensor_slices(
        (np.asarray(indexes, np.int32), np.asarray(labels, np.int32))
    )
    options = tf.data.Options()
    options.threading.private_threadpool_size = WORKERS
    options.threading.max_intra_op_parallelism = 1
    return ds.with_options(options).map(get, num_parallel_calls=WORKERS).batch(batch).prefetch(2)


def model_for(spec, count, weights):
    if spec["architecture"] == "pmg":
        from mediflow_datasets.web_skin_paper_suite import build_pmg_model

        return build_pmg_model(count, spec["size"], weights=weights)
    size = spec["size"]
    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(size, size, 3)
    )
    backbone.trainable = False
    inputs = keras.Input((size, size, 3))
    features = keras.layers.GlobalAveragePooling2D()(backbone(inputs, training=False))
    outputs = keras.layers.Dense(count, activation="softmax")(keras.layers.Dropout(0.3)(features))
    return keras.Model(inputs, outputs)


def predictor(model, spec):
    pmg = spec["architecture"] == "pmg"
    size = spec["size"]

    @tf.function(input_signature=[tf.TensorSpec([None, size, size, 3], tf.float32)])
    def predict(images):
        outputs = model(images, training=False)
        scores = tf.nn.softmax(tf.add_n(outputs), axis=-1) if pmg else outputs
        return tf.debugging.check_numerics(scores, "validation scores")

    return predict


def steps(model, spec, optimizer, count, *, alignment=None):
    pmg = spec["architecture"] == "pmg"
    loss_fn = keras.losses.CategoricalCrossentropy(
        from_logits=pmg, label_smoothing=spec["label_smoothing"]
    )
    variables = model.trainable_variables
    optimizer.build(variables)
    if pmg:
        from mediflow_datasets.web_skin_paper_suite import _jigsaw
    if alignment is not None:
        adapter = "sum_logits_softmax" if pmg else "single_softmax"
        extractor = exp.feature_model(model, adapter)
        features = keras.Model(model.input, extractor.outputs[-1])

    def update(images, labels, branch=None, factor=1.0, target=None):
        with tf.GradientTape() as tape:
            if target is None:
                outputs = model(images, training=True)
                logits = outputs[branch] if pmg else outputs
                mmd = coral = tf.constant(0.0)
            else:
                outputs = extractor(images, training=True)
                logits = outputs[branch if pmg else 0]
                mmd, coral = domain_alignment.losses(
                    outputs[-1], features(target, training=False), alignment["reference"]
                )
            ce = factor * loss_fn(labels, logits)
            loss = ce
            if target is not None:
                loss = (
                    loss + alignment["weights"]["mmd"] * mmd + alignment["weights"]["coral"] * coral
                )
            loss = tf.debugging.check_numerics(loss, "training loss")
        grads = tape.gradient(loss, variables)
        pairs = [
            (grad, var) for grad, var in zip(grads, variables, strict=True) if grad is not None
        ]
        clipped, norm = tf.clip_by_global_norm([g for g, _ in pairs], 1.0)
        with tf.control_dependencies([tf.debugging.check_numerics(norm, "gradient norm")]):
            optimizer.apply_gradients(zip(clipped, [v for _, v in pairs], strict=True))
        return loss, ce, mmd, coral

    signature = [
        tf.TensorSpec([None, spec["size"], spec["size"], 3], tf.float32),
        tf.TensorSpec([None], tf.int32),
    ]

    def update_batch(images, indices, target=None):
        labels = tf.one_hot(indices, count)
        if pmg:
            branch_losses = [
                update(_jigsaw(images, grid), labels, i)[0] for i, grid in enumerate((8, 4, 2))
            ]
            loss, ce, mmd, coral = update(images, labels, 3, 2.0, target)
            total = tf.add_n(branch_losses)
            return (total + loss) / 4, (total + ce) / 4, mmd, coral
        return update(images, labels, target=target)

    @tf.function(input_signature=signature)
    def train_step(images, indices):
        return update_batch(images, indices)[0]

    if alignment is not None:
        aligned_step = tf.function(update_batch, input_signature=[*signature, signature[0]])
        return train_step, predictor(model, spec), aligned_step

    return train_step, predictor(model, spec)


def epoch_records(folder, epochs):
    records = sorted(folder.glob("epoch_*.json"))
    history = [data.read_json(p) for p in records]
    if len(history) > epochs or [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError(f"Incomplete or inconsistent epoch records: {folder}")
    return history


def save_epoch(checkpoint, folder, local, entry):
    attempt = "snapshot_" + uuid.uuid4().hex
    destination = folder / attempt
    destination.mkdir()
    local.mkdir(exist_ok=True)
    local_prefix = checkpoint.write(str(local / "state"))
    record = {**entry, "snapshot": attempt, "files": {}}
    for path in sorted(Path(local_prefix).parent.glob("state.*")):
        shutil.copyfile(path, destination / path.name)
        record["files"][path.name] = data.sha256(path)
    if set(record["files"]) != {"state.index", "state.data-00000-of-00001"}:
        raise RuntimeError("Unexpected checkpoint shard layout")
    write_progress(folder / f"epoch_{entry['epoch']:03}.json", record)
    return record


def restore_epoch(checkpoint, folder, record):
    for name, sha in record["files"].items():
        if data.sha256(data.local_path(folder, f"{record['snapshot']}/{name}")) != sha:
            raise ValueError("Epoch checkpoint failed integrity verification")
    checkpoint.read(str(folder / record["snapshot"] / "state")).assert_consumed()


def publish_model(source, directory, name):
    temporary = directory / (uuid.uuid4().hex + ".keras.tmp")
    shutil.copyfile(source, temporary)
    destination = directory / f"{name}.keras"
    temporary.replace(destination)
    return {"file": destination.name, "sha256": data.sha256(source)}


def export_model(model, directory, name, local):
    local.mkdir(exist_ok=True)
    scratch = local / f"{name}.keras"
    model.save(scratch)
    return publish_model(scratch, directory, name)


def validation_pipelines(cache, shifted, n, val, batch, size):
    labels = [row.label for row in val]
    return {
        "clean": pipeline(cache, range(n, n + len(val)), labels, batch, size),
        **{
            name: pipeline(images, range(len(val)), labels, batch, size)
            for name, images in shifted.items()
        },
    }


def validate(predict, datasets, rows, class_count, label):
    metrics = {}
    for name, ds in datasets.items():
        print(f"{label} Validation {name} · {len(rows):,}장", flush=True)
        scores = np.concatenate([predict(images).numpy() for images, _ in ds])
        metrics[name] = exp.classification(rows, scores, class_count)
    return metrics


def phase(model, context, folder, local, spec, phase_name, epochs, rate, *, alignment=None):
    config, task, train, val, profile, cache, shifted, pairs = context
    folder.mkdir(exist_ok=True)
    history = epoch_records(folder, epochs)
    optimizer = keras.optimizers.Adam(rate)
    step, predict, *extra_steps = steps(
        model, spec, optimizer, len(task["classes"]), alignment=alignment
    )
    if (phase_name == "aligned") != (alignment is not None):
        raise ValueError("Alignment inputs must be present exactly for the aligned phase")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    tag = f"[{task['domain']} · {spec['role']} · {phase_name}]"
    if history:
        restore_epoch(checkpoint, folder, history[-1])
        print(f"{tag} Drive에서 {len(history)}/{epochs} epoch 복원", flush=True)
    n, batch, size = len(train), task["batch_size"], spec["size"]
    expanded = phase_name in ("expanded", "aligned")
    condition = "expanded" if expanded else "lighting"
    datasets = validation_pipelines(
        cache, shifted if expanded else {"lighting": shifted}, n, val, batch, size
    )
    for epoch in range(len(history), epochs):
        started = time.monotonic()
        keras.utils.set_random_seed(config["seed"] + epoch)
        order = np.random.default_rng(config["seed"] + epoch).permutation(n)
        indexes = order
        if phase_name.startswith("prepare") and spec.get("skin_augmented_preparation"):
            indexes = [pairs[i] if (i + epoch) % 2 == 0 else i for i in order]
        transform = None
        if expanded or phase_name == "lighting":
            seeds = [exp.item_seed(row, config["seed"], epoch) for row in train]
            apply = [exp.augmentation_applies(s, config["augmentation_probability"]) for s in seeds]

            def transform(i, image, seeds=seeds, apply=apply):
                if not apply[i]:
                    return image
                if expanded:
                    return camera_extended.transform(image, profile, "expanded", seeds[i])
                return exp.transform_image(image, profile, "target_guided", seeds[i])

        label = f"{tag} Epoch {epoch + 1}/{epochs}"
        batches = math.ceil(n / batch)
        print(f"{label} 시작 · 학습 {n:,}장 · {batches} 배치", flush=True)
        train_data = pipeline(
            cache, indexes, [train[i].label for i in order], batch, size, transform
        )
        losses = []
        aligned_metrics = []
        schedule = {}
        if alignment is not None:
            eligible = range(batches - int(n % batch == 1))
            schedule = domain_alignment.balanced_schedule(
                alignment["rows"], eligible, config["seed"], epoch
            )
            if not schedule:
                raise ValueError("No Source batch is eligible for feature alignment")
        for i, (images, labels) in enumerate(train_data, 1):
            if i - 1 in schedule:
                target = alignment["images"][schedule[i - 1]]
                loss, ce, mmd, coral = map(float, extra_steps[0](images, labels, target))
                losses.append(loss)
                aligned_metrics.append((ce, mmd, coral))
            else:
                losses.append(float(step(images, labels)))
            if i == 1 or i == batches or i % max(5, batches // 15) == 0:
                elapsed = time.monotonic() - started
                remaining = elapsed * (batches / i - 1)
                show(
                    label + " 학습",
                    i,
                    batches,
                    f"loss={np.mean(losses):.5f} · 남은 학습 약 {remaining:.0f}초",
                )
        train_seconds = time.monotonic() - started
        metrics = validate(predict, datasets, val, len(task["classes"]), label)
        validation_seconds = time.monotonic() - started - train_seconds
        entry = {
            "epoch": epoch + 1,
            "train_loss_mean_per_update": float(np.mean(losses)),
            "training_batches": batches,
            "optimizer_updates": batches * (4 if spec["architecture"] == "pmg" else 1),
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            **metrics,
        }
        if aligned_metrics:
            ce, mmd, coral = np.asarray(aligned_metrics).mean(0).tolist()
            entry["alignment"] = {
                "updates": len(aligned_metrics),
                "target_images_used": sum(map(len, schedule.values())),
                "classification_loss_on_aligned_batches": ce,
                "mmd_mean": mmd,
                "coral_mean": coral,
            }
            print(
                f"{label} 정렬 {len(aligned_metrics)}배치 · MMD={mmd:.6g} · CORAL={coral:.6g}",
                flush=True,
            )
        print(f"{label} Drive 체크포인트 저장", flush=True)
        history.append(save_epoch(checkpoint, folder, local, entry))
        print(
            f"{label} 완료 · loss={entry['train_loss_mean_per_update']:.5f} · "
            f"원본 Val F1={metrics['clean']['macro_f1']:.5f} · "
            f"조명 Val F1={metrics['lighting']['macro_f1']:.5f}"
            + (f" · 확장 Val F1={metrics['expanded']['macro_f1']:.5f}" if expanded else ""),
            flush=True,
        )
    best = max(history, key=lambda r: score(r, expanded or phase_name == "lighting", condition))
    restore_epoch(checkpoint, folder, best)
    write_progress(folder / "history.json", history)
    return best


def score(record, lighting=True, condition="lighting"):
    clean, shifted = record["clean"]["macro_f1"], record[condition]["macro_f1"]
    return (min(clean, shifted), clean) if lighting else (clean,)


def trial(context, output, local, spec, job):
    config, task = context[:2]
    output.mkdir(exist_ok=True)
    completed = output / "complete.json"
    if completed.exists():
        saved = data.read_json(completed)
        verify_models(output, saved)
        print(f"[{job}/6 · {task['domain']} · {spec['role']}] 완료 결과 재사용", flush=True)
        return saved
    print(f"\n[{job}/6 · {task['domain']} · {spec['role']}] 실험 시작", flush=True)
    keras.backend.clear_session()
    keras.utils.set_random_seed(config["seed"])
    count = len(task["classes"])
    initial_path = output / "initial.json"
    if initial_path.exists():
        initial = data.read_json(initial_path)
        path = data.local_path(output, initial["file"])
        if data.sha256(path) != initial["sha256"]:
            raise ValueError("Starting checkpoint changed")
        model = keras.models.load_model(path, compile=False)
    else:
        if spec.get("model"):
            path = data.local_path(ROOT, spec["model"])
            if data.sha256(path) != spec["model_sha256"]:
                raise ValueError("Selected checkpoint changed")
            model = keras.models.load_model(path, compile=False)
            initial = publish_model(path, output, "initial")
            print(f"[{task['domain']} · {spec['role']}] 저장 가중치로 조명 적응 시작", flush=True)
        else:
            weights = None if (output / "prepare_head/epoch_001.json").exists() else "imagenet"
            print(f"[{task['domain']} · {spec['role']}] 현재 Train으로 출발 모델 준비", flush=True)
            model = model_for(spec, count, weights)
            phase(
                model,
                context,
                output / "prepare_head",
                local,
                spec,
                "prepare_head",
                config["prepare_head_epochs"],
                config["prepare_head_lr"],
            )
            exp.configure_partial(model)
            phase(
                model,
                context,
                output / "prepare_partial",
                local,
                spec,
                "prepare_partial",
                config["prepare_partial_epochs"],
                config["prepare_partial_lr"],
            )
            initial = export_model(model, output, "initial", local)
        write_progress(initial_path, initial)
    if tuple(model.input_shape[1:]) != (spec["size"], spec["size"], 3):
        raise ValueError("Initial input shape differs from configuration")
    if len(model.outputs) != (4 if spec["architecture"] == "pmg" else 1) or any(
        layer.shape[-1] != count for layer in model.outputs
    ):
        raise ValueError("Initial output contract differs from class mapping")
    exp.configure_partial(model)
    best = phase(
        model,
        context,
        output / "lighting",
        local,
        spec,
        "lighting",
        config["epochs"],
        config["learning_rate"],
    )
    saved = {
        "initial": initial,
        "best_model": export_model(model, output, "best", local),
        "model_config": spec,
        "validation": best,
        "source_test_evaluated": False,
        "validation_scope": config.get("validation_scope", "current_source_validation"),
    }
    write_progress(completed, saved)
    return saved


def verify_models(folder, result):
    for key in ("initial", "best_model"):
        item = result[key]
        if data.sha256(data.local_path(folder, item["file"])) != item["sha256"]:
            raise ValueError(f"Completed checkpoint changed: {folder}/{item['file']}")


def run_domain(domain, source_zip, output, local):
    config = data.read_json(ROOT / "configs/adaptation_stage1.json")
    task = config["tasks"][domain]
    if config.get("initialization_policy") == "saved_only" and any(
        not spec.get("model") or not spec.get("model_sha256") for spec in task["models"].values()
    ):
        raise ValueError("A starting checkpoint is missing from the saved-only bundle")
    if config.get("validation_scope"):
        print(
            f"[{domain}] 과거 학습 이력의 겹침 여부 미확인 · Validation은 탐색 비교용", flush=True
        )
    reference = data.read_json(ROOT / f"references/{domain}.json")
    profile = reference["extended_profile"]["lighting"]
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": object_hash(config),
        "reference_sha256": object_hash(reference),
        "code_sha256": data.sha256(Path(__file__)),
    }
    marker = output / "identity.json"
    if marker.exists() and data.read_json(marker) != identity:
        raise ValueError("Different inputs cannot resume this output")
    write_progress(marker, identity)
    write_progress(
        output / "run_configuration.json",
        {
            "identity": identity,
            "configuration": config,
            "profile": profile,
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "gpu": [
                tf.config.experimental.get_device_details(device)
                for device in tf.config.list_physical_devices("GPU")
            ],
            "code_provenance": data.read_json(ROOT / "bundle_provenance.json"),
        },
    )
    if all((output / role / "complete.json").is_file() for role in ARMS):
        results = {role: data.read_json(output / role / "complete.json") for role in ARMS}
        for role, result in results.items():
            verify_models(output / role, result)
        print(f"[{domain}] 두 학습 모두 완료됨", flush=True)
        return results
    train, val, pairs = prepare_source(domain, source_zip, task, reference, output)
    caches = {}
    results = {}
    for i, role in enumerate(ARMS, 1):
        existing = output / role / "complete.json"
        if existing.is_file():
            results[role] = data.read_json(existing)
            verify_models(output / role, results[role])
            print(
                f"[{DOMAINS.index(domain) * 2 + i}/6 · {domain} · {role}] 완료 결과 재사용",
                flush=True,
            )
            continue
        spec = {**task["models"][role], "role": role}
        size = spec["size"]
        if size not in caches:
            caches[size] = image_cache(local, source_zip, train, val, pairs, task, size, profile)
        context = (config, task, train, val, profile, *caches[size])
        results[role] = trial(
            context, output / role, local / "saving", spec, DOMAINS.index(domain) * 2 + i
        )
    return results


def write_selection(root):
    selected, table = {}, []
    for domain in DOMAINS:
        records = {role: data.read_json(root / domain / role / "complete.json") for role in ARMS}
        for role, record in records.items():
            verify_models(root / domain / role, record)
            table.append(
                {
                    "domain": domain,
                    "role": role,
                    "best_epoch": record["validation"]["epoch"],
                    "clean_macro_f1": record["validation"]["clean"]["macro_f1"],
                    "lighting_macro_f1": record["validation"]["lighting"]["macro_f1"],
                    "clean_accuracy": record["validation"]["clean"]["accuracy"],
                    "lighting_accuracy": record["validation"]["lighting"]["accuracy"],
                    "validation_scope": record.get("validation_scope", "current_source_validation"),
                }
            )
        role = max(ARMS, key=lambda name: score(records[name]["validation"]))
        record = records[role]
        selected[domain] = {
            "role": role,
            "model_config": record["model_config"],
            "initial_checkpoint": {
                "path": f"{domain}/{role}/initial.keras",
                "sha256": record["initial"]["sha256"],
            },
            "stage1_checkpoint": {
                "path": f"{domain}/{role}/best.keras",
                "sha256": record["best_model"]["sha256"],
            },
        }
    selection = {
        "schema": "mediflow_adaptation_stage1_selection_v1",
        "selected": selected,
        "criteria": "Maximize worst Val Macro-F1. Break ties by clean Val, then basic.",
        "source_test_evaluated": False,
        "target_accuracy": None,
    }
    if any(row["validation_scope"].startswith("exploratory_") for row in table):
        selection["selection_scope"] = "provisional_followup_candidate"
        selection["historical_train_evaluation_independence_verified"] = False
    path = root / "selection.json"
    if path.exists() and data.read_json(path) != selection:
        raise ValueError("Existing Validation selection differs")
    with (root / "validation_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    write_progress(path, selection)
    for row in table:
        print(row, flush=True)
    print("1차 완료 · 분야별 선택:", {d: s["role"] for d, s in selected.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", type=Path)
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local", type=Path)
    args = parser.parse_args()
    if args.select:
        write_selection(args.select)
    elif all((args.domain, args.source_zip, args.output, args.local)):
        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("TensorFlow GPU를 사용할 수 없습니다.")
        run_domain(args.domain, args.source_zip, args.output, args.local)
    else:
        parser.error("Provide --select or all domain run parameters")


if __name__ == "__main__":
    main()
