"""Three camera adaptation arms per domain, with shared baselines and sealed evaluation."""

from __future__ import annotations

import argparse
import gc
import shutil
import traceback
import uuid
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mediflow_datasets import camera_extended as camera
from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets import domain_adaptation_resolution as resolution
from mediflow_datasets import domain_alignment as alignment
from mediflow_datasets.candidate_reproduction import ROOT
from mediflow_datasets.domain_adaptation_io import write_progress
from mediflow_datasets.domain_adaptation_training import object_hash

ARMS = ("lighting", "expanded", "aligned")
DOMAINS = ("hair", "web_skin", "skin")


def load_config(path):
    config = data.read_json(path)
    if config.get("schema") != "mediflow_nine_v1" or config.get("arms") != list(ARMS):
        raise ValueError("Expected exactly three reviewed adaptation arms")
    if set(config["tasks"]) != set(DOMAINS):
        raise ValueError("Nine experiments need three separate domains")
    for key in ("epochs", "batch_size", "baseline_head_epochs", "baseline_partial_epochs"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError("Invalid positive integer: " + key)
    if not 0 < config["learning_rate"] < 1 or not 0 <= config["augmentation_probability"] <= 1:
        raise ValueError("Invalid optimizer or augmentation configuration")
    if not 0 < config["alignment_fraction_per_loss"] <= 0.25:
        raise ValueError("Alignment regularization fraction must be bounded")
    return config


def target_inputs(directory, domain, spec):
    meta = data.read_json(directory / "target_inputs.json")
    path = directory / "target_inputs.npz"
    if meta["domain"] != domain or meta["input_policy"] != spec["input_policy"]:
        raise ValueError("Target domain or preprocessing differs")
    if data.sha256(path) != meta["array_sha256"]:
        raise ValueError("Target image array changed")
    rows = meta["rows"]
    if len({r["person"] for r in rows}) != 4 or any(
        r["split"] != "calibration" or r.get("label") is not None for r in rows
    ):
        raise ValueError("Target inputs must be the four unlabeled calibration participants")
    with np.load(path, allow_pickle=False) as arrays:
        images = arrays["images"]
    if images.shape != (len(rows), spec["input_size"], spec["input_size"], 3):
        raise ValueError("Target array dimensions differ from the model contract")
    if images.dtype != np.float32 or not np.isfinite(images).all():
        raise ValueError("Target images must be finite float32")
    if images.min() < 0 or images.max() > 255:
        raise ValueError("Target images must use RGB 0-255")
    return meta, images


def predictions(model, rows, reader, spec, batch_size, profile=None, mode="expanded"):
    import tensorflow as tf

    extractor = exp.feature_model(model, spec["output_adapter"])
    scores, features = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = []
        for row in batch:
            image = exp.image_tensor(reader(row), spec["input_size"], spec["input_policy"])
            if profile is not None:
                image = camera.transform(image, profile, mode, exp.item_seed(row, 1707))
            images.append(image)
        output = extractor(np.stack(images), training=False)
        score = (
            tf.nn.softmax(tf.add_n(output[:-1]), -1)
            if (spec["output_adapter"] == "sum_logits_softmax")
            else output[0]
        )
        scores.extend(score.numpy())
        features.extend(output[-1].numpy())
    scores, features = np.asarray(scores), np.asarray(features)
    if not np.isfinite(scores).all() or not np.isfinite(features).all():
        raise ValueError("Non-finite prediction output")
    return scores, features


def target_predictions(model, images, spec, batch_size):
    import tensorflow as tf

    extractor = exp.feature_model(model, spec["output_adapter"])
    scores, features = [], []
    for start in range(0, len(images), batch_size):
        output = extractor(images[start : start + batch_size], training=False)
        score = (
            tf.nn.softmax(tf.add_n(output[:-1]), -1)
            if (spec["output_adapter"] == "sum_logits_softmax")
            else output[0]
        )
        scores.extend(score.numpy())
        features.extend(output[-1].numpy())
    return np.asarray(scores), np.asarray(features)


def validation(model, source, reader, spec, config, profile):
    result = {}
    for name, change in (("clean", None), ("shifted", profile)):
        scores, _ = predictions(
            model,
            source["original", "val"],
            reader,
            spec,
            config["batch_size"],
            change,
        )
        result[name] = exp.classification(source["original", "val"], scores, spec["class_count"])
    return result


def make_reference(model, source, reader, target, images, spec, config):
    # Class-balanced deterministic Train sample, with no Validation/Test feature fitting.
    sample = [
        r
        for label in range(spec["class_count"])
        for r in [v for v in source["original", "train"] if v.label == label][:8]
    ]
    scores, features = predictions(model, sample, reader, spec, config["batch_size"])
    _, target_features = target_predictions(model, images, spec, config["batch_size"])
    reference = exp.feature_reference(features)
    participant_features = np.stack(
        [
            target_features[[i for i, r in enumerate(target) if r["person"] == p]].mean(0)
            for p in sorted({r["person"] for r in target})
        ]
    )
    a = (features - reference["mean"]) / reference["std"]
    b = (participant_features - reference["mean"]) / reference["std"]
    distances = ((a[:, None] - b[None]) ** 2).sum(-1)
    positive = distances[distances > 1e-8]
    if len(positive):
        reference["bandwidth_sq"] = float(np.median(positive))
    reference = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in reference.items()}
    reference["bandwidth_multipliers"] = [0.25, 1.0, 4.0]
    mmd, coral = alignment.losses(features, target_features, reference)
    labels = np.asarray([r.label for r in sample])
    classification = max(
        0.05, float(-np.log(np.maximum(scores[np.arange(len(sample)), labels], 1e-7)).mean())
    )
    weights = {
        key: min(10.0, config["alignment_fraction_per_loss"] * classification / float(value))
        if float(value) > 1e-8
        else 0.0
        for key, value in (("mmd", mmd), ("coral", coral))
    }
    if sum(weights.values()) == 0:
        raise ValueError("Both initial alignment losses are zero; cannot claim a distinct arm")
    return (
        reference,
        weights,
        {
            "source_train_paths": [r.path for r in sample],
            "target_role": "adaptation_train",
            "bandwidth_policy": "median Source-Train to Target-training participant-mean distance",
            "initial_mmd": float(mmd),
            "initial_coral": float(coral),
            "initial_classification_loss": classification,
            "weights": weights,
            "weight_policy": "each loss starts at 0.05 of Train CE, coefficient capped at 10",
        },
    )


def train_epoch(
    model,
    rows,
    reader,
    spec,
    config,
    profile,
    arm,
    epoch,
    optimizer,
    target_rows,
    target_images,
    reference,
    weights,
):
    import keras
    import tensorflow as tf

    order = np.random.default_rng(config["seed"] + epoch).permutation(len(rows))
    batches = [
        [rows[i] for i in order[start : start + config["batch_size"]]]
        for start in range(0, len(order), config["batch_size"])
    ]
    eligible = [i for i, batch in enumerate(batches) if len(batch) >= 2]
    schedule = alignment.balanced_schedule(target_rows, eligible, config["seed"], epoch)
    extractor = exp.feature_model(model, spec["output_adapter"])
    loss_fn = keras.losses.CategoricalCrossentropy(
        from_logits=spec["output_adapter"] == "sum_logits_softmax",
        label_smoothing=0.05 if spec["loss"] == "ls005" else 0,
    )
    summary = {
        "classification_sum": 0.0,
        "mmd_sum": 0.0,
        "coral_sum": 0.0,
        "alignment_steps": 0,
        "optimizer_updates": 0,
        "target_images_used": 0,
    }

    def update(images, labels, branch=None, grid=None, target=None, factor=1.0):
        if grid is not None:
            from mediflow_datasets.web_skin_paper_suite import _jigsaw

            images = _jigsaw(tf.convert_to_tensor(images), grid)
        with tf.GradientTape() as tape:
            outputs = extractor(images, training=True)
            logits = outputs[branch] if branch is not None else outputs[0]
            ce = factor * loss_fn(labels, logits)
            mmd = coral = tf.constant(0, dtype=tf.float32)
            if target is not None:
                target_features = extractor(target, training=False)[-1]
                mmd, coral = alignment.losses(outputs[-1], target_features, reference)
            total = ce + weights["mmd"] * mmd + weights["coral"] * coral
        gradients = tape.gradient(total, model.trainable_variables)
        pairs = [
            (g, v)
            for g, v in zip(gradients, model.trainable_variables, strict=True)
            if g is not None
        ]
        if not pairs:
            raise ValueError("No trainable gradients")
        tf.debugging.assert_all_finite(total, "Non-finite training loss")
        for gradient, _ in pairs:
            tf.debugging.assert_all_finite(gradient, "Non-finite gradient")
        clipped, _ = tf.clip_by_global_norm([g for g, _ in pairs], config["gradient_clip_norm"])
        optimizer.apply_gradients(zip(clipped, [v for _, v in pairs], strict=True))
        summary["classification_sum"] += float(ce)
        summary["mmd_sum"] += float(mmd)
        summary["coral_sum"] += float(coral)
        summary["optimizer_updates"] += 1

    for index, batch in enumerate(batches):
        images = []
        for row in batch:
            image = exp.image_tensor(reader(row), spec["input_size"], spec["input_policy"])
            seed = exp.item_seed(row, config["seed"], epoch)
            if exp.augmentation_applies(seed, config["augmentation_probability"]):
                image = camera.transform(image, profile, arm, seed)
            images.append(image)
        images = np.stack(images)
        labels = tf.one_hot([r.label for r in batch], spec["class_count"])
        target = target_images[schedule[index]] if arm == "aligned" and index in schedule else None
        if spec["output_adapter"] == "sum_logits_softmax":
            for branch, grid in enumerate((8, 4, 2)):
                update(images, labels, branch=branch, grid=grid)
            update(images, labels, branch=3, target=target, factor=2.0)
        else:
            update(images, labels, target=target)
        if target is not None:
            summary["alignment_steps"] += 1
            summary["target_images_used"] += len(target)
    if arm == "aligned" and summary["alignment_steps"] == 0:
        raise ValueError("No alignment update was possible")
    return summary


def run_arm(
    output,
    arm,
    checkpoint,
    source,
    reader,
    spec,
    config,
    profile,
    target_rows,
    images,
    reference,
    weights,
):
    import keras

    folder = output / arm
    complete = folder / "complete.json"
    if complete.exists():
        saved = data.read_json(complete)
        if data.sha256(data.local_path(folder, saved["model_relative"])) != saved["model_sha256"]:
            raise ValueError("Completed checkpoint changed: " + arm)
        return saved
    folder.mkdir(exist_ok=True)
    attempt = folder / ("attempt_" + uuid.uuid4().hex[:8])
    attempt.mkdir()
    keras.backend.clear_session()
    keras.utils.set_random_seed(config["seed"])
    model = keras.models.load_model(checkpoint, compile=False)
    exp.configure_partial(model)
    optimizer = keras.optimizers.Adam(config["learning_rate"])
    optimizer.build(model.trainable_variables)
    best, best_metrics, history = None, None, []
    for epoch in range(config["epochs"]):
        losses = train_epoch(
            model,
            source[spec["train_variant"], "train"],
            reader,
            spec,
            config,
            profile,
            arm,
            epoch,
            optimizer,
            target_rows,
            images,
            reference,
            weights,
        )
        metrics = validation(model, source, reader, spec, config, profile)
        history.append({"epoch": epoch + 1, **losses, **metrics})
        score = min(metrics["clean"]["macro_f1"], metrics["shifted"]["macro_f1"])
        if best is None or score > best:
            best, best_metrics = score, metrics
            model.save(attempt / "best.keras")
        write_progress(attempt / "history.json", history)
        print(
            spec["domain"],
            arm,
            epoch + 1,
            metrics["clean"]["macro_f1"],
            metrics["shifted"]["macro_f1"],
            "alignment",
            losses["alignment_steps"],
            flush=True,
        )
    saved = {
        **best_metrics,
        "model_relative": (attempt / "best.keras").relative_to(folder).as_posix(),
        "model_sha256": data.sha256(attempt / "best.keras"),
        "history_relative": (attempt / "history.json").relative_to(folder).as_posix(),
    }
    runner.write_json(complete, saved)
    del model
    keras.backend.clear_session()
    gc.collect()
    return saved


def source_and_spec(domain, config, source_zip, review):
    reference, checkpoint, classes = data.candidate(domain)
    task = config["tasks"][domain]
    spec = {
        k: reference[k]
        for k in (
            "model_sha256",
            "input_size",
            "output_adapter",
            "backbone",
            "architecture",
            "normal_class_included",
        )
        if k in reference
    }
    spec.update(
        domain=domain,
        input_policy=task["input_policy"],
        loss=task["loss"],
        train_variant=task["train_variant"],
        class_count=len(classes),
    )
    source, digest = data.source_index(source_zip, classes, task["source_sha256"])
    if domain != "hair":
        source = resolution.filtered_source(
            source, data.read_json(review / "quarantine_plan.json"), classes, digest
        )
    return source, spec, checkpoint, classes


def run_domain(args):
    import keras

    config = load_config(args.config)
    review, output = args.review, args.output.resolve()
    profile = data.read_json(review / "extended_profile.json")
    receipt = data.read_json(review / "readiness.json")
    if (
        receipt["profile_sha256"] != object_hash(profile)
        or receipt["config_sha256"] != object_hash(config)
        or receipt["status"] != "reviewed_nine_pilot"
    ):
        raise ValueError("Unreviewed or changed nine-experiment configuration")
    for name, digest in receipt["evidence_sha256"].items():
        if data.sha256(data.local_path(review, name)) != digest:
            raise ValueError("Pilot evidence changed: " + name)
    source, spec, original_checkpoint, classes = source_and_spec(
        args.domain, config, args.source_zip, review
    )
    meta, images = target_inputs(review, args.domain, spec)
    identity = {
        "domain": args.domain,
        "config": object_hash(config),
        "profile": object_hash(profile),
        "classes": classes,
        "target_arrays": meta["array_sha256"],
        "source_sha256": config["tasks"][args.domain]["source_sha256"],
        "source_rows": object_hash([asdict(r) for rows in source.values() for r in rows]),
        "code": {
            name: data.sha256(ROOT / f"src/mediflow_datasets/{name}.py")
            for name in ("domain_nine", "domain_alignment", "camera_extended")
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "run_identity.json"
    if marker.exists():
        if data.read_json(marker) != identity:
            raise ValueError("Cannot resume with changed inputs")
    else:
        if any(output.iterdir()):
            raise ValueError("Refuse to overwrite an unrelated output directory")
        runner.write_json(marker, identity)
        runner.snapshot(output, config)
        extra = output / "nine_code"
        extra.mkdir()
        for name in ("camera_extended", "domain_alignment", "domain_nine"):
            shutil.copyfile(ROOT / f"src/mediflow_datasets/{name}.py", extra / f"{name}.py")
        runner.write_json(
            output / "source_rows.json", [asdict(r) for v in source.values() for r in v]
        )
        runner.write_json(
            output / "target_roles.json",
            {
                "rows": meta["rows"],
                "role": "calibration and unlabeled adaptation training",
                "independent_target_test": False,
                "target_accuracy": None,
            },
        )
    write_progress(output / "status.json", {"domain": args.domain, "status": "checking_inputs"})
    if args.domain == "hair":
        audit, pixels = data.audit_source(args.source_zip, source)
        consistent = all(audit["original_augmented_consistency"].values())
    else:
        with zipfile.ZipFile(args.source_zip) as archive:
            entries = [
                (r, *data.image_signatures(archive.read(r.path)))
                for rows in source.values()
                for r in rows
            ]
        audit, pixels, consistent = data.audit_groups(entries), {r[2] for r in entries}, True
    audit["source_target_pixel_overlap"] = len(pixels & set(meta["pixel_hashes"]))
    write_progress(output / "data_audit.json", audit)
    if (
        audit["blocking_leaks_or_conflicts"]
        or audit["near_duplicate_review"]
        or audit["source_target_pixel_overlap"]
        or not consistent
    ):
        raise ValueError("Unresolved data findings; no model update was started")
    baseline_record = output / "baseline_complete.json"
    with zipfile.ZipFile(args.source_zip) as archive:
        reader = lambda r: archive.read(r.path)  # noqa: E731
        if baseline_record.exists():
            base = data.read_json(baseline_record)
            checkpoint = data.local_path(output, base["relative_path"])
            if data.sha256(checkpoint) != base["model_sha256"]:
                raise ValueError("Shared baseline checkpoint changed")
        else:
            if args.domain == "hair":
                directory = output / "baseline"
                directory.mkdir(exist_ok=True)
                checkpoint = directory / "existing_v2.keras"
                if not checkpoint.exists():
                    shutil.copyfile(original_checkpoint, checkpoint)
                if data.sha256(checkpoint) != spec["model_sha256"]:
                    raise ValueError("Hair baseline copy changed")
            else:
                directory = output / ("fresh_baseline_" + uuid.uuid4().hex[:8])
                checkpoint = resolution.train_fresh_baseline(
                    args.domain,
                    spec,
                    classes,
                    source,
                    archive,
                    config,
                    profile["lighting"],
                    directory,
                )
            base = {
                "relative_path": checkpoint.relative_to(output).as_posix(),
                "model_sha256": data.sha256(checkpoint),
                "initialization": "existing_v2" if args.domain == "hair" else "ImageNet",
            }
            runner.write_json(baseline_record, base)
        setup_path = output / "baseline_setup.json"
        if setup_path.exists():
            setup = data.read_json(setup_path)
            if setup.get("baseline_sha256") != base["model_sha256"] or setup.get(
                "identity_sha256"
            ) != object_hash(identity):
                raise ValueError("Baseline feature reference belongs to different inputs")
        else:
            model = keras.models.load_model(checkpoint, compile=False)
            reference, weights, evidence = make_reference(
                model, source, reader, meta["rows"], images, spec, config
            )
            setup = {
                "baseline_validation": validation(model, source, reader, spec, config, profile),
                "reference": reference,
                "weights": weights,
                "evidence": evidence,
                "baseline_sha256": base["model_sha256"],
                "identity_sha256": object_hash(identity),
            }
            runner.write_json(setup_path, setup)
            del model
            keras.backend.clear_session()
        results, failures = {}, {}
        for arm in ARMS:
            try:
                results[arm] = run_arm(
                    output,
                    arm,
                    checkpoint,
                    source,
                    reader,
                    spec,
                    config,
                    profile,
                    meta["rows"],
                    images,
                    setup["reference"],
                    setup["weights"],
                )
            except Exception as error:
                failures[arm] = {"error": str(error), "traceback": traceback.format_exc()}
                write_progress(output / f"{arm}_failure.json", failures[arm])
                keras.backend.clear_session()
                gc.collect()
            write_progress(
                output / "progress.json", {"complete": list(results), "failures": failures}
            )
        if failures:
            write_progress(
                output / "status.json",
                {"domain": args.domain, "status": "incomplete", "failures": list(failures)},
            )
            raise RuntimeError("Some training arms failed; successful checkpoints were preserved")
        selected = exp.select_candidate(setup["baseline_validation"], results, config)
        selection_file = output / "frozen_selection.json"
        frozen = {
            "selected": selected,
            "arms": results,
            "baseline": base,
            "identity_sha256": object_hash(identity),
            "profile_sha256": object_hash(profile),
            "baseline_setup_sha256": data.sha256(setup_path),
        }
        if selection_file.exists():
            if data.read_json(selection_file) != frozen:
                raise ValueError("Frozen Validation selection changed")
        else:
            runner.write_json(selection_file, frozen)
        evaluate(output, source, reader, spec, config, profile, meta, images, setup, frozen)


def evaluate(output, source, reader, spec, config, profile, meta, images, setup, frozen):
    import keras

    final = output / "final_evaluation"
    complete = final / "complete.json"
    if complete.exists():
        saved = data.read_json(complete)
        if data.sha256(final / "comparison.json") != saved["comparison_sha256"] or saved[
            "selection_sha256"
        ] != object_hash(frozen):
            raise ValueError("Final evaluation changed")
        publish(output, final, spec)
        return
    # Partial attempts are retained, never merged with a new final evaluation.
    attempt = output / ("evaluation_" + uuid.uuid4().hex[:8])
    attempt.mkdir()
    inventory = {
        "baseline": {
            "path": output / frozen["baseline"]["relative_path"],
            "sha256": frozen["baseline"]["model_sha256"],
        }
    }
    for arm, row in frozen["arms"].items():
        inventory[arm] = {
            "path": output / arm / row["model_relative"],
            "sha256": row["model_sha256"],
        }
    target_base = None
    results = {}
    test = source["original", "test"]
    sample_indexes = [
        i
        for label in range(spec["class_count"])
        for i in [j for j, row in enumerate(test) if row.label == label][:8]
    ]
    for name, checkpoint in inventory.items():
        if data.sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Frozen evaluation checkpoint changed")
        keras.backend.clear_session()
        model = keras.models.load_model(checkpoint["path"], compile=False)
        metrics = {}
        clean_features = None
        for condition, change, mode in (
            ("original_source_test", None, "expanded"),
            ("lighting_source_test", profile, "lighting"),
            ("expanded_source_test", profile, "expanded"),
        ):
            scores, features = predictions(
                model, test, reader, spec, config["batch_size"], change, mode
            )
            metrics[condition] = exp.classification(test, scores, spec["class_count"])
            arrays = {"scores": scores}
            if change is None:
                arrays["features"] = features
                clean_features = features[sample_indexes]
            np.savez_compressed(attempt / f"{name}_{condition}.npz", **arrays)
        scores, features = target_predictions(model, images, spec, config["batch_size"])
        if target_base is None:
            target_base = scores.argmax(1)
        mmd, coral = alignment.losses(clean_features, features, setup["reference"])
        metrics["target_descriptive"] = {
            "accuracy": None,
            "macro_f1": None,
            "independent_test": False,
            "changed_predictions": int(np.sum(scores.argmax(1) != target_base)),
            "mmd2_multikernel": float(mmd),
            "coral": float(coral),
            "meaning": "Target was calibration/training data; not generalization accuracy",
        }
        runner.write_json(
            attempt / f"{name}_target_predictions.json",
            [
                {
                    "path": row["path"],
                    "person": row["person"],
                    "label": None,
                    "predicted_index": int(score.argmax()),
                    "scores": score.tolist(),
                }
                for row, score in zip(meta["rows"], scores, strict=True)
            ],
        )
        runner.feature_plot(clean_features, features, attempt / f"{name}_feature_pca.svg")
        np.savez_compressed(attempt / f"{name}_target_features.npz", features=features)
        results[name] = metrics
        del model
        keras.backend.clear_session()
    report = {
        "selected_before_test": frozen["selected"],
        "results": results,
        "test_membership_preserved": True,
        "target_accuracy": None,
        "alignment_arm": "combined MMD + CORAL; individual effects are not separated",
    }
    runner.write_json(attempt / "comparison.json", report)
    runner.write_json(attempt / "source_test_paths.json", [asdict(r) for r in test])
    lines = [
        f"# {spec['domain']} 3개 실험 결과",
        "",
        f"Validation에서 고정한 선택: {frozen['selected']}",
        "",
        "| 모델 | 원본 Accuracy | 원본 Macro-F1 | 확장 변화 Accuracy | 확장 변화 Macro-F1 |",
        "|---|---|---|---|---|",
    ]
    for name, row in results.items():
        clean, shifted = row["original_source_test"], row["expanded_source_test"]
        values = [clean["accuracy"], clean["macro_f1"], shifted["accuracy"], shifted["macro_f1"]]
        lines.append("| " + name + " | " + " | ".join(str(v) for v in values) + " |")
    lines += [
        "",
        "Target 정답이 없어 실제 장비 정확도는 계산하지 않았다.",
        "정렬 조건은 확장 증강 + MMD·CORAL 결합이며 두 손실의 개별 기여는 분리하지 않는다.",
        "기준선 대비 조명 군의 차이에는 추가 학습 효과도 포함된다.",
        "expanded 대 lighting, aligned 대 expanded는 같은 추가 학습량으로 비교한다.",
    ]
    with (attempt / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    runner.write_json(
        attempt / "complete.json",
        {
            "comparison_sha256": data.sha256(attempt / "comparison.json"),
            "selection_sha256": object_hash(frozen),
        },
    )
    if final.exists() or not all(
        path.resolve().is_relative_to(output.resolve()) for path in (attempt, final)
    ):
        raise ValueError("Unexpected evaluation promotion path")
    attempt.rename(final)
    publish(output, final, spec)


def publish(output, final, spec):
    report = output / "RESULTS.md"
    if report.exists():
        if report.read_bytes() != (final / "RESULTS.md").read_bytes():
            raise ValueError("Published report changed")
    else:
        shutil.copyfile(final / "RESULTS.md", report)
    write_progress(
        output / "status.json",
        {
            "domain": spec["domain"],
            "status": "completed",
            "experiments": 3,
            "target_accuracy": None,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    for name in ("config", "source-zip", "review", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    run_domain(parser.parse_args())


if __name__ == "__main__":
    main()
