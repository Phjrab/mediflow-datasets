"""Reproducible image analysis, calibrated training and sealed final evaluation.

Usage: python -m mediflow_datasets.domain_adaptation --help
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets.candidate_reproduction import ROOT, sha256

DA_REASON = (
    "Four previously inspected participants; disease labels and target class coverage unknown. "
    "Skin and Hair-5 have no normal class. Unconditional MMD/CORAL alignment could align disease "
    "content or remove discriminative features. MMD/CORAL remain diagnostics; supervised target "
    "fine-tuning needs verified target labels and new independent participants."
)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_config(path):
    config = data.read_json(path)
    if config.get("schema") != "mediflow_domain_adaptation_v1":
        raise ValueError("Unknown experiment configuration")
    for key in ("epochs", "batch_size"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 < config["learning_rate"] < 1 or not 0 <= config["augmentation_probability"] <= 1:
        raise ValueError("Invalid learning rate or augmentation probability")
    return config


def task_candidate(domain, config):
    spec, model, classes = data.candidate(domain)
    policy = config["tasks"][domain].get("input_policy", "legacy_resize")
    if policy not in ("legacy_resize", "center_square"):
        raise ValueError("Unknown common input policy")
    return {**spec, "input_policy": policy}, model, classes


def snapshot(output, config):
    import cv2
    import keras
    import PIL
    import tensorflow as tf

    def git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result.stdout if result.returncode == 0 else "unavailable"

    source = output / "source_code"
    source.mkdir()
    hashes = {}
    names = (
        "domain_adaptation",
        "domain_adaptation_data",
        "domain_adaptation_analysis",
        "domain_adaptation_experiment",
        "domain_adaptation_preflight",
        "domain_adaptation_review",
        "domain_adaptation_training",
        "domain_adaptation_resolution",
        "domain_adaptation_io",
        "candidate_reproduction",
        "common_engine",
        "common_workflow",
        "web_skin_paper_suite",
        "web_skin_wsdan",
    )
    for name in names:
        path = ROOT / "src" / "mediflow_datasets" / f"{name}.py"
        shutil.copyfile(path, source / path.name)
        hashes[path.name] = sha256(path)
    write_json(
        output / "reproduction.json",
        {
            "config": config,
            "commit": git("rev-parse", "HEAD").strip(),
            "working_tree_status": git("status", "--short"),
            "uncommitted_diff": git("diff", "--", "src", "configs"),
            "code_sha256": hashes,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "tensorflow": tf.__version__,
                "keras": keras.__version__,
                "pillow": PIL.__version__,
                "opencv": cv2.__version__,
            },
            "devices": [str(item) for item in tf.config.list_physical_devices()],
        },
    )


def write_measured_csv(path, rows):
    metrics = sorted({key for row in rows for key in row["metrics"]})
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["kind", "path", "split", "person", "condition", "label", *metrics]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {**{key: row.get(key) for key in writer.fieldnames[:6]}, **row["metrics"]}
            )


def feature_plot(source, target, path):
    import html

    combined = np.concatenate([source, target])
    centered = combined - combined.mean(0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    xy = u[:, :2] * s[:2]
    lo, hi = xy.min(0), xy.max(0)
    points = (xy - lo) / np.maximum(hi - lo, 1e-9) * [680, 400] + [40, 45]
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="500">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="22">PCA — same frozen classifier, exploratory samples</text>',
    ]
    for i, point in enumerate(points):
        color = "#1769aa" if i < len(source) else "#cf4a23"
        name = "Source" if i < len(source) else "Target calibration"
        out.append(
            f'<circle cx="{point[0]}" cy="{point[1]}" r="4" fill="{color}">'
            f"<title>{html.escape(name)}</title></circle>"
        )
    out.append('<text x="20" y="480">Blue = Source; orange = Target calibration</text></svg>')
    path.write_text("\n".join(out), encoding="utf-8")


def baseline_inference(
    output,
    spec,
    model_path,
    source_rows,
    target_rows,
    source_reader,
    target_reader,
    classes,
    config,
    precomputed_target=None,
):
    import keras

    from mediflow_datasets import domain_adaptation_experiment as exp

    model = keras.models.load_model(model_path, compile=False)
    source = []
    for label in range(len(classes)):
        eligible = [r for r in source_rows if r.label == label]
        source.extend(sorted(eligible, key=lambda r: exp.item_seed(r, config["seed"]))[:8])
    # Examples have no validated ground-truth class; retain them for feature diagnostics only.
    if not source:
        source = source_rows
    source_scores, sf = exp.infer(
        model,
        source,
        source_reader,
        spec["input_size"],
        spec["output_adapter"],
        config["batch_size"],
        input_policy=spec.get("input_policy", "legacy_resize"),
    )
    if precomputed_target is None:
        target_scores, tf = exp.infer(
            model,
            target_rows,
            target_reader,
            spec["input_size"],
            spec["output_adapter"],
            config["batch_size"],
            input_policy=spec.get("input_policy", "legacy_resize"),
        )
    else:
        target_scores, tf = precomputed_target
    np.savez_compressed(
        output / "baseline_features.npz",
        source=sf,
        target=tf,
        source_scores=source_scores,
        target_scores=target_scores,
    )
    feature_plot(sf, tf, output / "feature_pca.svg")
    reference = exp.feature_reference(sf)
    people = defaultdict(list)
    for row, feature in zip(target_rows, tf, strict=True):
        people[row.person].append(feature)
    subject_features = np.asarray([np.mean(group, axis=0) for group in people.values()])
    diagnostics = {
        "sample_features": exp.feature_distance(sf, tf, reference),
        "participant_means": exp.feature_distance(sf, subject_features, reference),
        "source_paths": [asdict(row) for row in source],
        "target_paths": [asdict(row) for row in target_rows],
        "feature_purpose": "exploratory; unmatched class content and only four target participants",
        "target_accuracy": None,
    }
    write_json(output / "feature_diagnostics.json", diagnostics)
    write_json(
        output / "baseline_target_predictions.json",
        [
            {
                "path": row.path,
                "person": row.person,
                "condition": row.condition,
                "predicted_index": int(score.argmax()),
                "predicted_label": classes[int(score.argmax())],
                "scores": score.tolist(),
                "label": None,
                "status": "unlabeled_calibration",
            }
            for row, score in zip(target_rows, target_scores, strict=True)
        ],
    )


def load_calibration_packet(path, domain, model_sha256, input_policy="legacy_resize"):
    packet = data.read_json(path)
    if packet["domain"] != domain or packet["model_sha256"] != model_sha256:
        raise ValueError("Calibration packet domain or model mismatch")
    if packet.get("input_policy", "legacy_resize") != input_policy:
        raise ValueError("Calibration packet preprocessing differs from the experiment")
    if packet["role"] != "calibration" or len(packet["people"]) != 4:
        raise ValueError("Unexpected calibration roles or participants")
    rows = [data.ImageRow(**row) for row in packet["rows"]]
    if {row.person for row in rows} != set(packet["people"]) or any(
        row.split != "calibration" or row.label is not None for row in rows
    ):
        raise ValueError("Pilot packet must contain unlabeled calibration rows only")
    return packet, rows


def analyze(args):
    from mediflow_datasets.domain_adaptation_data import ImageRow

    config = load_config(args.config)
    task = config["tasks"][args.domain]
    spec, model, classes = task_candidate(args.domain, config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot(output, config)
    packet = None
    if args.calibration_file:
        packet, target = load_calibration_packet(
            args.calibration_file, args.domain, spec["model_sha256"], spec["input_policy"]
        )
    else:
        target = data.target_rows(args.target.resolve(), args.domain, classes, args.target_manifest)
    calibration = [row for row in target if row.split == "calibration"]
    target_entries = (
        []
        if packet
        else [
            (row, *data.image_signatures(data.local_path(args.target, row.path).read_bytes()))
            for row in target
        ]
    )
    target_audit = packet["audit"] if packet else data.audit_groups(target_entries)
    write_json(output / "target_audit.json", target_audit)
    if target_audit["blocking_leaks_or_conflicts"] or target_audit["near_duplicate_review"]:
        raise ValueError("Target cross-split overlap requires review; see target_audit.json")
    detector = analysis.face_detector()
    measured = []

    def add_measurement(row, payload, kind):
        measured.append(
            {
                "kind": kind,
                **asdict(row),
                "metrics": analysis.measure(
                    payload,
                    args.domain,
                    spec["input_size"],
                    detector,
                    row.face_box,
                    input_policy=spec["input_policy"],
                ),
            }
        )

    if packet:
        measured.extend(packet["measurements"])
    else:
        for row in calibration:
            add_measurement(row, data.local_path(args.target, row.path).read_bytes(), "target")
    archive = None
    source_digest = None
    if args.source_zip:
        source, source_digest = data.source_index(
            args.source_zip, classes, task["source_sha256"], args.provenance
        )
        source_audit, pixels = data.audit_source(
            args.source_zip, source, read_near_reviews(args.near_review)
        )
        source_audit["source_target_exact_overlap"] = [
            row.path for row, _, pixel, _ in target_entries if pixel in pixels
        ]
        if packet:
            source_audit["source_target_exact_overlap"] = sorted(
                pixels & set(packet["pixel_hashes"])
            )
        write_json(output / "source_audit.json", source_audit)
        archive = zipfile.ZipFile(args.source_zip)
        source_rows = source["original", "train"]
        for row in source_rows:
            add_measurement(row, archive.read(row.path), "source")
        reader = lambda row: archive.read(row.path)  # noqa: E731
        observed_source = [row for row in measured if row["kind"] == "source"]
        profile = analysis.calibrate(
            args.domain, observed_source, [row for row in measured if row["kind"] == "target"]
        )
        profile.update(
            source_sha256=source_digest,
            model_sha256=spec["model_sha256"],
            calibration_fingerprint=packet["fingerprint"]
            if packet
            else target_fingerprint(args.target, calibration),
            selected_from="verified_source_train_and_target_calibration",
            train_visual_review="pending",
            input_policy=spec["input_policy"],
        )
        write_json(output / "measured_profile.json", profile)
    else:
        source_rows = [
            ImageRow(path.relative_to(ROOT).as_posix(), "reference_unknown")
            for path in sorted((ROOT / "data_examples" / args.domain).iterdir())
            if path.suffix.lower() in data.EXTENSIONS and not path.name.startswith("스크린샷")
        ]
        reader = lambda row: data.local_path(ROOT, row.path).read_bytes()  # noqa: E731
        for row in source_rows:
            add_measurement(row, reader(row), "example_reference")
    try:
        source_kind = "source" if args.source_zip else "example_reference"
        sources = [row for row in measured if row["kind"] == source_kind]
        targets = [row for row in measured if row["kind"] == "target"]
        groups = {
            "Source Train" if args.source_zip else "Examples (split unknown)": analysis.statistics(
                sources
            ),
            **analysis.grouped_target(targets),
        }
        analysis.distribution_plot(groups, output / "metric_distributions.svg")
        write_measured_csv(output / "image_measurements.csv", measured)
        write_json(
            output / "analysis.json",
            {
                "domain": args.domain,
                "source_status": "verified_archive" if args.source_zip else "examples_only",
                "source_sha256": source_digest,
                "groups": groups,
                "images": measured,
                "source_class_groups": {
                    classes[label]: analysis.statistics(
                        [row for row in sources if row["label"] == label]
                    )
                    for label in range(len(classes))
                    if any(row["label"] == label for row in sources)
                },
                "target_role": "calibration; all four people were previously inspected",
                "target_test_count": sum(row.split == "test" for row in target),
                "target_accuracy": None,
                "domain_alignment": {"status": "deferred", "reason": DA_REASON},
            },
        )
        baseline_inference(
            output,
            spec,
            model,
            source_rows,
            calibration,
            reader,
            lambda row: data.local_path(args.target, row.path).read_bytes(),
            classes,
            config,
            (np.asarray(packet["scores"]), np.asarray(packet["features"])) if packet else None,
        )
    finally:
        if archive is not None:
            archive.close()
    write_json(
        output / "completed.json",
        {"mode": "analysis", "domain": args.domain, "status": "completed"},
    )
    print(output, flush=True)


def target_fingerprint(root, rows):
    return sha256_text(
        [{**asdict(row), "sha256": sha256(data.local_path(root, row.path))} for row in rows]
    )


def sha256_text(value):
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_near_reviews(path):
    if not path:
        return ()
    pairs = data.read_json(path)
    if any(row.get("decision") != "distinct" or not row.get("reason") for row in pairs):
        raise ValueError("Near duplicate clearance needs decision=distinct and a reason per pair")
    return [row["paths"] for row in pairs]


def train(args):
    from mediflow_datasets.domain_adaptation_experiment import training_run

    config = load_config(args.config)
    task = config["tasks"][args.domain]
    spec, model, classes = task_candidate(args.domain, config)
    profile = data.read_json(args.analysis / "measured_profile.json")
    if profile.get("train_visual_review") != "approved" or not profile.get("review_note"):
        raise ValueError(
            "Review transformed Source Train images and record approval with a note first"
        )
    if profile["model_sha256"] != spec["model_sha256"]:
        raise ValueError("Analysis and training model differ")
    if profile.get("input_policy", "legacy_resize") != spec["input_policy"]:
        raise ValueError("Analysis and training preprocessing differ")
    packet = None
    if args.calibration_file:
        packet, rows = load_calibration_packet(
            args.calibration_file, args.domain, spec["model_sha256"], spec["input_policy"]
        )
    else:
        rows = data.pilot_rows(args.target.resolve(), args.domain)
    fingerprint = packet["fingerprint"] if packet else target_fingerprint(args.target, rows)
    if profile["calibration_fingerprint"] != fingerprint:
        raise ValueError("Target calibration changed after analysis")
    source, digest = data.source_index(
        args.source_zip, classes, task["source_sha256"], args.provenance
    )
    if digest != profile["source_sha256"]:
        raise ValueError("Source ZIP changed since calibration")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot(output, config)
    audit, source_pixels = data.audit_source(
        args.source_zip, source, read_near_reviews(args.near_review)
    )
    target_pixels = (
        set(packet["pixel_hashes"])
        if packet
        else {
            data.image_signatures(data.local_path(args.target, r.path).read_bytes())[1]
            for r in rows
        }
    )
    audit["source_target_pixel_overlap"] = len(source_pixels & target_pixels)
    write_json(output / "leakage_audit.json", audit)
    if (
        audit["blocking_leaks_or_conflicts"]
        or audit["near_duplicate_review"]
        or not all(audit["original_augmented_consistency"].values())
        or audit["source_target_pixel_overlap"]
    ):
        raise ValueError("Unresolved data leakage findings; see leakage_audit.json")
    if (
        not all(v["complete"] for v in audit["identity_metadata"].values())
        and not args.acknowledge_incomplete_identities
    ):
        raise ValueError(
            "Person/lesion/session/augmentation origins are incomplete; acknowledgement needed"
        )
    write_json(output / "frozen_profile.json", profile)
    write_json(output / "source_pixel_hashes.json", sorted(source_pixels))
    write_json(
        output / "run_contract.json",
        {
            "domain": args.domain,
            "source_zip": str(args.source_zip.resolve()),
            "source_sha256": digest,
            "baseline_model": str(model),
            "model_sha256": spec["model_sha256"],
            "classes": classes,
            "spec": spec,
            "config": config,
            "profile_sha256": sha256(output / "frozen_profile.json"),
            "source_pixel_hashes_sha256": sha256(output / "source_pixel_hashes.json"),
            "target_calibration_fingerprint": fingerprint,
            "identity_limitations_acknowledged": args.acknowledge_incomplete_identities,
        },
    )
    training_run(
        {
            "source_zip": args.source_zip,
            "source": source,
            "output": output,
            "config": config,
            "profile": profile,
            "spec": {**spec, "loss": task["loss"]},
            "classes": classes,
            "model": model,
            "da_reason": DA_REASON,
        }
    )
    write_json(
        output / "frozen_selection.json",
        {
            "selection_sha256": sha256(output / "validation_selection.json"),
            "contract_sha256": sha256(output / "run_contract.json"),
        },
    )


def preview_train(args):
    from PIL import Image, ImageDraw

    from mediflow_datasets import domain_adaptation_experiment as exp

    config = load_config(args.config)
    spec, _, classes = task_candidate(args.domain, config)
    source, _ = data.source_index(
        args.source_zip, classes, config["tasks"][args.domain]["source_sha256"]
    )
    profile = data.read_json(args.analysis / "measured_profile.json")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    chosen = [
        next(row for row in source["original", "train"] if row.label == label)
        for label in range(len(classes))
    ]
    sheet = Image.new("RGB", (400, len(chosen) * 210 + 25), "#19232c")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), "Source Train / measured target-guided transformation", fill="white")
    manifest = []
    with zipfile.ZipFile(args.source_zip) as archive:
        for i, row in enumerate(chosen):
            before = exp.image_tensor(
                archive.read(row.path), spec["input_size"], spec["input_policy"]
            )
            seed = exp.item_seed(row, config["seed"])
            after = exp.transform_image(before, profile, "target_guided", seed)
            for j, image in enumerate((before, after)):
                tile = Image.fromarray(np.rint(image).astype(np.uint8)).resize((192, 192))
                sheet.paste(tile, (j * 200, 25 + i * 210))
            draw.text((5, 25 + i * 210 + 194), f"C{row.label} / fixed seed", fill="white")
            manifest.append(
                {
                    **asdict(row),
                    "seed": seed,
                    "newly_clipped_fraction": float(
                        np.mean(((after == 0) | (after == 255)) & ((before > 0) & (before < 255)))
                    ),
                }
            )
    sheet.save(output / "source_train_before_after.png")
    write_json(output / "preview_paths.json", manifest)


def baseline_checkpoint(run, contract):
    relative = contract.get("baseline_relative_path")
    if relative is not None:
        return data.local_path(Path(run), relative)
    return data.candidate(contract["domain"])[1]


def compare_target(args):
    import keras

    from mediflow_datasets import domain_adaptation_experiment as exp

    contract = data.read_json(args.run / "run_contract.json")
    selection = data.read_json(args.run / "validation_selection.json")
    if contract["domain"] != args.domain:
        raise ValueError("Task mismatch")
    target = data.target_rows(
        args.target.resolve(), args.domain, contract["classes"], args.target_manifest
    )
    if (
        target_fingerprint(args.target, [r for r in target if r.split == "calibration"])
        != contract["target_calibration_fingerprint"]
    ):
        raise ValueError("Target calibration changed")
    report = data.audit_groups(
        [
            (row, *data.image_signatures(data.local_path(args.target, row.path).read_bytes()))
            for row in target
        ]
    )
    if report["blocking_leaks_or_conflicts"] or report["near_duplicate_review"]:
        raise ValueError("Unresolved Target evaluation overlap")
    verify_new_target_overlap(
        args.run, contract, target, lambda row: data.local_path(args.target, row.path).read_bytes()
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline_scores = None
    feature_reference = None
    results = {}
    for arm, item in {
        "baseline": {"model_sha256": contract["model_sha256"]},
        **selection["arms"],
    }.items():
        path = (
            baseline_checkpoint(args.run, contract)
            if arm == "baseline" else args.run / arm / "best.keras"
        )
        if sha256(path) != item["model_sha256"]:
            raise ValueError("Checkpoint changed")
        model = keras.models.load_model(path, compile=False)
        scores, features = exp.infer(
            model,
            target,
            lambda r: data.local_path(args.target, r.path).read_bytes(),
            contract["spec"]["input_size"],
            contract["spec"]["output_adapter"],
            contract["config"]["batch_size"],
            input_policy=contract["spec"].get("input_policy", "legacy_resize"),
        )
        if arm == "baseline":
            baseline_scores = scores
        test_indexes = [i for i, r in enumerate(target) if r.split == "test"]
        results[arm] = {
            "target_test": exp.classification(
                [target[i] for i in test_indexes], scores[test_indexes], len(contract["classes"])
            ),
            "changed_predictions": int(np.sum(scores.argmax(1) != baseline_scores.argmax(1))),
            "interpretation": "scores and prediction changes need labels for accuracy",
        }
        source_features = args.run / "final_evaluation" / f"{arm}_original_source_test.npz"
        source_paths = args.run / "final_evaluation" / "source_test_paths.json"
        if source_features.is_file() and source_paths.is_file():
            inventory = data.read_json(source_paths)
            indexes = [
                index
                for label in range(len(contract["classes"]))
                for index in [i for i, row in enumerate(inventory) if row["label"] == label][:8]
            ]
            with np.load(source_features) as arrays:
                source_sample = arrays["features"][indexes]
            if arm == "baseline":
                feature_reference = exp.feature_reference(source_sample)
            results[arm]["feature_distance"] = exp.feature_distance(
                source_sample, features, feature_reference
            )
            feature_plot(source_sample, features, output / f"{arm}_feature_pca.svg")
        write_json(
            output / f"{arm}_predictions.json",
            [
                {**asdict(row), "scores": score.tolist(), "predicted_index": int(score.argmax())}
                for row, score in zip(target, scores, strict=True)
            ],
        )
        np.savez_compressed(output / f"{arm}_features.npz", features=features)
        del model
        keras.backend.clear_session()
    write_json(output / "comparison.json", results)


def verify_new_target_overlap(run, contract, target, reader):
    test = [row for row in target if row.split == "test"]
    if not test:
        return
    path = run / "source_pixel_hashes.json"
    if sha256(path) != contract["source_pixel_hashes_sha256"]:
        raise ValueError("Source image fingerprint inventory changed")
    source_pixels = set(data.read_json(path))
    if any(data.image_signatures(reader(row))[1] in source_pixels for row in test):
        raise ValueError("Target Test contains an image from the Source dataset")


def final_target_metrics(model, target, reader, contract, output, arm, baseline, clean, reference):
    from mediflow_datasets import domain_adaptation_experiment as exp

    if reader is None:
        return {
            "status": "awaiting_local_compare_target",
            "accuracy": None,
            "macro_f1": None,
        }, baseline
    scores, features = exp.infer(
        model,
        target,
        reader,
        contract["spec"]["input_size"],
        contract["spec"]["output_adapter"],
        contract["config"]["batch_size"],
        input_policy=contract["spec"].get("input_policy", "legacy_resize"),
    )
    if baseline is None:
        baseline = scores.argmax(1)
    test_indexes = [i for i, row in enumerate(target) if row.split == "test"]
    metrics = {
        "target_test": exp.classification(
            [target[i] for i in test_indexes], scores[test_indexes], len(contract["classes"])
        ),
        "prediction_change_count": int(np.sum(scores.argmax(1) != baseline)),
        "feature_distance": exp.feature_distance(clean, features, reference),
    }
    write_json(
        output / f"{arm}_target_predictions.json",
        [
            {**asdict(row), "predicted_index": int(score.argmax()), "scores": score.tolist()}
            for row, score in zip(target, scores, strict=True)
        ],
    )
    np.savez_compressed(output / f"{arm}_target_features.npz", features=features)
    return metrics, baseline


def finalize(args):
    import keras

    from mediflow_datasets import domain_adaptation_experiment as exp

    contract = data.read_json(args.run / "run_contract.json")
    selection = data.read_json(args.run / "validation_selection.json")
    frozen = data.read_json(args.run / "frozen_selection.json")
    if (
        sha256(args.run / "run_contract.json") != frozen["contract_sha256"]
        or sha256(args.run / "validation_selection.json") != frozen["selection_sha256"]
    ):
        raise ValueError("Selection or contract changed after validation")
    if contract["domain"] != args.domain:
        raise ValueError("Task mismatch")
    if contract.get("source_subset_sha256") and (
        sha256(args.run / "source_subset.json") != contract["source_subset_sha256"]
    ):
        raise ValueError("Frozen Source subset changed")
    source, _ = data.source_index(args.source_zip, contract["classes"], contract["source_sha256"])
    profile_file = args.run / "frozen_profile.json"
    if sha256(profile_file) != contract["profile_sha256"]:
        raise ValueError("Frozen transform profile changed")
    output = args.run / "final_evaluation"
    output.mkdir(exist_ok=False)
    write_json(output / "selection_before_test.json", selection)
    profile = data.read_json(profile_file)
    result = {}
    if args.calibration_file:
        packet, target = load_calibration_packet(
            args.calibration_file,
            args.domain,
            contract.get("calibration_measurement_model_sha256", contract["model_sha256"]),
            contract["spec"].get("input_policy", "legacy_resize"),
        )
        target_reader = None
        target_audit = packet["audit"]
        fingerprint = packet["fingerprint"]
    else:
        target = data.target_rows(
            args.target.resolve(), args.domain, contract["classes"], args.target_manifest
        )
        target_reader = lambda row: data.local_path(args.target, row.path).read_bytes()  # noqa: E731
        target_audit = data.audit_groups(
            [(row, *data.image_signatures(target_reader(row))) for row in target]
        )
        fingerprint = target_fingerprint(
            args.target, [r for r in target if r.split == "calibration"]
        )
    if target_audit["blocking_leaks_or_conflicts"] or target_audit["near_duplicate_review"]:
        raise ValueError("Target evaluation overlaps calibration; review target manifest")
    if target_reader is not None:
        verify_new_target_overlap(args.run, contract, target, target_reader)
    if fingerprint != contract["target_calibration_fingerprint"]:
        raise ValueError("Target calibration changed")
    baseline_target = None
    feature_ref = None
    paths = {
        "baseline": {"model": contract["baseline_model"], "model_sha256": contract["model_sha256"]},
        **selection["arms"],
    }
    with zipfile.ZipFile(args.source_zip) as archive:
        test = source["original", "test"]
        diagnostic_indexes = [
            index
            for label in range(len(contract["classes"]))
            for index in [i for i, row in enumerate(test) if row.label == label][:8]
        ]
        for arm, item in paths.items():
            checkpoint = (
                baseline_checkpoint(args.run, contract)
                if arm == "baseline"
                else args.run / arm / "best.keras"
            )
            if sha256(checkpoint) != item["model_sha256"]:
                raise ValueError("Frozen evaluation checkpoint changed")
            model = keras.models.load_model(checkpoint, compile=False)
            metrics = {}
            for key, transform in (
                ("original_source_test", None),
                ("synthetic_shift_source_test", profile),
            ):
                scores, features = exp.infer(
                    model,
                    test,
                    lambda r: archive.read(r.path),
                    contract["spec"]["input_size"],
                    contract["spec"]["output_adapter"],
                    contract["config"]["batch_size"],
                    transform,
                    input_policy=contract["spec"].get("input_policy", "legacy_resize"),
                )
                metrics[key] = exp.classification(test, scores, len(contract["classes"]))
                np.savez_compressed(output / f"{arm}_{key}.npz", scores=scores, features=features)
                if arm == "baseline" and transform is None:
                    feature_ref = exp.feature_reference(features[diagnostic_indexes])
            clean_features = np.load(output / f"{arm}_original_source_test.npz")["features"]
            metrics["target"], baseline_target = final_target_metrics(
                model,
                target,
                target_reader,
                contract,
                output,
                arm,
                baseline_target,
                clean_features[diagnostic_indexes],
                feature_ref,
            )
            result[arm] = metrics
            del model
            keras.backend.clear_session()
    write_json(output / "source_test_paths.json", [asdict(row) for row in test])
    write_json(
        output / "comparison.json",
        {
            "selected_before_test": selection["selected"],
            "results": result,
            "target_test_count": sum(row.split == "test" for row in target),
            "meaning": "synthetic Source Test; labeled independent Target Test still required",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("analyze", "preview", "train", "finalize", "compare-target")
    )
    parser.add_argument("--domain", required=True, choices=("web_skin", "skin", "hair"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/domain_adaptation.json")
    parser.add_argument("--target", type=Path, default=ROOT / "camera_capture")
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--near-review", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--acknowledge-incomplete-identities", action="store_true")
    args = parser.parse_args()
    if args.mode in ("analyze", "preview", "train", "compare-target") and args.output is None:
        parser.error("--output must be a new directory")
    if args.mode in ("preview", "train", "finalize") and args.source_zip is None:
        parser.error("--source-zip must be a verified existing dataset")
    if args.mode in ("preview", "train") and args.analysis is None:
        parser.error("--analysis is required")
    if args.mode in ("finalize", "compare-target") and args.run is None:
        parser.error("--run is required")
    {
        "analyze": analyze,
        "preview": preview_train,
        "train": train,
        "finalize": finalize,
        "compare-target": compare_target,
    }[args.mode](args)


if __name__ == "__main__":
    main()
