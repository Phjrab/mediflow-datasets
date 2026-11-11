"""Conservative image-component quarantine and fresh Skin/Web Skin pilot baselines.

Quarantine is not a declaration that all hash candidates are duplicates. Original
files and Test membership are preserved; offline augmented data is not used.
"""

from __future__ import annotations

import argparse
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from mediflow_datasets import common_engine as engine
from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets import domain_adaptation_preflight as preflight
from mediflow_datasets import domain_adaptation_review as review_tools
from mediflow_datasets.domain_adaptation_io import write_progress
from mediflow_datasets.domain_adaptation_training import object_hash


def quarantine_plan(domain, groups, image_manifest, classes, source_sha256, counts):
    if domain not in ("skin", "web_skin"):
        raise ValueError("Resolution is limited to Skin and Web Skin")
    graph = defaultdict(set)
    for group in groups:
        a, b = group["paths"]
        for path in (a, b):
            if path not in image_manifest:
                raise ValueError("Candidate path lacks verified image metadata")
        graph[a].add(b)
        graph[b].add(a)
    visited, components = set(), []
    excluded = {"train": set(), "val": set()}
    for node in sorted(graph):
        if node in visited:
            continue
        pending, component = [node], set()
        while pending:
            current = pending.pop()
            if current not in component:
                component.add(current)
                pending.extend(graph[current] - component)
        visited.update(component)
        active = sorted(p for p in component if "/original/" in p)
        if not active:
            continue
        splits = {image_manifest[p]["split"] for p in active}
        keep = next(s for s in ("test", "val", "train") if s in splits)
        for path in active:
            split = image_manifest[path]["split"]
            if split != keep:
                excluded[split].add(path)
        components.append({"paths": sorted(component), "retained_split": keep})
    expected_counts = {
        s: counts[f"original/{s}"] - len(excluded.get(s, ())) for s in data.SPLITS
    }
    return {
        "schema": "mediflow_image_quarantine_v1",
        "domain": domain, "source_sha256": source_sha256, "class_names": classes,
        "policy": "connected components; Test > Validation > Train; original images only",
        "candidate_groups_sha256": object_hash(groups),
        "candidate_groups": len(groups), "components": components,
        "exclude_original": {s: sorted(p) for s, p in excluded.items()},
        "excluded_by_class": {
            s: dict(Counter(classes[image_manifest[p]["label"]] for p in paths))
            for s, paths in excluded.items()
        },
        "expected_original_counts": expected_counts,
        "candidate_decisions": "quarantined conservatively; not automatically declared duplicates",
        "test_membership_changed": False,
        "offline_augmented_training": "not used, including untraceable derived images",
        "identity_independence_verified": False,
    }


def filtered_source(source, plan, classes, source_sha256):
    if (
        plan.get("schema") != "mediflow_image_quarantine_v1"
        or plan["class_names"] != classes or plan["source_sha256"] != source_sha256
        or plan.get("test_membership_changed") is not False
        or set(plan["exclude_original"]) != {"train", "val"}
    ):
        raise ValueError("Source quarantine contract differs from the archive")
    result = {}
    for split in data.SPLITS:
        rows = source["original", split]
        excluded = set(plan["exclude_original"].get(split, ()))
        if not excluded <= {r.path for r in rows}:
            raise ValueError("Quarantine contains an unknown or wrong-split path")
        kept = [r for r in rows if r.path not in excluded]
        if (
            len(kept) != plan["expected_original_counts"][split]
            or {r.label for r in kept} != set(range(len(classes)))
        ):
            raise ValueError("Quarantined split count changed or a class became empty")
        result["original", split] = kept
    return result


def calibrated_profile(domain, measurements, target, plan, old_profile):
    excluded = set(plan["exclude_original"]["train"])
    retained = [r for r in measurements if r["path"] not in excluded]
    if len(retained) != plan["expected_original_counts"]["train"]:
        raise ValueError("Cached Train measurements do not match the quarantined subset")
    profile = analysis.calibrate(domain, retained, target, brightness_policy="within_participant")
    profile.update(
        profile_version="quarantined_within_person_bounded_v4",
        photometric_operator="bounded_additive_v3",
        input_policy=old_profile["input_policy"],
        source_sha256=plan["source_sha256"],
        reference_model_sha256=old_profile["model_sha256"],
        calibration_fingerprint=old_profile["calibration_fingerprint"],
        source_subset_sha256=object_hash(plan),
        train_visual_review="pending",
    )
    return profile


def preview_from_exports(domain, previous, output, plan, profile):
    """Use retained original Train previews only, including the parameter-range corners."""
    output.mkdir(parents=True, exist_ok=False)
    records = data.read_json(previous / "transform_qa/measurements.json")
    excluded = set(plan["exclude_original"]["train"])
    items, results = [], []
    detector = analysis.face_detector()
    for index, row in enumerate(records):
        if "/original/train/" not in row["path"]:
            raise ValueError("Only original Train previews can inform transform settings")
        if row["path"] in excluded:
            continue
        before_path = previous / f"transform_qa/Q{index:04}_before.png"
        with Image.open(before_path) as opened:
            before = np.asarray(opened.convert("RGB"), dtype=np.float32)
        seed = exp.item_seed(data.ImageRow(row["path"], "train", row["label"]), 707)
        after = exp.transform_image(before, profile, "target_guided", seed)
        after_path = output / f"Q{index:04}_after.png"
        with after_path.open("xb") as stream:
            stream.write(preflight.png_bytes(after))
        metrics = analysis.measure(
            preflight.png_bytes(after), domain, before.shape[0], detector,
        )
        corners = []
        for offset, light, contrast in product(
            sorted(set(profile["brightness_offset"])), sorted(set(profile["left_light"])),
            sorted(set(profile["contrast"])),
        ):
            fixed = {**profile, "brightness_offset": [offset, offset],
                     "left_light": [light, light], "contrast": [contrast, contrast]}
            transformed = exp.transform_image(before, fixed, "target_guided", seed)
            corners.append(review_tools.clipping(before, transformed, None))
        result = {
            "id": f"Q{index:04}", "path": row["path"], "label": row["label"],
            "metrics": metrics, "clipping": review_tools.clipping(before, after, None),
            "corner_max_near_endpoint": max(r["new_near_endpoint_channels"] for r in corners),
            "corner_max_hard_endpoint": max(r["new_endpoint_channels"] for r in corners),
        }
        results.append(result)
        items.append({"images": [before_path, after_path],
                      "caption": f"Q{index:04} C{row['label']} / v4 retained Train"})
    pages = preflight.contact_pages(items, output, "Retained Train: original / measured bounded v4")
    analysis.distribution_plot({"v4 retained Train": analysis.statistics(results)},
                               output / "distributions.svg")
    summary = {
        "samples": len(results), "by_class": dict(Counter(r["label"] for r in results)),
        "pages": pages, "excluded_preview_count": len(records) - len(results),
        "max_new_hard_endpoint": max(r["clipping"]["new_endpoint_channels"] for r in results),
        "corner_max_new_hard_endpoint": max(r["corner_max_hard_endpoint"] for r in results),
        "corner_max_near_endpoint": max(r["corner_max_near_endpoint"] for r in results),
        "profile_sha256": object_hash(profile), "visual_review": "pending",
        "input_limitation": "uint8 model-input previews; all full originals re-audited at runtime",
    }
    runner.write_json(output / "measurements.json", results)
    runner.write_json(output / "summary.json", summary)
    return summary


def fresh_model(domain, spec, count, weights="imagenet"):
    if domain == "web_skin":
        from mediflow_datasets.web_skin_paper_suite import build_pmg_model

        return build_pmg_model(count, spec["input_size"], weights=weights, backbone_name="B0")
    if domain != "skin":
        raise ValueError("Fresh initialization is for Skin/Web Skin only")
    if weights != "imagenet":
        raise ValueError("Skin initialization requires ImageNet, never an old checkpoint")
    return engine.build_model({
        "backbone": "B0", "size": spec["input_size"], "class_count": count, "loss": "ce",
    })


def train_fresh_baseline(domain, spec, classes, source, archive, config, profile, directory):
    import keras

    directory.mkdir()
    keras.utils.set_random_seed(config["seed"])
    model = fresh_model(domain, spec, len(classes))
    best, history = None, []
    reader = lambda row: archive.read(row.path)  # noqa: E731
    for stage, epochs, rate in (
        ("head", config["baseline_head_epochs"], 1e-4),
        ("partial", config["baseline_partial_epochs"], 1e-5),
    ):
        if stage == "partial":
            del model
            keras.backend.clear_session()
            model = keras.models.load_model(directory / "best.keras", compile=False)
            exp.configure_partial(model)
        optimizer = keras.optimizers.Adam(rate)
        optimizer.build(model.trainable_variables)
        for epoch in range(epochs):
            loss = exp.train_epoch(
                model, source["original", "train"], reader, spec, config, profile,
                "source_only", len(history), optimizer,
            )
            scores, _ = exp.infer(
                model, source["original", "val"], reader, spec["input_size"],
                spec["output_adapter"], config["batch_size"], input_policy=spec["input_policy"],
            )
            metrics = exp.classification(source["original", "val"], scores, len(classes))
            history.append({"stage": stage, "epoch": epoch + 1, "loss": loss, **metrics})
            if best is None or metrics["macro_f1"] > best:
                best = metrics["macro_f1"]
                model.save(directory / "best.keras")
            write_progress(directory / "history.json", history)
            print("fresh baseline", domain, stage, epoch + 1, metrics["macro_f1"], flush=True)
    del model
    keras.backend.clear_session()
    runner.write_json(directory / "provenance.json", {
        "initialization": "ImageNet backbone plus randomly initialized task-specific layers",
        "old_mediflow_checkpoint_weights_used": False,
        "selection": "retained original Validation Macro-F1 only", "source_test_evaluated": False,
        "model_sha256": data.sha256(directory / "best.keras"),
    })
    return directory / "best.keras"


def run(args):
    domain = args.domain
    config = runner.load_config(args.config)
    if config.get("train_variant") != "original" or min(
        config["baseline_head_epochs"], config["baseline_partial_epochs"]
    ) < 1:
        raise ValueError("Fresh-baseline configuration must use original Train and both stages")
    reference_spec, _, classes = runner.task_candidate(domain, config)
    # Historical validation/test scores belong to the reference checkpoint, not the new model.
    spec = {key: reference_spec[key] for key in (
        "model_sha256", "input_size", "input_policy", "output_adapter", "backbone",
        "architecture", "normal_class_included",
    ) if key in reference_spec}
    spec["loss"] = config["tasks"][domain]["loss"]
    review = args.review
    plan, profile = (
        data.read_json(review / p) for p in ("quarantine_plan.json", "profile_v4.json")
    )
    receipt = data.read_json(review / "readiness.json")
    if (
        receipt.get("status") != "ready_for_fresh_baseline_pilot"
        or receipt.get("domain") != domain
        or receipt.get("profile_sha256") != object_hash(profile)
        or receipt.get("quarantine_plan_sha256") != object_hash(plan)
        or receipt.get("config_sha256") != object_hash(config)
        or profile.get("train_visual_review") != "approved"
    ):
        raise ValueError("Unreviewed or changed resolution package")
    for relative, digest in receipt["evidence_sha256"].items():
        if data.sha256(data.local_path(review, relative)) != digest:
            raise ValueError("Resolution review evidence changed")
    source, digest = data.source_index(args.source_zip, classes, plan["source_sha256"])
    source = filtered_source(source, plan, classes, digest)
    packet, target = runner.load_calibration_packet(
        args.calibration_file, domain, profile["reference_model_sha256"], spec["input_policy"],
    )
    if packet["fingerprint"] != profile["calibration_fingerprint"] or any(
        r.label is not None or r.split != "calibration" for r in target
    ):
        raise ValueError("Target calibration changed or received unsupported labels")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner.snapshot(output, config)
    runner.write_json(output / "readiness.json", receipt)
    runner.write_json(output / "source_subset.json", {
        "plan": plan, "rows": [asdict(r) for rows in source.values() for r in rows],
    })
    # Test pixels are used for image-leakage checks only at this point, never for tuning.
    with zipfile.ZipFile(args.source_zip) as archive:
        entries = [
            (row, *data.image_signatures(archive.read(row.path)))
            for rows in source.values() for row in rows
        ]
        audit = data.audit_groups(entries)
        source_pixels = {pixel for _, _, pixel, _ in entries}
        audit["source_target_pixel_overlap"] = len(source_pixels & set(packet["pixel_hashes"]))
        runner.write_json(output / "audit_after_quarantine.json", audit)
        if (
            audit["blocking_leaks_or_conflicts"] or audit["near_duplicate_review"]
            or audit["source_target_pixel_overlap"]
        ):
            raise ValueError("Unresolved retained-data leakage; training was not started")
        checkpoint = train_fresh_baseline(
            domain, spec, classes, source, archive, config, profile, output / "fresh_baseline",
        )
    model_sha = data.sha256(checkpoint)
    spec = {**spec, "reference_model_sha256": spec["model_sha256"], "model_sha256": model_sha}
    frozen_profile = {**profile, "model_sha256": model_sha}
    runner.write_json(output / "frozen_profile.json", frozen_profile)
    runner.write_json(output / "source_pixel_hashes.json", sorted(source_pixels))
    runner.write_json(output / "run_contract.json", {
        "domain": domain, "source_sha256": digest, "classes": classes, "spec": spec,
        "config": config, "baseline_model": str(checkpoint), "model_sha256": model_sha,
        "baseline_relative_path": "fresh_baseline/best.keras",
        "calibration_measurement_model_sha256": profile["reference_model_sha256"],
        "profile_sha256": data.sha256(output / "frozen_profile.json"),
        "source_subset_sha256": data.sha256(output / "source_subset.json"),
        "source_pixel_hashes_sha256": data.sha256(output / "source_pixel_hashes.json"),
        "target_calibration_fingerprint": packet["fingerprint"],
        "identity_limitations_acknowledged": True,
    })
    exp.training_run({
        "source_zip": args.source_zip, "source": source, "output": output,
        "config": config, "profile": frozen_profile, "spec": spec, "classes": classes,
        "model": checkpoint, "da_reason": runner.DA_REASON,
    })
    runner.write_json(output / "frozen_selection.json", {
        "selection_sha256": data.sha256(output / "validation_selection.json"),
        "contract_sha256": data.sha256(output / "run_contract.json"),
    })
    runner.finalize(SimpleNamespace(
        run=output, domain=domain, source_zip=args.source_zip,
        calibration_file=args.calibration_file, target=None, target_manifest=None,
    ))
    final = data.read_json(output / "final_evaluation/comparison.json")
    lines = [
        f"# {domain} 새 기준선·적응 비교 결과", "",
        f"Validation에서 고정한 선택: {final['selected_before_test']}", "",
        "| 모델 | 원본 Accuracy | 원본 Macro-F1 | 변화 Accuracy | 변화 Macro-F1 |",
        "|---|---|---|---|---|",
    ]
    for arm, metrics in final["results"].items():
        clean, shifted = metrics["original_source_test"], metrics["synthetic_shift_source_test"]
        values = [clean["accuracy"], clean["macro_f1"], shifted["accuracy"], shifted["macro_f1"]]
        lines.append("| " + arm + " | " + " | ".join(str(v) for v in values) + " |")
    lines += [
        "", "baseline은 새 데이터 목록과 ImageNet 초기값으로 새로 학습한 기준 모델이다.",
        "기존 보고 성능과는 학습·검증 목록과 이력이 달라 직접 향상 폭으로 계산하지 않는다.",
        "실제 사진에는 질환 정답이 없어 Target 정확도는 계산하지 않았다.",
        "후속 실제 사진 예측·특징 비교는 내려받은 모델로 PC에서 진행한다.",
        "사람·병변·세션 단위 독립성은 확인할 수 없으며 모든 누수의 배제를 주장하지 않는다.",
    ]
    with (output / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    runner.write_json(output / "status.json", {
        "status": "completed_fresh_baseline_adaptation_and_source_test", "domain": domain,
        "old_checkpoint_used_for_training": False, "original_test_preserved": True,
        "target_accuracy": None, "target_comparison": "local after download",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("skin", "web_skin"))
    for name in ("config", "source-zip", "calibration-file", "review", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
