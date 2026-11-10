"""Prepare reversible Skin/Web Skin quarantine plans and reviewable transformation previews."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_resolution as resolution
from mediflow_datasets.candidate_reproduction import ROOT


def prepare(output, preflight, previous, bundle):
    output.mkdir(parents=True, exist_ok=False)
    config = data.read_json(ROOT / "configs/domain_adaptation.json")
    config["tasks"] = {d: config["tasks"][d] for d in ("skin", "web_skin")}
    config.update(
        experiment_revision="original_only_component_quarantine_fresh_baseline_v4",
        train_variant="original", baseline_head_epochs=15, baseline_partial_epochs=10,
    )
    summaries = {}
    with zipfile.ZipFile(bundle) as archive:
        for domain in config["tasks"]:
            task = config["tasks"][domain]
            task.update(
                input_policy="center_square" if domain == "web_skin" else "legacy_resize",
                geometry=False,
            )
            _, _, classes = data.candidate(domain)
            old = data.read_json(previous / domain / "analysis/analysis.json")
            audit = data.read_json(previous / domain / "analysis/source_audit.json")
            if old["source_sha256"] != task["source_sha256"]:
                raise ValueError("Cached source measurements and archive pin differ")
            near = preflight / domain / "near_candidates"
            plan = resolution.quarantine_plan(
                domain, data.read_json(near / "candidate_groups.json"),
                data.read_json(near / "image_manifest.json"), classes,
                task["source_sha256"], audit["source_counts"],
            )
            packet = json.loads(archive.read(f"calibration/{domain}.json"))
            source_measurements = [r for r in old["images"] if r["kind"] == "source"]
            old_profile = data.read_json(preflight / domain / "profile_v2.json")
            profile = resolution.calibrated_profile(
                domain, source_measurements, packet["measurements"], plan, old_profile,
            )
            directory = output / domain
            directory.mkdir()
            runner.write_json(directory / "quarantine_plan.json", plan)
            runner.write_json(directory / "profile_v4.json", profile)
            runner.write_json(directory / "source_measurements.json", source_measurements)
            runner.write_json(directory / "class_names.json", classes)
            qa = resolution.preview_from_exports(
                domain, preflight / domain, directory / "transform_qa", plan, profile,
            )
            summaries[domain] = {
                "retained_counts": plan["expected_original_counts"],
                "excluded_counts": {s: len(p) for s, p in plan["exclude_original"].items()},
                "profile": {k: profile[k] for k in (
                    "brightness_offset", "contrast", "left_light", "input_policy",
                )},
                "qa": qa, "training_executed": False,
            }
    runner.write_json(output / "resolution_config.json", config)
    runner.write_json(output / "preparation_summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight", type=Path, default=ROOT / (
        "camera_capture/_review/preflight_v2_fbe37d44"
    ))
    parser.add_argument("--previous", type=Path, default=ROOT / (
        "camera_capture/_review/domain_adaptation_colab_20261109"
    ))
    parser.add_argument("--bundle", type=Path, default=ROOT / (
        "output/adaptation_preflight_v2_20261109/mediflow_preflight_runtime.zip"
    ))
    args = parser.parse_args()
    prepare(args.output, args.preflight, args.previous, args.bundle)
