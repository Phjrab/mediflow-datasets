"""Execute the reviewed Hair pilot, retaining all original data and sealed selection."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets.candidate_reproduction import sha256


def object_hash(value):
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def verify_readiness(plan_path, config, profile, spec, classes):
    """A review receipt approves a bounded pilot, never missing identities or accuracy."""
    plan = data.read_json(plan_path)
    if plan.get("schema") != "mediflow_reviewed_training_v1":
        raise ValueError("Unknown training readiness schema")
    if plan.get("domain") != "hair" or plan.get("status") != "ready_for_pilot_training":
        raise ValueError("Only the reviewed Hair pilot is ready for training")
    expected = {
        "profile_sha256": object_hash(profile),
        "config_sha256": object_hash(config),
        "model_sha256": spec["model_sha256"],
        "class_names": classes,
        "input_policy": spec["input_policy"],
        "source_sha256": config["tasks"]["hair"]["source_sha256"],
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Training settings differ from the reviewed plan")
    if (
        plan.get("reviewed_train_previews") != 120
        or plan.get("reviewed_contact_pages") != list(range(1, 16))
        or plan.get("target_labels_verified") is not False
        or plan.get("identity_independence_verified") is not False
        or profile.get("train_visual_review") != "approved"
        or profile.get("photometric_operator") != "bounded_additive_v3"
    ):
        raise ValueError("Incomplete or misleading Hair review receipt")
    if exp.arm_plan(profile) != (
        ("source_only", "brightness_only"),
        {"target_guided": "brightness_only"},
    ):
        raise ValueError("Hair pilot must contain only the reviewed brightness variable")
    evidence = plan.get("evidence_sha256", {})
    if not evidence:
        raise ValueError("Missing review evidence")
    for relative, expected_digest in evidence.items():
        path = data.local_path(Path(plan_path).parent, relative)
        if sha256(path) != expected_digest:
            raise ValueError("Review evidence changed: " + relative)
    return plan


def comparison_report(output):
    """Show exact saved measurements; final Test results do not alter the selection."""
    training = output / "training"
    selected = data.read_json(training / "validation_selection.json")
    final = data.read_json(training / "final_evaluation/comparison.json")
    lines = [
        "# Hair 촬영 밝기 적응 파일럿 결과",
        "",
        f"Validation으로 고정한 선택: `{selected['selected']}`",
        "",
        "| 모델 | 원본 Test Accuracy | 원본 Test Macro-F1 | "
        "변화 Test Accuracy | 변화 Test Macro-F1 |",
        "|---|---|---|---|---|",
    ]
    for name, row in final["results"].items():
        clean = row["original_source_test"]
        shifted = row["synthetic_shift_source_test"]
        values = [clean["accuracy"], clean["macro_f1"], shifted["accuracy"], shifted["macro_f1"]]
        lines.append("| " + name + " | " + " | ".join(str(v) for v in values) + " |")
    lines += [
        "",
        "baseline은 기존 저장 모델, source_only는 기존 방식의 추가 학습 대조군,",
        "brightness_only는 촬영 측정값을 반영한 추가 학습이다.",
        "target_guided는 brightness_only와 동일하므로 중복 학습하지 않았다.",
        "클래스별 Recall·Confusion Matrix는 training/final_evaluation/comparison.json에 있다.",
        "",
        "Test 결과로 선택을 바꾸지 않는다. 변화 Test는 설정한 촬영 변화에 대한 시험이다.",
        "실제 사진에는 질환 정답이 없어 Target Accuracy·Macro-F1은 계산하지 않았다.",
        "실제 촬영 사진 원본은 업로드하지 않았다. 결과를 PC로 내려받은 뒤 실제 사진의",
        "예측 변화와 특징 거리를 로컬 compare-target 명령으로 별도 측정한다.",
        "사람·병변·세션 및 일부 증강 원본 정보 부재로 모든 누수의 배제를 보장하지 않는다.",
    ]
    with (output / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def run(args):
    config = runner.load_config(args.config)
    spec, _, classes = runner.task_candidate("hair", config)
    profile = data.read_json(args.analysis / "measured_profile.json")
    plan = verify_readiness(args.plan, config, profile, spec, classes)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner.write_json(output / "reviewed_plan.json", plan)
    common = {
        "domain": "hair",
        "config": args.config,
        "source_zip": args.source_zip,
        "calibration_file": args.calibration_file,
        "analysis": args.analysis,
        "target": None,
        "provenance": None,
        "near_review": None,
        "acknowledge_incomplete_identities": True,
        "target_manifest": None,
    }
    # Existing train() repeats the full Source leakage audit before any model update.
    runner.train(SimpleNamespace(**common, output=output / "training"))
    # finalize() requires hashes of the already frozen Validation selection.
    runner.finalize(SimpleNamespace(**common, run=output / "training"))
    comparison_report(output)
    runner.write_json(
        output / "run_status.json",
        {
            "status": "completed_training_and_source_test",
            "domain": "hair",
            "target_accuracy": None,
            "target_comparison": "awaiting_local_photos_after_download",
            "held_domains": plan["held_domains"],
            "selection_before_test": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source-zip", "calibration-file", "analysis", "plan", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
