from __future__ import annotations

from types import SimpleNamespace

import pytest

from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_training as training


@pytest.fixture
def reviewed_pilot(tmp_path):
    config = {
        "schema": "mediflow_domain_adaptation_v1",
        "epochs": 3, "batch_size": 16, "seed": 42, "learning_rate": 1e-5,
        "augmentation_probability": 0.5,
        "tasks": {"hair": {"source_sha256": "source"}},
    }
    spec = {"model_sha256": "model", "input_policy": "legacy_resize"}
    classes = ["C0", "C1"]
    profile = {
        "brightness_offset": [-0.5, 12], "contrast": [1, 1], "left_light": [0, 0],
        "scale_x": [1, 1], "scale_y": [1, 1],
        "train_visual_review": "approved", "photometric_operator": "bounded_additive_v3",
    }
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("simulated review evidence", encoding="utf-8")
    plan = {
        "schema": "mediflow_reviewed_training_v1", "domain": "hair",
        "status": "ready_for_pilot_training", "reviewed_train_previews": 120,
        "reviewed_contact_pages": list(range(1, 16)),
        "profile_sha256": training.object_hash(profile),
        "config_sha256": training.object_hash(config),
        "model_sha256": "model", "source_sha256": "source", "class_names": classes,
        "input_policy": "legacy_resize", "target_labels_verified": False,
        "identity_independence_verified": False,
        "evidence_sha256": {"evidence.txt": data.sha256(evidence)},
        "held_domains": {"skin": "unresolved near copies", "web_skin": "review pending"},
    }
    path = tmp_path / "plan.json"
    runner.write_json(path, plan)
    return path, plan, config, profile, spec, classes


def test_receipt_binds_settings_class_order_and_evidence(reviewed_pilot):
    path, plan, config, profile, spec, classes = reviewed_pilot
    assert training.verify_readiness(path, config, profile, spec, classes) == plan
    with pytest.raises(ValueError, match="settings differ"):
        training.verify_readiness(path, {**config, "seed": 7}, profile, spec, classes)
    with pytest.raises(ValueError, match="settings differ"):
        training.verify_readiness(path, config, profile, spec, list(reversed(classes)))
    (path.parent / "evidence.txt").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        training.verify_readiness(path, config, profile, spec, classes)


@pytest.mark.parametrize("key,value", [
    ("domain", "skin"), ("status", "pending"), ("reviewed_train_previews", 119),
    ("identity_independence_verified", True), ("target_labels_verified", True),
])
def test_pending_or_misleading_readiness_is_rejected(reviewed_pilot, key, value):
    path, plan, config, profile, spec, classes = reviewed_pilot
    changed = path.with_name("changed_plan.json")
    runner.write_json(changed, {**plan, key: value})
    with pytest.raises(ValueError):
        training.verify_readiness(changed, config, profile, spec, classes)


def test_new_pilot_variable_cannot_enter_reviewed_brightness_run(reviewed_pilot):
    path, plan, config, profile, spec, classes = reviewed_pilot
    profile = {**profile, "contrast": [0.9, 1]}
    changed = path.with_name("changed_plan.json")
    runner.write_json(changed, {**plan, "profile_sha256": training.object_hash(profile)})
    with pytest.raises(ValueError, match="brightness variable"):
        training.verify_readiness(changed, config, profile, spec, classes)


def test_orchestration_waits_for_training_then_preserves_frozen_test_selection(
    reviewed_pilot, tmp_path, monkeypatch,
):
    path, _, config, profile, spec, classes = reviewed_pilot
    config_path = tmp_path / "config.json"
    runner.write_json(config_path, config)
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    runner.write_json(analysis / "measured_profile.json", profile)
    monkeypatch.setattr(runner, "task_candidate", lambda *args: (spec, None, classes))
    calls = []

    def train(args):
        calls.append("train")
        args.output.mkdir()
        runner.write_json(args.output / "validation_selection.json", {"selected": "baseline"})

    def finalize(args):
        assert calls == ["train"]
        calls.append("finalize")
        folder = args.run / "final_evaluation"
        folder.mkdir()
        metrics = {"accuracy": 0.75, "macro_f1": 0.7}
        runner.write_json(folder / "comparison.json", {
            "selected_before_test": "baseline",
            "results": {"baseline": {
                "original_source_test": metrics, "synthetic_shift_source_test": metrics,
            }},
        })

    monkeypatch.setattr(runner, "train", train)
    monkeypatch.setattr(runner, "finalize", finalize)
    args = SimpleNamespace(config=config_path, analysis=analysis, plan=path,
                           source_zip=tmp_path / "unused.zip",
                           calibration_file=tmp_path / "calibration.json", output=tmp_path / "run")
    training.run(args)
    status = data.read_json(args.output / "run_status.json")
    assert calls == ["train", "finalize"] and status["target_accuracy"] is None
    assert status["held_domains"]["skin"] == "unresolved near copies"
    assert "0.75" in (args.output / "RESULTS.md").read_text(encoding="utf-8")

    def fail(args):
        raise ValueError("Unresolved leakage")

    monkeypatch.setattr(runner, "train", fail)
    args.output = tmp_path / "failed"
    with pytest.raises(ValueError, match="Unresolved leakage"):
        training.run(args)
    assert calls == ["train", "finalize"]  # Failed audit did not trigger another Test evaluation.
    assert not (args.output / "run_status.json").exists()
