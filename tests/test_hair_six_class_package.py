"""Check that six-class packaging accepts only the fixed selected run."""

import json
from pathlib import Path

import pytest

from mediflow_datasets import hair_six_class_package as candidate


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch) -> Path:
    parent = tmp_path / candidate.RUN_NAME
    attempt = "attempt_fixed"
    model = parent / candidate.WINNER / attempt / candidate.MODEL_NAME
    model.parent.mkdir(parents=True)
    model.write_bytes(b"checkpoint")
    monkeypatch.setattr(candidate, "_sha", lambda path: candidate.EXPECTED_MODEL_SHA256)
    _write(parent / "class_names.json", candidate.EXPECTED_CLASSES)
    _write(
        parent / "dataset_contract.json",
        {"class_names": candidate.EXPECTED_CLASSES, "limitations": []},
    )
    _write(
        parent / "run_config.json",
        {
            "signature": "fixed",
            "settings": {
                "protocol": "hair_six_class_four_v1",
                "dataset_sha256": candidate.EXPECTED_DATA_SHA256,
            },
        },
    )
    _write(
        parent / "comparison_summary.json",
        {
            "protocol": "hair_six_class_four_v1",
            "selection": "validation_macro_f1_then_accuracy",
            "winner": candidate.WINNER,
            "test_evaluated": True,
            "selected_test_accuracy": 0.8,
            "selected_test_macro_f1": 0.81,
            "selected_model_sha256": candidate.EXPECTED_MODEL_SHA256,
            "experiments": [
                {"id": candidate.WINNER, "validation_accuracy": 0.79, "validation_macro_f1": 0.8}
            ],
        },
    )
    _write(
        parent / "selected_test_metrics.json",
        {
            "selected_model": candidate.WINNER,
            "accuracy": 0.8,
            "macro_f1": 0.81,
            "count": 1305,
            "support": [250, 250, 250, 250, 252, 53],
            "class_f1": [0.8] * 6,
            "confusion_matrix": [[0] * 6 for _ in range(6)],
        },
    )
    _write(
        parent / candidate.WINNER / "completed.json",
        {
            "id": candidate.WINNER,
            "signature": "fixed",
            "attempt": attempt,
            "selected_model": candidate.MODEL_NAME,
            "selected_stage": "stage2",
            "spec": {
                "id": candidate.WINNER,
                "backbone": "B1",
                "size": 384,
                "loss": "ls005",
                "variant": "original",
                "class_count": 6,
            },
            "parameters": 6582925,
            "model_bytes": len(b"checkpoint"),
            "validation": {"accuracy": 0.79, "macro_f1": 0.8},
            "artifact_hashes": {
                f"{attempt}/{candidate.MODEL_NAME}": candidate.EXPECTED_MODEL_SHA256
            },
        },
    )
    return parent


def test_fixed_records_are_accepted(tmp_path, monkeypatch):
    parent = _fixture(tmp_path, monkeypatch)
    assert candidate.verify_records(parent)["model_path"].name == candidate.MODEL_NAME


def test_changed_winner_is_rejected(tmp_path, monkeypatch):
    parent = _fixture(tmp_path, monkeypatch)
    summary_path = parent / "comparison_summary.json"
    summary = candidate._read(summary_path)
    summary["winner"] = "b1_384_augmented"
    _write(summary_path, summary)
    with pytest.raises(ValueError, match="Validation 선정"):
        candidate.verify_records(parent)


def test_changed_class_order_is_rejected(tmp_path, monkeypatch):
    parent = _fixture(tmp_path, monkeypatch)
    _write(parent / "class_names.json", list(reversed(candidate.EXPECTED_CLASSES)))
    with pytest.raises(ValueError, match="클래스 순서"):
        candidate.verify_records(parent)
