"""Verify the preselected six-class B0 lightweight alternate."""

import json
from pathlib import Path

import pytest

from mediflow_datasets import hair_six_class_light as light


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch) -> Path:
    parent = tmp_path / light.RUN_NAME
    model = parent / light.TRIAL / "attempt_fixed" / "stage2_best.keras"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"checkpoint")
    monkeypatch.setattr(light, "_sha", lambda path: light.MODEL_SHA256)
    _write(parent / "class_names.json", light.CLASSES)
    _write(parent / "dataset_contract.json", {"class_names": light.CLASSES})
    _write(
        parent / "run_config.json",
        {"signature": "fixed", "settings": {"dataset_sha256": light.DATA_SHA256}},
    )
    _write(
        parent / "comparison_summary.json",
        {"winner": "b1_384_original", "selected_model_sha256": light.HEAVY_SHA256},
    )
    _write(parent / "b0_256_original" / "completed.json", {"validation": {"macro_f1": 0.77}})
    _write(
        parent / light.TRIAL / "completed.json",
        {
            "id": light.TRIAL,
            "signature": "fixed",
            "attempt": "attempt_fixed",
            "selected_model": "stage2_best.keras",
            "selected_stage": "stage2",
            "spec": {
                "id": light.TRIAL,
                "backbone": "B0",
                "size": 256,
                "loss": "ce",
                "variant": "augmented",
                "class_count": 6,
            },
            "validation": {"macro_f1": 0.78},
            "artifact_hashes": {"attempt_fixed/stage2_best.keras": light.MODEL_SHA256},
            "model_bytes": len(b"checkpoint"),
        },
    )
    return parent


def test_selected_b0_is_accepted(tmp_path, monkeypatch):
    parent = _fixture(tmp_path, monkeypatch)
    assert light.verify_candidate(parent)["model_path"].name == "stage2_best.keras"


def test_b0_without_validation_advantage_is_rejected(tmp_path, monkeypatch):
    parent = _fixture(tmp_path, monkeypatch)
    _write(parent / "b0_256_original" / "completed.json", {"validation": {"macro_f1": 0.79}})
    with pytest.raises(ValueError, match="Validation 선정 근거"):
        light.verify_candidate(parent)


def test_changed_class_order_is_rejected(tmp_path, monkeypatch):
    parent = _fixture(tmp_path, monkeypatch)
    _write(parent / "class_names.json", list(reversed(light.CLASSES)))
    with pytest.raises(ValueError, match="클래스 순서"):
        light.verify_candidate(parent)
