import ast
import json
import zipfile
from copy import deepcopy

import cv2
import numpy as np
import pytest

from mediflow_datasets import camera_extended as camera
from mediflow_datasets import smartphone_prepare as prepare
from mediflow_datasets.domain_adaptation_data import sha256
from mediflow_datasets.smartphone_appearance import build_plan, validate_plan
from mediflow_datasets.smartphone_profile import empty_profile, write_new


def study_fixture():
    p = empty_profile(
        {
            "brightness_offset": [-9, 0],
            "contrast": [1, 1.09],
            "left_light": [-0.2, 0],
            "scale_x": [1, 1],
            "scale_y": [1, 1],
            "photometric_operator": "bounded_additive_v3",
        }
    )
    p["blur_input_scale"] = 0.41125000000000006
    p2, p4 = deepcopy(p), deepcopy(p)
    p2["color_a"] = [0, 1.01]
    p4["lighting"]["brightness_offset"] = [-18, 0]
    return {
        "source_manifest_sha256": "fixed_split",
        "profiles": {"stage1": p, "stage2": p2, "stage4": p4},
        "limits": {"blur_sigma_roi64_max": 0.5},
    }


def test_strength_plan_preserves_color_lighting_starting_scope_and_declares_exploration():
    study = study_fixture()
    original = deepcopy(study)
    plan = build_plan(study, "fixed_study")
    assert validate_plan(study, "fixed_study", plan) == plan
    assert study == original and plan["profiles"]["stage1"] == study["profiles"]["stage1"]
    for stage in ("stage2", "stage4"):
        p, base = plan["profiles"][stage], study["profiles"][stage]
        for key in ("lighting", "color_a", "color_b", "blur_input_scale"):
            assert p[key] == base[key]
        sigma = p["blur_sigma_roi64"][1] * p["blur_input_scale"]
        assert sigma == pytest.approx(0.5 if stage == "stage2" else 1)
    assert plan["distribution_improvement_established"] is False
    assert not plan["source_test_used"] and not plan["target_holdout_used"]


@pytest.mark.parametrize("size", [224, 256])
def test_input_sigma_conversion_reaches_runtime_and_stronger_blur_reduces_edges(size):
    plan = build_plan(study_fixture(), "fixed")
    yy, xx = np.indices((size, size))
    gray = (100 + 60 * ((xx + yy) % 2)).astype(np.float32)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    detail = []
    for stage in ("stage2", "stage4"):
        p = deepcopy(plan["profiles"][stage])
        p["lighting"].update(brightness_offset=[0, 0], contrast=[1, 1], left_light=[0, 0])
        p.update(color_a=[0, 0], color_b=[0, 0], texture_gain=[1, 1])
        p["blur_sigma_roi64"][0] = p["blur_sigma_roi64"][1]
        p["blur_input_scale"] *= size / 256
        changed = camera.transform(image, p, "expanded", 42)
        assert changed.shape == image.shape and changed.dtype == np.float32
        assert np.isfinite(changed).all() and 0 <= changed.min() <= changed.max() <= 255
        detail.append(float(cv2.Laplacian(changed[:, :, 0], cv2.CV_32F).var()))
    assert detail[1] < detail[0] < float(cv2.Laplacian(gray, cv2.CV_32F).var())


@pytest.mark.parametrize("change", ["lighting", "color", "holdout", "claim", "study"])
def test_plan_rejects_unrequested_changes_and_false_claims(change):
    study = study_fixture()
    plan = build_plan(study, "fixed")
    if change == "lighting":
        plan["profiles"]["stage4"]["lighting"]["brightness_offset"] = [-99, 0]
    elif change == "color":
        plan["profiles"]["stage2"]["color_a"] = [0, 3]
    elif change == "holdout":
        plan["target_holdout_used"] = True
    elif change == "claim":
        plan["distribution_improvement_established"] = True
    else:
        plan["source_study_sha256"] = "other"
    with pytest.raises(ValueError, match="frozen split scope"):
        validate_plan(study, "fixed", plan)


def test_refresh_preserves_payload_and_publishes_consistent_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    private, output = tmp_path / "private", tmp_path / "output"
    runtime = private / "runtime"
    runtime.mkdir(parents=True)
    output.mkdir()
    code = tmp_path / "src/mediflow_datasets"
    code.mkdir(parents=True)
    (code / "smartphone_appearance.py").write_text("current_code = True\n")
    (runtime / "src/mediflow_datasets").mkdir(parents=True)
    (runtime / "src/mediflow_datasets/smartphone_appearance.py").write_text("old = True\n")
    (tmp_path / "configs").mkdir()
    study = study_fixture()
    write_new(runtime / "study.json", study)
    write_new(private / "profiles.json", study["profiles"])
    (runtime / "target_encrypted").mkdir()
    payload = runtime / "target_encrypted/calibration_256.bin"
    payload.write_bytes(b"preserved encrypted input")
    payload_sha = sha256(payload)
    files = {
        p.relative_to(runtime).as_posix(): sha256(p) for p in runtime.rglob("*") if p.is_file()
    }
    write_new(runtime / "runtime_manifest.json", {"files": files})
    bundle = output / prepare.BUNDLE_NAME
    with zipfile.ZipFile(bundle, "w") as z:
        for p in runtime.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(runtime).as_posix())
    original_bundle_sha = sha256(bundle)
    book = tmp_path / "notebooks/phone.ipynb"
    book.parent.mkdir()
    write_new(book, {"cells": [{"outputs": []}]})
    write_new(
        output / "preparation.json",
        {"runtime_sha256": original_bundle_sha, "notebook": "notebooks/phone.ipynb"},
    )
    plan = build_plan(study, sha256(runtime / "study.json"))
    plan_path = output / "appearance_strength_plan.json"
    write_new(plan_path, plan)
    prepare.refresh_runtime(private, output, appearance_plan=plan_path)
    current = json.loads((runtime / "study.json").read_bytes())
    assert current["profiles"] == plan["profiles"] and current["expected_training_trials"] == 5
    assert current["reuse_identical_stage2"] is False and sha256(payload) == payload_sha
    with zipfile.ZipFile(bundle) as z:
        assert json.loads(z.read("study.json")) == current
        assert z.read("target_encrypted/calibration_256.bin") == payload.read_bytes()
        manifest = json.loads(z.read("runtime_manifest.json"))
        import hashlib

        for name, digest in manifest["files"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    updated_book = json.loads(book.read_bytes())
    source = "".join(updated_book["cells"][0]["source"])
    assert "탐색 실험" in source and "0~1px" in source and "0~0.5px" in source
    for cell in updated_book["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert sha256(bundle) in "".join(updated_book["cells"][1]["source"])
    old = list(private.glob("history/appearance_before_*/output/" + prepare.BUNDLE_NAME))
    assert len(old) == 1 and sha256(old[0]) == original_bundle_sha
