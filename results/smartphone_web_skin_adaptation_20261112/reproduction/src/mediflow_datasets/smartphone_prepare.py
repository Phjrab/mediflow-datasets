"""Build a private smartphone calibration audit and a web_skin-only Colab runtime."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image, ImageDraw

from mediflow_datasets import camera_similarity as audit
from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import smartphone_profile as profile

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "camera_capture/_review/smartphone_web_skin_20261111"
OUTPUT = ROOT / "output/smartphone_web_skin_20261111"
BUNDLE_NAME = "MediFlow_Smartphone_WebSkin_Runtime_20261111.zip"


def analyze(folder, destination):
    import cv2

    cv2.setNumThreads(1)
    destination.mkdir(parents=True, exist_ok=False)
    print("[스마트폰 분석] 파일·회전 정보 확인, 두 입력 해상도 준비", flush=True)
    original = profile.inventory(folder)
    detector = analysis.face_detector()
    inputs = {
        size: np.stack(
            [profile.canonical_input(Path(folder) / row["filename"], size) for row in original]
        )
        for size in (224, 256)
    }
    calibration = [i for i, row in enumerate(original) if row["split"] == "calibration"]
    images = inputs[256][calibration]
    rows = [{**original[i], "group": original[i]["person"]} for i in calibration]
    targets = profile.measured_rows(images, rows, detector)
    print("[스마트폰 분석] 42장 볼 ROI 측정 완료, Source Train 118장 대조", flush=True)
    if any(
        sum("y_p50" in r["metrics"] for r in targets if r["person"] == p) < 4
        for p in ("P01", "P02", "P03", "P04")
    ):
        raise ValueError("Insufficient detected cheek ROIs; inspect capture framing")
    source_images, source = profile.source_previews(ROOT)
    source = profile.measured_rows(source_images, source, detector, frozen_boxes=True)
    print("[스마트폰 분석] Source 측정 완료, 1·2·4차 변환 재생 비교", flush=True)
    profiles, evidence = profile.calibrate(source_images, source, targets, detector)
    measurements = {"source": source, "calibration": targets}
    summaries = {}
    for name, treatment in profiles.items():
        replayed, correlation = profile.replay(source_images, source, treatment, detector)
        measurements[name] = replayed
        summaries[name] = {
            "spatial_correlation_q05": correlation,
            "metrics": {key: profile.distribution(replayed, key, "label") for key in profile.KEYS},
        }
    comparison = {
        key: {
            "source": profile.distribution(source, key, "label"),
            "target": profile.distribution(targets, key, "person"),
            "wasserstein": audit.distribution_gap(source, targets, key),
            "stage_wasserstein": {
                name: audit.distribution_gap(measurements[name], targets, key) for name in profiles
            },
        }
        for key in profile.KEYS
    }
    # Do not inspect holdout image statistics/embeddings during any calibration decision.
    report = {
        "schema": "mediflow_smartphone_pilot_v1",
        "domain": "web_skin",
        "raw_count": len(original),
        "counts": dict(Counter(r["person"] for r in original)),
        "phones": {r["person"]: r["phone"] for r in original},
        "calibration_count": len(calibration),
        "reserved_holdout_count": len(original) - len(calibration),
        "calibration_roi_count": sum("y_p50" in r["metrics"] for r in targets),
        "source_train_preview_count": len(source),
        "source_train_roi_count": sum("y_p50" in r["metrics"] for r in source),
        "raw_exif_rotations_required": sum(r["exif_orientation"] != 1 for r in original),
        "icc_profiles": dict(Counter(r["icc_profile"] for r in original)),
        "target_input_policy": "EXIF orientation + ICC to sRGB + center square + TF resize",
        "source_test_used": False,
        "target_accuracy": None,
        "training_executed": False,
        "holdout_used_for_calibration": False,
        "engineering_limits": profile.LIMITS,
        "comparison": comparison,
        "replay": summaries,
        "evidence": evidence,
        "limitations": [
            "Four calibration people/devices; fifth prespecified person/device reserved",
            "One phone per person: camera and person effects cannot be separated",
            "No disease labels; normal/disease labels must not be inferred",
            "118 retained stratified Train previews, not full Train pixel replay",
            "Framing measured, not synthetically warped; clinical preservation unverified",
        ],
    }
    profile.write_new(destination / "inventory.json", original)
    profile.write_new(destination / "measurements.json", measurements)
    profile.write_new(destination / "profiles.json", profiles)
    profile.write_new(destination / "analysis.json", report)
    for size, values in inputs.items():
        np.savez_compressed(destination / f"target_inputs_{size}.npz", images=values)
    contact_sheet(
        source_images, source, images, targets, profiles, destination / "private_examples.jpg"
    )
    plot(report, destination / "metric_distributions.png")
    for row in original:
        if data.sha256(Path(folder) / row["filename"]) != row["sha256"]:
            raise ValueError("An original smartphone photo changed")
    return report


def contact_sheet(source_images, source, targets, target_rows, profiles, path):
    from mediflow_datasets import camera_extended

    sheet = Image.new("RGB", (5 * 220, 4 * 246), "white")
    draw = ImageDraw.Draw(sheet)
    for j, person in enumerate(("P01", "P02", "P03", "P04")):
        i = next(k for k, row in enumerate(target_rows) if row["person"] == person)
        source_index = next(k for k, row in enumerate(source) if row["label"] == j)
        original = source_images[source_index]
        versions = (
            [original]
            + [
                camera_extended.transform(original, profiles[name], "expanded", 42)
                for name in ("stage1", "stage2", "stage4")
            ]
            + [targets[i]]
        )
        for column, (label, image) in enumerate(
            zip(("Source Train", "Stage1", "Stage2/3", "Stage4", person), versions, strict=True)
        ):
            thumb = Image.fromarray(np.rint(image).clip(0, 255).astype(np.uint8)).resize((216, 216))
            sheet.paste(thumb, (column * 220, j * 246))
            draw.text((column * 220 + 2, j * 246 + 220), label, fill="black")
    sheet.save(path, quality=90)


def plot(report, path):
    try:
        import matplotlib
    except ModuleNotFoundError:
        sys.path.insert(0, str(ROOT / ".tmp/final-test-report-libraries"))
        import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    keys = (
        "y_p50",
        "contrast",
        "lab_a",
        "lab_b",
        "detail",
        "normalized_detail",
        "side_light",
        "face_w",
    )
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for key, axis in zip(keys, axes.flat, strict=True):
        for j, (name, color) in enumerate((("source", "#2864a0"), ("target", "#d86732"))):
            values = report["comparison"][key][name]
            if values:
                q05, q25, q50, q75, q95 = values["q05_q25_q50_q75_q95"]
                axis.plot([q05, q95], [j, j], color=color)
                axis.plot([q25, q75], [j, j], lw=8, color=color, alpha=0.4)
                axis.plot(q50, j, "o", color=color)
        axis.set_yticks([0, 1], ["Source Train", "Phone calibration"])
        axis.set_title(key)
        axis.grid(alpha=0.2)
    fig.suptitle("Smartphone pilot: equal class/person weights; holdout excluded")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def encrypt_inputs(destination, output, key):
    rows = data.read_json(destination / "inventory.json")
    encrypted = output / "target_encrypted"
    encrypted.mkdir()
    for size in (224, 256):
        with np.load(destination / f"target_inputs_{size}.npz", allow_pickle=False) as arrays:
            images = arrays["images"]
        for role in ("calibration", "holdout"):
            indices = [i for i, r in enumerate(rows) if r["split"] == role]
            stream = io.BytesIO()
            np.savez_compressed(stream, images=images[indices])
            plaintext = stream.getvalue()
            member = f"target_encrypted/{role}_{size}.bin"
            nonce = os.urandom(12)
            (output / member).write_bytes(
                nonce + AESGCM(key).encrypt(nonce, plaintext, member.encode())
            )
            import hashlib

            profile.write_new(
                output / member.replace(".bin", ".json"),
                {
                    "cipher": "AES-256-GCM",
                    "aad_utf8": member,
                    "input_size": size,
                    "input_policy": "exif_transpose_icc_srgb_then_center_square",
                    "pixel_range": [0, 255],
                    "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
                    "rows": [
                        {
                            k: rows[i][k]
                            for k in ("id", "person", "phone", "session", "split", "label")
                        }
                        for i in indices
                    ],
                },
            )


def build(folder, private=PRIVATE, output=OUTPUT):
    from mediflow_datasets.smartphone_notebook import build_notebook

    if output.exists():
        raise FileExistsError("Choose a new output; never overwrite previous artifacts")
    report = analyze(folder, private)
    output.mkdir(parents=True)
    runtime = private / "runtime"
    runtime.mkdir()
    source_bundle = (
        ROOT / "camera_capture/_review/stage1_release_v3_20261110/MediFlow_Adaptation_Common_v3.zip"
    )
    with zipfile.ZipFile(source_bundle) as archive:
        config = json.loads(archive.read("configs/adaptation_stage1.json"))
        task = config["tasks"]["web_skin"]
        reference = json.loads(archive.read("references/web_skin.json"))
    manifest_path = ROOT / "results/camera_adaptation_stage1_20261110/web_skin/source_manifest.json"
    phone_profiles = data.read_json(private / "profiles.json")
    identical = phone_profiles["stage1"] == phone_profiles["stage2"]
    study = {
        "schema": "mediflow_smartphone_web_skin_stages_v1",
        "domain": "web_skin",
        **{
            k: config[k]
            for k in (
                "seed",
                "epochs",
                "learning_rate",
                "augmentation_probability",
                "validation_scope",
            )
        },
        "task": task,
        "profiles": phone_profiles,
        "source_manifest_sha256": data.sha256(manifest_path),
        "starting_weights": "each treatment starts independently from its original checkpoint",
        "stage1_selection": "max worst(clean, phone_lighting) F1, then clean F1, then basic",
        "final_selection": (
            "max worst(clean, phone_lighting, phone_expanded, phone_refit) F1; clean; earlier arm"
        ),
        "target_training_people": ["P01", "P02", "P03", "P04"],
        "target_reserved_people": ["P05"],
        "target_labels_available": False,
        "reuse_identical_stage2": identical,
        "expected_training_trials": 4 if identical else 5,
        "validation_condition_aliases": {"phone_expanded": "phone_lighting"} if identical else {},
        "source_test_in_training": False,
        "calibration_report_sha256": data.sha256(private / "analysis.json"),
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "uncommitted_runtime_sha256": {
            p.name: data.sha256(p) for p in (ROOT / "src/mediflow_datasets").glob("smartphone_*.py")
        },
        "limits": profile.LIMITS,
        "prior_checkpoint_source_test_exposure_unverified": True,
    }
    profile.write_new(runtime / "study.json", study)
    profile.write_new(ROOT / "configs/smartphone_web_skin_20261111.json", study)
    # Quarantine and exact original split are inherited; old webcam pixel profiles are not.
    shutil.copyfile(manifest_path, runtime / "source_manifest.json")
    profile.write_new(
        runtime / "reference.json",
        {
            "quarantine": reference["quarantine"],
            "pixel_hashes": [r["pixel_sha256"] for r in data.read_json(private / "inventory.json")],
            "cross_domain_audit": (
                "Exact original RGB pixels; transformed/near identity not established"
            ),
        },
    )
    for spec in task["models"].values():
        path = data.local_path(ROOT, spec["model"])
        if data.sha256(path) != spec["model_sha256"]:
            raise ValueError("Starting model changed")
        target = runtime / spec["model"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    code = runtime / "src/mediflow_datasets"
    code.mkdir(parents=True)
    for path in (ROOT / "src/mediflow_datasets").glob("*.py"):
        shutil.copyfile(path, code / path.name)
    key = AESGCM.generate_key(bit_length=256)
    encrypt_inputs(private, runtime, key)
    key_path = output / "MediFlow_Smartphone_Target_Key_20261111.txt"
    key_path.write_text(base64.b64encode(key).decode() + "\n", encoding="utf-8")
    del key
    files = {
        p.relative_to(runtime).as_posix(): data.sha256(p)
        for p in sorted(runtime.rglob("*"))
        if p.is_file()
    }
    profile.write_new(
        runtime / "runtime_manifest.json",
        {"schema": "mediflow_phone_runtime_manifest_v1", "files": files},
    )
    archive_path = output / BUNDLE_NAME
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(runtime.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(runtime).as_posix())
    digest = data.sha256(archive_path)
    (output / (BUNDLE_NAME + ".sha256")).write_text(
        digest + "  " + BUNDLE_NAME + "\n", encoding="utf-8"
    )
    notebook = (
        ROOT / "notebooks/camera_adaptation/MediFlow_Smartphone_WebSkin_Stages_20261111.ipynb"
    )
    with notebook.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            build_notebook(
                digest, stage2_reused=identical, stage2_profile=phone_profiles["stage2"]
            ),
            handle,
            ensure_ascii=False,
            indent=1,
        )
        handle.write("\n")
    profile.write_new(
        output / "preparation.json",
        {
            "schema": study["schema"],
            "runtime_sha256": digest,
            "runtime_bytes": archive_path.stat().st_size,
            "notebook": notebook.relative_to(ROOT).as_posix(),
            "counts": report["counts"],
            "source_test_used": False,
            "training_executed": False,
        },
    )
    print(
        "준비 완료:", archive_path, "\n노트북:", notebook, "\n복호화 키 파일:", key_path, flush=True
    )


def accepted_review(root, private, study, review_path):
    review_path = Path(review_path).resolve()
    if not review_path.is_relative_to(root.resolve()):
        raise ValueError("Profile review must be a preserved repository-relative artifact")
    review = data.read_json(review_path)
    if (
        review.get("schema") != "mediflow_phone_stage2_common_nearer_review_v1"
        or review.get("status") != "measured"
        or review.get("source_test_used") is not False
        or review.get("target_holdout_used") is not False
        or review.get("training_executed") is not False
        or review["profile_source_sha256"] != data.sha256(private / "profiles.json")
        or review["measurement_source_sha256"] != data.sha256(private / "measurements.json")
        or review["source_manifest_sha256"] != study["source_manifest_sha256"]
        or any(review["profiles"][key] != study["profiles"][key] for key in ("stage1", "stage4"))
        or review["profiles"]["stage2"]["lighting"] != study["profiles"]["stage1"]["lighting"]
    ):
        raise ValueError("Stage2 review differs from the frozen calibration/Train-only scope")
    return review


def refresh_runtime(private=PRIVATE, output=OUTPUT, *, profile_review=None, appearance_plan=None):
    """Explicitly refresh this newly generated draft bundle without reusing old run outputs."""
    from mediflow_datasets.smartphone_notebook import build_notebook

    if profile_review is not None and appearance_plan is not None:
        raise ValueError("Choose a measured review or an exploratory appearance plan")
    runtime = private / "runtime"
    previous = data.read_json(output / "preparation.json")
    notebook = ROOT / previous["notebook"]
    if any(c.get("outputs") for c in data.read_json(notebook)["cells"]):
        raise ValueError("Do not refresh an executed notebook")
    if data.sha256(output / BUNDLE_NAME) != previous["runtime_sha256"]:
        raise ValueError("Existing draft archive changed unexpectedly")
    old_manifest = data.read_json(runtime / "runtime_manifest.json")
    for name, expected in old_manifest["files"].items():
        if data.sha256(data.local_path(runtime, name)) != expected:
            raise ValueError("Existing draft runtime changed unexpectedly")
    study = data.read_json(runtime / "study.json")
    previous_record = (
        runtime / "appearance_plan.json"
        if study.get("appearance_plan")
        else runtime / "profile_review.json"
        if study.get("profile_review")
        else private / "profiles.json"
    )
    previous_profiles = data.read_json(previous_record)
    if previous_record != private / "profiles.json":
        previous_profiles = previous_profiles["profiles"]
    if study["profiles"] != previous_profiles:
        raise ValueError("Camera profiles changed; a new calibration is required")
    if profile_review is not None:
        review = accepted_review(ROOT, private, study, profile_review)
        history = private / "history/stage2_before_review_20261111"
        history.mkdir(parents=True, exist_ok=False)
        old_files = [
            runtime / "study.json",
            runtime / "runtime_manifest.json",
            ROOT / previous["notebook"],
            output / "preparation.json",
            output / "verification.json",
            output / "artifact_verification.json",
        ]
        for path in old_files:
            if path.is_file():
                shutil.copyfile(path, history / path.name)
        code_backup = history / "code"
        code_backup.mkdir()
        for path in (runtime / "src/mediflow_datasets").glob("smartphone_*.py"):
            shutil.copyfile(path, code_backup / path.name)
        study["profiles"] = review["profiles"]
        study["profile_review"] = {
            "file": Path(profile_review).resolve().relative_to(ROOT).as_posix(),
            "sha256": data.sha256(profile_review),
            "changed": "Stage2 and Stage3 appearance only",
            "stage1_and_stage4_unchanged": True,
        }
        shutil.copyfile(profile_review, runtime / "profile_review.json")
    if appearance_plan is not None:
        from mediflow_datasets.smartphone_appearance import validate_plan

        plan_path = Path(appearance_plan).resolve()
        if not plan_path.is_relative_to(ROOT.resolve()):
            raise ValueError("Appearance plan must be preserved inside the repository")
        plan = validate_plan(study, data.sha256(runtime / "study.json"), data.read_json(plan_path))
        history = private / "history" / ("appearance_before_" + data.sha256(plan_path)[:12])
        history.mkdir(parents=True, exist_ok=False)
        for source in (
            runtime / "study.json",
            runtime / "runtime_manifest.json",
            ROOT / "configs/smartphone_web_skin_20261111.json",
            notebook,
            *output.glob("*.json"),
            output / BUNDLE_NAME,
            output / (BUNDLE_NAME + ".sha256"),
        ):
            if source.is_file():
                backup = history / source.relative_to(ROOT)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, backup)
        study["profiles"] = plan["profiles"]
        study["appearance_plan"] = {
            "file": plan_path.relative_to(ROOT).as_posix(),
            "sha256": data.sha256(plan_path),
            "kind": plan["status"],
            "reference_input_size": plan["reference_input_size"],
        }
        study["limits"]["blur_sigma_roi64_max"] = plan["blur_sigma_roi64_exploratory_cap"]
        study["limits"]["meaning"] = (
            "Engineering search caps; blur expanded for requested exploration, not device limits"
        )
        shutil.copyfile(plan_path, runtime / "appearance_plan.json")
    identical = study["profiles"]["stage1"] == study["profiles"]["stage2"]
    study.update(
        reuse_identical_stage2=identical,
        expected_training_trials=4 if identical else 5,
        validation_condition_aliases={"phone_expanded": "phone_lighting"} if identical else {},
    )
    study["uncommitted_runtime_sha256"] = {
        p.name: data.sha256(p) for p in (ROOT / "src/mediflow_datasets").glob("smartphone_*.py")
    }
    for path in (ROOT / "src/mediflow_datasets").glob("*.py"):
        shutil.copyfile(path, runtime / "src/mediflow_datasets" / path.name)
    for path in (runtime / "study.json", ROOT / "configs/smartphone_web_skin_20261111.json"):
        path.write_text(
            json.dumps(study, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    files = {
        p.relative_to(runtime).as_posix(): data.sha256(p)
        for p in sorted(runtime.rglob("*"))
        if p.is_file() and p.name != "runtime_manifest.json" and "__pycache__" not in p.parts
    }
    (runtime / "runtime_manifest.json").write_text(
        json.dumps(
            {"schema": "mediflow_phone_runtime_manifest_v1", "files": files},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary = output / (BUNDLE_NAME + ".partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in [*files, "runtime_manifest.json"]:
            archive.write(runtime / name, name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip():
            raise ValueError("Updated archive CRC failed")
    temporary.replace(output / BUNDLE_NAME)
    digest = data.sha256(output / BUNDLE_NAME)
    (output / (BUNDLE_NAME + ".sha256")).write_text(
        digest + "  " + BUNDLE_NAME + "\n", encoding="utf-8"
    )
    notebook.write_text(
        json.dumps(
            build_notebook(
                digest,
                stage2_reused=identical,
                stage2_profile=study["profiles"]["stage2"],
                stage4_profile=study["profiles"]["stage4"],
                appearance_requested=bool(study.get("appearance_plan")),
            ),
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    previous.update(
        runtime_sha256=digest,
        runtime_bytes=(output / BUNDLE_NAME).stat().st_size,
        expected_training_trials=study["expected_training_trials"],
        stage2_reused=identical,
        appearance_plan=study.get("appearance_plan"),
    )
    (output / "preparation.json").write_text(
        json.dumps(previous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("미실행 준비본 갱신 완료:", output / BUNDLE_NAME, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path)
    parser.add_argument("--refresh-runtime", action="store_true")
    parser.add_argument("--profile-review", type=Path)
    parser.add_argument("--appearance-plan", type=Path)
    args = parser.parse_args()
    if args.refresh_runtime:
        refresh_runtime(profile_review=args.profile_review, appearance_plan=args.appearance_plan)
    elif args.photos:
        build(args.photos)
    else:
        parser.error("--photos 또는 --refresh-runtime을 지정하세요.")


if __name__ == "__main__":
    main()
