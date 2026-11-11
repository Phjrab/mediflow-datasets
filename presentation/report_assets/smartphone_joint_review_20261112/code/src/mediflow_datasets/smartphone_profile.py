"""Train-only replay calibration for the five-person, unlabeled smartphone pilot."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

from mediflow_datasets import camera_extended as camera
from mediflow_datasets import camera_revision as revision
from mediflow_datasets import camera_similarity as audit
from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets.camera_refit import inverse_offset

GROUPS = (
    ("KakaoTalk_20261109_151910153", "P01", "iPhone 16 Pro"),
    ("KakaoTalk_20261109_151950329", "P02", "iPhone 13"),
    ("KakaoTalk_20261109_155838622", "P03", "iPhone 15 Pro"),
    ("KakaoTalk_20261111_202128448", "P04", "iPhone 17"),
    ("KakaoTalk_20261111_2025", "P05", "iPhone 14"),
)
KEYS = (
    "y_mean",
    "y_p50",
    "contrast",
    "r_p50",
    "g_p50",
    "b_p50",
    "lab_a",
    "lab_b",
    "side_light",
    "lighting_range",
    "detail",
    "normalized_detail",
    "face_w",
    "face_h",
    "face_x",
    "face_y",
    "very_bright",
    "very_dark",
)
LIMITS = {
    "brightness_offset": [-96.0, 96.0],
    "contrast": [0.5, 1.5],
    "side_light": [-24.0, 24.0],
    "lab_offset": 3.0,
    "texture_gain": [0.7, 1.3],
    "blur_sigma_roi64_max": 0.5,
    "spatial_correlation_q05_minimum": 0.95,
    "meaning": "Engineering search caps; not measured device limits or clinical validation",
}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def canonical_input(path, size):
    """Decode, honor EXIF/ICC, then TF center-square resize without a slow PNG round-trip."""
    import tensorflow as tf

    with Image.open(path) as original:
        upright = ImageOps.exif_transpose(original).convert("RGB")
        icc = original.info.get("icc_profile")
        if icc:
            upright = ImageCms.profileToProfile(
                upright,
                ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        raw = np.asarray(upright)
    h, w = raw.shape[:2]
    side = min(h, w)
    raw = raw[(h - side) // 2 : (h - side) // 2 + side, (w - side) // 2 : (w - side) // 2 + side]
    return tf.image.resize(raw, (size, size), method="bilinear", antialias=False).numpy()


def inventory(folder):
    rows = []
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("Unexpected smartphone input; inspect the folder before continuing")
        matches = [
            (person, phone) for prefix, person, phone in GROUPS if path.stem.startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError(f"Unmapped smartphone file: {path.name}")
        person, phone = matches[0]
        with Image.open(path) as image:
            tags = image.getexif()
            if tags.get(272) != phone:
                raise ValueError("Phone metadata differs from the confirmed capture grouping")
            rgb = image.convert("RGB")
            pixel_hash = hashlib.sha256(f"{rgb.size}".encode() + rgb.tobytes()).hexdigest()
            icc = image.info.get("icc_profile")
            color_profile = (
                ImageCms.getProfileName(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
                if icc
                else "absent_assume_sRGB"
            )
            rows.append(
                {
                    "id": f"{person}_{sum(r['person'] == person for r in rows) + 1:03}",
                    "filename": path.name,
                    "person": person,
                    "phone": phone,
                    "session": "S01",
                    "split": "holdout" if person == "P05" else "calibration",
                    "label": None,
                    "sha256": data.sha256(path),
                    "width": image.width,
                    "height": image.height,
                    "exif_orientation": tags.get(274, 1),
                    "pixel_sha256": pixel_hash,
                    "icc_profile": color_profile,
                }
            )
    counts = Counter(r["person"] for r in rows)
    if counts != {"P01": 12, "P02": 10, "P03": 10, "P04": 10, "P05": 10}:
        raise ValueError("Confirmed 52-image, five-person pilot changed")
    if len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate smartphone files require review")
    return rows


def source_previews(root):
    root = Path(root)
    manifest = data.read_json(
        root / "results/camera_adaptation_stage1_20261110/web_skin/source_manifest.json"
    )
    kept = {r["path"]: r["label"] for r in manifest["original"]["train"]}
    previous = root / "presentation/report_assets/camera_similarity_check_20261110"
    provenance = data.read_json(previous / "summary.json")["domains"]["web_skin"][
        "source_png_provenance"
    ]
    originals = {
        r["path"]: r
        for r in data.read_json(previous / "measurements.json")["web_skin"]["groups"]["original"]
    }
    images, rows = [], []
    for item in provenance:
        row = originals[item["path"]]
        if kept.get(row["path"]) != row["label"]:
            raise ValueError("Only retained Source Train previews may calibrate the phone profile")
        path = root / item["preview"]
        if data.sha256(path) != item["sha256"]:
            raise ValueError("Source preview provenance mismatch")
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB"), dtype=np.float32))
        rows.append(
            {**row, "split": "train", "preview": item["preview"], "preview_sha256": item["sha256"]}
        )
    if len(rows) != 118 or {r["label"] for r in rows} != set(range(5)):
        raise ValueError("Stratified Source Train preview collection changed")
    return images, rows


def measured_rows(images, rows, detector, *, frozen_boxes=False):
    result = []
    for image, row in zip(images, rows, strict=True):
        box = audit.box_from_metrics(row["metrics"], len(image)) if frozen_boxes else None
        if box is None and not frozen_boxes:
            box = analysis.find_face(
                cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY), detector
            )
        metrics = revision.fast_measure(image, "web_skin", detector, box)
        result.append({**row, "metrics": metrics, "box": box})
    return result


def validate_calibration(source, target):
    if not source or any(r["split"] != "train" or r.get("label") is None for r in source):
        raise ValueError("Source calibration accepts labeled Train only")
    if not target or any(r["split"] != "calibration" or r.get("label") is not None for r in target):
        raise ValueError(
            "Target calibration accepts unlabeled adaptation Train only, never holdout"
        )
    if {r["person"] for r in target} != {"P01", "P02", "P03", "P04"}:
        raise ValueError("Four calibration participants are required; P05 is reserved")


def values_by_group(rows, key, field):
    values = defaultdict(list)
    for row in rows:
        if key in row["metrics"]:
            values[row[field]].append(row["metrics"][key])
    return values


def anchor(rows, key, field):
    groups = values_by_group(rows, key, field)
    if not groups:
        raise ValueError(f"Missing metric {key}")
    return float(np.median([np.median(v) for v in groups.values()]))


def distribution(rows, key, field):
    groups = values_by_group(rows, key, field)
    if not groups:
        return None
    values, weights = [], []
    for group in groups.values():
        values.extend(group)
        weights.extend([1 / (len(groups) * len(group))] * len(group))
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    quantiles = np.interp([0.05, 0.25, 0.5, 0.75, 0.95], np.cumsum(w) - w / 2, v)
    mean = np.average(v, weights=w)
    return {
        "count": len(v),
        "group_count": len(groups),
        "mean": float(mean),
        "variance": float(np.average((v - mean) ** 2, weights=w)),
        "q05_q25_q50_q75_q95": quantiles.tolist(),
    }


def lighting_profile(source, target, fraction=0.5):
    validate_calibration(source, target)
    y = anchor(source, "y_p50", "label")
    people = values_by_group(target, "y_p50", "person")
    desired = np.quantile([np.median(v) for v in people.values()], [0.25, 0.75])
    offset = np.clip([inverse_offset(y, v) * fraction for v in desired], -96, 96)
    contrast_ratio = anchor(target, "contrast", "person") / anchor(source, "contrast", "label")
    contrast = float(np.clip(1 + fraction * (contrast_ratio - 1), 0.5, 1.5))
    side_delta = anchor(target, "side_light", "person") - anchor(source, "side_light", "label")
    side_budget = min(
        24.0,
        np.median(
            [
                np.quantile(v, 0.9) - np.quantile(v, 0.1)
                for v in values_by_group(target, "side_light", "person").values()
            ]
        ),
    )
    side = float(np.clip(side_delta * fraction, -side_budget, side_budget))
    return {
        "domain": "web_skin",
        "photometric_operator": "bounded_additive_v3",
        "brightness_offset": sorted([0.0, *offset.tolist()])[::2],
        "contrast": sorted([1.0, contrast]),
        "left_light": sorted([0.0, side]),
        "scale_x": [1.0, 1.0],
        "scale_y": [1.0, 1.0],
    }


def empty_profile(lighting):
    return {
        "schema": "measured_camera_extended_v1",
        "domain": "web_skin",
        "lighting": lighting,
        "color_a": [0.0, 0.0],
        "color_b": [0.0, 0.0],
        "texture_gain": [1.0, 1.0],
        "blur_sigma_roi64": [0.0, 0.0],
        "blur_input_scale": 1.0,
    }


def replay(images, rows, profile, detector, seeds=(42, 707, 1707)):
    output, correlations = [], []
    for image, row in zip(images, rows, strict=True):
        repetitions = []
        for seed in seeds:
            item = data.ImageRow(row["path"], "train", row["label"])
            result = camera.transform(image, profile, "expanded", exp.item_seed(item, seed))
            repetitions.append(revision.fast_measure(result, "web_skin", detector, row["box"]))
            correlations.append(
                float(np.corrcoef(image.mean(-1).ravel(), result.mean(-1).ravel())[0, 1])
            )
        output.append({**row, "metrics": audit.median_repetitions(repetitions)})
    return output, float(np.nanquantile(correlations, 0.05))


def objective(rows, target, keys):
    return sum(
        audit.distribution_gap(rows, target, key) / max(abs(anchor(target, key, "person")), 1.0)
        for key in keys
    )


def residual_color(source, target, key):
    base = anchor(source, key, "label")
    groups = values_by_group(target, key, "person")
    means = [np.median(v) for v in groups.values()]
    directions = []
    for omit in range(len(means)):
        lo, hi = np.quantile(np.delete(means, omit) - base, [0.25, 0.75])
        directions.append(1 if lo > 0 else -1 if hi < 0 else 0)
    stable = len(set(directions)) == 1 and directions[0] != 0
    budget = min(
        3.0, float(np.median([np.quantile(v, 0.9) - np.quantile(v, 0.1) for v in groups.values()]))
    )
    shift = float(np.clip(np.median(means) - base, -budget, budget)) if stable else 0.0
    return sorted([0.0, shift]), {
        "leave_one_person_out_directions": directions,
        "within_person_q90_minus_q10_cap": budget,
        "stable": stable,
    }


def expanded_profile(images, source, target, base, detector):
    rows, _ = replay(images, source, base, detector)
    profile = deepcopy(base)
    evidence = {}
    for key, metric in (("color_a", "lab_a"), ("color_b", "lab_b")):
        bounds, direction = residual_color(rows, target, metric)
        proposal = deepcopy(profile)
        proposal[key] = bounds
        changed, corr = replay(images, source, proposal, detector)
        accepted = corr >= 0.95 and revision.does_not_worsen(changed, rows, target, (metric,))[0]
        evidence[key] = {**direction, "proposed": bounds, "accepted": accepted}
        if accepted:
            profile, rows = proposal, changed
    # Blur/detail are accepted only if BOTH absolute and contrast-normalized detail improve.
    keys = ("detail", "normalized_detail")
    for key, candidates in (
        ("texture_gain", ([0.85, 1.0], [1.0, 1.15])),
        ("blur_sigma_roi64", ([0.0, 0.2], [0.0, 0.4])),
    ):
        before = objective(rows, target, keys)
        accepted = None
        for bounds in candidates:
            proposal = deepcopy(profile)
            proposal[key] = bounds
            changed, corr = replay(images, source, proposal, detector)
            if (
                corr >= 0.95
                and revision.does_not_worsen(changed, rows, target, keys)[0]
                and objective(changed, target, keys) < before
            ):
                accepted = bounds
                before = objective(changed, target, keys)
        if accepted:
            profile[key] = accepted
            rows, _ = replay(images, source, profile, detector)
        evidence[key] = {"accepted_range": profile[key], "two_detail_metrics_required": True}
    return profile, evidence


def calibrate(images, source, target, detector):
    validate_calibration(source, target)
    stage1 = empty_profile(lighting_profile(source, target))
    faces = [r["metrics"]["face_w"] for r in source if "face_w" in r["metrics"]]
    scale = 256 * float(np.median(faces)) * 0.14 / 64
    stage1["blur_input_scale"] = scale
    print("[스마트폰 분석] 1차 조명 범위 측정, 2차 색·초점 후보 비교", flush=True)
    stage2, evidence = expanded_profile(images, source, target, stage1, detector)
    # Stage4 tests a full observed-gap lighting fit, then rechecks color/detail directions.
    stage4 = empty_profile(lighting_profile(source, target, fraction=1.0))
    stage4["blur_input_scale"] = scale
    print("[스마트폰 분석] 2차 후보 비교 완료, 4차 조명 재점검", flush=True)
    keys = ("y_p50", "contrast", "side_light")
    for key in ("brightness_offset", "contrast", "left_light"):
        choices = [stage1["lighting"][key], stage4["lighting"][key]]
        best = None
        for bounds in choices:
            candidate = deepcopy(stage4)
            candidate["lighting"][key] = bounds
            rows, corr = replay(images, source, candidate, detector)
            cost = objective(rows, target, keys)
            if corr >= 0.95 and (best is None or cost < best[0]):
                best = (cost, candidate)
        if best is None:
            raise ValueError("No camera fit preserved the declared spatial guard")
        stage4 = best[1]
    stage4, stage4_evidence = expanded_profile(images, source, target, stage4, detector)
    return {"stage1": stage1, "stage2": stage2, "stage4": stage4}, {
        "stage2": evidence,
        "stage4": stage4_evidence,
    }
