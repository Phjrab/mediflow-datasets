"""Measured pilot color/microcontrast/focus variation, not calibrated camera optics."""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_experiment as exp


def focus_patch(image, row, domain):
    h, w = image.shape[:2]
    metrics = row["metrics"]
    if domain == "web_skin" and "face_w" in metrics:
        fw, fh = metrics["face_w"] * w, metrics["face_h"] * h
        x, y = metrics["face_x"] * w, metrics["face_y"] * h
        left, right = max(0, round(x - fw / 2)), min(w, round(x + fw / 2))
        top, bottom = max(0, round(y - fh / 2)), min(h, round(y + fh / 2))
        patch = image[top:bottom, left:right]
    else:
        patch = image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    if min(patch.shape[:2]) < 4:
        raise ValueError("Focus measurement region is too small")
    scale = float(np.sqrt(patch.shape[0] * patch.shape[1]) / 64)
    gray = cv2.cvtColor(patch.astype(np.float32), cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR), scale


def normalized_detail(gray):
    contrast = float(np.quantile(gray, 0.9) - np.quantile(gray, 0.1))
    if contrast < 1:
        return 0.0
    return float(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1)).mean() / contrast)


def measured_extended_profile(domain, lighting, target_rows, target_images, source_measurements):
    if len(target_rows) != len(target_images) or any(
        r["split"] != "calibration" for r in target_rows
    ):
        raise ValueError("Only aligned, unlabeled calibration images may determine a profile")
    if any(r.get("label") is not None for r in target_rows):
        raise ValueError("Unverified Target labels must not enter calibration")
    if any(r["split"] != "train" for r in source_measurements):
        raise ValueError("Source transform analysis must be Train only")
    groups = analysis.grouped_target(target_rows)
    people = sorted({r["person"] for r in target_rows})
    if len(people) != 4:
        raise ValueError("The current pilot requires the four recorded participants")

    def changes(metric, ratio=False):
        result = []
        for person in people:
            a = groups[f"{person}/ref"][metric]["q50"]
            b = groups[f"{person}/alt"][metric]["q50"]
            result.append(b / max(a, 1e-6) if ratio else b - a)
        return result

    chroma = {}
    for key in ("lab_a", "lab_b"):
        values = changes(key)
        chroma[key] = {
            "participant_changes": values,
            "range": np.clip(0.5 * np.quantile(values, [0.25, 0.75]), -3, 3).tolist(),
        }
    detail_groups = defaultdict(list)
    focus_groups = defaultdict(list)
    scales = []
    for row, image in zip(target_rows, target_images, strict=True):
        patch, scale = focus_patch(image, row, domain)
        scales.append(scale)
        m = row["metrics"]
        detail_groups[row["person"], row["condition"]].append(
            m["detail"] / max(m["contrast"], 1e-3)
        )
        focus_groups[row["person"], row["condition"]].append((row, patch))
    ratios = [
        float(np.median(detail_groups[p, "alt"]) / max(np.median(detail_groups[p, "ref"]), 1e-6))
        for p in people
    ]
    texture = np.clip(1 + 0.5 * (np.quantile(ratios, [0.25, 0.75]) - 1), 0.85, 1.15)
    fits = []
    for (person, condition), entries in sorted(focus_groups.items()):
        # Match scale/coverage inside the same participant and lighting condition.
        proxy = "face_w" if domain == "web_skin" else "dark_ridge" if domain == "hair" else None
        if proxy:
            center = np.median([r["metrics"][proxy] for r, _ in entries])
            entries = sorted(entries, key=lambda x: abs(x[0]["metrics"][proxy] - center))[:3]
        if len(entries) < 3:
            continue
        values = [normalized_detail(patch) for _, patch in entries]
        reference_index = int(np.argmax(values))
        desired = float(np.quantile(values, 0.25))
        reference = entries[reference_index][1]
        grid = np.linspace(0, 1.5, 31)
        simulated = [
            normalized_detail(
                reference if sigma == 0 else cv2.GaussianBlur(reference, (0, 0), float(sigma))
            )
            for sigma in grid
        ]
        best = int(np.argmin(np.abs(np.asarray(simulated) - desired)))
        fits.append(
            {
                "person": person,
                "condition": condition,
                "sigma_roi64": float(grid[best]),
                "reference_path": entries[reference_index][0]["path"],
                "compared_paths": [r["path"] for r, _ in entries],
                "normalized_detail_reference": values[reference_index],
                "normalized_detail_target_q25": desired,
                "fit_at_search_limit": best == 30,
            }
        )
    if len(fits) < 4:
        raise ValueError("Insufficient within-condition focus evidence for an expanded pilot")
    sigma_max = min(0.75, 0.5 * float(np.quantile([r["sigma_roi64"] for r in fits], 0.75)))
    return {
        "schema": "measured_camera_extended_v1",
        "domain": domain,
        "lighting": lighting,
        "color_a": chroma["lab_a"]["range"],
        "color_b": chroma["lab_b"]["range"],
        "texture_gain": texture.tolist(),
        "blur_sigma_roi64": [0.0, sigma_max],
        "blur_input_scale": float(np.median(scales)),
        "evidence": {
            "chroma": chroma,
            "normalized_detail_ratios": ratios,
            "focus_fits": fits,
            "source_train_statistics": analysis.statistics(source_measurements),
            "target_statistics": analysis.statistics(target_rows),
        },
        "engineering_bounds": {
            "measured_change_fraction": 0.5,
            "chroma_limit_lab": 3,
            "texture_gain_limits": [0.85, 1.15],
            "focus_fit_grid_roi64": [0, 1.5, 0.05],
            "blur_max_roi64": 0.75,
        },
        "limitations": [
            "Within-person variation includes pose/content; not an isolated camera calibration",
            "Texture and focus proxies are not independent physical measurements",
            "Global disease colors/textures are not matched to unlabeled participants",
            "Four participants are not a general camera population or independent Target Test",
        ],
    }


def transform(image, profile, mode, seed):
    if mode not in ("lighting", "expanded", "aligned"):
        raise ValueError("Unknown camera experiment")
    result = exp.transform_image(image, profile["lighting"], "target_guided", seed)
    if mode == "lighting":
        return result
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x455854]))
    base = result.copy()
    delta_a = rng.uniform(*profile["color_a"])
    delta_b = rng.uniform(*profile["color_b"])
    gain = rng.uniform(*profile["texture_gain"])
    sigma = rng.uniform(*profile["blur_sigma_roi64"]) * profile["blur_input_scale"]
    # Lab conversion alone uses 0-1; the classifier still receives RGB float32 0-255.
    if delta_a != 0 or delta_b != 0:
        lab = cv2.cvtColor(result / 255.0, cv2.COLOR_RGB2Lab)
        lab[:, :, 1] += delta_a
        lab[:, :, 2] += delta_b
        result = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB) * 255.0
    if gain != 1:
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        residual = gray - cv2.GaussianBlur(gray, (0, 0), max(result.shape[0] / 128, 0.1))
        result += (gain - 1) * residual[:, :, None]
    if sigma > 0:
        result = cv2.GaussianBlur(result, (0, 0), float(sigma))
    result = exp.bounded_photometric_update(base, result)
    if result.shape != image.shape or not np.isfinite(result).all():
        raise ValueError("Expanded transform failed its shape/finite-value contract")
    return result
