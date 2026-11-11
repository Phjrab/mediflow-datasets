"""Descriptive image analysis and conservative, Train-only camera calibration."""

from __future__ import annotations

import html
import io
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

CHEEKS = ((0.22, 0.55, 0.36, 0.66), (0.64, 0.55, 0.78, 0.66))
PLOT_METRICS = (
    "y_mean",
    "y_std",
    "y_p50",
    "contrast",
    "lab_a",
    "lab_b",
    "side_light",
    "detail",
    "face_w",
    "face_h",
    "face_x",
    "face_y",
    "dark_ridge",
    "very_bright",
    "r_p50",
    "g_p50",
    "b_p50",
    "lighting_range",
)


def face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")


def find_face(gray, detector):
    h, w = gray.shape
    scale = min(1, 640 / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
    minimum = max(20, round(min(small.shape) * 0.15))
    faces = detector.detectMultiScale(small, 1.07, 4, minSize=(minimum, minimum))
    if not len(faces):
        return None
    x, y, fw, fh = max(faces, key=lambda face: face[2] * face[3])
    return tuple(round(value / scale) for value in (x, y, x + fw, y + fh))


def measure(data, domain, size, detector, face_box=None, input_policy="legacy_resize"):
    with Image.open(io.BytesIO(data)) as photo:
        raw = np.asarray(photo.convert("RGB"))
    h, w = raw.shape[:2]
    native_w, native_h = w, h
    if input_policy == "center_square":
        side = min(h, w)
        dx, dy = (w - side) // 2, (h - side) // 2
        raw = raw[dy : dy + side, dx : dx + side]
        if face_box is not None:
            left, top, right, bottom = face_box
            face_box = (left - dx, top - dy, right - dx, bottom - dy)
        h, w = raw.shape[:2]
    elif input_policy != "legacy_resize":
        raise ValueError(f"Unknown input policy: {input_policy}")
    image = cv2.resize(raw.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    row = {"width": w, "height": h, "aspect": w / h, "frame_y_mean": float(gray.mean())}
    if input_policy != "legacy_resize":
        row.update(native_width=native_w, native_height=native_h)
    if domain == "web_skin":
        face = face_box or find_face(gray, detector)
        if face is None:
            return {**row, "roi_status": "face_not_found"}
        left, top, right, bottom = face
        if not (0 <= left < right <= w and 0 <= top < bottom <= h):
            raise ValueError("Face ROI outside its image")
        fw, fh = right - left, bottom - top
        row.update(
            face_w=fw / w,
            face_h=fh / h,
            face_x=(left + right) / (2 * w),
            face_y=(top + bottom) / (2 * h),
            roi_status="recorded" if face_box else "detected",
        )
        patches = [
            image[
                round((top + fh * b) * size / h) : round((top + fh * d) * size / h),
                round((left + fw * a) * size / w) : round((left + fw * c) * size / w),
            ]
            for a, b, c, d in CHEEKS
        ]
    else:
        patches = [image[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4]]
        row["roi_status"] = "central_field"
    gray_patches = [cv2.cvtColor(p, cv2.COLOR_RGB2GRAY) for p in patches if p.size]
    if len(gray_patches) != len(patches):
        return {**row, "roi_status": "roi_too_small"}
    y = np.concatenate([p.ravel() for p in gray_patches])
    pixels = np.concatenate([p.reshape(-1, 3) for p in patches])
    # Only Lab measurement needs [0,1]. Saved classifiers still receive RGB [0,255].
    lab = cv2.cvtColor((pixels / 255).reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    p10, p50, p90 = np.quantile(y, [0.1, 0.5, 0.9])
    details = []
    for patch in gray_patches:
        common = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_LINEAR)
        details.append(float(np.mean(np.abs(common - cv2.GaussianBlur(common, (0, 0), 1)))))
    row.update(
        y_mean=float(y.mean()),
        y_std=float(y.std()),
        y_p10=float(p10),
        y_p50=float(p50),
        y_p90=float(p90),
        contrast=float(
            np.median([np.quantile(p, 0.9) - np.quantile(p, 0.1) for p in gray_patches])
        ),
        very_bright=float(np.mean(y >= 250)),
        very_dark=float(np.mean(y <= 5)),
        detail=float(np.mean(details)),
        lab_a=float(np.median(lab[:, 1])),
        lab_b=float(np.median(lab[:, 2])),
    )
    quadrant_medians = []
    for patch in gray_patches:
        ph, pw = patch.shape
        quadrant_medians.extend(
            float(np.median(part))
            for part in (
                patch[: ph // 2, : pw // 2],
                patch[: ph // 2, pw // 2 :],
                patch[ph // 2 :, : pw // 2],
                patch[ph // 2 :, pw // 2 :],
            )
        )
    row["lighting_range"] = max(quadrant_medians) - min(quadrant_medians)
    for index, channel in enumerate("rgb"):
        for q, value in zip(
            (10, 50, 90), np.quantile(pixels[:, index], [0.1, 0.5, 0.9]), strict=True
        ):
            row[f"{channel}_p{q}"] = float(value)
    if domain == "web_skin":
        row["side_light"] = float(np.median(gray_patches[0]) - np.median(gray_patches[1]))
    if domain == "hair":
        # Dark thin structures proxy; includes shadows and texture, not a validated hair mask.
        field = gray_patches[0]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        ridge = cv2.morphologyEx(field, cv2.MORPH_BLACKHAT, kernel)
        row["dark_ridge"] = float(np.mean((ridge > 15) & (field < np.median(field))))
    return row


def statistics(rows):
    metrics = defaultdict(list)
    for row in rows:
        for key, value in row["metrics"].items():
            if isinstance(value, (float, int)):
                metrics[key].append(value)
    output = {}
    for key, values in metrics.items():
        q = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        output[key] = dict(
            n=len(values),
            mean=float(np.mean(values)),
            var=float(np.var(values)),
            **dict(zip(("q05", "q25", "q50", "q75", "q95"), q.tolist(), strict=True)),
        )
    return output


def grouped_target(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["person"], row["condition"]].append(row)
    return {
        f"{person}/{condition}": statistics(values)
        for (person, condition), values in sorted(groups.items())
    }


def coverage_matched_contrast(target, people):
    """Sensitivity analysis only: dark ridges are not verified hair segmentation."""
    evidence = {}
    ratios = []
    for person in people:
        groups = [
            [
                r
                for r in target
                if r["person"] == person
                and r["condition"] == condition
                and "dark_ridge" in r["metrics"]
            ]
            for condition in ("ref", "alt")
        ]
        if not all(groups):
            return [], {"status": "missing_coverage_proxy"}
        lower = max(min(r["metrics"]["dark_ridge"] for r in g) for g in groups)
        upper = min(max(r["metrics"]["dark_ridge"] for r in g) for g in groups)
        matched = [[r for r in g if lower <= r["metrics"]["dark_ridge"] <= upper] for g in groups]
        evidence[person] = {"overlap": [lower, upper], "rows": list(map(len, matched))}
        if min(map(len, matched)) < 2:
            return [], {"status": "insufficient_common_coverage", "people": evidence}
        a, b = (float(np.median([r["metrics"]["contrast"] for r in g])) for g in matched)
        if a <= 0:
            return [], {"status": "zero_reference_contrast", "people": evidence}
        ratios.append(b / a)
        evidence[person]["contrast_ratio"] = b / a
    return ratios, {"status": "measured_proxy_sensitivity", "people": evidence}


def calibrate(domain, train, target, brightness_policy="source_class_anchor"):
    """Pilot-derived ranges are damped and bounded, and still require Train review."""
    if (
        not train
        or not target
        or any(r["split"] != "train" for r in train)
        or any(r["split"] != "calibration" for r in target)
    ):
        raise ValueError("Calibration needs verified Source Train and Target calibration only")
    if brightness_policy not in ("source_class_anchor", "within_participant"):
        raise ValueError("Unknown brightness calibration policy")
    source_stats = statistics(train)
    groups = grouped_target(target)
    people = sorted({r["person"] for r in target})
    if len(people) < 4:
        raise ValueError("At least four distinct calibration participants are required")

    def paired(key):
        try:
            return [
                (groups[f"{p}/ref"][key]["q50"], groups[f"{p}/alt"][key]["q50"]) for p in people
            ]
        except KeyError:
            return []

    luminance = paired("y_p50")
    if not luminance or (
        brightness_policy == "source_class_anchor" and "y_p50" not in source_stats
    ):
        raise ValueError("Missing comparable brightness regions")
    ref_to_alt = [b - a for a, b in luminance]
    if domain == "web_skin":
        # Equal class weights prevent successful face detections in one class dominating.
        classes = defaultdict(list)
        for row in train:
            classes[row.get("label")].append(row)
        class_stats = {str(label): statistics(rows) for label, rows in classes.items()}
        class_counts = {str(label): len(rows) for label, rows in classes.items()}
        anchors = [v["y_p50"]["q50"] for v in class_stats.values() if "y_p50" in v]
        if brightness_policy == "source_class_anchor":
            source_y = float(np.median(anchors))
            shifts = [value - source_y for pair in luminance for value in pair]
        else:
            shifts = ref_to_alt
    else:
        # Unmatched disease/scalp coverage makes a global Source-to-Target pixel offset unsafe.
        shifts = ref_to_alt
    offsets = np.clip(0.5 * np.quantile(shifts, [0.25, 0.75]), -32, 32).tolist()
    result = {
        "domain": domain,
        "profile_version": "measured_photometric_v2",
        "brightness_offset": offsets,
        "contrast": [1.0, 1.0],
        "scale_x": [1.0, 1.0],
        "scale_y": [1.0, 1.0],
        "left_light": [0.0, 0.0],
        "evidence": {
            "target_people": people,
            "brightness_changes": ref_to_alt,
            "brightness_raw_shifts": shifts,
            "brightness_policy": (
                brightness_policy if domain == "web_skin" else "within_participant"
            ),
        },
        "bounds": {
            "fraction_of_measured_shift": 0.5,
            "max_brightness_offset": 32,
            "contrast": [0.90, 1.15],
            "geometry": [0.4, 1.0],
        },
        "held_back": [
            "color: unmatched disease color",
            "blur/noise: texture/scale confounding",
            "hair mask: not validated and could change class evidence",
            "geometry: no rescaling onto artificial padding; composition is an input policy",
        ],
    }
    if domain == "web_skin":
        if brightness_policy == "source_class_anchor":
            result["evidence"]["source_brightness_anchor"] = source_y
            result["evidence"]["source_anchor_policy"] = "median of per-class valid ROI medians"
        else:
            result["evidence"]["source_anchor_policy"] = (
                "not used; missing Source face ROIs cannot drive brightness calibration"
            )
        result["evidence"]["source_roi_by_class"] = {
            label: {"rows": class_counts[label], "valid_roi": v.get("y_p50", {}).get("n", 0)}
            for label, v in class_stats.items()
        }
    ratios = [b / a for a, b in paired("contrast") if a > 0]
    if domain == "hair":
        matched_ratios, sensitivity = coverage_matched_contrast(target, people)
        result["evidence"]["coverage_matched_contrast"] = sensitivity

        # Both the complete groups and comparable coverage subset must agree.
        def same_direction(values):
            return bool(values) and (min(values) > 1 or max(values) < 1)

        if not (
            same_direction(ratios)
            and same_direction(matched_ratios)
            and (ratios[0] > 1) == (matched_ratios[0] > 1)
        ):
            ratios = []
        else:
            ratios = matched_ratios
    if len(ratios) == len(people) and (all(r > 1 for r in ratios) or all(r < 1 for r in ratios)):
        result["contrast"] = np.clip(
            1 + 0.5 * (np.quantile(ratios, [0.25, 0.75]) - 1), 0.90, 1.15
        ).tolist()
        result["evidence"]["paired_contrast_ratios"] = ratios
    if domain == "web_skin":
        lateral = [b - a for a, b in paired("side_light")]
        if len(lateral) == len(people) and (min(lateral) > 0 or max(lateral) < 0):
            result["left_light"] = np.clip(
                0.5 * np.quantile(lateral, [0.25, 0.75]), -24, 24
            ).tolist()
            result["evidence"]["paired_left_minus_right_change"] = lateral
    if domain == "hair":
        result["held_back"].append(
            "contrast/highlights: hair coverage and scalp visibility changed"
        )
    return result


def distribution_plot(groups, destination):
    """Standalone, consistently scaled per-metric box plots using SVG."""
    metrics = [
        metric for metric in PLOT_METRICS if any(metric in group for group in groups.values())
    ]
    height = 36 + len(metrics) * (40 + 25 * len(groups))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="880" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="12" y="22" font-size="16">q05–q95, quartiles, median</text>',
    ]
    top = 38
    for metric in metrics:
        data = [(name, group[metric]) for name, group in groups.items() if metric in group]
        lower = min(s["q05"] for _, s in data)
        upper = max(s["q95"] for _, s in data)

        def x(value, lower=lower, upper=upper):
            return 188 + 618 * (value - lower) / max(upper - lower, 1e-9)

        out.append(f'<text x="12" y="{top}" font-size="14" font-weight="bold">{metric}</text>')
        out.append(f'<text x="188" y="{top}" font-size="12">{lower:.4g}</text>')
        out.append(f'<text x="780" y="{top}" font-size="12">{upper:.4g}</text>')
        for index, (name, s) in enumerate(data):
            y = top + 24 + 25 * index
            out.extend(
                [
                    f'<text x="12" y="{y + 4}" font-size="12">'
                    f"{html.escape(name)} ({s['n']})</text>",
                    f'<line x1="{x(s["q05"])}" y1="{y}" x2="{x(s["q95"])}" '
                    f'y2="{y}" stroke="#254B72"/>',
                    f'<rect x="{x(s["q25"])}" y="{y - 7}" '
                    f'width="{max(1, x(s["q75"]) - x(s["q25"]))}" '
                    'height="14" fill="#B8D8F1" stroke="#254B72"/>',
                    f'<circle cx="{x(s["q50"])}" cy="{y}" r="3" fill="#143D60"/>',
                ]
            )
        top += 40 + 25 * len(groups)
    destination.write_text("\n".join([*out, "</svg>"]), encoding="utf-8")
