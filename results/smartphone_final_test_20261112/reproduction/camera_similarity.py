"""Audit frozen camera transformations against real captures using retained Train previews."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mediflow_datasets import camera_extended as camera
from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets.domain_adaptation_preflight import png_bytes

DOMAINS = ("skin", "web_skin", "hair")
SEEDS = (42, 707, 1707)
METRICS = (
    "y_p50",
    "contrast",
    "lab_a",
    "lab_b",
    "detail",
    "lighting_range",
    "side_light",
    "face_w",
    "face_h",
    "dark_ridge",
    "very_bright",
    "very_dark",
)


def retained_previews(rows, manifest):
    """Only already-retained original Train images may enter the replay."""
    keep = {r["path"]: r["label"] for r in manifest["original"]["train"]}
    output = []
    for index, row in enumerate(rows):
        if row["split"] != "train":
            raise ValueError("Replay accepts Train previews only")
        if row["path"] not in keep:
            continue
        if row["label"] != keep[row["path"]]:
            raise ValueError("Preview class differs from the retained manifest")
        output.append((index, row))
    return output


def balanced_weights(rows, field):
    counts = Counter(row[field] for row in rows)
    return np.asarray([1 / (len(counts) * counts[row[field]]) for row in rows])


def distribution_gap(source, target, key):
    """Weighted 1D Wasserstein distance; original units, no similarity percentage."""
    source = [r for r in source if key in r["metrics"]]
    target = [r for r in target if key in r["metrics"]]
    if not source or not target:
        return None
    a = np.array([r["metrics"][key] for r in source])
    b = np.array([r["metrics"][key] for r in target])
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Non-finite image metric")
    wa, wb = balanced_weights(source, "label"), balanced_weights(target, "group")
    ia, ib = np.argsort(a), np.argsort(b)
    edges = np.unique(np.concatenate([a, b]))
    ca = np.r_[0.0, np.cumsum(wa[ia])][np.searchsorted(a[ia], edges[:-1], side="right")]
    cb = np.r_[0.0, np.cumsum(wb[ib])][np.searchsorted(b[ib], edges[:-1], side="right")]
    return float(np.sum(np.abs(ca - cb) * np.diff(edges)))


def box_from_metrics(metrics, size):
    if "face_w" not in metrics:
        return None
    return tuple(
        round(v * size)
        for v in (
            metrics["face_x"] - metrics["face_w"] / 2,
            metrics["face_y"] - metrics["face_h"] / 2,
            metrics["face_x"] + metrics["face_w"] / 2,
            metrics["face_y"] + metrics["face_h"] / 2,
        )
    )


def measure(image, domain, detector, box):
    if domain == "web_skin" and box is None:
        return {"roi_status": "missing_in_original_frozen"}
    return analysis.measure(png_bytes(image), domain, len(image), detector, box)


def median_repetitions(repetitions):
    keys = set.intersection(*(set(r) for r in repetitions))
    return {
        key: float(np.median([r[key] for r in repetitions]))
        for key in keys
        if all(isinstance(r[key], (int, float)) for r in repetitions)
    }


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    study_path = root / "configs/camera_strength_20261110.json"
    study = data.read_json(study_path)
    report = {
        "study_sha256": data.sha256(study_path),
        "analysis_code_sha256": data.sha256(Path(__file__)),
        "seeds": list(SEEDS),
        "augmentation_forced": True,
        "source_test_used": False,
        "training_settings_changed": False,
        "nearly_identical_established": False,
        "target_accuracy": None,
        "measurement_policy": (
            "Retained stratified Train QA PNGs; fixed original ROIs; uint8 PNG measurement "
            "for all views and Target; median metric across three seeds per Source image"
        ),
        "distance_policy": (
            "1D Wasserstein in original metric units; equal Source class weights and equal "
            "Target person/condition weights; descriptive, no accepted equivalence threshold"
        ),
        "domains": {},
    }
    all_rows, visual = {}, {}
    bundle = (
        root / "camera_capture/_review/stage1_release_v3_20261110/MediFlow_Adaptation_Common_v3.zip"
    )
    if data.sha256(bundle) != study["bundle_sha256"]:
        raise ValueError("Frozen common bundle changed")
    with ZipFile(bundle) as z:
        config = json.loads(z.read("configs/adaptation_stage1.json"))
        references = {d: json.loads(z.read(f"references/{d}.json")) for d in DOMAINS}
    detector = analysis.face_detector()
    for domain in DOMAINS:
        profile = study["domains"][domain]["profile"]
        weak = references[domain]["extended_profile"]
        manifest_path = (
            root / f"results/camera_adaptation_stage1_20261110/{domain}/source_manifest.json"
        )
        if data.sha256(manifest_path) != study["domains"][domain]["source_manifest_sha256"]:
            raise ValueError("Retained source manifest changed")
        manifest = data.read_json(manifest_path)
        preview_root = root / f"camera_capture/_review/preflight_v2_fbe37d44/{domain}/transform_qa"
        preview_records = data.read_json(preview_root / "measurements.json")
        source = retained_previews(preview_records, manifest)
        calibration = (
            root
            / f"camera_capture/_review/training_reference_inputs_20261110/{domain}/calibration.json"
        )
        if (
            data.sha256(calibration)
            != study["domains"][domain]["measurement_provenance"]["target_sha256"]
        ):
            raise ValueError("Target packet changed")
        packet = data.read_json(calibration)
        image_rows = [data.ImageRow(**r) for r in packet["rows"]]
        if any(r.split != "calibration" or r.label is not None for r in image_rows):
            raise ValueError("Target must be unlabeled calibration captures")
        if runner.target_fingerprint(root / "camera_capture", image_rows) != packet["fingerprint"]:
            raise ValueError("Actual capture files changed")
        cache_root = root / f"camera_capture/_review/nine_experiments_20261109/{domain}"
        meta = data.read_json(cache_root / "target_inputs.json")
        if (
            meta["array_sha256"] != data.sha256(cache_root / "target_inputs.npz")
            or meta["rows"] != packet["rows"]
            or meta["capture_fingerprint"] != packet["fingerprint"]
            or meta["input_policy"] != config["tasks"][domain]["input_policy"]
        ):
            raise ValueError("Target model-input cache differs from its source captures")
        with np.load(cache_root / "target_inputs.npz", allow_pickle=False) as archive:
            target_images = archive["images"]
        size = meta["input_size"]
        if target_images.shape != (len(image_rows), size, size, 3):
            raise ValueError("Target input array shape differs")
        groups = {key: [] for key in ("original", "previous", "stronger", "target")}
        provenance, repetitions = [], []
        view_source = None
        for index, row in source:
            path = preview_root / f"Q{index:04}_before.png"
            with Image.open(path) as photo:
                original = np.asarray(photo.convert("RGB"), dtype=np.float32)
            if original.shape != (size, size, 3):
                raise ValueError("Train preview input size differs from the model")
            box = box_from_metrics(row["original"], size)
            origin = measure(original, domain, detector, box)
            entry = {"path": row["path"], "label": row["label"], "metrics": origin}
            groups["original"].append(entry)
            row_id = data.ImageRow(row["path"], "train", row["label"])
            for name, setting in (("previous", weak), ("stronger", profile)):
                metrics = []
                for seed_base in SEEDS:
                    seed = exp.item_seed(row_id, seed_base)
                    transformed = camera.transform(original, setting, "expanded", seed)
                    metric = measure(transformed, domain, detector, box)
                    metrics.append(metric)
                    repetitions.append(
                        {
                            "path": row["path"],
                            "condition": name,
                            "seed_base": seed_base,
                            "seed": seed,
                            "metrics": metric,
                            "new_endpoint_fraction": float(
                                np.mean(
                                    ((transformed <= 0) | (transformed >= 255))
                                    & (original > 0)
                                    & (original < 255)
                                )
                            ),
                        }
                    )
                groups[name].append({**entry, "metrics": median_repetitions(metrics)})
            provenance.append(
                {
                    "path": row["path"],
                    "preview": path.relative_to(root).as_posix(),
                    "sha256": data.sha256(path),
                }
            )
            if view_source is None and "y_p50" in origin:
                view_source = (original, row_id)
        for index, row in enumerate(image_rows):
            if domain == "web_skin":
                with Image.open(root / "camera_capture" / row.path) as photo:
                    w, h = photo.size
                side = min(w, h)
                dx, dy = (w - side) // 2, (h - side) // 2
                left, top, right, bottom = row.face_box
                box = tuple(
                    round(v * size / side) for v in (left - dx, top - dy, right - dx, bottom - dy)
                )
            else:
                box = None
            groups["target"].append(
                {
                    "path": row.path,
                    "group": f"{row.person}/{row.condition}",
                    "metrics": measure(target_images[index], domain, detector, box),
                }
            )
        stats = {key: analysis.statistics(rows) for key, rows in groups.items()}
        gaps = {
            metric: {
                key: distribution_gap(groups[key], groups["target"], metric)
                for key in ("original", "previous", "stronger")
            }
            for metric in METRICS
        }
        # Target examples are selected by proximity to their group brightness median.
        actual = []
        for condition in ("ref", "alt"):
            ids = [
                i
                for i, r in enumerate(image_rows)
                if r.person == "P01" and r.condition == condition
            ]
            center = np.median([groups["target"][i]["metrics"]["y_p50"] for i in ids])
            chosen = min(ids, key=lambda i: abs(groups["target"][i]["metrics"]["y_p50"] - center))
            actual.append((target_images[chosen], image_rows[chosen].path))
        original, row_id = view_source
        seed = exp.item_seed(row_id, 42)
        visual[domain] = [
            (original, row_id.path),
            (camera.transform(original, weak, "expanded", seed), row_id.path),
            (camera.transform(original, profile, "expanded", seed), row_id.path),
            *actual,
        ]
        result = {
            "source_unique_count": len(source),
            "source_excluded": len(preview_records) - len(source),
            "source_count_by_class": dict(Counter(row["label"] for _, row in source)),
            "source_roi_count": stats["original"].get("y_p50", {}).get("n", 0),
            "target_count": len(image_rows),
            "target_people": packet["people"],
            "target_capture_fingerprint_verified": True,
            "target_cache_sha256": meta["array_sha256"],
            "source_png_provenance": provenance,
            "groups": stats,
            "distribution_gaps": gaps,
            "stronger_new_endpoint_max": max(
                r["new_endpoint_fraction"] for r in repetitions if r["condition"] == "stronger"
            ),
            "visual_paths": [path for _, path in visual[domain]],
        }
        report["domains"][domain] = result
        all_rows[domain] = {"groups": groups, "repetitions": repetitions}
        print(domain, "retained Train", len(source), "Target", len(image_rows), flush=True)
    (output / "measurements.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    contact_sheet(visual, output)
    distribution_figure(report, output)
    return report


def contact_sheet(visual, output):
    thumb, gap = 300, 20
    sheet = Image.new("RGB", (5 * (thumb + gap) + gap, 3 * (thumb + 70) + 120), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 23)
    draw.text(
        (20, 12),
        "유지된 Train 표본 · 기존 변환 · 새 변환 · 실제 촬영 비교",
        fill="black",
        font=font,
    )
    for column, title in enumerate(
        ("원본 Train", "기존 2차", "새 후속 변환", "실제 P01 기준", "실제 P01 변경")
    ):
        draw.text((20 + column * (thumb + gap), 60), title, fill="black", font=font)
    for row_index, domain in enumerate(DOMAINS):
        y = 105 + row_index * (thumb + 70)
        for column, (array, _) in enumerate(visual[domain]):
            picture = Image.fromarray(np.rint(array).clip(0, 255).astype(np.uint8))
            picture = picture.resize((thumb, thumb), Image.Resampling.BILINEAR)
            sheet.paste(picture, (20 + column * (thumb + gap), y))
        draw.text(
            (20, y + thumb + 8), f"{domain} · 실제 촬영은 질환 정답 미확인", fill="black", font=font
        )
    draw.text(
        (20, sheet.height - 28),
        "질환·사람·구도 차이 포함 · 거의 동일함 또는 성능 개선을 입증한 그림 아님",
        fill="black",
        font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 18),
    )
    sheet.save(output / "actual_comparison.png")


def distribution_figure(report, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    metrics = ("y_p50", "contrast", "lab_a", "lab_b", "detail", "lighting_range")
    names = ("original", "previous", "stronger", "target")
    fig, axes = plt.subplots(3, len(metrics), figsize=(18, 10), layout="constrained")
    for row_index, domain in enumerate(DOMAINS):
        for column, metric in enumerate(metrics):
            axis = axes[row_index, column]
            groups = report["domains"][domain]["groups"]
            for y, (name, color) in enumerate(
                zip(names, ("#777777", "#3882a3", "#cf6f20", "#279441"), strict=True)
            ):
                s = groups[name].get(metric)
                if not s:
                    continue
                axis.plot([s["q05"], s["q95"]], [y, y], color=color, alpha=0.5)
                axis.plot([s["q25"], s["q75"]], [y, y], color=color, linewidth=6)
                axis.scatter([s["q50"]], [y], color=color, s=35, zorder=4)
            axis.set_yticks(range(4), names)
            axis.set_title(f"{domain} / {metric}")
            axis.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Retained Train previews vs actual captures: medians, IQR, 5-95%\n"
        "Three-seed per-image metric medians; descriptive comparison",
        fontsize=15,
    )
    fig.savefig(output / "distributions.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
