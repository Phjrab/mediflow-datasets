"""Read-only Source review, independent transform QA and v2 experiment preparation."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp


def png_bytes(image):
    stream = io.BytesIO()
    Image.fromarray(np.rint(image).clip(0, 255).astype(np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def ssim_gray(a, b):
    """11x11, sigma 1.5, population SSIM; a ranking aid, never duplicate clearance."""
    x, y = (
        cv2.cvtColor(v.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64) for v in (a, b)
    )
    mx, my = (cv2.GaussianBlur(v, (11, 11), 1.5) for v in (x, y))
    vx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mx * mx
    vy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - my * my
    cov = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    score = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(score[5:-5, 5:-5].mean())


def contact_pages(items, output, title, columns=2, rows=4):
    """Each item contains paths to two existing QA thumbnails and an ASCII caption."""
    pages = []
    for start in range(0, len(items), columns * rows):
        chunk = items[start : start + columns * rows]
        sheet = Image.new("RGB", (columns * 400, rows * 230 + 35), "#15212d")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), title, fill="white")
        for index, item in enumerate(chunk):
            x, y = (index % columns) * 400, 35 + (index // columns) * 230
            for j, path in enumerate(item["images"]):
                with Image.open(path) as opened:
                    tile = opened.convert("RGB")
                    tile.thumbnail((194, 194))
                    sheet.paste(tile, (x + j * 200 + (194 - tile.width) // 2, y))
            draw.text((x + 3, y + 196), item["caption"], fill="white")
        path = output / f"page_{start // (columns * rows) + 1:04}.jpg"
        sheet.save(path, quality=94)
        pages.append(path.name)
    return pages


def review_near_pairs(archive_path, source, audit, output):
    """Export every candidate, grouped only by recorded provenance and ranked at 256px."""
    output.mkdir(exist_ok=False)
    thumbs = output / "images"
    thumbs.mkdir()
    index = {row.path: row for rows in source.values() for row in rows}

    def canonical(path):
        row = index[path]
        return row.origin if row.origin in index else path

    groups = defaultdict(list)
    for pair in audit["near_duplicate_review"]:
        if any(path not in index for path in pair):
            raise ValueError("Near candidate does not belong to the pinned Source ZIP")
        groups[tuple(sorted(canonical(path) for path in pair))].append(pair)
    records = []
    image_manifest = {}
    with zipfile.ZipFile(archive_path) as archive:

        @lru_cache(maxsize=96)
        def load(path):
            body = archive.read(path)
            with Image.open(io.BytesIO(body)) as opened:
                raw = opened.convert("RGB")
                small = np.asarray(raw.resize((256, 256), Image.Resampling.BILINEAR))
                name = hashlib.sha256(path.encode()).hexdigest()[:20] + ".jpg"
                destination = thumbs / name
                if not destination.exists():
                    tile = raw.copy()
                    tile.thumbnail((512, 512))
                    tile.save(destination, quality=95)
                image_manifest[path] = {
                    "thumbnail": name,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "label": index[path].label,
                    "split": index[path].split,
                }
            return small

        for number, (pair, aliases) in enumerate(sorted(groups.items()), 1):
            a, b = map(load, pair)
            record = {
                "id": f"N{number:05}",
                "paths": list(pair),
                "raw_pairs": aliases,
                "ssim_gray_256": ssim_gray(a, b),
                "mean_absolute_rgb_difference": float(np.abs(a.astype(float) - b).mean()),
                "decision": "pending",
                "reason": "",
                "scope": "image derivation review only",
            }
            records.append(record)
            if number % 100 == 0:
                print(f"Near review: {number}/{len(groups)}", flush=True)
    records.sort(key=lambda item: item["ssim_gray_256"], reverse=True)
    items = [
        {
            "images": [thumbs / image_manifest[p]["thumbnail"] for p in r["paths"]],
            "caption": f"{r['id']} SSIM={r['ssim_gray_256']:.3f} / pending",
        }
        for r in records
    ]
    pages = contact_pages(items, output, "Near candidates: higher similarity first; ALL pending")
    runner.write_json(output / "candidate_groups.json", records)
    runner.write_json(output / "image_manifest.json", image_manifest)
    runner.write_json(
        output / "summary.json",
        {
            "raw_pairs": len(audit["near_duplicate_review"]),
            "provenance_groups": len(records),
            "review_pages": pages,
            "auto_cleared": 0,
            "identity_independence_verified": False,
            "meaning": "Test images are inspected for derivation only; never augmentation tuning",
        },
    )
    markup = [
        '<meta charset="utf-8"><title>MediFlow candidate review</title>',
        "<h1>Image-derivation candidates — all pending</h1>",
        "<p>Similarity is a ranking aid. It does not identify people or approve a pair.</p>",
    ]
    for record in records:
        markup.append(f"<h3>{record['id']} / {record['ssim_gray_256']:.4f}</h3>")
        for path in record["paths"]:
            name = image_manifest[path]["thumbnail"]
            markup.append(
                f'<figure style="display:inline-block"><img loading="lazy" '
                f'src="images/{name}" width="320"><figcaption>'
                f"{html.escape(path)}</figcaption></figure>"
            )
    (output / "review.html").write_text("\n".join(markup), encoding="utf-8")
    return {
        "raw_pairs": len(audit["near_duplicate_review"]),
        "groups": len(records),
        "auto_cleared": 0,
    }


def quality_check(
    archive_path,
    train_rows,
    source_measurements,
    target_measurements,
    profile,
    spec,
    output,
    samples_per_class=24,
):
    """Measure only Train transformations; Test pixels are never used here."""
    if any(row.split != "train" for row in train_rows):
        raise ValueError("Transform QA accepts Source Train only")
    output.mkdir(exist_ok=False)
    detector = analysis.face_detector()
    measures = {"Source sample": [], "Transformed sample": []}
    samples, clips, page_items = [], [], []
    originals = {r["path"]: r for r in source_measurements}
    labels = sorted({row.label for row in train_rows})
    with zipfile.ZipFile(archive_path) as archive:
        for label in labels:
            selected = sorted(
                [r for r in train_rows if r.label == label], key=lambda r: exp.item_seed(r, 9181)
            )[:samples_per_class]
            for row in selected:
                before = exp.image_tensor(
                    archive.read(row.path), spec["input_size"], spec["input_policy"]
                )
                after = exp.transform_image(
                    before, profile, "target_guided", exp.item_seed(row, 707)
                )
                old = originals[row.path]["metrics"]
                box = None
                if "face_w" in old:
                    size = spec["input_size"]
                    box = tuple(
                        round(v * size)
                        for v in (
                            old["face_x"] - old["face_w"] / 2,
                            old["face_y"] - old["face_h"] / 2,
                            old["face_x"] + old["face_w"] / 2,
                            old["face_y"] + old["face_h"] / 2,
                        )
                    )
                pair = []
                for key, image in (("Source sample", before), ("Transformed sample", after)):
                    metrics = analysis.measure(
                        png_bytes(image), profile["domain"], spec["input_size"], detector, box
                    )
                    pair.append(metrics)
                    measures[key].append({"metrics": metrics, "label": label})
                fraction = float(
                    np.mean(((after == 0) | (after == 255)) & ((before > 0) & (before < 255)))
                )
                clips.append(fraction)
                samples.append(
                    {
                        "path": row.path,
                        "split": "train",
                        "label": label,
                        "newly_clipped_channel_fraction": fraction,
                        "original": pair[0],
                        "transformed": pair[1],
                    }
                )
                if len(page_items) < len(labels) * samples_per_class:
                    number = len(page_items)
                    paths = []
                    for suffix, image in (("before", before), ("after", after)):
                        path = output / f"Q{number:04}_{suffix}.png"
                        with path.open("xb") as stream:
                            stream.write(png_bytes(image))
                        paths.append(path)
                    page_items.append(
                        {"images": paths, "caption": f"Q{number:04} C{label} clip={fraction:.3f}"}
                    )
    groups = {key: analysis.statistics(rows) for key, rows in measures.items()}
    groups.update(analysis.grouped_target(target_measurements))
    analysis.distribution_plot(groups, output / "after_transform_distributions.svg")
    pages = contact_pages(page_items, output, "Train only: original / v2 photometric transform")
    runner.write_json(output / "measurements.json", samples)
    summary = {
        "sample_count": len(samples),
        "samples_per_class": samples_per_class,
        "newly_clipped_channel_fraction_max": max(clips, default=0),
        "distributions": groups,
        "pages": pages,
        "visual_review": "pending",
        "meaning": "QA metrics use uint8 preview rounding; classifier input stays float32 0-255",
    }
    runner.write_json(output / "summary.json", summary)
    return {
        key: summary[key]
        for key in ("sample_count", "newly_clipped_channel_fraction_max", "visual_review")
    }


def face_failure_review(archive_path, source_rows, source_measurements, output):
    output.mkdir(exist_ok=False)
    lookup = {r.path: r for r in source_rows}
    failed = [r for r in source_measurements if r["metrics"].get("roi_status") == "face_not_found"]
    items = []
    with zipfile.ZipFile(archive_path) as archive:
        for i, row in enumerate(failed):
            if row["split"] != "train" or row["path"] not in lookup:
                raise ValueError("Face failure review must use Source Train")
            path = output / f"F{i:04}.jpg"
            with Image.open(io.BytesIO(archive.read(row["path"]))) as original:
                image = original.convert("RGB")
                image.thumbnail((512, 512))
                image.save(path, quality=95)
            items.append({"images": [path], "caption": f"F{i:04} C{row['label']} / ROI missing"})
    pages = contact_pages(
        items, output, "Source Train missing face detections; inspect composition"
    )
    runner.write_json(output / "failed_paths.json", failed)
    return {"failed_train_rows": len(failed), "pages": pages, "visual_review": "pending"}


def preflight(args):
    config = runner.load_config(args.config)
    spec, model, classes = runner.task_candidate(args.domain, config)
    previous = data.read_json(args.previous / "analysis.json")
    audit = data.read_json(args.previous / "source_audit.json")
    source, digest = data.source_index(
        args.source_zip, classes, config["tasks"][args.domain]["source_sha256"]
    )
    if previous["domain"] != args.domain or previous["source_sha256"] != digest:
        raise ValueError("Previous Source analysis differs from the pinned archive")
    measured = [row for row in previous["images"] if row["kind"] == "source"]
    if {r["path"] for r in measured} != {r.path for r in source["original", "train"]}:
        raise ValueError("Previous Train measurements do not cover this archive")
    labels_by_path = {r.path: r.label for r in source["original", "train"]}
    if any(r["split"] != "train" or r["label"] != labels_by_path[r["path"]] for r in measured):
        raise ValueError("Previous Source labels or split differ from the model contract")
    if spec["input_policy"] == "center_square" and any(
        r["metrics"]["width"] != r["metrics"]["height"] for r in measured
    ):
        raise ValueError("Changed Source crop needs new measurements; cached rows cannot be reused")
    packet, target = runner.load_calibration_packet(
        args.calibration_file, args.domain, spec["model_sha256"], spec["input_policy"]
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner.snapshot(output, config)
    profile = analysis.calibrate(args.domain, measured, packet["measurements"])
    profile.update(
        source_sha256=digest,
        model_sha256=spec["model_sha256"],
        calibration_fingerprint=packet["fingerprint"],
        input_policy=spec["input_policy"],
        train_visual_review="pending",
        selected_from="verified_train_and_target_calibration",
    )
    runner.write_json(output / "profile_v2.json", profile)
    runner.write_json(output / "class_names.json", classes)
    arms, aliases = exp.arm_plan(profile)
    runner.write_json(
        output / "experiment_plan.json",
        {
            "arms": arms,
            "equivalent_arms": aliases,
            "training_executed": False,
            "randomness_policy": "independent_apply_stream_v2",
            "input_policy": spec["input_policy"],
        },
    )
    qa = quality_check(
        args.source_zip,
        source["original", "train"],
        measured,
        packet["measurements"],
        profile,
        spec,
        output / "transform_qa",
        args.samples_per_class,
    )
    faces = (
        face_failure_review(
            args.source_zip, source["original", "train"], measured, output / "face_failures"
        )
        if args.domain == "web_skin"
        else None
    )
    near = review_near_pairs(args.source_zip, source, audit, output / "near_candidates")
    features = output / "feature_diagnostics"
    features.mkdir()
    with zipfile.ZipFile(args.source_zip) as archive:
        runner.baseline_inference(
            features,
            spec,
            model,
            source["original", "train"],
            target,
            lambda row: archive.read(row.path),
            None,
            classes,
            config,
            (np.asarray(packet["scores"]), np.asarray(packet["features"])),
        )
    runner.write_json(
        output / "preflight_status.json",
        {
            "domain": args.domain,
            "status": "data_collected_review_pending",
            "transform_qa": qa,
            "near_candidates": near,
            "face_failures": faces,
            "training_executed": False,
            "source_test_accuracy_evaluated": False,
            "target_accuracy": None,
            "identity_limitations": audit["identity_metadata"],
        },
    )
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("web_skin", "hair", "skin"))
    for name in ("source-zip", "config", "calibration-file", "previous", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=24)
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("samples-per-class must be positive")
    preflight(args)


if __name__ == "__main__":
    main()
