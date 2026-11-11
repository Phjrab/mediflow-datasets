"""Replay photometric QA on exported Train previews without modifying original results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets import domain_adaptation_preflight as preflight


def clipping(before, after, roi):
    hard = ((after == 0) | (after == 255)) & ((before > 0) & (before < 255))
    near = ((after <= 5) & (before > 5)) | ((after >= 250) & (before < 250))
    return {
        "new_endpoint_channels": float(hard.mean()),
        "new_near_endpoint_channels": float(near.mean()),
        "roi_new_endpoint_channels": float(roi(hard).mean()) if roi is not None else None,
        "roi_new_near_endpoint_channels": float(roi(near).mean()) if roi is not None else None,
    }


def replay(domain, source, output):
    output.mkdir(parents=True, exist_ok=False)
    original = data.read_json(source / "profile_v2.json")
    revised = {
        **original,
        "profile_version": "measured_bounded_v3",
        "photometric_operator": "bounded_additive_v3",
        "train_visual_review": "pending",
    }
    records = data.read_json(source / "transform_qa/measurements.json")
    detector = analysis.face_detector()
    comparisons = []
    plot_rows = {"Original Train previews": [], "Affine v2 replay": [], "Bounded v3 replay": []}
    items = []
    for i, row in enumerate(records):
        if "/original/train/" not in row["path"]:
            raise ValueError("Only original Train previews may be used for transform decisions")
        before = np.asarray(
            Image.open(source / f"transform_qa/Q{i:04}_before.png").convert("RGB"), dtype=np.float32
        )
        image_row = data.ImageRow(row["path"], "train", row["label"])
        seed = exp.item_seed(image_row, 707)
        old = exp.transform_image(before, original, "target_guided", seed)
        new = exp.transform_image(before, revised, "target_guided", seed)
        m = row["original"]
        h, w = before.shape[:2]
        face_box = None
        if domain == "web_skin" and "face_w" in m:
            left = (m["face_x"] - m["face_w"] / 2) * w
            top = (m["face_y"] - m["face_h"] / 2) * h
            fw, fh = m["face_w"] * w, m["face_h"] * h
            face_box = tuple(round(v) for v in (left, top, left + fw, top + fh))
            boxes = [
                (
                    round(left + fw * a),
                    round(top + fh * b),
                    round(left + fw * c),
                    round(top + fh * d),
                )
                for a, b, c, d in analysis.CHEEKS
            ]

            def roi(array, boxes=boxes):
                return np.concatenate([array[y:y2, x:x2].reshape(-1, 3) for x, y, x2, y2 in boxes])
        elif domain != "web_skin":

            def roi(array, h=h, w=w):
                return array[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        else:
            roi = None
        comparison = {
            "path": row["path"],
            "label": row["label"],
            "seed": seed,
            "v2": clipping(before, old, roi),
            "v3": clipping(before, new, roi),
        }
        comparisons.append(comparison)
        for name, image in zip(plot_rows, (before, old, new), strict=True):
            metrics = analysis.measure(preflight.png_bytes(image), domain, h, detector, face_box)
            plot_rows[name].append({"metrics": metrics, "label": row["label"]})
        destination = output / f"Q{i:04}_bounded.png"
        with destination.open("xb") as stream:
            stream.write(preflight.png_bytes(new))
        items.append(
            {
                "images": [source / f"transform_qa/Q{i:04}_before.png", destination],
                "caption": f"Q{i:04} C{row['label']} / bounded v3",
            }
        )
    stats = {}
    for version in ("v2", "v3"):
        stats[version] = {}
        for metric in comparisons[0][version]:
            values = [r[version][metric] for r in comparisons if r[version][metric] is not None]
            stats[version][metric] = {
                "n": len(values),
                "mean": float(np.mean(values)) if values else None,
                "q95": float(np.quantile(values, 0.95)) if values else None,
                "max": max(values) if values else None,
            }
    groups = {key: analysis.statistics(rows) for key, rows in plot_rows.items()}
    analysis.distribution_plot(groups, output / "replay_distributions.svg")
    pages = preflight.contact_pages(items, output, "Known Train previews: original / bounded v3")
    runner.write_json(output / "profile_v3.json", revised)
    runner.write_json(output / "replay_measurements.json", comparisons)
    result = {
        "domain": domain,
        "samples": len(comparisons),
        "clipping": stats,
        "pages": pages,
        "controlled_variable": "operator only; same Train preview, seed and parameter ranges",
        "input_limitation": "uint8 model-input previews; not original full-resolution files",
        "training_executed": False,
        "target_accuracy": None,
        "visual_review": "pending",
    }
    runner.write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for domain in ("web_skin", "hair", "skin"):
        results[domain] = replay(domain, args.source / domain, args.output / domain)
    runner.write_json(args.output / "replay_summary.json", results)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
