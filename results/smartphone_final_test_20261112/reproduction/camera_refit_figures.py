"""Static, reproducible comparisons for measured pilot photometric fitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mediflow_datasets import camera_extended as camera
from mediflow_datasets import camera_similarity as audit
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp


def figures(output, visuals, summaries):
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 22)
    sheet = Image.new("RGB", (1620, 1260), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), "실제 원본 재측정 · 기존 4차 / 재설정 / 실제 촬영", fill="black", font=font)
    titles = ("기존 Train 원본", "조정 전 4차", "재설정 4차", "실제 P01 기준", "실제 P01 변경")
    for column, title in enumerate(titles):
        draw.text((20 + 320 * column, 60), title, fill="black", font=font)
    manifest = {}
    for index, domain in enumerate(audit.DOMAINS):
        images, source, target_images, target, previous, fitted = visuals[domain]
        path = summaries[domain]["visual_paths"][0]
        source_index = next(i for i, r in enumerate(source) if r["path"] == path)
        original, row = images[source_index], source[source_index]
        seed = exp.item_seed(data.ImageRow(row["path"], "train", row["label"]), 42)
        panels = [
            original,
            camera.transform(original, previous, "expanded", seed),
            camera.transform(original, fitted, "expanded", seed),
        ]
        target_paths = []
        for condition in ("ref", "alt"):
            ids = [i for i, r in enumerate(target) if r["group"] == "P01/" + condition]
            center = np.median([target[i]["metrics"]["y_p50"] for i in ids])
            chosen = min(ids, key=lambda i: abs(target[i]["metrics"]["y_p50"] - center))
            panels.append(target_images[chosen])
            target_paths.append(target[chosen]["path"])
        for column, array in enumerate(panels):
            image = Image.fromarray(np.rint(array).clip(0, 255).astype(np.uint8))
            sheet.paste(
                image.resize((300, 300), Image.Resampling.BILINEAR),
                (20 + 320 * column, 105 + 370 * index),
            )
        draw.text(
            (20, 413 + 370 * index), f"{domain} · 촬영의 질환 정답 미확인", fill="black", font=font
        )
        manifest[domain] = {"source": path, "target": target_paths, "seed": seed}
    draw.text(
        (20, 1220),
        "밝기·색·초점 지표의 근접성 점검 · 질환 성능·장비 동일성 미검증",
        fill="black",
        font=font,
    )
    sheet.save(output / "actual_refit_examples.png")
    review_pages = []
    for domain in audit.DOMAINS:
        images, source, _, _, previous, fitted = visuals[domain]
        selected = []
        for label in sorted({r["label"] for r in source}):
            ids = [
                i for i, r in enumerate(source) if r["label"] == label and "y_p50" in r["metrics"]
            ]
            ids.sort(key=lambda i: source[i]["metrics"]["y_p50"])
            selected.extend(dict.fromkeys((ids[0], ids[-1])))
        for page, start in enumerate(range(0, len(selected), 6), 1):
            review = Image.new("RGB", (1020, 1910), "white")
            pen = ImageDraw.Draw(review)
            pen.text(
                (15, 12),
                f"{domain} · 클래스별 밝기 양끝 표본 · 원본 / 이전 / 재설정",
                font=font,
                fill="black",
            )
            for local_row, index in enumerate(selected[start : start + 6]):
                row, original = source[index], images[index]
                seed = exp.item_seed(data.ImageRow(row["path"], "train", row["label"]), 42)
                panels = [
                    original,
                    camera.transform(original, previous, "expanded", seed),
                    camera.transform(original, fitted, "expanded", seed),
                ]
                y = 60 + local_row * 300
                for column, array in enumerate(panels):
                    image = Image.fromarray(np.rint(array).clip(0, 255).astype(np.uint8))
                    review.paste(
                        image.resize((260, 260), Image.Resampling.BILINEAR), (15 + column * 335, y)
                    )
                pen.text(
                    (15, y + 263),
                    f"class index {row['label']} · " + row["path"][-50:],
                    font=font,
                    fill="black",
                )
            name = f"train_review_{domain}_{page:02}.png"
            review.save(output / name)
            review_pages.append(
                {"file": name, "paths": [source[i]["path"] for i in selected[start : start + 6]]}
            )
    from mediflow_datasets.camera_revision import write_new

    write_new(output / "example_manifest.json", manifest)
    write_new(output / "train_review_manifest.json", review_pages)
    # A pre-existing optional local plotting library is available in this workspace.
    library = Path(__file__).resolve().parents[2] / ".tmp/final-test-report-libraries"
    if library.is_dir():
        sys.path.append(str(library))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    metrics = ("y_p50", "contrast", "lab_a", "lab_b", "detail", "normalized_detail")
    fig, axes = plt.subplots(3, 6, figsize=(19, 10), layout="constrained")
    groups = ("original", "previous_4", "fitted", "target")
    colors = ("#777777", "#a07738", "#2875bd", "#279441")
    for i, domain in enumerate(audit.DOMAINS):
        for j, metric in enumerate(metrics):
            axis = axes[i, j]
            for y, (group, color) in enumerate(zip(groups, colors, strict=True)):
                value = summaries[domain]["groups"][group].get(metric)
                if not value:
                    continue
                axis.plot([value["q05"], value["q95"]], [y, y], color=color, alpha=0.4)
                axis.plot([value["q25"], value["q75"]], [y, y], color=color, linewidth=6)
                axis.scatter([value["q50"]], [y], color=color, s=30)
            axis.set_yticks(range(4), groups)
            axis.set_title(f"{domain} / {metric}")
            axis.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Pilot pixel refit · retained Train / unlabeled captures\n"
        "Dots: median; bars: IQR and 5–95%; not a disease accuracy result"
    )
    fig.savefig(output / "distributions.png", dpi=150)
    plt.close(fig)
