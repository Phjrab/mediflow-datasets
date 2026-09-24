"""Render archived experiment values as consistent publication PNG/PDF figures."""

import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
NAVY = "#19364e"
BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#16a34a"
RED = "#ef4444"
COLORS = [BLUE, ORANGE, GREEN, RED, "#7c3aed", "#db2777"]
LABELS = {
    "b0_224_ce": "B0 · 224 · CE",
    "b0_256_ce": "B0 · 256 · CE",
    "b0_256_ls005": "B0 · 256 · Label Smoothing 0.05",
    "b0_256_focal15": "B0 · 256 · Focal Loss 1.5",
    "b1_256_ls005": "B1 · 256 · Label Smoothing 0.05",
    "b1_256_ls005_extend5": "B1 · 256 · LS 0.05 · 추가 5 Epoch",
    "original": "B0 · 224 · CE · Original",
    "augmented": "B0 · 224 · CE · Augmented",
}


def font(size, bold=False):
    return ImageFont.truetype(
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf", size
    )


def text(draw, xy, value, size=26, fill=NAVY, anchor="la", bold=False):
    draw.text(xy, str(value), font=font(size, bold), fill=fill, anchor=anchor)


def canvas(title, subtitle, height):
    image = Image.new("RGB", (3200, height), "#f5f8fc")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 3200, 130), fill=NAVY)
    text(draw, (1600, 24), title, 48, "white", "ma", True)
    text(draw, (1600, 88), subtitle, 25, "#d4e3f2", "ma")
    return image, draw


def lineplot(draw, box, histories, metric, accuracy_min):
    x, y, w, h = box
    left, right, top, bottom = x + 76, x + w - 20, y + 42, y + h - 72
    text(draw, (x + w / 2, y), "Accuracy" if metric == "accuracy" else "Loss", 28, anchor="ma")
    values = [v for _, hist in histories for k in (metric, "val_" + metric) for v in hist[k]]
    lo, hi = (accuracy_min, 1.0) if metric == "accuracy" else (0, math.ceil(max(values) * 5) / 5)
    n = sum(len(hist[metric]) for _, hist in histories)

    def px(epoch):
        return left + (right - left) * (epoch - 1) / max(n - 1, 1)

    def py(v):
        return bottom - (bottom - top) * (v - lo) / (hi - lo)

    for i in range(6):
        v = lo + (hi - lo) * i / 5
        yy = py(v)
        draw.line((left, yy, right, yy), fill="#d9e2ec", width=2)
        text(draw, (left - 12, yy), f"{v:.2f}", 21, anchor="rm")
    for epoch in [1] + list(range(5, n + 1, 5)):
        xx = px(epoch)
        text(draw, (xx, bottom + 12), epoch, 21, anchor="ma")
    offset = 0
    for stage, hist in histories:
        if offset:
            xx = px(offset + 0.5)
            for yy in range(int(top), int(bottom), 16):
                draw.line((xx, yy, xx, min(yy + 8, bottom)), fill="#8193a5", width=2)
        for j, key in enumerate((metric, "val_" + metric)):
            points = [(px(offset + e + 1), py(v)) for e, v in enumerate(hist[key])]
            draw.line(points, fill=COLORS[(stage - 1) * 2 + j], width=4)
        offset += len(hist[metric])
    draw.line((left, top, left, bottom, right, bottom), fill="#8193a5", width=2)
    text(draw, ((left + right) / 2, bottom + 43), "Epoch", 23, anchor="ma")


def save(image, directory, name):
    image.save(directory / (name + ".png"), dpi=(300, 300))
    image.save(directory / (name + ".pdf"), resolution=300)


def render(domain, run_name, prepared=None):
    base = ROOT / "results" / domain / "experiments" / run_name
    output = ROOT / "results" / domain / "presentation_20260914"
    output.mkdir(exist_ok=True)
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    selection = prepared["selection"] if prepared else read(base / "selection_before_test.json")
    final = prepared["test"] if prepared else read(base / "final_test_metrics.json")
    index = read(ROOT / "results/CANDIDATE_INDEX.json")
    classes = read(
        (ROOT / "results" / index["candidates"][domain]["model"]).with_name("class_names.json")
    )  # noqa: E501
    keys = list(LABELS)[:6] if domain == "web_skin" else ["original", "augmented"]
    rows = prepared["experiments"] if prepared else []
    if prepared:
        sources.update(prepared["sources"])
        keys = []
    for key in keys:
        completed = read(base / key / "completed.json")
        attempt = base / key / completed["attempt"]
        histories = []
        if key.endswith("extend5"):
            histories = list(rows[-1]["histories"])
            histories.append((3, read(attempt / "extension_history.json")))
        else:
            for stage in (1, 2):
                path = attempt / f"stage{stage}_history.json"
                if path.exists():
                    histories.append((stage, read(path)))
        for _, hist in histories:
            assert (
                len({len(hist[k]) for k in ("accuracy", "val_accuracy", "loss", "val_loss")}) == 1
            )
            assert all(
                math.isfinite(v)
                for k in ("accuracy", "val_accuracy", "loss", "val_loss")
                for v in hist[k]
            )
        rows.append(dict(key=key, histories=histories, metrics=completed["validation"]))
    title = {"web_skin": "Web Skin", "skin": "Skin", "hair": "Hair"}[domain]
    nrows = math.ceil(len(rows) / 2)
    image, draw = canvas(
        title + " 모델 전체 학습곡선 비교",
        "보존된 원본 곡선 이미지 · 6번은 Stage 2 Validation만 표시"
        if prepared
        else "저장된 Epoch 기록 그대로 · 파선: 학습 단계 전환 · 평활화 없음",
        230 + nrows * 690,
    )
    legend = (
        [
            "Stage 1 Train",
            "Stage 1 Validation",
            "Stage 2 Train",
            "Stage 2 Validation",
            "추가 Train",
            "추가 Validation",
        ]
        if domain == "web_skin"
        else ["Train", "Validation"]
    )
    for i, label in enumerate(legend):
        if prepared:
            break
        xx = 120 + i * 505
        draw.line((xx, 169, xx + 40, 169), fill=COLORS[i], width=5)
        text(draw, (xx + 55, 149), label, 25)
    for i, row in enumerate(rows):
        x, y = 30 + (i % 2) * 1590, 205 + (i // 2) * 690
        draw.rounded_rectangle((x, y, x + 1550, y + 660), 12, fill="white")
        selected = row["key"] == selection["winner"]
        draw.rounded_rectangle(
            (x, y, x + 1550, y + 65), 12, fill="#dcfce7" if selected else "#e5effa"
        )
        label = f"{i + 1}. " + LABELS[row["key"]] + ("  [선정]" if selected else "")
        text(draw, (x + 775, y + 15), label, 30, anchor="ma", bold=True)
        if prepared:
            panel = Image.open(ROOT / row["curve_image"]).convert("RGB")
            panel.thumbnail((1530, 570), Image.Resampling.LANCZOS)
            image.paste(panel, (x + (1550 - panel.width) // 2, y + 80 + (570 - panel.height) // 2))
            continue
        for col, metric in enumerate(("accuracy", "loss")):
            lineplot(
                draw,
                (x + 10 + col * 770, y + 100, 755, 535),
                row["histories"],
                metric,
                0.3,
            )
    foot = (
        "추가 학습 패널은 부모 25 Epoch + 추가 5 Epoch. 최종 평가는 부모 최고 모델 유지. "
        "Loss 정의가 달라 실험 간 절대값 순위 비교 불가."
        if domain == "web_skin"
        else (
            "Original / Augmented 모두 분류부만 15 Epoch 학습. "
            "동일 Validation 사용; 증강은 업데이트 수가 더 많음."
        )
    )
    if prepared:
        foot = (
            "원본 Epoch JSON 미확보: 저장된 곡선을 재배치. Loss 정의가 달라 절대값 순위 비교 불가."
        )
    text(draw, (1600, image.height - 43), foot, 24, anchor="ma")
    save(image, output, domain + "_all_training_curves_comparison")

    image, draw = canvas(
        title + " 모델 성능 종합 비교",
        "기존 6개 실험의 Validation / Test 기록 · 후보는 Validation 기준"
        if prepared
        else "전체 실험: Validation 비교  |  Test: 최종 선정 모델만 평가",
        2150,
    )
    text(draw, (110, 165), "Validation Accuracy", 31, bold=True)
    text(
        draw,
        (780, 165),
        "Test Macro F1" if prepared else "Validation Macro F1",
        31,
        GREEN,
        bold=True,
    )
    if prepared:
        text(draw, (1450, 165), "Test Accuracy", 31, ORANGE, bold=True)
    left, right, top, bottom = 190, 3070, 260, 770
    for tick in range(0, 101, 20):
        yy = bottom - (bottom - top) * tick / 100
        draw.line((left, yy, right, yy), fill="#d9e2ec", width=2)
        text(draw, (left - 20, yy), str(tick) + "%", 25, anchor="rm")
    step = (right - left) / len(rows)
    for i, row in enumerate(rows):
        cx = left + step * (i + 0.5)
        if row["key"] == selection["winner"]:
            draw.rounded_rectangle(
                (cx - step * 0.45, 215, cx + step * 0.45, 910), 20, fill="#e2f9ec"
            )
        metrics = (
            ("accuracy", "macro_f1", "test_accuracy") if prepared else ("accuracy", "macro_f1")
        )
        for j, metric in enumerate(metrics):
            value = row["metrics"][metric]
            xx = cx + (j - 1) * 95
            yy = bottom - (bottom - top) * value
            draw.rounded_rectangle((xx, yy, xx + 88, bottom), 7, fill=(BLUE, GREEN, ORANGE)[j])
            if j == 0:
                text(
                    draw,
                    (cx, yy - 45),
                    "Acc " + str(value),
                    20 if prepared else 25,
                    anchor="ma",
                    bold=True,
                )
        label = LABELS[row["key"]].replace(" · ", "\n")
        text(draw, (cx, 795), label, 24, anchor="ma")
    text(
        draw,
        (1600, 955),
        "클래스별 Test F1 · 색이 진할수록 높음"
        if prepared
        else "클래스별 Validation F1 · 색이 진할수록 높음",
        35,
        anchor="ma",
        bold=True,
    )

    def heatmap(values, labels, y, cell_h):
        start, width = 700, 2370 / len(classes)
        for col, name in enumerate(classes):
            text(draw, (start + width * (col + 0.5), y - 42), name, 25, anchor="ma", bold=True)
        for rr, (vals, label) in enumerate(zip(values, labels, strict=True)):
            text(draw, (start - 20, y + rr * cell_h + cell_h / 2), label, 23, anchor="rm")
            for cc, v in enumerate(vals):
                assert 0 <= v <= 1
                rgb = tuple(
                    round(a + (b - a) * v)
                    for a, b in zip((239, 246, 255), (30, 103, 170), strict=True)
                )  # noqa: E501
                xx, yy = start + cc * width, y + rr * cell_h
                draw.rectangle((xx, yy, xx + width - 3, yy + cell_h - 3), fill=rgb)
                text(
                    draw,
                    (xx + width / 2, yy + cell_h / 2),
                    str(v),
                    19 if len(classes) > 5 else 24,
                    "white" if v > 0.6 else NAVY,
                    anchor="mm",
                )

    heatmap(
        [r["metrics"]["class_f1"] for r in rows],
        [LABELS[r["key"]] for r in rows],
        1050,
        62 if len(rows) > 2 else 125,
    )
    text(
        draw,
        (1600, 1470),
        "선정 모델 Test 결과 · " + LABELS[selection["winner"]],
        36,
        GREEN,
        "ma",
        True,
    )
    text(
        draw,
        (1600, 1535),
        f"Test {final['count']}장  |  Accuracy {final['accuracy']}  |  "
        f"Macro F1 {final['macro_f1']}",
        31,
        anchor="ma",
    )
    heatmap([final["class_f1"]], ["선정 모델 Test F1"], 1670, 115)
    for i in range(100):
        v = i / 99
        rgb = tuple(
            round(a + (b - a) * v) for a, b in zip((239, 246, 255), (30, 103, 170), strict=True)
        )  # noqa: E501
        draw.rectangle((1150 + i * 9, 1850, 1159 + i * 9, 1880), fill=rgb)
    text(draw, (1120, 1865), "0", 24, anchor="rm")
    text(draw, (2080, 1865), "1", 24, anchor="lm")
    text(
        draw,
        (1600, 1930),
        "F1 원값은 source_data.json에 보존 · 평가하지 않은 Test 값은 표시하지 않음",
        27,
        anchor="ma",
    )
    note = (
        "동률은 사전 실험 순서로 선정 · B1 추가 학습 행은 부모 최고 유지 결과"
        if domain == "web_skin"
        else "증강 학습 후보 선정"
    )
    if prepared:
        note = "B1 · 256 · LS 0.05 · Stage 2 총 15 Epoch 후보"
    text(draw, (1600, 1990), note, 28, GREEN, "ma", True)
    text(
        draw,
        (1600, 2060),
        "공개 데이터 결과 · 실제 장비 성능 및 임상 정확도를 의미하지 않음",
        24,
        anchor="ma",
    )
    save(image, output, domain + "_model_performance_dashboard")
    (output / "source_data.json").write_text(
        json.dumps(
            dict(
                sources=sources, experiments=rows, selection=selection, test=final, classes=classes
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# " + title + " 발표용 그림\n\nPNG 300 dpi, PDF는 동일 고해상도 래스터 그림입니다.\n"
        "원본 기록과 기존 이미지는 변경하지 않았습니다. "
        "source_data.json에 원값과 출처 SHA-256을 보존했습니다.\n"
        "막대 축은 0에서 시작하며 F1 열지도 범위는 0~1입니다. 수치 반올림 없이 원값을 보존하고, "
        "F1 값은 열지도에 원값 그대로 표시합니다.\n" + foot + "\n",
        encoding="utf-8",
    )
    print(domain, "created", len(rows), "experiments", output)


if __name__ == "__main__":
    render("web_skin", "suite_20260908_014452_72768a42")
    render("skin", "comparison_20260909_075056_72d865bf")
