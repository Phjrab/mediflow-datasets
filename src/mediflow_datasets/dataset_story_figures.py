"""Create presentation figures from documented dataset audit results."""

import hashlib
import json

from PIL import Image, ImageDraw

from mediflow_datasets.presentation_charts import BLUE, GREEN, NAVY, ORANGE, ROOT, text

OUT = ROOT / "results/dataset_validation_presentation_20260914"


def page(title, subtitle):
    image = Image.new("RGB", (3200, 1800), "#f5f8fc")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 3200, 230), fill=NAVY)
    text(draw, (120, 55), title, 65, "white", bold=True)
    text(draw, (120, 155), subtitle, 33, "#d4e3f2")
    return image, draw


def card(draw, x, title, lines, color=BLUE):
    draw.rounded_rectangle((x, 330, x + 920, 1040), 25, fill="white")
    text(draw, (x + 45, 375), title, 52, color, bold=True)
    for i, line in enumerate(lines):
        text(draw, (x + 45, 490 + i * 115), line, 34)


def footer(draw, source):
    text(
        draw,
        (120, 1640),
        "검사 범위: 동일 파일·픽셀 등  |  사람·병변·촬영 세션 단위 겹침은 미검증",
        30,
    )
    text(draw, (120, 1710), source, 25, "#64748b")


def save(image, name):
    image.save(OUT / (name + ".png"), dpi=(300, 300))


def detail(domain, before, after, splits, overlaps, unit, steps, augmentation, source):
    image, draw = page(
        domain + " · 중복 발견에서 데이터 재구성까지",
        "원본 이미지 기준 · 기존 검사·정리 기록을 시각화한 자료",
    )
    text(draw, (120, 300), "01  분할 사이 중복 발견", 44, bold=True)
    labels = ["Train ↔ Validation", "Train ↔ Test", "Validation ↔ Test"]
    for i, (label, value) in enumerate(zip(labels, overlaps, strict=True)):
        y = 415 + i * 160
        text(draw, (140, y), label, 36)
        text(draw, (980, y - 10), str(value) + " " + unit, 53, ORANGE, anchor="ra", bold=True)
    text(draw, (120, 970), "쌍별 겹침 수의 합 ≠ 전체 제거 이미지 수", 30)
    text(draw, (1190, 300), "02  정리 전후 원본 이미지 수", 44, bold=True)
    for i, (label, count, color) in enumerate(
        [("정리 전", before, "#94a3b8"), ("정리 후", after, GREEN)]
    ):
        y = 425 + i * 195
        text(draw, (1190, y), label, 35)
        width = int(1250 * count / before)
        draw.rounded_rectangle((1380, y, 1380 + width, y + 90), 12, fill=color)
        text(draw, (2690, y + 15), f"{count:,}장", 43, bold=True)
    text(draw, (1190, 855), f"원본 수 차이  {before - after:,}장", 40, GREEN, bold=True)
    text(draw, (1190, 950), splits, 34)
    text(draw, (120, 1110), "03  처리 순서", 44, bold=True)
    for i, label in enumerate(steps):
        x = 120 + i * 765
        draw.rounded_rectangle((x, 1215, x + 670, 1360), 20, fill="#e5effa")
        text(draw, (x + 335, 1260), label, 35, anchor="ma", bold=True)
        if i < 3:
            text(draw, (x + 710, 1260), "→", 42)
    text(draw, (120, 1440), augmentation, 34, GREEN, bold=True)
    footer(draw, source)
    save(image, domain.lower() + "_dataset_cleanup")


def main():
    OUT.mkdir(exist_ok=True)
    image, draw = page(
        "데이터 검증 · 점수 비교 전에 평가 조건부터 확인",
        "학습에서 본 사진이 평가에도 포함되면 새로운 사진에 대한 성능을 판단하기 어려움",
    )
    card(
        draw,
        120,
        "Hair  |  두피",
        [
            "완전 동일 파일 겹침 발견",
            "서로 다른 라벨의 동일 파일도 존재",
            "중복·라벨 충돌 정리",
            "원본 재분할 + Train 증강 재생성",
        ],
    )
    card(
        draw,
        1140,
        "Skin  |  확대 피부",
        [
            "동일 RGB 픽셀 겹침 발견",
            "원본 중복 51장 제거",
            "원본 재분할 + Train 증강 재생성",
            "생성 단계 분할 간 중복 0그룹",
        ],
    )
    card(
        draw,
        2160,
        "Web Skin  |  얼굴",
        [
            "파일·픽셀 중복 발견되지 않음",
            "라벨 충돌 발견되지 않음",
            "기존 데이터 분할 유지",
            "검사한 ZIP을 사용해 재학습",
        ],
        GREEN,
    )
    text(draw, (120, 1170), "공통 원칙", 44, bold=True)
    text(
        draw,
        (120, 1280),
        "학습용 Train과 평가용 Validation·Test를 구분하고, 평가 사진은 증강하지 않음",
        39,
    )
    text(
        draw,
        (120, 1400),
        "Hair·Skin은 분할이 바뀌었으므로 정리 전후 점수 차이를 학습 기법의 효과로만 해석하지 않음",
        35,
    )
    footer(
        draw, "출처: Hair 실험 정리 · Skin 데이터 감사 · Hair/Web Skin 실험 설명 (2026-09-07~09)"
    )
    save(image, "01_dataset_validation_overview")
    detail(
        "Hair",
        14500,
        12536,
        "새 분할  Train 10,032 / Validation 1,252 / Test 1,252",
        [195, 178, 16],
        "개",
        ["원본 모으기", "중복·충돌 정리", "80:10:10 재분할", "Train만 새 증강"],
        "학습 구성: 원본 10,032장 + 새 증강 5,015장 = 15,047장 · 평가에는 원본만 사용",
        "출처: docs/research/HAIR_EXPERIMENT_SUMMARY_20260907.md, §3–4 · 개: 완전 동일 파일 수",
    )
    detail(
        "Skin",
        9000,
        8949,
        "새 분할  Train 7,249 / Validation 1,000 / Test 700",
        [10, 5, 1],
        "그룹",
        ["원본 모으기", "동일 픽셀 정리", "클래스별 재분할", "Train만 새 증강"],
        "생성 단계 검사: 분할 간 중복 0그룹 · 원본 7,249장 + 새 증강 7,249장 = 14,498장",
        "출처: docs/research/SKIN_DATA_AUDIT_20260909.md · 생성 후 별도 공통 감사는 생략",
    )
    files = [
        "HAIR_EXPERIMENT_SUMMARY_20260907.md",
        "SKIN_DATA_AUDIT_20260909.md",
        "HAIR_WEB_SKIN_EXPERIMENT_EXPLAINED_20260908.md",
    ]
    manifest = {}
    for name in files:
        path = ROOT / "docs/research" / name
        manifest[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 데이터 검증 발표 자료\n\n16:9, 3200×1800 PNG 3장. 기존 문서의 보고값을 사용했습니다.\n"
        "실제 중복 사진 예시는 포함하지 않았습니다. "
        "새 데이터 검사나 학습을 수행한 결과가 아닙니다.\n"
        "Hair 쌍별 동일 파일 수와 Skin 픽셀 중복 그룹은 다른 집계 단위입니다.\n"
        "Hair 원본 수 차이는 중복 및 라벨 충돌 정리 후 차이이며, 쌍별 겹침 수 합이 아닙니다.\n"
        "발표 순서: 전체 검증 개요 → Hair 정리 → Skin 정리. Web Skin은 기존 분할을 유지했습니다.\n",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
