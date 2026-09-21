from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 1080
BG = "#F7F9FC"
NAVY = "#172B4D"
TEXT = "#233044"
MUTED = "#66758C"
LINE = "#DDE5F0"
WHITE = "#FFFFFF"
BLUE = "#3B82F6"
BLUE_LIGHT = "#EAF3FF"
TEAL = "#14B8A6"
TEAL_LIGHT = "#E7F8F5"
ORANGE = "#F59E0B"
ORANGE_LIGHT = "#FFF4D8"
PURPLE = "#8B5CF6"
PURPLE_LIGHT = "#F0EAFF"
RED = "#E45B5B"
RED_LIGHT = "#FFF0F0"

FONT_REGULAR = Path("C:/Windows/Fonts/NotoSansKR-Regular.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/NotoSansKR-Bold.ttf")
OUT_DIR = Path("results/presentation_assets")


def font(size: int, bold: bool = False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, value, size, color=TEXT, bold=False, anchor=None, spacing=6):
    draw.multiline_text(
        xy,
        value,
        font=font(size, bold),
        fill=color,
        anchor=anchor,
        spacing=spacing,
    )


def pill(draw, x, y, label, fill, color, size=20, pad_x=25, height=42):
    current_font = font(size, True)
    bbox = draw.textbbox((0, 0), label, font=current_font)
    width = bbox[2] - bbox[0] + pad_x * 2
    rounded(draw, (x, y, x + width, y + height), height // 2, fill)
    draw.text(
        (x + width / 2, y + height / 2 - 1),
        label,
        font=current_font,
        fill=color,
        anchor="mm",
    )
    return width


def header(draw, eyebrow, title_value, subtitle):
    pill(draw, 70, 48, eyebrow, BLUE_LIGHT, BLUE, 21)
    text(draw, (70, 112), title_value, 49, NAVY, True)
    text(draw, (70, 177), subtitle, 25, MUTED)
    draw.line((70, 226, 1850, 226), fill=LINE, width=2)


def footer(draw, right_label):
    text(
        draw,
        (70, 1045),
        "대상  Skin 10-class · Hair 5-class",
        18,
        MUTED,
    )
    text(draw, (1850, 1045), right_label, 18, MUTED, True, "ra")


def domain_badge(draw, x, y, color, label, sublabel):
    rounded(draw, (x, y, x + 170, y + 185), 30, color)
    text(draw, (x + 85, y + 57), label, 30, WHITE, True, "mm")
    text(draw, (x + 85, y + 113), sublabel, 22, "#EFF6FF", False, "mm")


def arrow(draw, x1, x2, cy, color="#A9B9CE"):
    draw.line((x1, cy, x2 - 16, cy), fill=color, width=5)
    draw.polygon([(x2 - 16, cy - 11), (x2, cy), (x2 - 16, cy + 11)], fill=color)


def stage_card(draw, box, number, title_value, accent, lines):
    x1, y1, x2, y2 = box
    rounded(draw, box, 28, WHITE, LINE, 2)
    rounded(draw, (x1, y1, x2, y1 + 12), 6, accent)
    rounded(draw, (x1 + 28, y1 + 30, x1 + 80, y1 + 82), 18, accent)
    text(draw, (x1 + 54, y1 + 56), str(number), 24, WHITE, True, "mm")
    text(draw, (x1 + 97, y1 + 55), title_value, 28, NAVY, True, "lm")
    y = y1 + 115
    for value, size, color, bold in lines:
        text(draw, (x1 + 30, y), value, size, color, bold)
        y += 43 if size >= 24 else 36


def save(image, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    image.save(path, quality=96)
    print(path.resolve())


def create_split_visual():
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    header(
        draw,
        "02  DATA SPLIT",
        "선정 데이터를 학습·검증·평가용으로 분리",
        "모델이 학습한 데이터와 성능을 확인하는 데이터를 역할별로 나누어 구성",
    )

    rows = [
        {
            "y": 275,
            "color": BLUE,
            "light": BLUE_LIGHT,
            "label": "SKIN",
            "sub": "10-class",
            "selected": "선정 풀\n1,000장 / class",
            "used": "초기 사용\n900장 / class",
            "segments": [
                ("Train", "730", 730),
                ("Validation", "100", 100),
                ("Test", "70", 70),
            ],
            "totals": "전체  Train 7,300 · Validation 1,000 · Test 700",
            "note": (
                "나머지 100장/class는 당시 초기 분할에 포함되지 않았으며 "
                "세부 기준 기록이 남아 있지 않음"
            ),
        },
        {
            "y": 620,
            "color": TEAL,
            "light": TEAL_LIGHT,
            "label": "HAIR",
            "sub": "5-class",
            "selected": "선정 데이터\n2,900장 / class",
            "used": "전체 사용\n2,900장 / class",
            "segments": [
                ("Train", "2,320", 2320),
                ("Validation", "290", 290),
                ("Test", "290", 290),
            ],
            "totals": "전체  Train 11,600 · Validation 1,450 · Test 1,450",
            "note": "각 클래스를 동일한 수량으로 맞춘 뒤 8:1:1로 분리",
        },
    ]

    segment_colors = [BLUE, ORANGE, PURPLE]
    for row in rows:
        y = row["y"]
        domain_badge(draw, 70, y, row["color"], row["label"], row["sub"])
        rounded(draw, (275, y, 565, y + 185), 28, WHITE, LINE, 2)
        text(draw, (420, y + 60), row["selected"], 26, NAVY, True, "mm", 11)
        arrow(draw, 585, 640, y + 92)
        rounded(draw, (655, y, 945, y + 185), 28, row["light"])
        text(draw, (800, y + 60), row["used"], 26, row["color"], True, "mm", 11)
        arrow(draw, 965, 1020, y + 92)

        rounded(draw, (1035, y, 1850, y + 185), 28, WHITE, LINE, 2)
        text(draw, (1065, y + 28), "클래스당 분할", 21, NAVY, True)
        total = sum(item[2] for item in row["segments"])
        bar_x, bar_y, bar_w, bar_h = 1065, y + 70, 745, 52
        cursor = bar_x
        for index, (_name, _count, amount) in enumerate(row["segments"]):
            width = round(bar_w * amount / total)
            if index == len(row["segments"]) - 1:
                width = bar_x + bar_w - cursor
            draw.rectangle(
                (cursor, bar_y, cursor + width, bar_y + bar_h),
                fill=segment_colors[index],
            )
            cursor += width
        legend_x = 1065
        for index, (name, count, _) in enumerate(row["segments"]):
            draw.ellipse(
                (legend_x, y + 140, legend_x + 14, y + 154),
                fill=segment_colors[index],
            )
            text(draw, (legend_x + 23, y + 135), f"{name} {count}", 18, TEXT, True)
            legend_x += 205 if index == 0 else 230

        text(draw, (275, y + 217), row["totals"], 22, row["color"], True)
        text(draw, (275, y + 259), row["note"], 18, MUTED)

    rounded(draw, (70, 936, 1850, 1010), 24, "#EEF2F8")
    roles = [
        ("Train", "모델이 특징을 학습", BLUE),
        ("Validation", "학습 중 가장 좋은 모델을 선택", ORANGE),
        ("Test", "선택이 끝난 뒤 최종 성능을 평가", PURPLE),
    ]
    x = 115
    for name, desc, color in roles:
        pill(draw, x, 952, name, WHITE, color, 18, 20, 40)
        text(draw, (x + 135, 969), desc, 18, TEXT, False, "lm")
        x += 565
    footer(draw, "데이터 분할 계획")
    save(image, "02_initial_data_split.png")


def transform_chip(draw, x, y, label, fill, color, width=175):
    rounded(draw, (x, y, x + width, y + 46), 16, fill)
    text(draw, (x + width / 2, y + 22), label, 19, color, True, "mm")


def create_augmentation_visual():
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    header(
        draw,
        "03  AUGMENTATION",
        "Train 데이터만 증강해 촬영 환경 변화에 대비",
        "원본을 먼저 분리한 뒤 학습용 이미지에만 현실적인 변화를 적용",
    )

    # Principle flow
    flow_y = 270
    flow = [
        ("① 먼저 분할", "Train · Validation · Test", BLUE, BLUE_LIGHT),
        ("② Train만 변형", "원본 + 새 증강 이미지", TEAL, TEAL_LIGHT),
        ("③ 평가는 원본", "Validation · Test 유지", PURPLE, PURPLE_LIGHT),
    ]
    x_positions = [70, 670, 1270]
    for i, (title_value, sub, color, light) in enumerate(flow):
        x = x_positions[i]
        rounded(draw, (x, flow_y, x + 510, flow_y + 118), 26, light)
        text(draw, (x + 30, flow_y + 30), title_value, 25, color, True)
        text(draw, (x + 30, flow_y + 74), sub, 20, TEXT)
        if i < 2:
            arrow(draw, x + 530, x + 575, flow_y + 59)

    # Domain cards
    card_y1, card_y2 = 435, 925
    cards = [
        (70, 930, BLUE, BLUE_LIGHT, "SKIN", "피부 병변의 모양은 유지하고 촬영 조건만 변화"),
        (990, 1850, TEAL, TEAL_LIGHT, "HAIR", "작은 각질·모공 구조를 해치지 않도록 약하게 변화"),
    ]
    for x1, x2, color, light, label, description in cards:
        rounded(draw, (x1, card_y1, x2, card_y2), 30, WHITE, LINE, 2)
        rounded(draw, (x1, card_y1, x2, card_y1 + 12), 6, color)
        pill(draw, x1 + 30, card_y1 + 34, label, light, color, 22)
        text(draw, (x1 + 30, card_y1 + 105), description, 23, NAVY, True)

    skin_transforms = [
        "좌우 반전", "미세 회전", "밝기", "대비", "화이트밸런스",
        "감마", "가우시안 블러", "모션 블러", "노이즈", "JPEG 열화", "원근 변형",
    ]
    hair_transforms = [
        "좌우 반전", "미세 회전", "밝기", "대비",
        "색상", "블러", "노이즈", "확대·잘라내기",
    ]
    for i, label in enumerate(skin_transforms):
        x = 100 + (i % 4) * 198
        y = 590 + (i // 4) * 62
        transform_chip(draw, x, y, label, BLUE_LIGHT, "#245DAE", 180)
    for i, label in enumerate(hair_transforms):
        x = 1020 + (i % 3) * 255
        y = 590 + (i // 3) * 62
        transform_chip(draw, x, y, label, TEAL_LIGHT, "#087F73", 235)

    rounded(draw, (100, 806, 900, 885), 22, "#F2F6FC")
    text(draw, (125, 827), "목적", 19, BLUE, True)
    text(draw, (200, 827), "조명·각도·초점·압축 차이에 대한 적응", 20, TEXT, True)
    rounded(draw, (1020, 806, 1820, 885), 22, "#F1FAF8")
    text(draw, (1045, 827), "목적", 19, TEAL, True)
    text(draw, (1120, 827), "현미경 위치·조명 변화에 대한 적응", 20, TEXT, True)

    rounded(draw, (70, 950, 1850, 1012), 20, RED_LIGHT)
    text(
        draw,
        (960, 981),
        "핵심 원칙  Validation과 Test에는 증강 이미지를 넣지 않음",
        21,
        RED,
        True,
        "mm",
    )
    footer(draw, "초기 데이터 증강 계획")
    save(image, "03_initial_augmentation_plan.png")


def create_training_visual():
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    header(
        draw,
        "04  TRAINING PLAN",
        "ImageNet 사전학습 모델로 초기 분류 기준선 구축",
        "같은 학습 조건에서 Original과 Augmented를 따로 학습해 증강 효과를 비교",
    )

    # Main pipeline
    pipeline = [
        ("입력 이미지", "RGB\n224 × 224", BLUE, BLUE_LIGHT),
        ("사전학습 모델", "EfficientNet-B0\nImageNet", PURPLE, PURPLE_LIGHT),
        ("특징 추출", "기존 가중치 고정\n일반 이미지 특징 활용", TEAL, TEAL_LIGHT),
        ("새 분류층 학습", "Skin 10-class\nHair 5-class", ORANGE, ORANGE_LIGHT),
    ]
    x_positions = [70, 520, 970, 1420]
    for i, (title_value, body, color, light) in enumerate(pipeline):
        x = x_positions[i]
        rounded(draw, (x, 290, x + 360, 515), 30, WHITE, LINE, 2)
        rounded(draw, (x, 290, x + 360, 302), 6, color)
        rounded(draw, (x + 30, 330, x + 88, 388), 20, color)
        text(draw, (x + 59, 359), str(i + 1), 26, WHITE, True, "mm")
        text(draw, (x + 105, 359), title_value, 23, NAVY, True, "lm")
        rounded(draw, (x + 30, 415, x + 330, 488), 20, light)
        text(draw, (x + 180, 450), body, 20, color, True, "mm", 8)
        if i < 3:
            arrow(draw, x + 375, x + 430, 402)

    # Conditions
    text(draw, (70, 580), "초기 공통 학습 조건", 29, NAVY, True)
    settings = [
        ("Batch", "32", BLUE, BLUE_LIGHT),
        ("Epoch", "15", TEAL, TEAL_LIGHT),
        ("Optimizer", "Adam", PURPLE, PURPLE_LIGHT),
        ("Loss", "Cross Entropy", ORANGE, ORANGE_LIGHT),
    ]
    for i, (name, value, color, light) in enumerate(settings):
        x = 70 + i * 305
        rounded(draw, (x, 635, x + 270, 745), 24, light)
        text(draw, (x + 24, 660), name, 18, color, True)
        text(draw, (x + 24, 697), value, 26, NAVY, True)

    rounded(draw, (1310, 580, 1850, 745), 28, WHITE, LINE, 2)
    text(draw, (1340, 610), "두 번의 독립 학습", 24, NAVY, True)
    pill(draw, 1340, 662, "Original Train", BLUE_LIGHT, BLUE, 18, 18, 40)
    text(draw, (1540, 681), "VS", 19, MUTED, True, "mm")
    pill(draw, 1580, 662, "Augmented Train", TEAL_LIGHT, TEAL, 18, 18, 40)

    # Selection logic
    rounded(draw, (70, 805, 1850, 1008), 30, WHITE, LINE, 2)
    stages = [
        ("학습 기록", "Epoch마다 Train·Validation\nAccuracy와 Loss 저장", BLUE),
        ("모델 선택", "Validation Accuracy가\n가장 높은 모델 저장", ORANGE),
        ("최종 평가", "모델 선택 완료 후\nTest 데이터로 한 번 평가", PURPLE),
    ]
    for i, (title_value, body, color) in enumerate(stages):
        x = 110 + i * 585
        rounded(draw, (x, 840, x + 490, 970), 24, "#F7F9FC")
        pill(draw, x + 25, 862, str(i + 1), color, WHITE, 18, 15, 38)
        text(draw, (x + 88, 865), title_value, 22, NAVY, True)
        text(draw, (x + 88, 908), body, 18, TEXT, False, None, 7)
        if i < 2:
            arrow(draw, x + 510, x + 555, 906)
    footer(draw, "초기 모델 학습 계획")
    save(image, "04_initial_training_plan.png")


def main():
    create_split_visual()
    create_augmentation_visual()
    create_training_visual()


if __name__ == "__main__":
    main()
