from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 1080
BG = "#F7F9FC"
NAVY = "#172B4D"
TEXT = "#233044"
MUTED = "#66758C"
BLUE = "#3B82F6"
BLUE_LIGHT = "#EAF3FF"
TEAL = "#14B8A6"
TEAL_LIGHT = "#E7F8F5"
RED_LIGHT = "#FFF0F0"
RED = "#DC5A5A"
WHITE = "#FFFFFF"
LINE = "#DDE5F0"

FONT_REGULAR = Path("C:/Windows/Fonts/NotoSansKR-Regular.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/NotoSansKR-Bold.ttf")


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


def pill(draw, x, y, label, fill, color, max_width=None):
    f = font(22, True)
    bbox = draw.textbbox((0, 0), label, font=f)
    w = bbox[2] - bbox[0] + 30
    if max_width:
        w = min(w, max_width)
    rounded(draw, (x, y, x + w, y + 42), 21, fill)
    draw.text((x + w / 2, y + 20), label, font=f, fill=color, anchor="mm")
    return w


def arrow(draw, x1, x2, cy, color):
    draw.line((x1, cy, x2 - 16, cy), fill=color, width=5)
    draw.polygon([(x2 - 16, cy - 11), (x2, cy), (x2 - 16, cy + 11)], fill=color)


def card(draw, box, top_color, title_value, step, body_lines, accent):
    x1, y1, x2, y2 = box
    rounded(draw, box, 28, WHITE, LINE, 2)
    rounded(draw, (x1, y1, x2, y1 + 12), 6, top_color)
    rounded(draw, (x1 + 26, y1 + 30, x1 + 78, y1 + 82), 18, accent)
    text(draw, (x1 + 52, y1 + 56), str(step), 24, WHITE, True, "mm")
    text(draw, (x1 + 94, y1 + 55), title_value, 28, NAVY, True, "lm")
    y = y1 + 112
    for line_value, size, color, bold in body_lines:
        text(draw, (x1 + 30, y), line_value, size, color, bold)
        y += 38 if size <= 22 else 46


def chip_grid(draw, x, y, labels, fill, color, width=160, height=37, gap=10, cols=3, size=18):
    for i, label in enumerate(labels):
        cx = x + (i % cols) * (width + gap)
        cy = y + (i // cols) * (height + gap)
        rounded(draw, (cx, cy, cx + width, cy + height), 13, fill)
        text(draw, (cx + width / 2, cy + height / 2 - 1), label, size, color, True, "mm")


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    # Header
    pill(draw, 70, 48, "DATA SELECTION", BLUE_LIGHT, BLUE)
    text(draw, (70, 112), "질환 분류를 위한 초기 데이터 선별", 50, NAVY, True)
    text(
        draw,
        (70, 177),
        "AI Hub 원천 데이터에서 이미지를 직접 검수하고, 시각적으로 구분 가능한 클래스만 선정",
        25,
        MUTED,
    )
    draw.line((70, 226, 1850, 226), fill=LINE, width=2)

    # Domain labels
    rounded(draw, (70, 275, 255, 625), 32, BLUE)
    text(draw, (162, 335), "SKIN", 25, WHITE, True, "mm")
    text(draw, (162, 395), "피부질환", 37, WHITE, True, "mm")
    text(draw, (162, 447), "근접 이미지", 22, "#DCEBFF", False, "mm")
    draw.ellipse((127, 506, 197, 576), fill="#FFFFFF")
    draw.ellipse((145, 525, 179, 559), fill="#7FB0FF")

    rounded(draw, (70, 657, 255, 1007), 32, TEAL)
    text(draw, (162, 717), "HAIR", 25, WHITE, True, "mm")
    text(draw, (162, 777), "두피 증상", 37, WHITE, True, "mm")
    text(draw, (162, 829), "확대 이미지", 22, "#DDF9F4", False, "mm")
    for px in [125, 145, 165, 185]:
        draw.arc((px, 883, px + 38, 965), 180, 355, fill="#FFFFFF", width=5)

    # Shared geometry
    source_x = (295, 275, 665, 625)
    review_x = (735, 275, 1125, 625)
    selected_x = (1195, 275, 1850, 625)
    source_h = (295, 657, 665, 1007)
    review_h = (735, 657, 1125, 1007)
    selected_h = (1195, 657, 1850, 1007)

    card(
        draw,
        source_x,
        BLUE,
        "원천 데이터 확보",
        1,
        [
            ("AI Hub", 22, BLUE, True),
            ("피부종양 이미지 합성 데이터", 24, NAVY, True),
            ("15개 클래스", 33, BLUE, True),
            ("클래스당 약 1,000장", 22, TEXT, False),
            ("총 약 15,000장", 27, NAVY, True),
        ],
        BLUE,
    )
    card(
        draw,
        review_x,
        BLUE,
        "이미지 직접 검수",
        2,
        [
            ("병변 형태와 촬영 특성 확인", 23, NAVY, True),
            ("클래스 간 시각적 구분 가능성 검토", 21, TEXT, False),
            ("제외 기준", 20, RED, True),
            ("선택 클래스와 특징이 비슷하거나", 20, TEXT, False),
            ("육안 구분이 어려운 5개 클래스", 20, TEXT, False),
        ],
        BLUE,
    )
    card(
        draw,
        selected_x,
        BLUE,
        "분류 대상 확정",
        3,
        [
            ("10개 클래스 · 클래스당 1,000장", 27, BLUE, True),
            ("총 10,000장", 28, NAVY, True),
        ],
        BLUE,
    )
    skin_labels = [
        "광선각화증", "기저세포암", "보웬병", "사마귀", "지루각화증",
        "편평세포암", "표피낭종", "피부섬유종", "혈관종", "흑색점",
    ]
    chip_grid(draw, 1225, 480, skin_labels, BLUE_LIGHT, "#245DAE", width=140, cols=4, size=17)

    card(
        draw,
        source_h,
        TEAL,
        "원천 데이터 확보",
        1,
        [
            ("AI Hub", 22, TEAL, True),
            ("유형별 두피 이미지", 25, NAVY, True),
            ("고유 이미지", 22, TEXT, False),
            ("101,027장", 36, TEAL, True),
            ("두피 증상 이미지", 21, TEXT, False),
        ],
        TEAL,
    )
    card(
        draw,
        review_h,
        TEAL,
        "이미지 직접 검수",
        2,
        [
            ("두피 상태와 촬영 특성 확인", 23, NAVY, True),
            ("클래스 간 시각적 구분 가능성 검토", 21, TEXT, False),
            ("제외 기준", 20, RED, True),
            ("선택 클래스와 특징이 비슷하거나", 20, TEXT, False),
            ("육안 구분이 어려운 항목 제외", 20, TEXT, False),
        ],
        TEAL,
    )
    card(
        draw,
        selected_h,
        TEAL,
        "분류 대상 확정",
        3,
        [
            ("5개 클래스 · 클래스당 2,900장", 27, TEAL, True),
            ("총 14,500장", 28, NAVY, True),
        ],
        TEAL,
    )
    hair_labels = ["모낭사이홍반", "미세각질", "비듬", "탈모", "피지과다"]
    chip_grid(draw, 1225, 870, hair_labels, TEAL_LIGHT, "#087F73", width=180, cols=3, size=18)

    arrow(draw, 680, 720, 450, "#A9B9CE")
    arrow(draw, 1140, 1180, 450, "#A9B9CE")
    arrow(draw, 680, 720, 832, "#A9B9CE")
    arrow(draw, 1140, 1180, 832, "#A9B9CE")

    # Footer
    source_note = (
        "출처  AI Hub 피부종양 이미지 합성 데이터(71864)"
        " · AI Hub 유형별 두피 이미지(216)"
    )
    text(draw, (70, 1045), source_note, 18, MUTED)
    text(draw, (1850, 1045), "데이터 수집·클래스 선별 단계", 18, MUTED, True, "ra")

    out_dir = Path("results/presentation_assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "initial_skin_hair_data_selection.png"
    image.save(output, quality=96)
    print(output.resolve())


if __name__ == "__main__":
    main()
