import argparse
from pathlib import Path
import shutil, random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from tqdm import tqdm

# =========================
# 설정
# =========================
parser = argparse.ArgumentParser(
    description="USB 현미경 두피 5-class 데이터셋을 전처리합니다."
)
parser.add_argument("--source", type=Path, required=True, help="클래스 폴더가 있는 원본 폴더")
parser.add_argument("--output", type=Path, required=True, help="처리 결과를 저장할 폴더")
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="출력 폴더가 이미 있으면 삭제 후 다시 생성",
)
args = parser.parse_args()

SOURCE_ROOT = args.source.expanduser().resolve()
OUTPUT_ROOT = args.output.expanduser().resolve()

CLASSES = ["모낭사이홍반", "미세각질", "비듬", "탈모", "피지과다"]

TARGET = 2900
TRAIN = 2320
VAL = 290
TEST = 290

# Train 원본의 50%만 증강
# 2320 + 1160 = 3480장/class
AUG_COUNT = 1160

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# 유틸
# =========================
def images(folder):
    return sorted([
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    ])

def save_jpg(img, path):
    img.convert("RGB").save(
        path, "JPEG", quality=95, optimize=True
    )

# =========================
# 현미경 환경용 적당한 증강
# =========================
# 과도한 변형 대신 실제 USB 현미경 촬영에서 발생할 수 있는
# 밝기/대비/색상/초점/센서 노이즈/미세 회전 등을 중심으로 사용.
def augment(img):
    img = img.convert("RGB")

    if random.random() < 0.65:
        img = img.rotate(
            random.uniform(-8, 8),
            resample=Image.Resampling.BILINEAR
        )

    if random.random() < 0.25:
        img = ImageOps.mirror(img)

    if random.random() < 0.75:
        img = ImageEnhance.Brightness(
            img
        ).enhance(random.uniform(0.82, 1.18))

    if random.random() < 0.65:
        img = ImageEnhance.Contrast(
            img
        ).enhance(random.uniform(0.85, 1.15))

    if random.random() < 0.35:
        img = ImageEnhance.Color(
            img
        ).enhance(random.uniform(0.90, 1.10))

    if random.random() < 0.25:
        img = img.filter(
            ImageFilter.GaussianBlur(
                random.uniform(0.15, 0.45)
            )
        )

    if random.random() < 0.25:
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.normal(
            0, random.uniform(1.5, 4.0), arr.shape
        )
        img = Image.fromarray(
            np.clip(arr, 0, 255).astype(np.uint8)
        )

    # 약한 crop/확대
    if random.random() < 0.30:
        w, h = img.size
        scale = random.uniform(0.92, 1.0)
        cw, ch = int(w * scale), int(h * scale)

        if cw < w and ch < h:
            left = random.randint(0, w - cw)
            top = random.randint(0, h - ch)
            img = img.crop((left, top, left + cw, top + ch))
            img = img.resize(
                (w, h),
                Image.Resampling.BILINEAR
            )

    return img

# =========================
# 입력 확인
# =========================
if not SOURCE_ROOT.exists():
    raise FileNotFoundError(f"원본 폴더가 없습니다: {SOURCE_ROOT}")

for cls in CLASSES:
    if not (SOURCE_ROOT / cls).is_dir():
        raise FileNotFoundError(
            f"클래스 폴더가 없습니다: {SOURCE_ROOT / cls}"
        )

# =========================
# 출력 폴더 초기화
# =========================
if OUTPUT_ROOT.exists():
    if not args.overwrite:
        raise FileExistsError(
            f"출력 폴더가 이미 있습니다: {OUTPUT_ROOT}\n"
            "삭제 후 다시 생성하려면 --overwrite 옵션을 사용하세요."
        )
    print(f"기존 출력 폴더 삭제: {OUTPUT_ROOT}")
    shutil.rmtree(OUTPUT_ROOT)

for typ in ["original", "augmented"]:
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (OUTPUT_ROOT / typ / split / cls).mkdir(
                parents=True, exist_ok=True
            )

# =========================
# 클래스별 처리
# =========================
for cls in CLASSES:
    src_files = images(SOURCE_ROOT / cls)

    print(f"\n[{cls}] 발견: {len(src_files):,}장")

    if len(src_files) < TARGET:
        raise ValueError(
            f"{cls}: {TARGET}장이 필요하지만 "
            f"{len(src_files)}장만 있습니다."
        )

    # 2900장을 먼저 무작위 선택
    selected = random.sample(src_files, TARGET)
    random.shuffle(selected)

    train_files = selected[:TRAIN]
    val_files = selected[TRAIN:TRAIN + VAL]
    test_files = selected[TRAIN + VAL:]

    assert len(train_files) == TRAIN
    assert len(val_files) == VAL
    assert len(test_files) == TEST

    # -------------------------
    # Original: 8:1:1
    # -------------------------
    for split, files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files)
    ]:
        dst_dir = OUTPUT_ROOT / "original" / split / cls

        for i, src in enumerate(
            tqdm(files, desc=f"{cls} original/{split}"),
            1
        ):
            try:
                with Image.open(src) as im:
                    save_jpg(
                        im,
                        dst_dir / f"{cls}_{i:04d}.jpg"
                    )
            except Exception as e:
                print(f"\n[원본 실패] {src}\n{e}")

    # -------------------------
    # Augmented Val/Test
    # 증강하지 않고 Original을 그대로 복사
    # -------------------------
    for split in ["val", "test"]:
        src_dir = OUTPUT_ROOT / "original" / split / cls
        dst_dir = OUTPUT_ROOT / "augmented" / split / cls

        for src in tqdm(
            images(src_dir),
            desc=f"{cls} augmented/{split} 원본복사"
        ):
            shutil.copy2(src, dst_dir / src.name)

    # -------------------------
    # Augmented Train
    # 원본 2320 + 증강 1160
    # -------------------------
    src_dir = OUTPUT_ROOT / "original" / "train" / cls
    dst_dir = OUTPUT_ROOT / "augmented" / "train" / cls

    # 원본 Train 그대로 복사
    train_originals = images(src_dir)

    for src in tqdm(
        train_originals,
        desc=f"{cls} augmented/train 원본복사"
    ):
        shutil.copy2(src, dst_dir / src.name)

    # 1160개의 추가 증강 이미지 생성
    for i in tqdm(
        range(AUG_COUNT),
        desc=f"{cls} augmented/train 증강"
    ):
        src = random.choice(train_originals)

        try:
            with Image.open(src) as im:
                aug = augment(im)

            save_jpg(
                aug,
                dst_dir / f"{cls}_aug_{i+1:04d}.jpg"
            )

        except Exception as e:
            print(f"\n[증강 실패] {src}\n{e}")

# =========================
# 최종 검증
# =========================
print("\n" + "=" * 70)
print("최종 데이터 개수 확인")
print("=" * 70)

for typ in ["original", "augmented"]:
    print(f"\n[{typ}]")

    for split in ["train", "val", "test"]:
        print(f"  {split}")

        for cls in CLASSES:
            n = len(
                images(
                    OUTPUT_ROOT / typ / split / cls
                )
            )

            expected = {
                ("original", "train"): 2320,
                ("original", "val"): 290,
                ("original", "test"): 290,
                ("augmented", "train"): 3480,
                ("augmented", "val"): 290,
                ("augmented", "test"): 290
            }[(typ, split)]

            status = "OK" if n == expected else "ERROR"

            print(
                f"    {cls:10s}: {n:4d} / "
                f"{expected:4d} [{status}]"
            )

print("\n완료!")
print(f"Original : {OUTPUT_ROOT / 'original'}")
print(f"Augmented: {OUTPUT_ROOT / 'augmented'}")
