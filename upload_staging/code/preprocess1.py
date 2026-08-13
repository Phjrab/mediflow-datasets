# ============================================================
# 웹캠 피부질환 데이터셋 전처리
# ============================================================
#
# 입력
# C:\Users\sejun\OneDrive\바탕 화면\web_skin\images
#
# 현재 구조
# train/
#   TS_건선_정면       800
#   TS_아토피_정면     800
#   TS_여드름_정면     800
#   TS_정상_정면       800
#   TS_주사_정면       800
#
# val/
#   VS_건선_정면       100
#   VS_아토피_정면     100
#   VS_여드름_정면     100
#   VS_정상_정면       100
#   VS_주사_정면       100
#
#
# 최종
#
# web_skin_processed/
#
# ├── ori
# │   ├── train
# │   ├── val
# │   └── test
# │
# └── aug
#     ├── train
#     ├── val
#     └── test
#
#
# Train
#   원본 720
#   + 증강 720
#   = 1440
#
# Val
#   원본 그대로 100
#
# Test
#   원본 그대로 80
#
# ============================================================


# ============================================================
# 1. 라이브러리
# ============================================================

import random
import shutil
import io

from pathlib import Path

import numpy as np

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter
)

from tqdm import tqdm


# ============================================================
# 2. 경로 설정
# ============================================================

SOURCE_DIR = Path(
    r"C:\Users\sejun\OneDrive\바탕 화면\web_skin\images"
)

OUTPUT_DIR = Path(
    r"C:\Users\sejun\OneDrive\바탕 화면\web_skin_processed"
)


# ============================================================
# 3. 기본 설정
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 4. 클래스
# ============================================================

CLASSES = [
    "건선",
    "아토피",
    "여드름",
    "정상",
    "주사"
]


# ============================================================
# 5. 이미지 확장자
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# 6. 기본 확인
# ============================================================

print()
print("=" * 75)
print("웹캠 피부질환 데이터셋 전처리")
print("=" * 75)

print()
print("입력:")
print(SOURCE_DIR)

print()
print("출력:")
print(OUTPUT_DIR)


if not SOURCE_DIR.exists():

    raise FileNotFoundError(
        f"\n입력 폴더가 없습니다.\n{SOURCE_DIR}"
    )


# ============================================================
# 7. 기존 결과 삭제
# ============================================================

if OUTPUT_DIR.exists():

    print()
    print("=" * 75)
    print("기존 처리 결과 삭제")
    print("=" * 75)

    print(OUTPUT_DIR)

    shutil.rmtree(
        OUTPUT_DIR
    )

    print("삭제 완료")


# ============================================================
# 8. 폴더 생성
# ============================================================

for dataset_type in [
    "ori",
    "aug"
]:

    for split in [
        "train",
        "val",
        "test"
    ]:

        for class_name in CLASSES:

            (
                OUTPUT_DIR
                / dataset_type
                / split
                / class_name
            ).mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# 9. 원본 폴더 찾기
# ============================================================

train_source = SOURCE_DIR / "train"
val_source = SOURCE_DIR / "val"


# ============================================================
# 10. 이미지 파일 검색 함수
# ============================================================

def get_images(folder):

    if not folder.exists():
        return []

    return sorted([
        p
        for p in folder.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ])


# ============================================================
# 11. 클래스별 원본 폴더 찾기
# ============================================================

def find_class_folder(parent, prefix, class_name):

    candidates = [

        folder

        for folder in parent.iterdir()

        if (
            folder.is_dir()
            and folder.name.startswith(prefix)
            and class_name in folder.name
        )

    ]

    if len(candidates) == 0:

        raise FileNotFoundError(
            f"\n{class_name} 폴더를 찾지 못했습니다.\n"
            f"위치: {parent}\n"
            f"prefix: {prefix}"
        )

    if len(candidates) > 1:

        print()
        print("[주의] 여러 폴더 발견:")
        for c in candidates:
            print(c)

    return candidates[0]


# ============================================================
# 12. 데이터 수집
# ============================================================

print()
print("=" * 75)
print("원본 데이터 확인")
print("=" * 75)


train_images = {}
val_images = {}


for class_name in CLASSES:

    train_folder = find_class_folder(
        train_source,
        "TS_",
        class_name
    )

    val_folder = find_class_folder(
        val_source,
        "VS_",
        class_name
    )


    train_list = get_images(
        train_folder
    )

    val_list = get_images(
        val_folder
    )


    train_images[class_name] = train_list
    val_images[class_name] = val_list


    print()
    print(f"[{class_name}]")
    print(
        f"Train 원본 : {len(train_list):,}장"
    )
    print(
        f"Val 원본   : {len(val_list):,}장"
    )


    # --------------------------------------------------------
    # 데이터 개수 검사
    # --------------------------------------------------------

    if len(train_list) != 800:

        print(
            f"⚠ Train이 800장이 아닙니다: "
            f"{len(train_list)}장"
        )

    if len(val_list) != 100:

        print(
            f"⚠ Val이 100장이 아닙니다: "
            f"{len(val_list)}장"
        )


# ============================================================
# 13. Train → Train/Test 분할
#
# 각 클래스에서 정확히 80장씩 Test로 분리
#
# Train = 720
# Test  = 80
#
# ============================================================

print()
print("=" * 75)
print("Train → Train / Test 분할")
print("=" * 75)


processed_train = {}
processed_test = {}


for class_name in CLASSES:

    images = train_images[
        class_name
    ].copy()


    # 랜덤 셔플
    random.shuffle(
        images
    )


    # 앞 80장 → Test
    test_list = images[:80]


    # 나머지 720장 → Train
    train_list = images[80:]


    processed_train[
        class_name
    ] = train_list


    processed_test[
        class_name
    ] = test_list


    print()
    print(
        f"{class_name:<8} "
        f"Train={len(train_list):3d} "
        f"Test={len(test_list):3d}"
    )


# ============================================================
# 14. ORIGINAL(ori) 저장
# ============================================================

print()
print("=" * 75)
print("ORI 데이터 생성")
print("=" * 75)


def copy_with_new_name(
    source_path,
    destination_dir,
    class_name,
    index
):

    extension = source_path.suffix.lower()

    destination_path = (
        destination_dir
        /
        f"{class_name}_{index:06d}{extension}"
    )

    shutil.copy2(
        source_path,
        destination_path
    )


# ============================================================
# 15. ORI Train
# ============================================================

for class_name in CLASSES:

    images = processed_train[
        class_name
    ]


    destination = (
        OUTPUT_DIR
        / "ori"
        / "train"
        / class_name
    )


    print()
    print(
        f"{class_name} / ori / train"
    )


    for index, source_path in enumerate(

        tqdm(
            images,
            desc="원본 Train 복사",
            unit="img"
        ),

        start=1

    ):

        copy_with_new_name(
            source_path,
            destination,
            class_name,
            index
        )


# ============================================================
# 16. ORI Val
# ============================================================

for class_name in CLASSES:

    images = val_images[
        class_name
    ]


    destination = (
        OUTPUT_DIR
        / "ori"
        / "val"
        / class_name
    )


    print()
    print(
        f"{class_name} / ori / val"
    )


    for index, source_path in enumerate(

        tqdm(
            images,
            desc="원본 Val 복사",
            unit="img"
        ),

        start=1

    ):

        copy_with_new_name(
            source_path,
            destination,
            class_name,
            index
        )


# ============================================================
# 17. ORI Test
# ============================================================

for class_name in CLASSES:

    images = processed_test[
        class_name
    ]


    destination = (
        OUTPUT_DIR
        / "ori"
        / "test"
        / class_name
    )


    print()
    print(
        f"{class_name} / ori / test"
    )


    for index, source_path in enumerate(

        tqdm(
            images,
            desc="원본 Test 복사",
            unit="img"
        ),

        start=1

    ):

        copy_with_new_name(
            source_path,
            destination,
            class_name,
            index
        )


# ============================================================
# 18. 웹캠 환경용 증강 함수
# ============================================================
#
# 너무 과격한 변형은 사용하지 않는다.
#
# 웹캠에서 실제로 발생할 수 있는:
#
# - 조명 변화
# - 밝기 변화
# - 색온도 변화
# - 카메라 노이즈
# - 약간의 초점 흐림
# - 약간의 움직임
# - 얼굴 각도 변화
# - JPEG 압축
#
# 등을 모사한다.
#
# ============================================================


# ------------------------------------------------------------
# 좌우 반전
# ------------------------------------------------------------

def aug_horizontal_flip(image):

    return image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )


# ------------------------------------------------------------
# 미세 회전
# ------------------------------------------------------------

def aug_rotation(image):

    angle = random.uniform(
        -7,
        7
    )

    return image.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=False
    )


# ------------------------------------------------------------
# 밝기
# ------------------------------------------------------------

def aug_brightness(image):

    factor = random.uniform(
        0.75,
        1.25
    )

    return ImageEnhance.Brightness(
        image
    ).enhance(
        factor
    )


# ------------------------------------------------------------
# 대비
# ------------------------------------------------------------

def aug_contrast(image):

    factor = random.uniform(
        0.80,
        1.20
    )

    return ImageEnhance.Contrast(
        image
    ).enhance(
        factor
    )


# ------------------------------------------------------------
# 채도
# ------------------------------------------------------------

def aug_saturation(image):

    factor = random.uniform(
        0.85,
        1.15
    )

    return ImageEnhance.Color(
        image
    ).enhance(
        factor
    )


# ------------------------------------------------------------
# 색온도 / 화이트밸런스 변화
# ------------------------------------------------------------

def aug_color_temperature(image):

    array = np.array(
        image
    ).astype(
        np.float32
    )


    shift = random.uniform(
        -8,
        8
    )


    # R / B 채널을 서로 반대 방향으로 조절
    array[:, :, 0] += shift
    array[:, :, 2] -= shift


    array = np.clip(
        array,
        0,
        255
    ).astype(
        np.uint8
    )


    return Image.fromarray(
        array
    )


# ------------------------------------------------------------
# 약한 Gaussian Blur
# ------------------------------------------------------------

def aug_blur(image):

    radius = random.uniform(
        0.2,
        0.7
    )

    return image.filter(
        ImageFilter.GaussianBlur(
            radius
        )
    )


# ------------------------------------------------------------
# 센서 노이즈
# ------------------------------------------------------------

def aug_noise(image):

    array = np.array(
        image
    ).astype(
        np.float32
    )


    noise = np.random.normal(
        0,
        random.uniform(
            1.0,
            3.0
        ),
        array.shape
    )


    array += noise


    array = np.clip(
        array,
        0,
        255
    ).astype(
        np.uint8
    )


    return Image.fromarray(
        array
    )


# ------------------------------------------------------------
# 미세한 Gamma 변화
# ------------------------------------------------------------

def aug_gamma(image):

    gamma = random.uniform(
        0.85,
        1.15
    )


    array = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0


    array = np.power(
        array,
        gamma
    )


    array = (
        array * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )


    return Image.fromarray(
        array
    )


# ------------------------------------------------------------
# JPEG 압축
# ------------------------------------------------------------

def aug_jpeg(image):

    quality = random.randint(
        75,
        95
    )


    buffer = io.BytesIO()


    image.save(
        buffer,
        format="JPEG",
        quality=quality
    )


    buffer.seek(0)


    return Image.open(
        buffer
    ).convert(
        "RGB"
    )


# ------------------------------------------------------------
# 미세 확대 / 크롭
# ------------------------------------------------------------

def aug_zoom(image):

    width, height = image.size


    scale = random.uniform(
        1.02,
        1.08
    )


    crop_width = int(
        width / scale
    )

    crop_height = int(
        height / scale
    )


    left = (
        width - crop_width
    ) // 2


    top = (
        height - crop_height
    ) // 2


    cropped = image.crop(
        (
            left,
            top,
            left + crop_width,
            top + crop_height
        )
    )


    return cropped.resize(
        (width, height),
        Image.Resampling.BILINEAR
    )


# ============================================================
# 19. 증강 목록
# ============================================================

AUGMENTATIONS = [

    aug_horizontal_flip,
    aug_rotation,
    aug_brightness,
    aug_contrast,
    aug_saturation,
    aug_color_temperature,
    aug_blur,
    aug_noise,
    aug_gamma,
    aug_jpeg,
    aug_zoom

]


# ============================================================
# 20. 랜덤 증강 생성
# ============================================================

def create_augmented_image(image):

    result = image.copy()


    # --------------------------------------------------------
    # 증강 개수
    #
    # 1~3개를 랜덤하게 선택
    # --------------------------------------------------------

    count = random.choices(

        [1, 2, 3],

        weights=[
            0.45,
            0.40,
            0.15
        ],

        k=1

    )[0]


    selected = random.sample(
        AUGMENTATIONS,
        count
    )


    # 순차 적용
    for augmentation in selected:

        result = augmentation(
            result
        )


    return result


# ============================================================
# 21. AUGMENTED TRAIN 생성
#
# 핵심:
#
# ori/train
#     720
#
# +
#
# 증강본
#     720
#
# =
#
# aug/train
#     1440
#
# ============================================================

print()
print("=" * 75)
print("AUG TRAIN 생성")
print("=" * 75)


for class_name in CLASSES:

    source_dir = (
        OUTPUT_DIR
        / "ori"
        / "train"
        / class_name
    )


    destination_dir = (
        OUTPUT_DIR
        / "aug"
        / "train"
        / class_name
    )


    source_images = sorted([

        p

        for p in source_dir.iterdir()

        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )

    ])


    print()
    print(
        f"========== {class_name} =========="
    )


    print(
        f"원본 Train: "
        f"{len(source_images)}장"
    )


    # --------------------------------------------------------
    # 21-1. 원본 720장 복사
    # --------------------------------------------------------

    print()
    print("① 원본 Train 복사")


    for source_path in tqdm(

        source_images,

        desc="원본 복사",

        unit="img"

    ):

        destination_path = (
            destination_dir
            / source_path.name
        )


        shutil.copy2(
            source_path,
            destination_path
        )


    # --------------------------------------------------------
    # 21-2. 원본 각각 1장씩 증강
    # --------------------------------------------------------

    print()
    print("② 증강본 720장 생성")


    success = 0
    fail = 0


    for index, source_path in enumerate(

        tqdm(

            source_images,

            desc="웹캠 환경 증강",

            unit="img"

        ),

        start=1

    ):

        try:

            # 이미지 열기
            image = Image.open(
                source_path
            ).convert(
                "RGB"
            )


            # 랜덤 증강
            augmented = (
                create_augmented_image(
                    image
                )
            )


            # 저장
            output_path = (
                destination_dir
                /
                f"{class_name}_aug_{index:06d}.jpg"
            )


            augmented.save(
                output_path,
                "JPEG",
                quality=95
            )


            success += 1


        except Exception as e:

            fail += 1

            print()
            print(
                "[증강 실패]"
            )

            print(
                source_path
            )

            print(
                e
            )


    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    final_count = len([

        p

        for p in destination_dir.iterdir()

        if p.is_file()

    ])


    print()
    print(
        f"원본       : {len(source_images):,}"
    )

    print(
        f"증강 성공  : {success:,}"
    )

    print(
        f"증강 실패  : {fail:,}"
    )

    print(
        f"최종       : {final_count:,}"
    )


# ============================================================
# 22. AUG VAL / TEST
#
# 증강하지 않는다.
#
# ori/val  → aug/val
# ori/test → aug/test
#
# 그대로 복사
# ============================================================

print()
print("=" * 75)
print("AUG VAL / TEST 생성")
print("=" * 75)


for split in [
    "val",
    "test"
]:

    for class_name in CLASSES:

        source_dir = (
            OUTPUT_DIR
            / "ori"
            / split
            / class_name
        )


        destination_dir = (
            OUTPUT_DIR
            / "aug"
            / split
            / class_name
        )


        source_images = sorted([

            p

            for p in source_dir.iterdir()

            if p.is_file()

        ])


        print()
        print(
            f"{split} / {class_name}"
        )


        for source_path in tqdm(

            source_images,

            desc="원본 그대로 복사",

            unit="img"

        ):

            destination_path = (
                destination_dir
                / source_path.name
            )


            shutil.copy2(
                source_path,
                destination_path
            )


# ============================================================
# 23. 최종 검증
# ============================================================

print()
print("=" * 75)
print("최종 데이터셋 검증")
print("=" * 75)


for class_name in CLASSES:

    print()
    print(
        f"========== {class_name} =========="
    )


    for split in [
        "train",
        "val",
        "test"
    ]:

        ori_dir = (
            OUTPUT_DIR
            / "ori"
            / split
            / class_name
        )


        aug_dir = (
            OUTPUT_DIR
            / "aug"
            / split
            / class_name
        )


        ori_count = len([

            p

            for p in ori_dir.iterdir()

            if p.is_file()

        ])


        aug_count = len([

            p

            for p in aug_dir.iterdir()

            if p.is_file()

        ])


        if split == "train":

            expected_aug = (
                ori_count * 2
            )

        else:

            expected_aug = ori_count


        if aug_count == expected_aug:

            status = "✓ 정상"

        else:

            status = "⚠ 확인 필요"


        print(

            f"{split:<5} | "
            f"ori={ori_count:4d} | "
            f"aug={aug_count:4d} | "
            f"예상={expected_aug:4d} | "
            f"{status}"

        )


# ============================================================
# 24. 전체 개수
# ============================================================

print()
print("=" * 75)
print("전체 데이터 개수")
print("=" * 75)


for dataset_type in [
    "ori",
    "aug"
]:

    print()
    print(
        f"[{dataset_type.upper()}]"
    )


    total_train = 0
    total_val = 0
    total_test = 0


    for class_name in CLASSES:

        for split in [
            "train",
            "val",
            "test"
        ]:

            folder = (
                OUTPUT_DIR
                / dataset_type
                / split
                / class_name
            )


            count = len([

                p

                for p in folder.iterdir()

                if p.is_file()

            ])


            if split == "train":
                total_train += count

            elif split == "val":
                total_val += count

            else:
                total_test += count


    print(
        f"Train : {total_train:,}장"
    )

    print(
        f"Val   : {total_val:,}장"
    )

    print(
        f"Test  : {total_test:,}장"
    )

    print(
        f"전체  : "
        f"{total_train + total_val + total_test:,}장"
    )


# ============================================================
# 25. 최종 구조 출력
# ============================================================

print()
print("=" * 75)
print("최종 폴더 구조")
print("=" * 75)

print(
r"""
C:\Users\sejun\OneDrive\바탕 화면\web_skin_processed
│
├── ori
│   ├── train
│   │   ├── 건선       720
│   │   ├── 아토피     720
│   │   ├── 여드름     720
│   │   ├── 정상       720
│   │   └── 주사       720
│   │
│   ├── val
│   │   ├── 건선       100
│   │   ├── 아토피     100
│   │   ├── 여드름     100
│   │   ├── 정상       100
│   │   └── 주사       100
│   │
│   └── test
│       ├── 건선        80
│       ├── 아토피      80
│       ├── 여드름      80
│       ├── 정상        80
│       └── 주사        80
│
└── aug
    ├── train
    │   ├── 건선      1440
    │   ├── 아토피    1440
    │   ├── 여드름    1440
    │   ├── 정상      1440
    │   └── 주사      1440
    │
    ├── val
    │   ├── 건선       100
    │   ├── 아토피     100
    │   ├── 여드름     100
    │   ├── 정상       100
    │   └── 주사       100
    │
    └── test
        ├── 건선        80
        ├── 아토피      80
        ├── 여드름      80
        ├── 정상        80
        └── 주사        80
"""
)


# ============================================================
# 26. 완료
# ============================================================

print()
print("=" * 75)
print("전처리 완료")
print("=" * 75)

print()
print(
    "ORI  = 원본 데이터"
)

print(
    "AUG  = Train 원본 + 증강 / Val·Test 원본 그대로"
)

print()
print(
    "Train : 클래스당 720 원본 + 720 증강 = 1440"
)

print(
    "Val   : 클래스당 100"
)

print(
    "Test  : 클래스당 80"
)

print()
print(
    "원본 C:\\web_skin 데이터는 수정하지 않았습니다."
)