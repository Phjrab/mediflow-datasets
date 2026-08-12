import cv2
import numpy as np
import random
import shutil
from pathlib import Path


# ============================================================
# 1. 기본 설정
# ============================================================

SOURCE_ROOT = Path(r"C:\skin_dataset\image")
OUTPUT_ROOT = Path(r"C:\skin_dataset\processed")

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 2. 데이터 개수
# ============================================================

TRAIN_COUNT = 730
TEST_COUNT = 70
VAL_COUNT = 100

# 증강 데이터셋의 Train
AUGMENTED_TRAIN_COUNT = 1460


# ============================================================
# 3. 이미지 확장자
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# 4. 이미지 읽기
#    ★ 한글 경로 대응
# ============================================================

def read_image_unicode(path):

    try:
        data = np.fromfile(
            str(path),
            dtype=np.uint8
        )

        img = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR
        )

        return img

    except Exception as e:

        print(
            f"\n[읽기 오류] {path}"
        )

        print(e)

        return None


# ============================================================
# 5. 이미지 저장
#    ★ 한글 경로 대응
# ============================================================

def save_image_unicode(img, path):

    try:

        extension = path.suffix.lower()

        if extension == ".jpg":
            encode_ext = ".jpg"

        elif extension == ".jpeg":
            encode_ext = ".jpg"

        elif extension == ".png":
            encode_ext = ".png"

        elif extension == ".bmp":
            encode_ext = ".bmp"

        else:
            encode_ext = ".png"

        success, encoded = cv2.imencode(
            encode_ext,
            img
        )

        if not success:
            return False

        encoded.tofile(
            str(path)
        )

        return True

    except Exception as e:

        print(
            f"\n[저장 오류] {path}"
        )

        print(e)

        return False


# ============================================================
# 6. 이미지 목록 가져오기
# ============================================================

def get_images(folder):

    if not folder.exists():
        return []

    return sorted([
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTENSIONS
    ])


# ============================================================
# 7. 클래스 이름 정리
# ============================================================

def clean_class_name(folder_name):

    if folder_name.startswith("TS_"):
        return folder_name[3:]

    if folder_name.startswith("VS_"):
        return folder_name[3:]

    return folder_name


# ============================================================
# 8. 증강 함수
# ============================================================

def horizontal_flip(img):

    return cv2.flip(img, 1)


def rotation(img):

    h, w = img.shape[:2]

    angle = random.uniform(
        -10,
        10
    )

    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        angle,
        1.0
    )

    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REFLECT_101
    )


def brightness(img):

    value = random.randint(
        -25,
        25
    )

    return cv2.convertScaleAbs(
        img,
        alpha=1.0,
        beta=value
    )


def contrast(img):

    alpha = random.uniform(
        0.85,
        1.15
    )

    return cv2.convertScaleAbs(
        img,
        alpha=alpha,
        beta=0
    )


def white_balance(img):

    img_float = img.astype(
        np.float32
    )

    temperature = random.uniform(
        0.95,
        1.05
    )

    # BGR
    img_float[:, :, 0] *= temperature
    img_float[:, :, 2] *= (
        2.0 - temperature
    )

    img_float = np.clip(
        img_float,
        0,
        255
    )

    return img_float.astype(
        np.uint8
    )


def gamma_correction(img):

    gamma_value = random.uniform(
        0.85,
        1.15
    )

    inv_gamma = 1.0 / gamma_value

    table = np.array([
        (
            (i / 255.0)
            ** inv_gamma
        ) * 255
        for i in range(256)
    ]).astype(
        np.uint8
    )

    return cv2.LUT(
        img,
        table
    )


def gaussian_blur(img):

    kernel_size = random.choice([
        3,
        5
    ])

    return cv2.GaussianBlur(
        img,
        (
            kernel_size,
            kernel_size
        ),
        0
    )


def motion_blur(img):

    kernel_size = random.choice([
        3,
        5
    ])

    kernel = np.zeros(
        (
            kernel_size,
            kernel_size
        )
    )

    kernel[
        kernel_size // 2,
        :
    ] = 1

    kernel /= kernel_size

    return cv2.filter2D(
        img,
        -1,
        kernel
    )


def gaussian_noise(img):

    noise = np.random.normal(
        0,
        5,
        img.shape
    )

    result = (
        img.astype(
            np.float32
        )
        + noise
    )

    result = np.clip(
        result,
        0,
        255
    )

    return result.astype(
        np.uint8
    )


def jpeg_compression(img):

    quality = random.randint(
        60,
        85
    )

    params = [
        cv2.IMWRITE_JPEG_QUALITY,
        quality
    ]

    success, encoded = cv2.imencode(
        ".jpg",
        img,
        params
    )

    if not success:
        return img

    return cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )


def perspective(img):

    h, w = img.shape[:2]

    shift = 0.03

    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    dx = w * shift
    dy = h * shift

    dst = np.float32([
        [
            random.uniform(0, dx),
            random.uniform(0, dy)
        ],

        [
            w - random.uniform(0, dx),
            random.uniform(0, dy)
        ],

        [
            w - random.uniform(0, dx),
            h - random.uniform(0, dy)
        ],

        [
            random.uniform(0, dx),
            h - random.uniform(0, dy)
        ]
    ])

    matrix = cv2.getPerspectiveTransform(
        src,
        dst
    )

    return cv2.warpPerspective(
        img,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REFLECT_101
    )


# ============================================================
# 9. 증강 목록
# ============================================================

AUGMENTATIONS = {

    "horizontal_flip":
        horizontal_flip,

    "rotation":
        rotation,

    "brightness":
        brightness,

    "contrast":
        contrast,

    "white_balance":
        white_balance,

    "gamma":
        gamma_correction,

    "gaussian_blur":
        gaussian_blur,

    "motion_blur":
        motion_blur,

    "gaussian_noise":
        gaussian_noise,

    "jpeg_compression":
        jpeg_compression,

    "perspective":
        perspective
}


# ============================================================
# 10. 랜덤 증강
# ============================================================

def apply_random_augmentation(img):

    name = random.choice(
        list(AUGMENTATIONS.keys())
    )

    function = AUGMENTATIONS[
        name
    ]

    result = function(img)

    return result, name


# ============================================================
# 11. 진행률
# ============================================================

def show_progress(
    current,
    total,
    label
):

    percent = (
        current / total
    ) * 100

    bar_length = 30

    filled = int(
        bar_length
        * current
        / total
    )

    bar = (
        "█" * filled
        +
        "-" * (
            bar_length - filled
        )
    )

    print(
        f"\r{label} "
        f"[{bar}] "
        f"{current}/{total} "
        f"({percent:5.1f}%)",
        end=""
    )

    if current == total:
        print()


# ============================================================
# 12. 메인
# ============================================================

def main():

    print()
    print("=" * 70)
    print("피부질환 데이터셋 생성 시작")
    print("=" * 70)

    print()
    print(
        "원본 위치:"
    )
    print(
        SOURCE_ROOT
    )

    print()
    print(
        "결과 위치:"
    )
    print(
        OUTPUT_ROOT
    )

    # --------------------------------------------------------
    # 원본 폴더 확인
    # --------------------------------------------------------

    train_root = (
        SOURCE_ROOT /
        "train"
    )

    val_root = (
        SOURCE_ROOT /
        "val"
    )

    if not train_root.exists():

        print()
        print(
            "❌ Train 폴더가 없습니다."
        )

        print(
            train_root
        )

        return

    if not val_root.exists():

        print()
        print(
            "❌ Val 폴더가 없습니다."
        )

        print(
            val_root
        )

        return


    # --------------------------------------------------------
    # 기존 결과 삭제
    # --------------------------------------------------------

    if OUTPUT_ROOT.exists():

        print()
        print(
            "⚠️ 기존 processed 폴더가 있습니다."
        )

        answer = input(
            "삭제하고 다시 생성할까요? "
            "(y/n): "
        )

        if answer.lower() != "y":

            print(
                "작업을 취소했습니다."
            )

            return

        shutil.rmtree(
            OUTPUT_ROOT
        )


    # --------------------------------------------------------
    # 결과 폴더 생성
    # --------------------------------------------------------

    for dataset in [
        "original",
        "augmented"
    ]:

        for split in [
            "train",
            "val",
            "test"
        ]:

            (
                OUTPUT_ROOT
                / dataset
                / split
            ).mkdir(
                parents=True,
                exist_ok=True
            )


    # --------------------------------------------------------
    # 클래스 자동 탐색
    # --------------------------------------------------------

    class_dirs = sorted([
        p
        for p in train_root.iterdir()
        if p.is_dir()
    ])


    print()
    print(
        f"발견된 클래스: "
        f"{len(class_dirs)}개"
    )

    for folder in class_dirs:

        print(
            f"  - {folder.name}"
        )


    if len(class_dirs) != 10:

        print()
        print(
            "⚠️ 현재 클래스가 "
            f"{len(class_dirs)}개입니다."
        )

        print(
            "10개가 맞는지 확인하세요."
        )


    # --------------------------------------------------------
    # 증강 통계
    # --------------------------------------------------------

    augmentation_stats = {
        name: 0
        for name in AUGMENTATIONS
    }


    # ========================================================
    # 클래스별 처리
    # ========================================================

    for class_number, train_class_dir in enumerate(
        class_dirs,
        start=1
    ):

        class_name = clean_class_name(
            train_class_dir.name
        )

        print()
        print()
        print("=" * 70)

        print(
            f"[{class_number}/{len(class_dirs)}] "
            f"{class_name}"
        )

        print("=" * 70)


        # ----------------------------------------------------
        # Train 이미지 찾기
        # ----------------------------------------------------

        train_images = get_images(
            train_class_dir
        )

        print(
            f"Train 원본: "
            f"{len(train_images)}장"
        )


        if len(train_images) < 800:

            print()
            print(
                "❌ Train 이미지가 "
                "800장보다 적습니다."
            )

            continue


        # ----------------------------------------------------
        # Train/Test 분리
        # ----------------------------------------------------

        random.shuffle(
            train_images
        )

        test_images = train_images[
            :TEST_COUNT
        ]

        train_selected = train_images[
            TEST_COUNT:
            TEST_COUNT + TRAIN_COUNT
        ]


        # ----------------------------------------------------
        # Val 폴더 찾기
        # ----------------------------------------------------

        val_class_dir = (
            val_root /
            f"VS_{class_name}"
        )

        # 혹시 접두사가 없는 경우
        if not val_class_dir.exists():

            val_class_dir = (
                val_root /
                class_name
            )


        if not val_class_dir.exists():

            print()
            print(
                "❌ Val 폴더를 찾을 수 없습니다:"
            )

            print(
                val_root
            )

            continue


        val_images = get_images(
            val_class_dir
        )


        print(
            f"Val 원본: "
            f"{len(val_images)}장"
        )


        if len(val_images) < 100:

            print(
                "❌ Val 이미지가 "
                "100장보다 적습니다."
            )

            continue


        random.shuffle(
            val_images
        )

        val_selected = val_images[
            :VAL_COUNT
        ]


        # ====================================================
        # 출력 폴더
        # ====================================================

        paths = {}

        for dataset in [
            "original",
            "augmented"
        ]:

            for split in [
                "train",
                "val",
                "test"
            ]:

                path = (
                    OUTPUT_ROOT
                    / dataset
                    / split
                    / class_name
                )

                path.mkdir(
                    parents=True,
                    exist_ok=True
                )

                paths[
                    f"{dataset}_{split}"
                ] = path


        # ====================================================
        # ORIGINAL TRAIN
        # ====================================================

        print()
        print(
            "▶ Original Train"
        )

        success_count = 0

        for i, image_path in enumerate(
            train_selected,
            start=1
        ):

            img = read_image_unicode(
                image_path
            )

            if img is None:
                continue

            output = (
                paths["original_train"]
                /
                f"{class_name}_{i:04d}.png"
            )

            if save_image_unicode(
                img,
                output
            ):

                success_count += 1

            show_progress(
                i,
                TRAIN_COUNT,
                "Original Train"
            )


        # ====================================================
        # ORIGINAL VAL
        # ====================================================

        print()
        print(
            "▶ Original Val"
        )

        for i, image_path in enumerate(
            val_selected,
            start=1
        ):

            img = read_image_unicode(
                image_path
            )

            if img is None:
                continue

            output = (
                paths["original_val"]
                /
                f"{class_name}_{i:04d}.png"
            )

            save_image_unicode(
                img,
                output
            )

            show_progress(
                i,
                VAL_COUNT,
                "Original Val"
            )


        # ====================================================
        # ORIGINAL TEST
        # ====================================================

        print()
        print(
            "▶ Original Test"
        )

        for i, image_path in enumerate(
            test_images,
            start=1
        ):

            img = read_image_unicode(
                image_path
            )

            if img is None:
                continue

            output = (
                paths["original_test"]
                /
                f"{class_name}_{i:04d}.png"
            )

            save_image_unicode(
                img,
                output
            )

            show_progress(
                i,
                TEST_COUNT,
                "Original Test"
            )


        # ====================================================
        # AUGMENTED TRAIN
        # ====================================================

        print()
        print(
            "▶ Augmented Train"
        )

        # ----------------------------------------------------
        # 730장 원본
        # ----------------------------------------------------

        for i, image_path in enumerate(
            train_selected,
            start=1
        ):

            img = read_image_unicode(
                image_path
            )

            if img is None:
                continue

            output = (
                paths["augmented_train"]
                /
                f"{class_name}_{i:04d}.png"
            )

            save_image_unicode(
                img,
                output
            )

            show_progress(
                i,
                AUGMENTED_TRAIN_COUNT,
                "Augmented 원본"
            )


        # ----------------------------------------------------
        # 730장 증강
        # ----------------------------------------------------

        for i in range(
            TRAIN_COUNT
        ):

            # 원본 730장 중 랜덤 선택
            source_path = random.choice(
                train_selected
            )

            img = read_image_unicode(
                source_path
            )

            if img is None:
                continue

            # ★ 정확히 하나의 증강만 적용
            augmented_img, aug_name = (
                apply_random_augmentation(
                    img
                )
            )

            augmentation_stats[
                aug_name
            ] += 1

            output_number = (
                TRAIN_COUNT + i + 1
            )

            output = (
                paths["augmented_train"]
                /
                f"{class_name}_{output_number:04d}.png"
            )

            save_image_unicode(
                augmented_img,
                output
            )

            show_progress(
                TRAIN_COUNT + i + 1,
                AUGMENTED_TRAIN_COUNT,
                "Augmented Train"
            )


        # ====================================================
        # AUGMENTED VAL
        # ====================================================

        print()
        print(
            "▶ Augmented Val"
        )

        # 증강하지 않고 원본 그대로
        for i, image_path in enumerate(
            val_selected,
            start=1
        ):

            img = read_image_unicode(
                image_path
            )

            if img is None:
                continue

            output = (
                paths["augmented_val"]
                /
                f"{class_name}_{i:04d}.png"
            )

            save_image_unicode(
                img,
                output
            )

            show_progress(
                i,
                VAL_COUNT,
                "Augmented Val"
            )


        # ====================================================
        # AUGMENTED TEST
        # ====================================================

        print()
        print(
            "▶ Augmented Test"
        )

        # 증강하지 않고 원본 그대로
        for i, image_path in enumerate(
            test_images,
            start=1
        ):

            img = read_image_unicode(
                image_path
            )

            if img is None:
                continue

            output = (
                paths["augmented_test"]
                /
                f"{class_name}_{i:04d}.png"
            )

            save_image_unicode(
                img,
                output
            )

            show_progress(
                i,
                TEST_COUNT,
                "Augmented Test"
            )


        # ====================================================
        # 클래스 결과 확인
        # ====================================================

        original_train_count = len(
            get_images(
                paths["original_train"]
            )
        )

        original_val_count = len(
            get_images(
                paths["original_val"]
            )
        )

        original_test_count = len(
            get_images(
                paths["original_test"]
            )
        )

        augmented_train_count = len(
            get_images(
                paths["augmented_train"]
            )
        )

        augmented_val_count = len(
            get_images(
                paths["augmented_val"]
            )
        )

        augmented_test_count = len(
            get_images(
                paths["augmented_test"]
            )
        )


        print()
        print(
            f"✅ {class_name} 완료"
        )

        print(
            f"   Original  : "
            f"{original_train_count} / "
            f"{original_val_count} / "
            f"{original_test_count}"
        )

        print(
            f"   Augmented : "
            f"{augmented_train_count} / "
            f"{augmented_val_count} / "
            f"{augmented_test_count}"
        )


    # ========================================================
    # 최종 증강 통계
    # ========================================================

    print()
    print("=" * 70)
    print(
        "증강 기법 사용 통계"
    )
    print("=" * 70)


    total_augmented = sum(
        augmentation_stats.values()
    )


    for name, count in (
        augmentation_stats.items()
    ):

        percentage = (
            count /
            total_augmented *
            100
            if total_augmented > 0
            else 0
        )

        print(
            f"{name:20s}"
            f": {count:5d}회 "
            f"({percentage:5.1f}%)"
        )


    # ========================================================
    # 전체 파일 수 확인
    # ========================================================

    print()
    print("=" * 70)
    print(
        "전체 데이터셋 최종 확인"
    )
    print("=" * 70)


    for dataset in [
        "original",
        "augmented"
    ]:

        print()
        print(
            f"[{dataset}]"
        )

        for split in [
            "train",
            "val",
            "test"
        ]:

            split_path = (
                OUTPUT_ROOT
                / dataset
                / split
            )

            total = 0

            for class_dir in split_path.iterdir():

                if class_dir.is_dir():

                    total += len(
                        get_images(
                            class_dir
                        )
                    )

            print(
                f"  {split:5s}: "
                f"{total:,}장"
            )


    # ========================================================
    # 완료
    # ========================================================

    print()
    print("=" * 70)
    print(
        "🎉 데이터셋 생성 완료!"
    )
    print("=" * 70)

    print()
    print(
        f"저장 위치:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "※ 이미지 크기는 원본 그대로 유지"
    )

    print(
        "※ 224×224 Resize는 Colab 학습 단계에서 수행"
    )

    print(
        "※ JSON 파일은 사용하지 않음"
    )


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()