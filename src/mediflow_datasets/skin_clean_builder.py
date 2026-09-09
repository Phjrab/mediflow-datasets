"""Build a leakage-reduced Skin dataset from the audited processed ZIP.

The builder never changes its input. Exact RGB-pixel duplicates are grouped before
the new split. Person, lesion, and capture-session grouping remains unavailable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import stat
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
AUGMENTATIONS = (
    "horizontal_flip",
    "rotation",
    "brightness",
    "contrast",
    "white_balance",
    "gamma",
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "jpeg_compression",
    "perspective",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def pixel_hash(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256(str(rgb.size).encode("ascii") + b":" + rgb.tobytes())
        return digest.hexdigest(), rgb.size


def validate_members(archive):
    seen = set()
    for item in archive.infolist():
        name = item.orig_filename
        path = PurePosixPath(name)
        key = name.rstrip("/").casefold()
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or stat.S_ISLNK(item.external_attr >> 16)
        ):
            raise ValueError("Unsafe ZIP member: " + name)
        if key in seen:
            raise ValueError("Duplicate ZIP destination: " + name)
        seen.add(key)


def extract_original(source_zip, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(source_zip) as archive:
        validate_members(archive)
        members = [
            item
            for item in archive.infolist()
            if "original" in (part.lower() for part in PurePosixPath(item.filename).parts)
        ]
        if not members:
            raise ValueError("ZIP에서 original 폴더를 찾지 못했습니다.")
        destination.mkdir(parents=True)
        for item in members:
            archive.extract(item, destination)


def find_original_root(extracted):
    matches = [
        path
        for path in Path(extracted).rglob("*")
        if path.is_dir()
        and path.name.lower() == "original"
        and all((path / split).is_dir() for split in ("train", "val", "test"))
    ]
    if len(matches) != 1:
        raise ValueError(f"original/train,val,test 구조가 하나여야 합니다: {matches}")
    return matches[0]


def inventory(root, classes):
    rows = []
    for split in ("train", "val", "test"):
        found = sorted(path.name for path in (root / split).iterdir() if path.is_dir())
        if found != sorted(classes):
            raise ValueError(f"클래스 폴더 불일치: {root / split}: {found}")
        for class_name in classes:
            files = sorted(
                path
                for path in (root / split / class_name).rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not files:
                raise ValueError(f"빈 클래스: {split}/{class_name}")
            for path in files:
                digest, size = pixel_hash(path)
                rows.append(
                    {
                        "class_name": class_name,
                        "old_split": split,
                        "source": path,
                        "source_relative": path.relative_to(root).as_posix(),
                        "file_sha256": file_hash(path),
                        "pixel_sha256": digest,
                        "width": size[0],
                        "height": size[1],
                    }
                )
    return rows


def split_plan(rows, classes, seed=42, validation_count=100, test_count=70):
    by_pixel = defaultdict(list)
    for row in rows:
        by_pixel[row["pixel_sha256"]].append(row)
    conflicts = [group for group in by_pixel.values() if len({r["class_name"] for r in group}) > 1]
    if conflicts:
        raise ValueError(f"동일 픽셀의 라벨 충돌 {len(conflicts)}그룹")
    unique_by_class = defaultdict(list)
    removed = []
    for digest, group in by_pixel.items():
        ordered = sorted(group, key=lambda row: row["source_relative"])
        representative = ordered[0]
        representative = {**representative, "duplicate_group_size": len(ordered)}
        unique_by_class[representative["class_name"]].append(representative)
        for duplicate in ordered[1:]:
            removed.append(
                {
                    "class_name": duplicate["class_name"],
                    "pixel_sha256": digest,
                    "kept_source": representative["source_relative"],
                    "removed_source": duplicate["source_relative"],
                    "reason": "same_rgb_pixels",
                }
            )
    plan = []
    for class_name in classes:
        values = unique_by_class[class_name]
        minimum = validation_count + test_count + 1
        if len(values) < minimum:
            raise ValueError(f"{class_name}: 고유 이미지 {len(values)}장, 최소 {minimum}장 필요")

        def score(row, class_value=class_name):
            value = f"skin_clean_v1|{seed}|{class_value}|{row['pixel_sha256']}"
            return hashlib.sha256(value.encode("utf-8")).hexdigest()

        ordered = sorted(values, key=score)
        assignments = (
            [("test", row) for row in ordered[:test_count]]
            + [("val", row) for row in ordered[test_count : test_count + validation_count]]
            + [("train", row) for row in ordered[test_count + validation_count :]]
        )
        for split, row in assignments:
            plan.append({**row, "new_split": split, "split_score": score(row)})
    return plan, removed


def read_cv(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지를 읽지 못했습니다: " + str(path))
    return image


def save_png(path, image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("PNG 인코딩 실패: " + str(path))
    encoded.tofile(path)


def augment(image, name, py_rng, np_rng):
    height, width = image.shape[:2]
    if name == "horizontal_flip":
        return cv2.flip(image, 1)
    if name == "rotation":
        angle = py_rng.uniform(-10, 10)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    if name == "brightness":
        return cv2.convertScaleAbs(image, alpha=1.0, beta=py_rng.randint(-25, 25))
    if name == "contrast":
        return cv2.convertScaleAbs(image, alpha=py_rng.uniform(0.85, 1.15), beta=0)
    if name == "white_balance":
        result = image.astype(np.float32)
        temperature = py_rng.uniform(0.95, 1.05)
        result[:, :, 0] *= temperature
        result[:, :, 2] *= 2.0 - temperature
        return np.clip(result, 0, 255).astype(np.uint8)
    if name == "gamma":
        gamma = py_rng.uniform(0.85, 1.15)
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(
            np.uint8
        )
        return cv2.LUT(image, table)
    if name == "gaussian_blur":
        size = py_rng.choice((3, 5))
        return cv2.GaussianBlur(image, (size, size), 0)
    if name == "motion_blur":
        size = py_rng.choice((3, 5))
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1 / size
        return cv2.filter2D(image, -1, kernel)
    if name == "gaussian_noise":
        result = image.astype(np.float32) + np_rng.normal(0, 5, image.shape)
        return np.clip(result, 0, 255).astype(np.uint8)
    if name == "jpeg_compression":
        quality = py_rng.randint(60, 85)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise ValueError("JPEG 증강 인코딩 실패")
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if name == "perspective":
        shift = 0.03
        source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        dx, dy = width * shift, height * shift
        target = np.float32(
            [
                [py_rng.uniform(0, dx), py_rng.uniform(0, dy)],
                [width - py_rng.uniform(0, dx), py_rng.uniform(0, dy)],
                [width - py_rng.uniform(0, dx), height - py_rng.uniform(0, dy)],
                [py_rng.uniform(0, dx), height - py_rng.uniform(0, dy)],
            ]
        )
        matrix = cv2.getPerspectiveTransform(source, target)
        return cv2.warpPerspective(
            image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101
        )
    raise ValueError(name)


def csv_write(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    original_root, destination, classes, seed=42, validation_count=100, test_count=70
):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    rows = inventory(Path(original_root), classes)
    plan, removed = split_plan(rows, classes, seed, validation_count, test_count)
    for kind in ("original", "augmented"):
        for split in ("train", "val", "test"):
            for class_name in classes:
                (destination / kind / split / class_name).mkdir(parents=True, exist_ok=False)
    lineage = []
    split_rows = []
    output_pixels = {"train": set(), "val": set(), "test": set()}
    for row in plan:
        code = f"C{classes.index(row['class_name']):02d}"
        extension = row["source"].suffix.lower()
        name = f"{code}_{row['pixel_sha256'][:20]}{extension}"
        split = row["new_split"]
        original_target = destination / "original" / split / row["class_name"] / name
        augmented_target = destination / "augmented" / split / row["class_name"] / name
        shutil.copyfile(row["source"], original_target)
        shutil.copyfile(row["source"], augmented_target)
        output_pixels[split].add(row["pixel_sha256"])
        split_rows.append(
            {
                **row,
                "source": row["source_relative"],
                "output": original_target.relative_to(destination).as_posix(),
            }
        )
    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    if any(output_pixels[left] & output_pixels[right] for left, right in split_pairs):
        raise RuntimeError("새 분할 사이에 픽셀 중복이 남았습니다.")
    generated_pixels = set(output_pixels["train"])
    train_rows = [row for row in plan if row["new_split"] == "train"]
    output_by_pixel = {row["pixel_sha256"]: row["output"] for row in split_rows}
    for row in train_rows:
        source_image = read_cv(row["source"])
        created = None
        for attempt in range(12):
            material = f"{seed}|{row['pixel_sha256']}|{attempt}"
            local_seed = int(hashlib.sha256(material.encode()).hexdigest()[:16], 16)
            py_rng = random.Random(local_seed)
            np_rng = np.random.default_rng(local_seed)
            name = AUGMENTATIONS[local_seed % len(AUGMENTATIONS)]
            candidate = augment(source_image, name, py_rng, np_rng)
            rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
            digest = hashlib.sha256(
                str((rgb.shape[1], rgb.shape[0])).encode("ascii") + b":" + rgb.tobytes()
            ).hexdigest()
            if (
                digest not in generated_pixels
                and digest not in output_pixels["val"]
                and digest not in output_pixels["test"]
            ):
                created = candidate, digest, name, attempt
                break
        if created is None:
            raise RuntimeError("고유한 증강 이미지를 만들지 못했습니다: " + row["source_relative"])
        candidate, digest, augmentation_name, attempt = created
        generated_pixels.add(digest)
        code = f"C{classes.index(row['class_name']):02d}"
        filename = f"{code}_aug_{row['pixel_sha256'][:12]}_{digest[:12]}.png"
        target = destination / "augmented" / "train" / row["class_name"] / filename
        save_png(target, candidate)
        lineage.append(
            {
                "augmented_path": target.relative_to(destination).as_posix(),
                "source_original_path": output_by_pixel[row["pixel_sha256"]],
                "class_name": row["class_name"],
                "source_pixel_sha256": row["pixel_sha256"],
                "augmented_pixel_sha256": digest,
                "augmentation": augmentation_name,
                "retry_index": attempt,
            }
        )
    csv_write(
        destination / "split_manifest.csv",
        split_rows,
        [
            "output",
            "class_name",
            "new_split",
            "source",
            "old_split",
            "file_sha256",
            "pixel_sha256",
            "width",
            "height",
            "duplicate_group_size",
            "split_score",
        ],
    )
    csv_write(
        destination / "removed_exact_duplicates.csv",
        removed,
        ["class_name", "pixel_sha256", "kept_source", "removed_source", "reason"],
    )
    csv_write(
        destination / "augmentation_lineage.csv",
        lineage,
        [
            "augmented_path",
            "source_original_path",
            "class_name",
            "source_pixel_sha256",
            "augmented_pixel_sha256",
            "augmentation",
            "retry_index",
        ],
    )
    counts = []
    for kind in ("original", "augmented"):
        for split in ("train", "val", "test"):
            for class_name in classes:
                count = sum(
                    1
                    for path in (destination / kind / split / class_name).iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
                counts.append(
                    {"kind": kind, "split": split, "class_name": class_name, "count": count}
                )
    csv_write(destination / "dataset_counts.csv", counts, ["kind", "split", "class_name", "count"])
    manifest = {
        "protocol": "skin_clean_v1",
        "seed": seed,
        "class_names": classes,
        "input_original_images": len(rows),
        "unique_original_images": len(plan),
        "removed_duplicate_files": len(removed),
        "generated_augmented_images": len(lineage),
        "validation_count_per_class": validation_count,
        "test_count_per_class": test_count,
        "checks": {
            "cross_split_pixel_duplicates": 0,
            "label_conflicts": 0,
            "unique_generated_pixels": len(generated_pixels) - len(output_pixels["train"]),
            "evaluation_copies_identical": True,
            "augmentation_lineage_complete": len(lineage) == len(train_rows),
        },
        "limitations": [
            "person, lesion and capture-session identifiers unavailable",
            "perceptual near-duplicate detection not performed",
            "folder labels were not clinically re-annotated",
        ],
    }
    return manifest, counts
    manifest = {
        "protocol": "skin_clean_v1",
        "seed": seed,
        "class_names": classes,
        "validation_count_per_class": validation_count,
        "test_count_per_class": test_count,
        "input_original_images": len(rows),
        "unique_original_images": len(plan),
        "removed_exact_duplicate_files": len(removed),
        "generated_train_augmentations": len(lineage),
        "split_policy": "deduplicate RGB pixels, SHA-256 seeded order, test then val then train",
        "augmentation_policy": "one deterministic augmentation per unique train image",
        "augmentation_types": AUGMENTATIONS,
        "limitations": [
            "Person, lesion, and capture-session grouping unavailable",
            "Perceptual near-duplicate detection not performed",
            "Clinical label correctness not verified",
            "New split invalidates historical validation and test metrics",
        ],
    }
    write_json(destination / "build_manifest.json", manifest)
    return manifest, counts


def make_zip(dataset_root, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    root = Path(dataset_root)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, root.name + "/" + path.relative_to(root).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise OSError("Dataset ZIP CRC failure")
    return file_hash(destination)
