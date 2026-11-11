"""Read-only cross audit of Hair clean v1 and user-confirmed all-zero scalp images."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train", "val", "test"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _subject_id(name: str) -> str | None:
    first = PurePosixPath(name).name.split("_", 1)[0]
    return first if first.isdecimal() else None


def _old_label(path: PurePosixPath) -> tuple[str, str] | None:
    parts = path.parts
    for index, part in enumerate(parts):
        if part.lower() == "original" and len(parts) > index + 3:
            split = parts[index + 1].lower()
            if split in SPLITS:
                return split, parts[index + 2]
    return None


def _scan(path: Path, *, existing: bool) -> tuple[list[dict], dict]:
    records = []
    skipped = Counter()
    seen_paths = set()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = member.filename
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or "\\" in name:
                raise ValueError(f"Unsafe ZIP member: {name}")
            if name.casefold() in seen_paths:
                raise ValueError(f"Repeated ZIP path: {name}")
            seen_paths.add(name.casefold())
            if parts.suffix.lower() not in IMAGE_SUFFIXES:
                skipped["non_image"] += 1
                continue
            label = _old_label(parts) if existing else None
            if existing and label is None:
                skipped["not_original"] += 1
                continue
            data = archive.read(member)
            with Image.open(io.BytesIO(data)) as image:
                rgb = image.convert("RGB")
                pixels = hashlib.sha256()
                pixels.update(f"{rgb.width}x{rgb.height}:".encode("ascii"))
                pixels.update(rgb.tobytes())
                size = [rgb.width, rgb.height]
            row = {
                "path": name,
                "subject_id": _subject_id(name),
                "byte_sha256": hashlib.sha256(data).hexdigest(),
                "pixel_sha256": pixels.hexdigest(),
                "size": size,
            }
            if existing:
                row["split"], row["class_name"] = label
            records.append(row)
            if len(records) % 2000 == 0:
                print(f"{'Hair v1' if existing else '양호'}: {len(records)}장 확인")
    if not records:
        raise ValueError(f"No {'original split' if existing else 'good'} images: {path}")
    return records, dict(skipped)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(existing_zip: Path, good_zip: Path, output_dir: Path) -> dict:
    """Inspect the two ZIPs without changing either input or making train/test splits."""
    existing_zip = Path(existing_zip).resolve()
    good_zip = Path(good_zip).resolve()
    output_dir = Path(output_dir).resolve()
    if existing_zip == good_zip:
        raise ValueError("기존 Hair ZIP과 양호 ZIP을 각각 지정하세요.")
    if output_dir.exists():
        raise FileExistsError(f"보고서 폴더가 이미 있습니다: {output_dir}")
    if not zipfile.is_zipfile(existing_zip) or not zipfile.is_zipfile(good_zip):
        raise ValueError("두 입력 모두 ZIP 파일이어야 합니다.")

    old, old_skipped = _scan(existing_zip, existing=True)
    good, good_skipped = _scan(good_zip, existing=False)
    old_bytes = defaultdict(list)
    old_pixels = defaultdict(list)
    old_subjects = defaultdict(list)
    for row in old:
        old_bytes[row["byte_sha256"]].append(row)
        old_pixels[row["pixel_sha256"]].append(row)
        if row["subject_id"] is not None:
            old_subjects[row["subject_id"]].append(row)

    details = []
    for row in good:
        byte_matches = old_bytes[row["byte_sha256"]]
        pixel_matches = old_pixels[row["pixel_sha256"]]
        subject_matches = old_subjects[row["subject_id"]] if row["subject_id"] else []
        details.append(
            {
                "good_path": row["path"],
                "subject_id": row["subject_id"] or "",
                "byte_matches": len(byte_matches),
                "pixel_matches": len(pixel_matches),
                "subject_matches": len(subject_matches),
                "old_splits_for_subject": "|".join(sorted({m["split"] for m in subject_matches})),
                "old_classes_for_subject": "|".join(
                    sorted({m["class_name"] for m in subject_matches})
                ),
                "exact_old_paths": "|".join(
                    sorted({m["path"] for m in byte_matches + pixel_matches})
                ),
            }
        )

    good_people = {r["subject_id"] for r in good if r["subject_id"] is not None}
    old_people = {r["subject_id"] for r in old if r["subject_id"] is not None}
    summary = {
        "protocol": "hair_good_cross_audit_v1",
        "status": "read_only_audit_not_a_v2_dataset",
        "existing_zip_name": existing_zip.name,
        "existing_zip_sha256": sha256_file(existing_zip),
        "good_zip_name": good_zip.name,
        "good_zip_sha256": sha256_file(good_zip),
        "existing_original_images": len(old),
        "existing_split_counts": dict(Counter(row["split"] for row in old)),
        "existing_class_counts": dict(Counter(row["class_name"] for row in old)),
        "good_images": len(good),
        "good_subject_ids": len(good_people),
        "existing_subject_ids": len(old_people),
        "good_images_without_subject_id": sum(r["subject_id"] is None for r in good),
        "existing_images_without_subject_id": sum(r["subject_id"] is None for r in old),
        "shared_subject_ids": len(good_people & old_people),
        "good_images_matching_old_bytes": sum(row["byte_matches"] > 0 for row in details),
        "good_images_matching_old_pixels": sum(row["pixel_matches"] > 0 for row in details),
        "good_images_with_old_subject": sum(row["subject_matches"] > 0 for row in details),
        "old_skipped": old_skipped,
        "good_skipped": good_skipped,
        "good_label_source": "user_confirmed_all_six_values_zero; no_json_in_good_zip",
        "limits": [
            "A shared subject ID is a grouping concern, not proof of identical images.",
            "Exact byte/pixel hashes do not exclude near duplicates.",
            "Subject matching uses the first numeric filename component; "
            "unmatched IDs remain unverified.",
            "No train/validation/test split or augmentation was created.",
        ],
    }
    output_dir.mkdir(parents=True)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        output_dir / "good_cross_matches.csv",
        details,
        [
            "good_path",
            "subject_id",
            "byte_matches",
            "pixel_matches",
            "subject_matches",
            "old_splits_for_subject",
            "old_classes_for_subject",
            "exact_old_paths",
        ],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
