import csv
import io
import json
import zipfile
from collections import defaultdict

import pytest
from PIL import Image

from mediflow_datasets.hair_six_class_suite import (
    CLASSES,
    TRIALS,
    build_dataset,
    dataset_factory,
    file_sha256,
)


def _image(color):
    stream = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(stream, format="JPEG")
    return stream.getvalue()


def _sources(tmp_path):
    old = tmp_path / "old.zip"
    good = tmp_path / "good.zip"
    number = 0
    with zipfile.ZipFile(old, "w") as archive:
        for split in ("train", "val", "test"):
            for label in CLASSES[:-1]:
                for index in range(2):
                    number += 1
                    archive.writestr(
                        f"root/original/{split}/{label}/{label}_{index:05d}.jpg",
                        _image((number * 5, number * 3, 0)),
                    )
        archive.writestr("root/augmented/train/탈모/ignored.jpg", _image((250, 0, 0)))
    with zipfile.ZipFile(good, "w") as archive:
        for subject in range(6):
            for index in range(2):
                number += 1
                archive.writestr(
                    f"hair_good_raw_v1/{subject:04d}_camera_{index}_TH.jpg",
                    _image((number * 5, number * 3, 0)),
                )
    return old, good


def test_build_six_class_split_and_train_only_augmentation(tmp_path):
    old, good = _sources(tmp_path)
    output = tmp_path / "v2"
    contract = build_dataset(
        old, good, output, expected_old=30, expected_good=12,
        old_sha256=file_sha256(old), good_sha256=file_sha256(good),
    )
    assert len(TRIALS) == 4
    assert contract["class_names"] == CLASSES
    assert contract["good_subjects"] == 6
    assert sum(
        contract["original_counts"][split]["양호"] for split in ("train", "val", "test")
    ) == 12
    assert contract["augmented_extra_counts"]["양호"] == (
        5 * contract["original_counts"]["train"]["양호"]
    )
    assert all(contract["augmented_extra_counts"][name] == 1 for name in CLASSES[:-1])
    assert not (output / "augmentation" / "val").exists()
    assert not (output / "augmentation" / "test").exists()
    with (output / "image_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    splits_by_subject = defaultdict(set)
    for row in rows:
        if row["class_name"] == "양호" and row["variant"] == "original":
            splits_by_subject[row["subject_id"]].add(row["split"])
        if row["variant"] == "augmentation":
            assert row["split"] == "train"
            assert json.loads(row["augmentation"])["seed"] >= 0
    assert all(len(splits) == 1 for splits in splits_by_subject.values())
    original_train, original_paths = dataset_factory(output, "original", 8)("train", 32, True)
    augmented_train, augmented_paths = dataset_factory(output, "augmented", 8)(
        "train", 32, True
    )
    assert len(augmented_paths) > len(original_paths)
    assert len(list(original_train.unbatch())) == len(original_paths)
    assert len(list(augmented_train.unbatch())) == len(augmented_paths)
    val_original, val_paths = dataset_factory(output, "original", 8)("val", 32, False)
    val_augmented, val_aug_paths = dataset_factory(output, "augmented", 8)("val", 32, False)
    assert val_paths == val_aug_paths
    assert next(iter(val_original))[0].shape[1:] == (32, 32, 3)
    assert next(iter(val_augmented))[0].numpy().max() > 1.0
    with pytest.raises(FileExistsError):
        build_dataset(
            old, good, output, expected_old=30, expected_good=12,
            old_sha256=file_sha256(old), good_sha256=file_sha256(good),
        )


def test_build_rejects_wrong_source_hash(tmp_path):
    old, good = _sources(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        build_dataset(old, good, tmp_path / "v2", expected_old=30, expected_good=12)
