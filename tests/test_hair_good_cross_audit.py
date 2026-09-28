import csv
import io
import json
import zipfile

from PIL import Image

from mediflow_datasets.hair_good_cross_audit import audit


def _jpg(color):
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(stream, format="JPEG")
    return stream.getvalue()


def test_cross_audit_flags_same_image_and_subject_without_creating_split(tmp_path):
    existing = tmp_path / "hair_datasets.zip"
    good = tmp_path / "hair_good_raw_v1.zip"
    same = _jpg("red")
    with zipfile.ZipFile(existing, "w") as archive:
        archive.writestr("dataset/original/train/비듬/0013_cam_a_1_TH.jpg", same)
        archive.writestr("dataset/original/val/탈모/0021_cam_b_1_TH.jpg", _jpg("blue"))
        archive.writestr("dataset/augmented/train/비듬/0013_aug_1.jpg", _jpg("green"))
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("hair_good_raw_v1/0013_cam_new_2_TH.jpg", same)
        archive.writestr("hair_good_raw_v1/0099_cam_new_1_TH.jpg", _jpg("yellow"))

    output = tmp_path / "report"
    result = audit(existing, good, output)
    assert result["existing_original_images"] == 2
    assert result["good_images"] == 2
    assert result["shared_subject_ids"] == 1
    assert result["good_images_matching_old_bytes"] == 1
    assert result["good_images_matching_old_pixels"] == 1
    assert result["good_images_with_old_subject"] == 1
    assert result["existing_split_counts"] == {"train": 1, "val": 1}
    assert json.loads((output / "audit_summary.json").read_text(encoding="utf-8")) == result
    with (output / "good_cross_matches.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["old_splits_for_subject"] == "train"
    assert rows[1]["subject_matches"] == "0"
    assert not (output / "original").exists()


def test_cross_audit_does_not_overwrite_report(tmp_path):
    existing = tmp_path / "old.zip"
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(existing, "w") as archive:
        archive.writestr("original/train/비듬/0013_old.jpg", _jpg("red"))
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("0014_good.jpg", _jpg("green"))
    output = tmp_path / "report"
    output.mkdir()
    try:
        audit(existing, good, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing report directory should be preserved")
