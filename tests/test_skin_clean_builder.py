"""Deterministic Skin deduplication, re-splitting, augmentation and notebook checks."""

import ast
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mediflow_datasets import skin_clean_builder as builder

ROOT = Path(__file__).resolve().parents[1]


def image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((12, 12, 3), value, dtype="uint8")
    pixels[0, 0] = [value, value // 2, 255 - value]
    Image.fromarray(pixels).save(path)


def dataset(root, classes):
    for class_index, class_name in enumerate(classes):
        values = [20 + class_index * 50 + index for index in range(6)]
        for index, value in enumerate(values):
            split = ("train", "val", "test")[index % 3]
            image(root / split / class_name / f"{index}.png", value)
        # Add one exact cross-split duplicate and one within-split duplicate.
        source = root / "train" / class_name / "0.png"
        (root / "val" / class_name).mkdir(parents=True, exist_ok=True)
        (root / "train" / class_name).mkdir(parents=True, exist_ok=True)
        (root / "val" / class_name / "cross.png").write_bytes(source.read_bytes())
        (root / "train" / class_name / "within.png").write_bytes(source.read_bytes())
    return root


def output_pixel_sets(root, kind, classes):
    values = {}
    for split in ("train", "val", "test"):
        values[split] = {
            builder.pixel_hash(path)[0]
            for class_name in classes
            for path in (root / kind / split / class_name).iterdir()
        }
    return values


def test_clean_build_is_deterministic_leak_free_and_traced(tmp_path):
    classes = ["a", "b"]
    source = dataset(tmp_path / "source", classes)
    rows = builder.inventory(source, classes)
    first, removed = builder.split_plan(rows, classes, seed=42, validation_count=1, test_count=1)
    second, _ = builder.split_plan(rows, classes, seed=42, validation_count=1, test_count=1)
    assert [(r["pixel_sha256"], r["new_split"]) for r in first] == [
        (r["pixel_sha256"], r["new_split"]) for r in second
    ]
    assert len(removed) == 4
    output = tmp_path / "skin_clean"
    manifest, counts = builder.build_dataset(
        source, output, classes, seed=42, validation_count=1, test_count=1
    )
    assert manifest["input_original_images"] == 16
    assert manifest["unique_original_images"] == 12
    assert manifest["removed_duplicate_files"] == 4
    assert manifest["generated_augmented_images"] == 8
    assert manifest["checks"]["augmentation_lineage_complete"]
    original = output_pixel_sets(output, "original", classes)
    assert not original["train"] & original["val"]
    assert not original["train"] & original["test"]
    assert not original["val"] & original["test"]
    augmented = output_pixel_sets(output, "augmented", classes)
    assert augmented["val"] == original["val"]
    assert augmented["test"] == original["test"]
    assert len(augmented["train"]) == 2 * len(original["train"])
    assert all(row["count"] > 0 for row in counts)
    with (output / "augmentation_lineage.csv").open(encoding="utf-8-sig") as stream:
        lineage = list(csv.DictReader(stream))
    assert len(lineage) == 8
    archive = tmp_path / "skin_clean.zip"
    digest = builder.make_zip(output, archive)
    assert digest == builder.file_hash(archive)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert "skin_clean/augmentation_lineage.csv" in zipped.namelist()


def test_label_conflict_blocks_automatic_cleaning(tmp_path):
    source = dataset(tmp_path / "source", ["a", "b"])
    conflict = source / "test/b/conflict.png"
    conflict.write_bytes((source / "train/a/0.png").read_bytes())
    rows = builder.inventory(source, ["a", "b"])
    with pytest.raises(ValueError, match="라벨 충돌"):
        builder.split_plan(rows, ["a", "b"], validation_count=1, test_count=1)


def test_skin_clean_notebook_is_standalone_and_embeds_exact_source():
    path = ROOT / "notebooks/04_one_time_skin_clean_builder_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == []
        source = "".join(cell["source"])
        sources.append(source)
        code = "\n".join(line for line in source.splitlines() if not line.startswith("%pip "))
        compile(code, f"cell-{index}", "exec")
    embedded = next(source for source in sources if source.startswith("BUILDER_SOURCE = "))
    assignment = ast.parse(embedded).body[0]
    assert ast.literal_eval(assignment.value) == Path(builder.__file__).read_text(encoding="utf-8")
    assert "skin_clean_v1" in "\n".join(sources)
    assert "AUDIT_DIR" in "\n".join(sources)
