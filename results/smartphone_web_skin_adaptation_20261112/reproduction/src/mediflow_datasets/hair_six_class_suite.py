"""Build an isolated Hair six-class dataset and run four Colab comparisons."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

CLASSES = ["모낭사이홍반", "미세각질", "비듬", "탈모", "피지과다", "양호"]
SPLITS = ("train", "val", "test")
OLD_SHA256 = "2ac7260663cf69835ba50edb6ae8c7e7ac13be9c73b9f7ea24f60e0342e7e156"
GOOD_SHA256 = "d602fdaf3ff3b4dbe288603dd1684316f468702ace751dde1c5e1a3e3215b86f"
TRIALS = (
    {"id": "b0_256_original", "backbone": "B0", "size": 256, "loss": "ce", "variant": "original"},
    {"id": "b0_256_augmented", "backbone": "B0", "size": 256, "loss": "ce", "variant": "augmented"},
    {
        "id": "b1_384_original", "backbone": "B1", "size": 384,
        "loss": "ls005", "variant": "original",
    },
    {
        "id": "b1_384_augmented", "backbone": "B1", "size": 384,
        "loss": "ls005", "variant": "augmented",
    },
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_members(archive: zipfile.ZipFile):
    seen = set()
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError(f"Unsafe ZIP path: {name}")
        key = name.casefold()
        if key in seen:
            raise ValueError(f"Repeated ZIP path: {name}")
        seen.add(key)
        if not info.is_dir() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield info


def _class_name(value: str) -> str:
    if value in CLASSES:
        return value
    try:
        repaired = value.encode("cp437").decode("cp949")
    except (UnicodeError, ValueError):
        repaired = value
    if repaired not in CLASSES[:-1]:
        raise ValueError(f"Unexpected Hair class: {value!r}")
    return repaired


def _old_entry(name: str) -> tuple[str, str] | None:
    parts = PurePosixPath(name).parts
    for index, part in enumerate(parts):
        if part.lower() == "original" and len(parts) == index + 4:
            split = parts[index + 1].lower()
            if split in SPLITS:
                return split, _class_name(parts[index + 2])
    return None


def split_good_subjects(groups: dict[str, list[str]], seed: int = 42) -> dict[str, str]:
    """Keep each user-confirmed filename ID wholly in one split."""
    if len(groups) < 3:
        raise ValueError("At least three good subject IDs are needed")
    rng = random.Random(seed)
    subjects = list(groups)
    rng.shuffle(subjects)
    subjects.sort(key=lambda subject: -len(groups[subject]))
    total = sum(map(len, groups.values()))
    targets = {"train": total * 0.8, "val": total * 0.1, "test": total * 0.1}
    counts = dict.fromkeys(SPLITS, 0)
    allocation = {}
    for subject in subjects:
        size = len(groups[subject])

        def cost(split, size=size):
            proposed = {key: counts[key] + (size if key == split else 0) for key in SPLITS}
            return sum((proposed[key] - targets[key]) ** 2 / targets[key] for key in SPLITS)

        chosen = min(SPLITS, key=cost)
        allocation[subject] = chosen
        counts[chosen] += size
    if any(counts[split] == 0 for split in SPLITS):
        raise ValueError(f"Good split is empty: {counts}")
    return allocation


def _pixel_hash(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode("ascii"))
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def _augment(source: Path, target: Path, seed: int) -> dict:
    rng = random.Random(seed)
    angle = rng.uniform(-10.0, 10.0)
    brightness = rng.uniform(0.85, 1.15)
    contrast = rng.uniform(0.85, 1.15)
    saturation = rng.uniform(0.90, 1.10)
    mirrored = rng.random() < 0.5
    blur = rng.random() < 0.12
    with Image.open(source) as image:
        result = image.convert("RGB")
        if mirrored:
            result = ImageOps.mirror(result)
        result = result.rotate(
            angle, resample=Image.Resampling.BICUBIC,
            fillcolor=result.getpixel((0, 0)),
        )
        result = ImageEnhance.Brightness(result).enhance(brightness)
        result = ImageEnhance.Contrast(result).enhance(contrast)
        result = ImageEnhance.Color(result).enhance(saturation)
        if blur:
            result = result.filter(ImageFilter.GaussianBlur(radius=0.35))
        target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target, format="JPEG", quality=92)
    return dict(
        angle=angle, brightness=brightness, contrast=contrast,
        saturation=saturation, mirrored=mirrored, blur=blur, seed=seed,
    )


def build_dataset(
    old_zip: Path,
    good_zip: Path,
    destination: Path,
    *,
    expected_old: int = 12536,
    expected_good: int = 534,
    seed: int = 42,
    old_sha256: str = OLD_SHA256,
    good_sha256: str = GOOD_SHA256,
) -> dict:
    """Create new files only; preserve old split and group new good images by ID."""
    old_zip, good_zip, destination = map(Path, (old_zip, good_zip, destination))
    if destination.exists():
        raise FileExistsError(destination)
    if not zipfile.is_zipfile(old_zip) or not zipfile.is_zipfile(good_zip):
        raise ValueError("Both source files must be ZIP archives")
    hashes = {"old": file_sha256(old_zip), "good": file_sha256(good_zip)}
    if hashes != {"old": old_sha256, "good": good_sha256}:
        raise ValueError("Source ZIP SHA-256 differs from the audited input")

    with zipfile.ZipFile(old_zip) as old_archive, zipfile.ZipFile(good_zip) as good_archive:
        old_items = [(info, _old_entry(info.filename)) for info in _safe_members(old_archive)]
        old_items = [(info, entry) for info, entry in old_items if entry is not None]
        good_items = list(_safe_members(good_archive))
        if len(old_items) != expected_old or len(good_items) != expected_good:
            raise ValueError(f"Image counts changed: old={len(old_items)}, good={len(good_items)}")
        groups = defaultdict(list)
        for info in good_items:
            subject = PurePosixPath(info.filename).name.split("_", 1)[0]
            if not subject.isdecimal():
                raise ValueError(f"Missing good subject ID: {info.filename}")
            groups[subject].append(info.filename)
        allocation = split_good_subjects(groups, seed)
        destination.mkdir(parents=True)
        rows = []
        pixel_splits = {}
        for archive, items, source_name in (
            (old_archive, old_items, "old"),
            (good_archive, [(i, (allocation[i.filename.split('/')[-1].split('_', 1)[0]], "양호"))
                            for i in good_items], "good"),
        ):
            for info, (split, label) in items:
                basename = PurePosixPath(info.filename).name
                target = destination / "original" / split / label / basename
                if target.exists():
                    raise ValueError(f"Colliding image filename: {target}")
                data = archive.read(info)
                pixel_hash = _pixel_hash(data)
                previous = pixel_splits.get(pixel_hash)
                if previous is not None:
                    raise ValueError(f"Identical image appears twice: {previous} / {info.filename}")
                pixel_splits[pixel_hash] = info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                rows.append(dict(
                    variant="original", split=split, class_name=label,
                    path=target.relative_to(destination).as_posix(), source_zip=source_name,
                    source_member=info.filename,
                    subject_id=basename.split("_", 1)[0] if source_name == "good" else "",
                    source_sha256=hashlib.sha256(data).hexdigest(),
                    augmentation="",
                ))

    train_by_class = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            train_by_class[row["class_name"]].append(row)
    for label in CLASSES:
        originals = sorted(train_by_class[label], key=lambda row: row["path"])
        if not originals:
            raise ValueError(f"Empty training class: {label}")
        if label == "양호":
            choices = [(row, index) for row in originals for index in range(1, 6)]
        else:
            rng = random.Random(seed + CLASSES.index(label))
            choices = [(row, 1) for row in rng.sample(originals, k=len(originals) // 2)]
        for row, index in choices:
            source = destination / row["path"]
            stem = PurePosixPath(row["path"]).stem
            name = f"{stem}_aug{index}.jpg"
            target = destination / "augmentation" / "train" / label / name
            if target.exists():
                raise ValueError(f"Colliding augmentation filename: {target}")
            transform_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{row['path']}:{index}".encode()).digest()[:8],
                "big",
            )
            params = _augment(source, target, transform_seed)
            rows.append(dict(
                variant="augmentation", split="train", class_name=label,
                path=target.relative_to(destination).as_posix(),
                source_zip=row["source_zip"], source_member=row["source_member"],
                subject_id=row["subject_id"], source_sha256=row["source_sha256"],
                augmentation=json.dumps(params, ensure_ascii=False, sort_keys=True),
            ))

    counts = Counter((row["variant"], row["split"], row["class_name"]) for row in rows)
    contract = dict(
        protocol="hair_six_class_v2", seed=seed, class_names=CLASSES,
        source_sha256=hashes, expected_old=expected_old, expected_good=expected_good,
        good_subjects=len(groups), good_group_assignment=allocation,
        original_counts={split: {label: counts[("original", split, label)] for label in CLASSES}
                         for split in SPLITS},
        augmented_extra_counts={label: counts[("augmentation", "train", label)]
                                for label in CLASSES},
        good_label_basis="user_confirmed_all_six_source_values_zero; source_json_unavailable",
        limitations=[
            "Old five-class filenames lost person/session IDs; person leakage remains unverified.",
            "Cross-ZIP byte and RGB-pixel audit found zero exact matches; "
            "near duplicates are not excluded.",
            "Augmented training changes both image variation and class counts.",
        ],
    )
    with (destination / "image_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "dataset_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return contract


def archive_dataset(dataset: Path, target: Path) -> str:
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(dataset.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dataset.parent).as_posix())
    return file_sha256(target)


def dataset_factory(root: Path, variant: str, batch_size: int, seed: int = 42):
    """Make fixed-order evaluation and weighted training datasets from one v2 root."""
    import numpy as np
    import tensorflow as tf

    class_to_index = {name: index for index, name in enumerate(CLASSES)}

    def make(split: str, size: int, is_train: bool):
        paths, labels = [], []
        for label in CLASSES:
            folders = [root / "original" / split / label]
            if is_train and variant == "augmented":
                folders.append(root / "augmentation" / "train" / label)
            for folder in folders:
                files = [p for p in sorted(folder.iterdir()) if p.is_file()]
                paths.extend(str(p) for p in files)
                labels.extend([class_to_index[label]] * len(files))
        if not paths or (is_train and set(labels) != set(range(len(CLASSES)))):
            raise ValueError("Missing dataset images or class")
        labels_array = np.asarray(labels, dtype=np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels_array))
        if is_train:
            dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
            counts = np.bincount(labels_array, minlength=len(CLASSES))
            weights = np.sqrt(counts.mean() / counts)
            weights /= np.mean(weights[labels_array])

        def decode(path, label):
            image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
            image.set_shape((None, None, 3))
            image = tf.image.resize(tf.cast(image, tf.float32), (size, size))
            target = tf.one_hot(label, len(CLASSES))
            if is_train:
                return image, target, tf.gather(tf.constant(weights, tf.float32), label)
            return image, target

        dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset, paths

    return make


def run_four(
    dataset: Path,
    output: Path,
    *,
    data_sha256: str,
    batch_size: int = 32,
    stage1_epochs: int = 15,
    stage2_epochs: int = 15,
    seed: int = 42,
    code_hashes: dict | None = None,
    code_commit: str = "",
) -> dict:
    """Run/resume four trials; select by validation macro F1, then test once."""
    import platform

    import keras
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    from mediflow_datasets import common_engine as engine

    contract = json.loads((dataset / "dataset_contract.json").read_text(encoding="utf-8"))
    if contract["class_names"] != CLASSES:
        raise ValueError("Six-class order differs from the fixed contract")
    settings = dict(
        protocol="hair_six_class_four_v1", dataset_sha256=data_sha256,
        dataset_contract=hashlib.sha256(
            (dataset / "dataset_contract.json").read_bytes()
        ).hexdigest(),
        trials=TRIALS, seed=seed, batch_size=batch_size,
        stage1_epochs=stage1_epochs, stage2_epochs=stage2_epochs,
        keras=keras.__version__, tensorflow=tf.__version__, numpy=np.__version__,
        python=platform.python_version(), code_hashes=code_hashes or {},
    )
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError("RESUME_DIR has different data, code settings, or environment")
    else:
        engine.write_json(config_path, {
            "signature": signature, "settings": settings,
            "code_commit_at_generation": code_commit,
            "code_state": "exact embedded source copies saved; may include uncommitted changes",
        })
        engine.write_json(output / "class_names.json", CLASSES)
        shutil.copyfile(dataset / "dataset_contract.json", output / "dataset_contract.json")
    records = []
    for trial in TRIALS:
        spec = {**trial, "class_count": len(CLASSES)}
        factory = dataset_factory(dataset, trial["variant"], batch_size, seed)
        print("실험 시작:", trial["id"])
        records.append(engine.run_trial(
            spec, factory, output, signature, seed=seed,
            epochs1=stage1_epochs, epochs2=stage2_epochs,
        ))
        print("Validation Macro F1:", records[-1]["validation"]["macro_f1"])

    winner = max(records, key=lambda row: (
        row["validation"]["macro_f1"], row["validation"]["accuracy"]
    ))
    summary = dict(
        protocol="hair_six_class_four_v1", selection="validation_macro_f1_then_accuracy",
        winner=winner["id"], test_evaluated=False,
        experiments=[{
            "id": row["id"], "validation_accuracy": row["validation"]["accuracy"],
            "validation_macro_f1": row["validation"]["macro_f1"],
            "validation_good_recall": row["validation"]["recall"][CLASSES.index("양호")],
            "training_seconds": row["training_seconds"],
            "model_bytes": row["model_bytes"],
            "selected_stage": row["selected_stage"],
        } for row in records],
    )
    engine.write_json(output / "comparison_summary.json", summary)
    with (output / "validation_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["experiments"][0]))
        writer.writeheader()
        writer.writerows(summary["experiments"])
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    for row_index, record in enumerate(records):
        history = record["history"]
        for col, key in enumerate(("accuracy", "loss")):
            axes[row_index, col].plot(history[key], label="Train")
            axes[row_index, col].plot(history["val_" + key], label="Validation")
            axes[row_index, col].axvline(record["stage_boundary"] - 0.5, color="gray", ls="--")
            axes[row_index, col].set_title(f"{record['id']} — {key}")
            axes[row_index, col].set_xlabel("Epoch")
            axes[row_index, col].grid(alpha=0.2)
            axes[row_index, col].legend()
    fig.tight_layout()
    fig.savefig(output / "four_training_curves.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    names = [r["id"] for r in summary["experiments"]]
    axes[0].bar(names, [r["validation_accuracy"] for r in summary["experiments"]])
    axes[1].bar(names, [r["validation_macro_f1"] for r in summary["experiments"]])
    for ax, title in zip(axes, ("Validation Accuracy", "Validation Macro F1"), strict=True):
        ax.set(title=title, ylim=(0, 1))
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output / "validation_dashboard.png", dpi=170)
    plt.close(fig)

    test_path = output / "selected_test_metrics.json"
    if not test_path.exists():
        keras.backend.clear_session()
        model = keras.models.load_model(engine.selected_model_path(output, winner), compile=False)
        factory = dataset_factory(dataset, winner["spec"]["variant"], batch_size, seed)
        test_data, test_paths = factory("test", winner["spec"]["size"], False)
        metrics = engine.evaluate_to_files(model, test_data, test_paths, output, "selected_test")
        metrics["selected_model"] = winner["id"]
        engine.write_json(test_path, metrics)
        del model
    else:
        metrics = engine.read_json(test_path)
    summary["test_evaluated"] = True
    summary["selected_test_accuracy"] = metrics["accuracy"]
    summary["selected_test_macro_f1"] = metrics["macro_f1"]
    selected_path = engine.selected_model_path(output, winner)
    summary["selected_model_sha256"] = engine.file_hash(selected_path)
    engine.write_json(output / "comparison_summary.json", summary)
    with (output / "validation_class_f1.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["experiment", *CLASSES])
        for row in records:
            writer.writerow([row["id"], *row["validation"]["class_f1"]])
    fig, ax = plt.subplots(figsize=(10, 5))
    matrix = np.array([row["validation"]["class_f1"] for row in records])
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(CLASSES)), xticklabels=CLASSES,
           yticks=range(len(records)), yticklabels=[row["id"] for row in records],
           title="Validation class F1")
    for index in range(len(records)):
        for column in range(len(CLASSES)):
            ax.text(column, index, f"{matrix[index, column]:.2f}",
                    ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "validation_class_f1.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    confusion = np.asarray(metrics["confusion_matrix"])
    image = ax.imshow(confusion, cmap="Blues")
    ax.set(xticks=range(len(CLASSES)), xticklabels=CLASSES,
           yticks=range(len(CLASSES)), yticklabels=CLASSES,
           xlabel="Predicted", ylabel="True", title="Selected model — Test")
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            ax.text(column, row, str(confusion[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "selected_test_confusion.png", dpi=170)
    plt.close(fig)
    report_zip = output.parent / (output.name + "_reports.zip")
    if not report_zip.exists():
        with zipfile.ZipFile(report_zip, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in output.rglob("*"):
                if path.is_file() and path.suffix != ".keras":
                    archive.write(path, path.relative_to(output.parent).as_posix())
    return {
        "summary": summary, "report_zip": str(report_zip),
        "output": str(output), "selected_model": str(selected_path),
    }
