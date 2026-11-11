"""Pinned source archives and explicit target roles for camera experiments."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import stat
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from mediflow_datasets.candidate_reproduction import ROOT, sha256

SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IDENTITIES = ("person", "lesion", "session", "origin")
CORRECTIONS = "_review/p02_skin_s01_condition_corrections.json"


@dataclass(frozen=True)
class ImageRow:
    path: str
    split: str
    label: int | None = None
    person: str | None = None
    session: str | None = None
    lesion: str | None = None
    origin: str | None = None
    condition: str | None = None
    face_box: tuple[int, int, int, int] | None = None
    variant: str = "original"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing file or path outside data root: {relative}")
    return path


def candidate(domain):
    item = read_json(ROOT / "results" / "CANDIDATE_INDEX.json")["candidates"][domain]
    model = local_path(ROOT / "results", item["model"])
    classes = read_json(model.with_name("class_names.json"))
    if len(set(classes)) != len(classes) or sha256(model) != item["model_sha256"]:
        raise ValueError(f"Candidate contract changed: {domain}")
    return item, model, classes


def source_index(path, classes, expected_hash, provenance=None):
    """Check identity and list original/augmented members; keep the ZIP unopened for writes."""
    digest = sha256(path)
    if digest != expected_hash:
        raise ValueError(f"Source ZIP does not match selected checkpoint: SHA-256 {digest}")
    identities = read_json(provenance) if provenance else {}
    found = defaultdict(list)
    roots = defaultdict(set)
    with zipfile.ZipFile(path) as archive:
        member_names = set(archive.namelist())
        lineage = {}
        lineage_members = [
            member
            for member in archive.namelist()
            if PurePosixPath(member).name == "augmentation_lineage.csv"
        ]
        if len(lineage_members) > 1:
            raise ValueError("Ambiguous augmentation lineage tables")
        if lineage_members:
            text = archive.read(lineage_members[0]).decode("utf-8-sig")
            prefix = PurePosixPath(lineage_members[0]).parent
            for record in csv.DictReader(io.StringIO(text)):
                if "augmented_path" in record and "source_original_path" in record:
                    child = (prefix / record["augmented_path"]).as_posix()
                    parent = (prefix / record["source_original_path"]).as_posix()
                    lineage[child] = parent
        seen = set()
        for info in archive.infolist():
            name = info.orig_filename
            parts = PurePosixPath(name).parts
            if (
                not parts
                or PurePosixPath(name).is_absolute()
                or ".." in parts
                or "\\" in name
                or ":" in name
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe ZIP entry: {name}")
            key = name.rstrip("/").casefold()
            if key in seen:
                raise ValueError(f"Ambiguous ZIP entry: {name}")
            seen.add(key)
            if info.is_dir() or PurePosixPath(name).suffix.lower() not in EXTENSIONS:
                continue
            for i, part in enumerate(parts[:-3]):
                variant, split = part.lower(), parts[i + 1].lower()
                if variant not in ("original", "augmented") or split not in SPLITS:
                    continue
                label = parts[i + 2]
                if label not in classes:
                    raise ValueError(f"Unexpected source class folder: {name}")
                ids = {field: identities.get(name, {}).get(field) for field in IDENTITIES}
                if ids["origin"] is None:
                    suffix = PurePosixPath(*parts[i + 1 :]).as_posix()
                    matching_original = PurePosixPath(*parts[:i], "original", suffix).as_posix()
                    if variant == "original" or matching_original in member_names:
                        ids["origin"] = matching_original
                    elif name in lineage:
                        ids["origin"] = lineage[name]
                found[variant, split].append(
                    ImageRow(name, split, classes.index(label), variant=variant, **ids)
                )
                roots[variant].add(parts[: i + 1])
                break
    keys = [(v, s) for v in ("original", "augmented") for s in SPLITS]
    for variant, split in keys:
        rows = found[variant, split]
        if len(roots[variant]) != 1 or set(r.label for r in rows) != set(range(len(classes))):
            raise ValueError(f"Expected one {variant}/{split} tree and all selected classes")
    known = {row.path for rows in found.values() for row in rows}
    if not set(identities).issubset(known):
        raise ValueError("Source provenance contains unmatched image paths")
    return {key: sorted(found[key], key=lambda r: r.path) for key in keys}, digest


def pilot_rows(root, domain):
    """The four observed people are calibration data, including previously corrected labels."""
    correction_file = root / CORRECTIONS
    corrections = (
        {item["photo"]: item for item in read_json(correction_file)["corrections"]}
        if correction_file.is_file()
        else {}
    )
    if domain == "skin" and not corrections:
        raise ValueError("P02 pilot condition correction is required for skin")
    review_file = root / "_review/pilot_analysis_20261108/measurements.json"
    boxes = (
        {row["photo"]: row["face_box"] for row in read_json(review_file)["photos"]}
        if review_file.is_file()
        else {}
    )
    rows = []
    used = set()
    for photo in sorted((root / domain).rglob("*.png")):
        path = photo.relative_to(root).as_posix()
        _, person, session, name = photo.relative_to(root).parts
        record = read_json(photo.with_suffix(".json"))
        if (record["domain"], record["participant"], record["session"], record["file"]) != (
            domain,
            person,
            session,
            name,
        ) or record.get("label_status") != "unverified":
            raise ValueError(f"Pilot metadata mismatch: {path}")
        condition = record["condition"]
        if path in corrections:
            correction = corrections[path]
            if (condition, record["captured_at"]) != (
                correction["recorded_condition"],
                correction["captured_at"],
            ):
                raise ValueError(f"Stale condition correction: {path}")
            condition = correction["reviewed_condition"]
            used.add(path)
        if condition not in ("ref", "alt"):
            raise ValueError(f"Unexpected pilot condition: {path}")
        rows.append(
            ImageRow(
                path,
                "calibration",
                person=person,
                session=session,
                condition=condition,
                face_box=boxes.get(path),
            )
        )
    if not rows or used != {path for path in corrections if path.startswith(domain + "/")}:
        raise ValueError(f"Missing pilot captures or corrections: {domain}")
    return rows


def target_rows(root, domain, classes, manifest=None):
    """Optional new target test images may be labeled only with documented evidence."""
    rows = pilot_rows(root, domain)
    seen = {r.path for r in rows}
    pilot_people = {r.person for r in rows}
    if manifest:
        for item in read_json(manifest):
            if item["domain"] != domain:
                continue
            path = local_path(root, item["path"]).relative_to(root.resolve()).as_posix()
            if path in seen or item["person"] in pilot_people:
                raise ValueError(f"Target test shares a pilot participant or image: {path}")
            seen.add(path)
            label = item.get("label")
            if not item["person"] or not item["session"] or not item.get("origin"):
                raise ValueError(f"Target test needs participant, session and original ID: {path}")
            if label is not None and (label not in classes or not item.get("label_evidence")):
                raise ValueError(f"Target label needs class membership and evidence: {path}")
            rows.append(
                ImageRow(
                    path,
                    "test",
                    classes.index(label) if label is not None else None,
                    item["person"],
                    item["session"],
                    item.get("lesion"),
                    item["origin"],
                    item.get("condition"),
                    item.get("face_box"),
                )
            )
    return rows


def image_signatures(data):
    with Image.open(io.BytesIO(data)) as photo:
        rgb = photo.convert("RGB")
        pixel = hashlib.sha256(f"{rgb.size}".encode() + rgb.tobytes()).hexdigest()
        thumb = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
        near = int.from_bytes(np.packbits(thumb[:, :-1] > thumb[:, 1:]).tobytes(), "big")
    return hashlib.sha256(data).hexdigest(), pixel, near


def audit_groups(entries, *, near_reviews=()):
    """Entries are (row, byte hash, pixel hash, dHash). Review dHash <= 3 candidates."""
    exact = defaultdict(list)
    bands = defaultdict(list)
    near = set()
    reviewed = {tuple(sorted(pair)) for pair in near_reviews}
    for row, byte, pixel, dhash in entries:
        for key in (("bytes", byte), ("pixels", pixel)):
            exact[key].append(row)
        for offset in (0, 16, 32, 48):
            band = (offset, (dhash >> offset) & 65535)
            for other_hash, other in bands[band]:
                if row.split != other.split and (dhash ^ other_hash).bit_count() <= 3:
                    pair = tuple(sorted((row.path, other.path)))
                    if pair not in reviewed:
                        near.add(pair)
            bands[band].append((dhash, row))
    leaks = [
        {"kind": key[0], "paths": sorted(r.path for r in rows)}
        for key, rows in exact.items()
        if len({r.split for r in rows}) > 1 or len({r.label for r in rows}) > 1
    ]
    metadata = {}
    for field in IDENTITIES:
        ids = defaultdict(set)
        for row, *_ in entries:
            identity = getattr(row, field)
            if identity:
                if field == "session":
                    identity = (row.person, identity)
                ids[identity].add(row.split)
        leaks.extend(
            {"kind": field, "id": str(key), "splits": sorted(splits)}
            for key, splits in ids.items()
            if len(splits) > 1
        )
        known = sum(bool(getattr(row, field)) for row, *_ in entries)
        metadata[field] = {
            "identified_rows": known,
            "total": len(entries),
            "complete": known == len(entries),
        }
    return {
        "blocking_leaks_or_conflicts": leaks,
        "near_duplicate_review": sorted(near),
        "identity_metadata": metadata,
        "near_duplicate_rule": "64-bit dHash, Hamming <=3",
    }


def audit_source(path, source, near_reviews=()):
    entries = []
    checks = {}
    with zipfile.ZipFile(path) as archive:
        for rows in source.values():
            entries.extend((row, *image_signatures(archive.read(row.path))) for row in rows)
    for split in SPLITS:
        a, b = (
            Counter(
                (r.label, h) for r, h, *_ in entries if r.variant == variant and r.split == split
            )
            for variant in ("original", "augmented")
        )
        checks[split] = not (a - b) if split == "train" else a == b
    report = audit_groups(entries, near_reviews=near_reviews)
    report["original_augmented_consistency"] = checks
    report["source_counts"] = {f"{v}/{s}": len(rows) for (v, s), rows in source.items()}
    return report, {pixel for _, _, pixel, _ in entries}
