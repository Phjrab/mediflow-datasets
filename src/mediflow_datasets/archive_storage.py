"""Restore retired duplicate ZIPs from compact sources and retained model files."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

DEFAULT_MANIFEST = "camera_capture/_review/history/project_cleanup_20261112/retired_archives.json"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def local_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError("Archive reference must stay inside the repository")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Archive reference resolves outside the repository")
    return resolved


def zip_metadata(info):
    value = {
        key: getattr(info, key)
        for key in (
            "filename",
            "date_time",
            "compress_type",
            "create_system",
            "create_version",
            "extract_version",
            "reserved",
            "flag_bits",
            "volume",
            "internal_attr",
            "external_attr",
        )
    }
    value.update(
        comment=base64.b64encode(info.comment).decode(), extra=base64.b64encode(info.extra).decode()
    )
    return value


def zip_info(value):
    info = zipfile.ZipInfo(value["filename"], tuple(value["date_time"]))
    for key, item in value.items():
        if key not in ("filename", "date_time", "comment", "extra"):
            setattr(info, key, item)
    info.comment = base64.b64decode(value["comment"])
    info.extra = base64.b64decode(value["extra"])
    return info


def reference_bytes(root, reference):
    path = local_path(root, reference["path"])
    if reference["kind"] == "file":
        return path.read_bytes()
    if reference["kind"] == "zip_member":
        with zipfile.ZipFile(path) as archive:
            return archive.read(reference["member"])
    raise ValueError("Unknown archive reference")


def restore_archive(root, record, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Restored ZIP must stay inside the repository")
    if output.exists():
        raise FileExistsError("Existing archive is preserved")
    if "copy_reference" in record:
        content = reference_bytes(root, record["copy_reference"])
        if digest(content) != record["sha256"]:
            raise ValueError("Retained ZIP hash mismatch")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(content)
        return output
    # Validate every retained input before creating the restored archive.
    for entry in record["entries"]:
        if digest(reference_bytes(root, entry["reference"])) != entry["sha256"]:
            raise ValueError("Retained member hash mismatch: " + entry["metadata"]["filename"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=record["compresslevel"]
    ) as archive:
        archive.comment = base64.b64decode(record["comment"])
        for entry in record["entries"]:
            archive.writestr(
                zip_info(entry["metadata"]),
                reference_bytes(root, entry["reference"]),
                compresslevel=record["compresslevel"],
            )
    if digest(output.read_bytes()) != record["sha256"]:
        raise ValueError("Restored ZIP differs from the original; inspect this new output")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--id", required=True)
    parser.add_argument("--output", help="Optional new path relative to the repository")
    args = parser.parse_args()
    root = args.root.resolve()
    records = json.loads(local_path(root, args.manifest).read_text("utf-8"))["archives"]
    record = records[args.id]
    path = restore_archive(root, record, local_path(root, args.output or record["original_path"]))
    print("Restored and SHA-256 verified:", path)


if __name__ == "__main__":
    main()
