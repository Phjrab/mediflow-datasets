import base64
import zipfile

import pytest

from mediflow_datasets.archive_storage import digest, local_path, restore_archive, zip_metadata


def test_retired_zip_restores_exact_bytes_from_shared_model_and_small_source(tmp_path):
    model = b"retained-model-content" * 90
    (tmp_path / "model.keras").write_bytes(model)
    source = tmp_path / "original.zip"
    with zipfile.ZipFile(source, "x", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("models/model.keras", model)
        archive.writestr("study.json", b'{"epochs":5}')
    entries = []
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(tmp_path / "small.zip", "x") as small:
        for info in original.infolist():
            content = original.read(info)
            if info.filename.endswith(".keras"):
                reference = {"kind": "file", "path": "model.keras"}
            else:
                small.writestr(info.filename, content)
                reference = {"kind": "zip_member", "path": "small.zip", "member": info.filename}
            entries.append(
                {"metadata": zip_metadata(info), "sha256": digest(content), "reference": reference}
            )
    record = {
        "sha256": digest(source.read_bytes()),
        "entries": entries,
        "comment": base64.b64encode(b"").decode(),
        "compresslevel": 6,
    }
    output = tmp_path / "restored.zip"
    restore_archive(tmp_path, record, output)
    assert output.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        restore_archive(tmp_path, record, output)


def test_corrupt_retained_input_fails_before_output_is_created(tmp_path):
    (tmp_path / "model.keras").write_bytes(b"changed")
    record = {
        "entries": [
            {
                "reference": {"kind": "file", "path": "model.keras"},
                "sha256": digest(b"original"),
                "metadata": {"filename": "model.keras"},
            }
        ]
    }
    output = tmp_path / "restored.zip"
    with pytest.raises(ValueError, match="member hash"):
        restore_archive(tmp_path, record, output)
    assert not output.exists()


@pytest.mark.parametrize("relative", ["../outside", "C:/outside", "folder\\outside"])
def test_retained_references_cannot_escape_the_repository(tmp_path, relative):
    with pytest.raises(ValueError, match="inside"):
        local_path(tmp_path.resolve(), relative)
