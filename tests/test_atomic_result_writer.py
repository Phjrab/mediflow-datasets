import json
from pathlib import Path

import pytest

from mediflow_datasets import domain_adaptation_io as writer


def locked_error():
    error = PermissionError("destination temporarily locked")
    error.winerror = 32
    return error


def test_result_writer_retries_temporary_windows_lock_without_losing_previous_file(
    tmp_path, monkeypatch,
):
    destination = tmp_path / "progress.json"
    destination.write_text('{"old": true}', encoding="utf-8")
    original = Path.replace
    calls, delays = [], []

    def replace(source, target):
        calls.append(target)
        if len(calls) < 3:
            assert json.loads(destination.read_text()) == {"old": True}
            raise locked_error()
        return original(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(writer.time, "sleep", delays.append)
    writer.write_progress(destination, {"new": True})
    assert json.loads(destination.read_text()) == {"new": True}
    assert len(calls) == 3 and delays == [0.05, 0.1]


def test_persistent_lock_is_reported_and_previous_result_is_preserved(tmp_path, monkeypatch):
    destination = tmp_path / "progress.json"
    destination.write_text('{"old": true}', encoding="utf-8")
    calls = []

    def replace(source, target):
        calls.append(target)
        raise locked_error()

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(writer.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        writer.write_progress(destination, {"new": True})
    assert json.loads(destination.read_text()) == {"old": True} and len(calls) == 6
