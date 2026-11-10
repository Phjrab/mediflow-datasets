from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

from mediflow_datasets import domain_adaptation_batch as batch


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fake_runtime(tmp_path):
    runtime, drive = tmp_path / "runtime", tmp_path / "drive"
    drive.mkdir()
    tasks = {}
    for domain in batch.DOMAINS:
        # These are real ZIP containers even though the orchestration test does not train models.
        path = drive / (domain + "_datasets")
        with zipfile.ZipFile(path, "x") as archive:
            archive.writestr("fixture.txt", domain)
        tasks[domain] = {"source_name": path.name, "source_sha256": batch.digest(path)}
    write_json(runtime / "configs/domain_adaptation.json", {"tasks": tasks})
    return runtime, drive


def test_two_domains_produce_separate_review_zips_without_training(tmp_path):
    runtime, drive = fake_runtime(tmp_path)
    calls = []

    def execute(command, *, env, log_path):
        calls.append(command)
        domain = command[command.index("--domain") + 1]
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        mode = command[3]
        if mode == "analyze":
            write_json(output / "completed.json", {"domain": domain, "status": "completed"})
            write_json(output / "analysis.json", {"domain": domain})
            write_json(output / "measured_profile.json", {"train_visual_review": "pending"})
            write_json(
                output / "source_audit.json",
                {
                    "source_counts": {"original/train": 2},
                    "blocking_leaks_or_conflicts": [],
                    "near_duplicate_review": [["train/a", "test/b"]],
                    "identity_metadata": {"person": {"complete": False}},
                },
            )
        elif mode == "preview":
            (output / "source_train_before_after.png").write_bytes(b"preview fixture")
            write_json(output / "preview_paths.json", [])
        else:
            raise AssertionError("Only analyze and preview may run")
        Path(log_path).write_text("ok", encoding="utf-8")

    parent, results = batch.run_batch(
        runtime,
        drive,
        sys.executable,
        executor=execute,
        environment={"fixture": "yes"},
    )
    assert [cmd[3] for cmd in calls] == ["analyze", "preview", "analyze", "preview"]
    assert [cmd[cmd.index("--domain") + 1] for cmd in calls] == ["hair", "hair", "skin", "skin"]
    for domain, record in results.items():
        assert record["status"] == "analysis_completed_review_pending"
        assert record["near_pairs"] == 1  # No automatically approved near-duplicate exceptions.
        assert record["target_accuracy"] is None
        assert record["training_executed"] is False
        with zipfile.ZipFile(record["review_zip"]) as archive:
            profile = json.loads(archive.read("analysis/measured_profile.json"))
            assert profile["train_visual_review"] == "pending"
            assert "train_preview/source_train_before_after.png" in archive.namelist()
            assert not any("training/" in name for name in archive.namelist())
            assert json.loads(archive.read("analysis/analysis.json"))["domain"] == domain
    # A repeat creates a new folder and leaves existing results unchanged.
    first_bytes = Path(results["hair"]["review_zip"]).read_bytes()
    second, _ = batch.run_batch(runtime, drive, sys.executable, executor=execute, environment={})
    assert second != parent
    assert Path(results["hair"]["review_zip"]).read_bytes() == first_bytes


def test_one_domain_failure_does_not_prevent_other_domain_export(tmp_path):
    runtime, drive = fake_runtime(tmp_path)
    visited = []

    def fail(command, *, env, log_path):
        visited.append(command[command.index("--domain") + 1])
        Path(log_path).write_text("fixture failure", encoding="utf-8")
        raise subprocess.CalledProcessError(1, command)

    _, results = batch.run_batch(runtime, drive, sys.executable, executor=fail, environment={})
    assert visited == ["hair", "skin"]
    for record in results.values():
        assert record["status"] == "failed"
        assert Path(record["review_zip"]).name.startswith("mediflow_failed_")
        with zipfile.ZipFile(record["review_zip"]) as archive:
            assert "analysis.log" in archive.namelist()
            assert json.loads(archive.read("batch_status.json"))["status"] == "failed"


def test_bundle_verification_rejects_changed_or_unsafe_files(tmp_path):
    bundle = tmp_path / "runtime.zip"
    body = b"original runtime"
    with zipfile.ZipFile(bundle, "x") as archive:
        archive.writestr("src/module.py", body)
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(
                {
                    "src/module.py": hashlib.sha256(body).hexdigest(),
                }
            ),
        )
    unpacked = batch.unpack_bundle(bundle, tmp_path / "good", batch.digest(bundle))
    assert (unpacked / "src/module.py").read_bytes() == body
    with pytest.raises(ValueError, match="ZIP"):
        batch.unpack_bundle(bundle, tmp_path / "wrong", "0" * 64)
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "x") as archive:
        archive.writestr("../escaped.py", body)
        archive.writestr("bundle_manifest.json", "{}")
    with pytest.raises(ValueError, match="ZIP"):
        batch.unpack_bundle(unsafe, tmp_path / "unsafe", batch.digest(unsafe))
    assert not (tmp_path / "escaped.py").exists()
    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(changed, "x") as archive:
        archive.writestr("src/module.py", body + b"changed")
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(
                {
                    "src/module.py": hashlib.sha256(body).hexdigest(),
                }
            ),
        )
    with pytest.raises(ValueError, match="검사 실패"):
        batch.unpack_bundle(changed, tmp_path / "changed", batch.digest(changed))


def test_incompatible_kernel_uses_isolated_installs_without_uninstall(tmp_path, monkeypatch):
    probes = iter([None, {"tensorflow": "2.20.0", "keras": "3.13.2", "gpu": []}])
    monkeypatch.setattr(batch, "probe_python", lambda python: next(probes))
    calls = []

    def record(command, *, env, log_path):
        calls.append((command, env))

    python, versions = batch.prepare_python(tmp_path, executor=record)
    assert python.is_relative_to(tmp_path / "analysis_environment")
    assert versions["tensorflow"] == "2.20.0"
    assert "--target" in calls[0][0]
    assert "virtualenv" in calls[1][0]
    assert batch.PACKAGE_PINS[0] in calls[2][0]
    assert calls[2][0][0] == str(python)
    assert all("uninstall" not in cmd for cmd, _ in calls)


def test_colab_upload_once_and_download_two_zips(tmp_path, monkeypatch):
    bundle = tmp_path / "runtime.zip"
    body = b"runtime fixture"
    with zipfile.ZipFile(bundle, "x") as archive:
        archive.writestr("module.py", body)
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(
                {
                    "module.py": hashlib.sha256(body).hexdigest(),
                }
            ),
        )
    downloads, mounts, callbacks = [], [], {}
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(
        upload=lambda: {str(bundle): bundle.read_bytes()},
        download=lambda path: downloads.append(Path(path)),
    )
    colab.drive = types.SimpleNamespace(mount=lambda path: mounts.append(path))
    colab.output = types.SimpleNamespace(
        register_callback=lambda name, function: callbacks.update({name: function}),
    )
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    display = types.ModuleType("IPython.display")
    display.display = lambda *args: None
    display.HTML = lambda *args, **kwargs: None
    display.Image = lambda *args, **kwargs: None
    display.SVG = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    monkeypatch.setattr(
        batch,
        "prepare_python",
        lambda workspace: (
            Path(sys.executable),
            {"gpu": [], "tensorflow": "2.20.0", "keras": "3.13.2"},
        ),
    )

    def fake_batch(runtime, drive_root, python):
        parent = drive_root / "mediflow_da_hair_skin_fixture"
        parent.mkdir(parents=True)
        results = {}
        for domain in batch.DOMAINS:
            path = parent / f"mediflow_review_{domain}_fixture.zip"
            with zipfile.ZipFile(path, "x") as archive:
                archive.writestr("batch_status.json", "{}")
            results[domain] = {
                "status": "analysis_completed_review_pending",
                "review_zip": str(path),
            }
        return parent, results

    monkeypatch.setattr(batch, "run_batch", fake_batch)
    results = batch.run_colab(batch.digest(bundle), content_root=tmp_path)
    assert len(mounts) == 1
    assert len(downloads) == 2
    assert len(callbacks) == 2
    assert all(path.is_file() for path in downloads)
    assert set(results) == {"hair", "skin"}
    next(iter(callbacks.values()))()  # Retry button keeps a valid file after the run ends.
    assert len(downloads) == 3
