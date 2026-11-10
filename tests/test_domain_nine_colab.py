from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

from mediflow_datasets import domain_adaptation_batch as batch
from mediflow_datasets import domain_nine_colab as colab


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_resume_is_bundle_specific_and_one_failure_keeps_other_domains(tmp_path):
    runtime, workspace, drive = (tmp_path / p for p in ("runtime", "workspace", "drive"))
    workspace.mkdir()
    drive.mkdir()
    tasks = {}
    for domain in colab.DOMAINS:
        source = drive / (domain + "_datasets.zip")
        with zipfile.ZipFile(source, "x") as archive:
            archive.writestr("fixture.txt", domain)
        tasks[domain] = {"source_name": source.name, "source_sha256": batch.digest(source)}
    save(runtime / "configs/nine_experiments.json", {"tasks": tasks})
    parent = colab.matching_parent(drive, "bundle1")
    assert colab.matching_parent(drive, "bundle1") == parent
    assert colab.matching_parent(drive, "bundle2") != parent
    visited, fail = [], {"web_skin"}

    def execute(command, *, env, log_path):
        domain = command[command.index("--domain") + 1]
        visited.append(domain)
        output = Path(command[command.index("--output") + 1])
        source = Path(command[command.index("--source-zip") + 1])
        assert batch.digest(source) == tasks[domain]["source_sha256"]
        log_path.write_text("fixture execution", encoding="utf-8")
        if domain in fail:
            raise RuntimeError("simulated interrupted domain")
        for arm in colab.ARMS:
            checkpoint = output / arm / "fixture.keras"
            if not checkpoint.exists():
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint fixture")
                save(
                    checkpoint.parent / "complete.json",
                    {"model_relative": checkpoint.name, "model_sha256": batch.digest(checkpoint)},
                )
        save(output / "status.json", {"status": "completed"})

    first = colab.run_jobs(
        runtime, workspace, parent, drive, sys.executable, executor=execute, environment={}
    )
    assert visited == list(colab.DOMAINS)
    assert first["arms_completed"] == 6 and not first["all_completed"]
    assert first["domains"]["web_skin"]["status"] == "failed"
    assert first["target_accuracy"] is None
    assert not list(workspace.glob("*_source.zip"))
    assert len(list(drive.glob("*_datasets.zip"))) == 3
    original = (parent / "hair/lighting/fixture.keras").read_bytes()
    fail.clear()
    second = colab.run_jobs(
        runtime, workspace, parent, drive, sys.executable, executor=execute, environment={}
    )
    assert second["arms_completed"] == 9 and second["all_completed"]
    assert (parent / "hair/lighting/fixture.keras").read_bytes() == original
    # A changed checkpoint is never reported as completed.
    (parent / "hair/aligned/fixture.keras").write_bytes(b"changed")
    assert colab.arm_inventory(parent / "hair")["aligned"] != "complete"
    changed = colab.run_jobs(
        runtime, workspace, parent, drive, sys.executable, executor=execute, environment={}
    )
    assert changed["arms_completed"] == 8 and not changed["all_completed"]


def test_colab_gpu_check_stops_before_any_model_job(tmp_path, monkeypatch):
    mounts, jobs = [], []
    google = types.ModuleType("google.colab")
    google.drive = types.SimpleNamespace(mount=mounts.append)
    google.files = types.SimpleNamespace(download=lambda path: None)
    google.output = types.SimpleNamespace(register_callback=lambda *args: None)
    monkeypatch.setitem(sys.modules, "google.colab", google)
    display = types.ModuleType("IPython.display")
    display.HTML = lambda *args: None
    display.display = lambda *args: None
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    monkeypatch.setattr(batch, "prepare_python", lambda p: (Path(sys.executable), {"gpu": []}))
    monkeypatch.setattr(colab, "run_jobs", lambda *args: jobs.append(args))
    with pytest.raises(RuntimeError, match="GPU"):
        colab.run_prepared(tmp_path, tmp_path, "bundle")
    assert mounts == ["/content/drive"] and not jobs
