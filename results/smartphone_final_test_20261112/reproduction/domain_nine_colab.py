"""Standard-library-only Colab orchestration for all nine experiments."""

from __future__ import annotations

import json
import shutil
import traceback
import uuid
from pathlib import Path

from mediflow_datasets import domain_adaptation_batch as batch
from mediflow_datasets.domain_adaptation_io import write_progress

DOMAINS = ("hair", "web_skin", "skin")
ARMS = ("lighting", "expanded", "aligned")


def matching_parent(drive_root, bundle_sha256):
    candidates = []
    for folder in Path(drive_root).glob("mediflow_nine_*"):
        marker = folder / "pipeline.json"
        if marker.is_file():
            try:
                if json.loads(marker.read_text(encoding="utf-8"))["bundle_sha256"] == bundle_sha256:
                    candidates.append(folder)
            except (OSError, ValueError, KeyError):
                continue
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    folder = Path(drive_root) / ("mediflow_nine_" + uuid.uuid4().hex[:8])
    folder.mkdir()
    batch.save_json(
        folder / "pipeline.json",
        {
            "bundle_sha256": bundle_sha256,
            "domains": list(DOMAINS),
            "arms": list(ARMS),
            "experiment_count": 9,
            "includes_identifiable_target_images": True,
        },
    )
    return folder


def arm_inventory(folder):
    result = {}
    for arm in ARMS:
        path = folder / arm / "complete.json"
        state = "not_complete"
        if path.is_file():
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                checkpoint = (path.parent / meta["model_relative"]).resolve()
                if checkpoint.is_relative_to(path.parent.resolve()) and (
                    batch.digest(checkpoint) == meta["model_sha256"]
                ):
                    state = "complete"
            except (OSError, ValueError, KeyError):
                state = "invalid_checkpoint"
        result[arm] = state
    return result


def run_jobs(runtime, workspace, parent, drive_root, python, *, executor=None, environment=None):
    executor = executor or batch.run_logged
    config_path = Path(runtime) / "configs/nine_experiments.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    env = environment if environment is not None else batch.python_environment(python, runtime)
    summary = {}
    for domain in DOMAINS:
        local_source = Path(workspace) / (domain + "_source.zip")
        print("\n===", domain, "세 조건을 차례로 실행합니다 ===", flush=True)
        error = None
        try:
            source = batch.find_source(drive_root, config["tasks"][domain])
            if shutil.disk_usage(workspace).free < source.stat().st_size + 1024**3:
                raise RuntimeError("Source ZIP 복사를 위한 실행 공간이 부족합니다.")
            shutil.copyfile(source, local_source)
            executor(
                [
                    str(python),
                    "-m",
                    "mediflow_datasets.domain_nine",
                    "--domain",
                    domain,
                    "--config",
                    str(config_path),
                    "--source-zip",
                    str(local_source),
                    "--review",
                    str(Path(runtime) / "review" / domain),
                    "--output",
                    str(Path(parent) / domain),
                ],
                env=env,
                log_path=Path(parent) / f"{domain}_{uuid.uuid4().hex[:8]}.log",
            )
        except Exception as exception:
            error = str(exception)
            batch.save_json(
                Path(parent) / f"{domain}_failure_{uuid.uuid4().hex[:8]}.json",
                {
                    "error": error,
                    "traceback": traceback.format_exc(),
                },
            )
            print(domain, "미완료:", error, flush=True)
        finally:
            # Only the new disposable file in this exact workspace is removed.
            if local_source.exists() and local_source.resolve().parent == Path(workspace).resolve():
                local_source.unlink()
        state = {
            "status": "failed" if error else "completed",
            "error": error,
            "arms": arm_inventory(Path(parent) / domain),
            "target_accuracy": None,
        }
        status = Path(parent) / domain / "status.json"
        state["evaluation_completed"] = (
            error is None
            and status.is_file()
            and json.loads(status.read_text(encoding="utf-8")).get("status") == "completed"
        )
        summary[domain] = state
        write_progress(
            Path(parent) / "summary.json",
            {
                "domains": summary,
                "arms_completed": sum(
                    v == "complete" for s in summary.values() for v in s["arms"].values()
                ),
                "requested_arms": 9,
                "all_completed": len(summary) == 3
                and all(
                    s["evaluation_completed"] and all(v == "complete" for v in s["arms"].values())
                    for s in summary.values()
                ),
                "target_accuracy": None,
            },
        )
    return json.loads((Path(parent) / "summary.json").read_text(encoding="utf-8"))


def run_prepared(runtime, workspace, bundle_sha256):
    from google.colab import drive, files, output
    from IPython.display import HTML, display

    drive.mount("/content/drive")
    python, versions = batch.prepare_python(workspace)
    if not versions["gpu"]:
        raise RuntimeError("GPU가 없습니다. 런타임 유형을 GPU로 변경한 뒤 모두 실행하세요.")
    drive_root = Path("/content/drive/MyDrive")
    parent = matching_parent(drive_root, bundle_sha256)
    print("동일 실행 꾸러미의 완료 모델은 검증 후 재사용합니다. 결과:", parent, flush=True)
    result = run_jobs(runtime, workspace, parent, drive_root, python)
    write_progress(parent / "environment.json", versions)
    for path in Path(workspace).glob("environment_*.log"):
        destination = parent / (path.stem + "_" + uuid.uuid4().hex[:8] + ".log")
        shutil.copyfile(path, destination)
    destination = parent.parent / (parent.name + "_results_" + uuid.uuid4().hex[:8] + ".zip")
    batch.export_review(parent, destination)
    local = Path(workspace) / destination.name
    shutil.copyfile(destination, local)

    def retry():
        files.download(str(local))

    callback = "mediflow.nine_" + uuid.uuid4().hex[:8]
    output.register_callback(callback, retry)
    display(
        HTML(
            "<button onclick=\"google.colab.kernel.invokeFunction('"
            + callback
            + "', [], {})\">9개 실험 결과 ZIP 다시 다운로드</button>"
        )
    )
    print("완료 실험:", result["arms_completed"], "/ 9")
    print("모든 평가까지 완료:", result["all_completed"])
    print("Drive 결과:", parent)
    print("실제 사진은 Target 학습 자료로 사용했으며 실제 장비 정확도는 계산하지 않았습니다.")
    files.download(str(local))
    return result
