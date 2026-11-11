"""Drive-based stage 1 launcher with live output and scoped interruption."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from mediflow_datasets import domain_adaptation_batch as batch
from mediflow_datasets.domain_adaptation_io import write_progress

DOMAINS = ("skin", "web_skin", "hair")


def run_live(command, env, log_path, *, input_text=None):
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            if input_text is not None:
                with process.stdin:
                    process.stdin.write(input_text)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def copy_source(source, destination):
    total = source.stat().st_size
    if shutil.disk_usage(destination.parent).free < total + 1024**3:
        raise RuntimeError("Colab 임시 디스크에 데이터 ZIP을 복사할 공간이 부족합니다.")
    with source.open("rb") as reader, destination.open("xb") as writer:
        done, last = 0, -1
        while block := reader.read(16 * 1024**2):
            writer.write(block)
            done += len(block)
            progress = min(100, 100 * done // total)
            if progress >= last + 5 or done == total:
                print(f"  Drive → Colab 로컬 복사 {progress}% ({done / 1e9:.2f} GB)", flush=True)
                last = progress
    if done != total:
        raise OSError("Drive ZIP 복사가 중단됐습니다.")


def run(runtime, work, project, bundle_sha, my_drive):
    config = json.loads((runtime / "configs/adaptation_stage1.json").read_text(encoding="utf-8"))
    python, versions = batch.prepare_python(work)
    if not versions["gpu"]:
        raise RuntimeError("Colab 런타임 유형에서 GPU를 선택한 뒤 다시 실행하세요.")
    env = batch.python_environment(python, runtime)
    env.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    destination = my_drive / "MediFlow_adaptation_runs" / bundle_sha[:16] / "stage1"
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "bundle.json"
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8"))["sha256"] != bundle_sha:
        raise RuntimeError("이 결과 폴더는 다른 버전의 공통 ZIP에 연결돼 있습니다.")
    write_progress(marker, {"sha256": bundle_sha, "stage": 1, "environment": versions})
    print("1차 결과 저장:", destination, flush=True)
    failures = {}
    for domain in DOMAINS:
        output = destination / domain
        log = destination / f"{domain}_{uuid.uuid4().hex[:10]}.log"
        command = [str(python), "-m", "mediflow_datasets.adaptation_stage1"]
        try:
            with tempfile.TemporaryDirectory(prefix=f"{domain}_", dir=work) as folder:
                local = Path(folder)
                data_zip = local / "source.zip"
                if not all(
                    (output / arm / "complete.json").is_file() for arm in ("basic", "selected")
                ):
                    print(f"\n[{domain}] Drive 데이터셋 준비", flush=True)
                    source = batch.find_source(project / "datasets", config["tasks"][domain])
                    copy_source(source, data_zip)
                run_live(
                    command
                    + [
                        "--domain",
                        domain,
                        "--source-zip",
                        str(data_zip),
                        "--output",
                        str(output),
                        "--local",
                        str(local),
                    ],
                    env,
                    log,
                )
        except Exception as error:
            failures[domain] = {"error": str(error), "log": log.name if log.is_file() else None}
            print(f"[{domain}] 미완료: {error}", flush=True)
    completed = [
        f"{d}/{a}"
        for d in DOMAINS
        for a in ("basic", "selected")
        if (destination / d / a / "complete.json").is_file() and d not in failures
    ]
    write_progress(
        destination / "summary.json",
        {"completed_trials": completed, "failures": failures, "selection_written": False},
    )
    if failures:
        raise RuntimeError(
            f"미완료 분야가 있습니다. 완료된 기록은 Drive에 보존했습니다: {destination}"
        )
    run_live(
        [str(python), "-m", "mediflow_datasets.adaptation_stage1", "--select", str(destination)],
        env,
        destination / f"selection_{uuid.uuid4().hex[:10]}.log",
    )
    write_progress(
        destination / "summary.json",
        {"completed_trials": completed, "failures": failures, "selection_written": True},
    )
    print("비교표:", destination / "validation_comparison.csv", flush=True)
    return destination


def launch(runtime, work, project, bundle_sha, my_drive):
    import fcntl

    # Keep two cells in the same Colab VM from touching the same run concurrently.
    lock = Path("/content") / f"mediflow_stage1_{bundle_sha}.lock"
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("같은 Colab 세션에서 이미 실행 중입니다.") from error
        return run(runtime, work, project, bundle_sha, my_drive)


if __name__ == "__main__":
    launch(*(Path(arg) if index != 3 else arg for index, arg in enumerate(sys.argv[1:])))
