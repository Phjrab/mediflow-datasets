"""Colab batch analysis and review exports, without training or final Test evaluation.

Only standard-library modules are imported here so this file can bootstrap a clean
TensorFlow environment without changing Colab's notebook-kernel dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from pathlib import Path, PurePosixPath

DOMAINS = ("hair", "skin")
PACKAGE_PINS = (
    "tensorflow[and-cuda]==2.20.0",
    "keras==3.13.2",
    "numpy>=1.26,<3",
    "pillow>=10,<13",
    "opencv-python-headless>=4.10,<5",
    "tqdm>=4.66,<5",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def unpack_bundle(bundle, destination, expected_hash):
    """Verify the previously supplied bundle before using any of its Python files."""
    if digest(bundle) != expected_hash:
        raise ValueError(
            "선택한 ZIP이 준비된 실행 ZIP과 다릅니다. mediflow_da_runtime.zip을 선택하세요."
        )
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("bundle_manifest.json"))
        members = {}
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
                or name.casefold() in members
            ):
                raise ValueError("안전하지 않거나 중복된 ZIP 경로: " + name)
            members[name.casefold()] = info
        files = {info.filename for info in members.values() if not info.is_dir()}
        if files != set(manifest) | {"bundle_manifest.json"}:
            raise ValueError("실행 ZIP의 파일 목록이 검증 목록과 다릅니다.")
        for name, expected in manifest.items():
            body = archive.read(name)
            if hashlib.sha256(body).hexdigest() != expected:
                raise ValueError("실행 파일 검사 실패: " + name)
            path = destination / name
            if not path.resolve().is_relative_to(destination):
                raise ValueError("ZIP 경로가 실행 폴더 밖을 가리킵니다.")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(body)
    return destination


def run_logged(command, *, env, log_path):
    """Keep complete logs and show a heartbeat during long ZIP/image scans."""
    log_path = Path(log_path)
    started = time.monotonic()
    last_update = started
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(0.5)
            now = time.monotonic()
            if process.poll() is None and now - last_update >= 30:
                minutes = (now - started) / 60
                print(f"  {log_path.stem}: 진행 중 ({minutes:.1f}분)", flush=True)
                last_update = now
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    if process.returncode:
        print(tail, flush=True)
        raise subprocess.CalledProcessError(process.returncode, command)
    print(f"  {log_path.stem}: 완료", flush=True)


def python_environment(python, runtime=None):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
    env.update(TF_NUM_INTEROP_THREADS="2", TF_NUM_INTRAOP_THREADS="2")
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["KERAS_BACKEND"] = "tensorflow"
    env.pop("TF_USE_LEGACY_KERAS", None)
    env.pop("PYTHONPATH", None)
    if runtime is not None:
        env["PYTHONPATH"] = str(Path(runtime) / "src")
    result = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    libraries = sorted(Path(result.stdout.strip()).glob("nvidia/*/lib"))
    if libraries:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*(str(path) for path in libraries), env.get("LD_LIBRARY_PATH", "")]
        )
    return env


def probe_python(python):
    """Reject incompatible TensorFlow add-ons before importing TensorFlow."""
    env = python_environment(python)
    check = subprocess.run(
        [str(python), "-m", "pip", "check"],
        env=env,
        capture_output=True,
        text=True,
    )
    if check.returncode and any(
        name in check.stdout.lower() for name in ("tensorflow", "keras", "ydf-tf")
    ):
        return None
    program = """import importlib.metadata as m
assert m.version('tensorflow') == '2.20.0'
assert m.version('keras') == '3.13.2'
import tensorflow as tf, keras, numpy, cv2, PIL, tqdm, json
assert float(tf.reduce_sum(tf.ones((2, 2))).numpy()) == 4.0
print(json.dumps({'tensorflow': tf.__version__, 'keras': keras.__version__,
                 'gpu': [str(d) for d in tf.config.list_physical_devices('GPU')]}))
"""
    result = subprocess.run(
        [str(python), "-c", program],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    return json.loads(result.stdout.strip().splitlines()[-1])


def prepare_python(workspace, executor=run_logged):
    """Reuse a compatible runtime or install into a new isolated environment."""
    compatible = probe_python(sys.executable)
    if compatible:
        print("현재 호환 환경 사용:", compatible, flush=True)
        return Path(sys.executable), compatible
    workspace = Path(workspace)
    tools = workspace / "environment_tools"
    environment = workspace / "analysis_environment"
    print("분석 전용 환경을 준비합니다. Colab 기본 TensorFlow는 변경하지 않습니다.", flush=True)
    base_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    base_env.pop("PYTHONPATH", None)
    executor(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--target",
            str(tools),
            "virtualenv",
        ],
        env=base_env,
        log_path=workspace / "environment_tools.log",
    )
    executor(
        [sys.executable, "-m", "virtualenv", "--no-download", str(environment)],
        env={**base_env, "PYTHONPATH": str(tools)},
        log_path=workspace / "environment_create.log",
    )
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    executor(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", *PACKAGE_PINS],
        env=base_env,
        log_path=workspace / "environment_install.log",
    )
    compatible = probe_python(python)
    if not compatible:
        raise RuntimeError("전용 환경 검사에 실패했습니다. environment_*.log를 확인하세요.")
    print("분석 전용 환경 준비 완료:", compatible, flush=True)
    return python, compatible


def find_source(drive_root, task):
    paths = sorted(
        path
        for path in Path(drive_root).rglob(task["source_name"] + "*")
        if path.is_file() and zipfile.is_zipfile(path)
    )
    if not paths:
        raise FileNotFoundError("Drive에서 데이터셋을 찾지 못했습니다: " + task["source_name"])
    if len(paths) == 1:
        return paths[0]  # The analysis CLI verifies the pinned hash before reading images.
    print("동일 이름 후보가 여러 개라 파일 해시로 확인합니다:", paths, flush=True)
    for path in paths:
        if digest(path) == task["source_sha256"]:
            return path
    raise ValueError("준비된 모델과 일치하는 데이터셋이 없습니다: " + task["source_name"])


def export_review(folder, destination):
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(folder).as_posix())


def run_batch(runtime, drive_root, python, *, executor=run_logged, environment=None):
    """Run both independent analysis jobs; never call train or finalize."""
    runtime, drive_root = Path(runtime), Path(drive_root)
    config_path = runtime / "configs/domain_adaptation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = uuid.uuid4().hex[:8]
    parent = drive_root / ("mediflow_da_hair_skin_" + run_id)
    parent.mkdir(exist_ok=False)
    results = {}
    env = environment if environment is not None else python_environment(python, runtime)
    print("두 분야의 결과 저장 위치:", parent, flush=True)
    for domain in DOMAINS:
        print(f"\n===== {domain}: 분석과 변환 미리보기 시작 =====", flush=True)
        folder = parent / domain
        folder.mkdir()
        record = {"domain": domain, "training_executed": False, "target_accuracy": None}
        try:
            source = find_source(drive_root, config["tasks"][domain])
            print("기존 데이터셋:", source, flush=True)
            base = [
                str(python),
                "-m",
                "mediflow_datasets.domain_adaptation",
            ]
            options = [
                "--domain",
                domain,
                "--config",
                str(config_path),
                "--source-zip",
                str(source),
                "--calibration-file",
                str(runtime / "calibration" / (domain + ".json")),
            ]
            executor(
                [*base, "analyze", *options, "--output", str(folder / "analysis")],
                env=env,
                log_path=folder / "analysis.log",
            )
            executor(
                [
                    *base,
                    "preview",
                    *options,
                    "--analysis",
                    str(folder / "analysis"),
                    "--output",
                    str(folder / "train_preview"),
                ],
                env=env,
                log_path=folder / "preview.log",
            )
            required = (
                "analysis/completed.json",
                "analysis/analysis.json",
                "analysis/source_audit.json",
                "analysis/measured_profile.json",
                "train_preview/source_train_before_after.png",
                "train_preview/preview_paths.json",
            )
            for name in required:
                if not (folder / name).is_file():
                    raise FileNotFoundError("결과 생성이 완료되지 않았습니다: " + name)
            audit = json.loads((folder / "analysis/source_audit.json").read_text(encoding="utf-8"))
            record.update(
                status="analysis_completed_review_pending",
                source_zip=str(source),
                source_counts=audit["source_counts"],
                confirmed_findings=len(audit["blocking_leaks_or_conflicts"]),
                near_pairs=len(audit["near_duplicate_review"]),
                identity_metadata=audit["identity_metadata"],
            )
            print("분석 완료, 학습 전 검토 대기:", record, flush=True)
        except Exception as error:
            record.update(status="failed", error=str(error), traceback=traceback.format_exc())
            print(
                f"{domain} 실행 실패. 원인을 결과 ZIP에 기록하고 다음 분야를 진행합니다.",
                flush=True,
            )
            print(record["traceback"], flush=True)
        save_json(folder / "batch_status.json", record)
        prefix = "mediflow_failed" if record["status"] == "failed" else "mediflow_review"
        destination = parent / f"{prefix}_{domain}_{run_id}.zip"
        export_review(folder, destination)
        results[domain] = {**record, "review_zip": str(destination)}
    save_json(parent / "batch_summary.json", results)
    return parent, results


def run_colab(expected_bundle_hash, content_root="/content"):
    """Entry point embedded verbatim in the one-run Colab notebook."""
    from google.colab import drive as colab_drive
    from google.colab import files, output
    from IPython.display import HTML, SVG, Image, display

    print("mediflow_da_runtime.zip 하나를 선택하세요. 실제 촬영 원본 업로드는 필요 없습니다.")
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("실행 ZIP 하나만 선택하세요.")
    bundle = Path(next(iter(uploaded))).resolve()
    del uploaded  # Release the in-memory copy of the model bundle.
    content_root = Path(content_root)
    workspace = content_root / ("mediflow_hair_skin_" + uuid.uuid4().hex[:8])
    workspace.mkdir()
    runtime = unpack_bundle(bundle, workspace / "runtime", expected_bundle_hash)
    colab_drive.mount(str(content_root / "drive"))
    python, versions = prepare_python(workspace)
    if not versions["gpu"]:
        print("GPU가 감지되지 않아 CPU로 분석합니다. 처리 시간이 길어질 수 있습니다.")
    parent, results = run_batch(runtime, content_root / "drive/MyDrive", python)
    for path in workspace.glob("environment_*.log"):
        shutil.copyfile(path, parent / path.name)
    save_json(parent / "batch_environment.json", versions)
    downloads = {}
    for domain, result in results.items():
        print(f"\n{domain}: {result['status']}")
        folder = parent / domain
        if result["status"] != "failed":
            display(SVG(filename=str(folder / "analysis/metric_distributions.svg")))
            display(Image(filename=str(folder / "train_preview/source_train_before_after.png")))
        local = workspace / Path(result["review_zip"]).name
        shutil.copyfile(result["review_zip"], local)
        downloads[domain] = local

    # Keep clickable retry buttons if a browser blocks the second automatic download.
    for domain, path in downloads.items():
        callback = f"mediflow.download_{domain}_{workspace.name}"

        def retry_download(path=path):
            files.download(str(path))

        output.register_callback(callback, retry_download)
        display(
            HTML(
                f"<button onclick=\"google.colab.kernel.invokeFunction('{callback}', [], {{}})\">"
                f"{domain} 결과 ZIP 다시 다운로드</button>"
            )
        )
    print("두 ZIP의 다운로드를 요청합니다. 여러 파일 다운로드 알림이 나오면 허용하세요.")
    for path in downloads.values():
        try:
            files.download(str(path))
        except Exception as error:
            print("자동 다운로드를 완료하지 못했습니다. 위 버튼을 사용하세요:", error)
    print("Drive 보관 위치:", parent)
    print("분석과 미리보기만 실행했습니다. 새 모델 학습과 최종 Test 평가는 실행하지 않았습니다.")
    return results


BATCH_RESULTS = run_colab("144e9bf7d9087deb6bb75fcb193daf08f885733de2eff3b806a285fa946b41b3")
