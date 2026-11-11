"""Train three expanded camera treatments from the pinned Stage 1 starting models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from mediflow_datasets import adaptation_stage1_colab as colab
from mediflow_datasets import domain_adaptation_batch as batch
from mediflow_datasets.domain_adaptation_io import write_progress


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def starting_plan(runtime, stage1, bundle_sha):
    config = read_json(runtime / "configs/adaptation_stage1.json")
    previous = read_json(stage1 / "bundle.json")
    summary = read_json(stage1 / "summary.json")
    selected = read_json(stage1 / "selection.json")
    expected = {f"{d}/{role}" for d in colab.DOMAINS for role in ("basic", "selected")}
    if (
        previous["sha256"] != bundle_sha
        or previous["stage"] != 1
        or set(summary["completed_trials"]) != expected
        or summary["failures"]
        or not summary["selection_written"]
        or selected["schema"] != "mediflow_adaptation_stage1_selection_v1"
    ):
        raise ValueError("공통 ZIP v3와 일치하는 1차 완료 결과가 필요합니다.")
    plan = {}
    for domain in colab.DOMAINS:
        task = config["tasks"][domain]
        choice = selected["selected"][domain]
        role = choice["role"]
        spec = {**task["models"][role], "role": role}
        record = read_json(stage1 / domain / role / "complete.json")
        initial = {"path": f"{domain}/{role}/initial.keras", "sha256": spec["model_sha256"]}
        adapted = {"path": f"{domain}/{role}/best.keras", "sha256": record["best_model"]["sha256"]}
        if (
            choice["initial_checkpoint"] != initial
            or choice["stage1_checkpoint"] != adapted
            or choice["model_config"] != spec
            or record["model_config"] != spec
            or record["initial"]["sha256"] != initial["sha256"]
        ):
            raise ValueError(f"{domain}의 1차 선택 기록과 출발 모델이 일치하지 않습니다.")
        plan[domain] = {"spec": spec, "stage1_best": adapted}
    return config, plan


def require_stage2(folder, bundle_sha):
    marker, summary = read_json(folder / "bundle.json"), read_json(folder / "summary.json")
    if (
        marker["stage"] != 2
        or marker["sha256"] != bundle_sha
        or set(summary["completed_domains"]) != set(colab.DOMAINS)
        or summary["failures"]
        or not summary["comparison_written"]
    ):
        raise ValueError("공통 ZIP v3와 일치하는 2차 완료 결과가 필요합니다.")


def run_domain(runtime, stage1, bundle_sha, domain, archive, output, local, *, stage=2, key=None):
    import cv2
    import keras
    import tensorflow as tf

    from mediflow_datasets import adaptation_stage1 as train

    config, plan = starting_plan(runtime, stage1, bundle_sha)
    task = config["tasks"][domain]
    spec, previous = plan[domain]["spec"], plan[domain]["stage1_best"]
    reference = read_json(runtime / "references" / f"{domain}.json")
    profile = reference["extended_profile"]
    if profile["schema"] != "measured_camera_extended_v1" or profile["domain"] != domain:
        raise ValueError(f"Unexpected expanded profile: {domain}")
    original_path = train.data.local_path(runtime, spec["model"])
    previous_path = train.data.local_path(stage1, previous["path"])
    if batch.digest(original_path) != spec["model_sha256"] or (
        batch.digest(previous_path) != previous["sha256"]
    ):
        raise ValueError(f"{domain} 모델 파일이 1차 기록과 일치하지 않습니다.")
    source_manifest = stage1 / domain / "source_manifest.json"
    identity = {
        "bundle_sha256": bundle_sha,
        "selection_sha256": batch.digest(stage1 / "selection.json"),
        "source_manifest_sha256": batch.digest(source_manifest),
        "reference_sha256": train.object_hash(reference),
        "configuration_sha256": train.object_hash(config),
        "code_sha256": {"stage2": batch.digest(__file__), "training": batch.digest(train.__file__)},
    }
    treatment = "expanded"
    benchmarks_paths = {"initial": original_path, "stage1_lighting": previous_path}
    if stage == 3:
        from mediflow_datasets import adaptation_stage3 as aligned

        stage2 = stage1.with_name("stage2")
        require_stage2(stage2, bundle_sha)
        prior_folder = stage2 / domain
        prior = read_json(prior_folder / "complete.json")
        prior_config = read_json(prior_folder / "run_configuration.json")
        prior_path = prior_folder / "best.keras"
        if (
            prior["model_config"] != spec
            or prior["stage1_best"] != previous
            or prior_config["training_configuration"] != config
            or prior_config["profile"] != profile
            or read_json(prior_folder / "source_manifest.json") != read_json(source_manifest)
            or batch.digest(prior_path) != prior["best_model"]["sha256"]
        ):
            raise ValueError(f"{domain}의 2차 대조군이 현재 실험 설정과 일치하지 않습니다.")
        benchmarks_paths["stage2_expanded"] = prior_path
        identity["stage2_best_sha256"] = prior["best_model"]["sha256"]
        identity["code_sha256"]["alignment"] = batch.digest(aligned.__file__)
        treatment = "aligned"
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "identity.json"
    if marker.exists() and read_json(marker) != identity:
        raise ValueError(f"{domain} 기존 {stage}차 결과와 이번 입력이 일치하지 않습니다.")
    write_progress(marker, identity)
    complete = output / "complete.json"
    if complete.is_file():
        result = read_json(complete)
        if batch.digest(output / "best.keras") != result["best_model"]["sha256"]:
            raise ValueError(f"{domain} 저장된 {stage}차 모델이 완료 기록과 일치하지 않습니다.")
        print(f"[{domain}] {stage}차 완료 결과 재사용", flush=True)
        return

    target = None
    if stage == 3:
        target = aligned.target_inputs(runtime, domain, spec, task, key)
        print(f"[{domain}] 실제 촬영 입력 {len(target['rows'])}장 확인 · 비라벨 정렬용", flush=True)

    write_progress(
        output / "run_configuration.json",
        {
            "stage": stage,
            "identity": identity,
            "training_configuration": config,
            "model": spec,
            "stage1_best": previous,
            "treatment": treatment,
            "profile": profile,
            "validation_conditions": ["clean", "lighting", "expanded"],
            "validation_seed": 1707,
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": train.np.__version__,
            "opencv": cv2.__version__,
            "gpu": [
                tf.config.experimental.get_device_details(d)
                for d in tf.config.list_physical_devices("GPU")
            ],
            "bundle_provenance": read_json(runtime / "bundle_provenance.json"),
        },
    )
    cv2.setNumThreads(1)
    rows, val, pairs = train.prepare_source(domain, archive, task, reference, output)
    if read_json(output / "source_manifest.json") != read_json(source_manifest):
        raise ValueError(f"{domain}의 Train/Validation/Test 목록이 1차와 다릅니다.")
    size = spec["size"]
    cache, shifted, pair_indexes = train.image_cache(
        local, archive, rows, val, pairs, task, size, profile["lighting"], expanded=profile
    )
    keras.utils.set_random_seed(config["seed"])
    model = keras.models.load_model(original_path, compile=False)
    count = len(task["classes"])
    if tuple(model.input_shape[1:]) != (size, size, 3) or (
        len(model.outputs) != (4 if spec["architecture"] == "pmg" else 1)
        or any(tensor.shape[-1] != count for tensor in model.outputs)
    ):
        raise ValueError(f"{domain} 시작 모델의 입력 또는 클래스 설정이 다릅니다.")
    train.exp.configure_partial(model)
    alignment = None
    if target is not None:
        calibration_path = output / "alignment_setup.json"
        if calibration_path.is_file():
            setup = read_json(calibration_path)
        else:
            setup = aligned.calibrate(model, cache, rows, task, spec, target)
            write_progress(calibration_path, setup)
        if setup["target_sha256"] != target["sha256"]:
            raise ValueError("Target calibration does not match the bundled images")
        alignment = {**target, **setup}
    print(f"[{domain} · {spec['role']}] 1차 학습 전 저장본에서 {treatment} 시작", flush=True)
    best = train.phase(
        model,
        (config, task, rows, val, profile, cache, shifted, pair_indexes),
        output / treatment,
        local / "saving",
        spec,
        treatment,
        config["epochs"],
        config["learning_rate"],
        alignment=alignment,
    )
    saved_model = train.export_model(model, output, "best", local / "saving")
    diagnostics = {}
    predicted = {}
    if alignment is not None:
        diagnostics["stage3_aligned"], predicted["stage3_aligned"] = aligned.diagnostic(
            model, shifted["expanded"], val, task, spec, target, setup["reference"]
        )
    del model
    datasets = train.validation_pipelines(cache, shifted, len(rows), val, task["batch_size"], size)
    benchmarks = {}
    for name, path in benchmarks_paths.items():
        keras.backend.clear_session()
        model = keras.models.load_model(path, compile=False)
        benchmarks[name] = train.validate(
            train.predictor(model, spec), datasets, val, count, f"[{domain} · {name}]"
        )
        if alignment is not None:
            diagnostics[name], predicted[name] = aligned.diagnostic(
                model, shifted["expanded"], val, task, spec, target, setup["reference"]
            )
        del model
    if alignment is not None:
        for name, diagnostic in diagnostics.items():
            diagnostic["predicted_class_changes_vs_initial"] = int(
                (predicted[name] != predicted["initial"]).sum()
            )
        write_progress(
            output / "feature_comparison.json",
            {
                "scope": "Expanded Source Validation sample vs unlabeled Target training images",
                "class_names": task["classes"],
                "target_accuracy": None,
                "models": diagnostics,
            },
        )
    benchmarks[f"stage{stage}_{treatment}"] = {name: best[name] for name in datasets}
    write_progress(
        complete,
        {
            "best_model": saved_model,
            "best_epoch": best["epoch"],
            "model_config": spec,
            "stage1_best": previous,
            "validation": benchmarks,
            "validation_scope": config["validation_scope"],
            "source_test_evaluated": False,
        },
    )
    print(f"[{domain}] {stage}차 완료 · 최적 {best['epoch']}회차 · Drive 저장 완료", flush=True)


def write_comparison(destination):
    rows = []
    for domain in colab.DOMAINS:
        complete = read_json(destination / domain / "complete.json")
        for model, metrics in complete["validation"].items():
            rows.append(
                {
                    "domain": domain,
                    "start_role": complete["model_config"]["role"],
                    "model": model,
                    "validation_scope": complete["validation_scope"],
                    **{f"{name}_macro_f1": result["macro_f1"] for name, result in metrics.items()},
                    **{f"{name}_accuracy": result["accuracy"] for name, result in metrics.items()},
                }
            )
    path = destination / "validation_comparison.csv"
    temporary = path.with_name(f".{uuid.uuid4().hex}.csv.tmp")
    with temporary.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def launch(runtime, work, project, bundle_sha, my_drive, *, stage=2, key=None):
    import fcntl

    parent = my_drive / "MediFlow_adaptation_runs" / bundle_sha[:16]
    stage1, destination = parent / "stage1", parent / f"stage{stage}"
    config, _ = starting_plan(runtime, stage1, bundle_sha)
    if stage == 3:
        require_stage2(parent / "stage2", bundle_sha)
    lock = Path("/content") / f"mediflow_stage{stage}_{bundle_sha}.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        python, versions = batch.prepare_python(work, executor=colab.run_live)
        if not versions["gpu"]:
            raise RuntimeError("Colab 런타임 유형에서 GPU를 선택한 뒤 다시 실행하세요.")
        env = batch.python_environment(python, runtime)
        env.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        command = [str(python), "-m", "mediflow_datasets.adaptation_stage2", "--stage", str(stage)]
        key_input = None
        if stage == 3:
            crypto_probe = [
                str(python),
                "-c",
                "from cryptography.hazmat.primitives.ciphers.aead import AESGCM",
            ]
            if subprocess.run(crypto_probe, env=env, capture_output=True).returncode:
                crypto = work / "target_crypto"
                colab.run_live(
                    [
                        str(python),
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--target",
                        str(crypto),
                        "cryptography>=42,<49",
                    ],
                    env,
                    work / "target_crypto.log",
                )
                env["PYTHONPATH"] += os.pathsep + str(crypto)
            key_input = key.strip() + "\n"
            colab.run_live(
                command + ["--check-target", str(runtime), str(stage1), bundle_sha],
                env,
                work / "target_check.log",
                input_text=key_input,
            )
        destination.mkdir(exist_ok=True)
        write_progress(
            destination / "bundle.json",
            {
                "sha256": bundle_sha,
                "stage": stage,
                "environment": versions,
                "launcher_sha256": batch.digest(__file__),
            },
        )
        print(f"{stage}차 결과 저장:", destination, flush=True)
        completed, failures = [], {}
        for index, domain in enumerate(colab.DOMAINS, 1):
            output = destination / domain
            log = destination / f"{domain}_{uuid.uuid4().hex[:10]}.log"
            print(f"\n[{index}/3 · {domain}] {stage}차 준비", flush=True)
            try:
                with tempfile.TemporaryDirectory(prefix=f"{domain}_", dir=work) as folder:
                    local = Path(folder)
                    source = local / "source.zip"
                    if not (output / "complete.json").is_file():
                        drive_source = batch.find_source(
                            project / "datasets", config["tasks"][domain]
                        )
                        colab.copy_source(drive_source, source)
                    args = (runtime, stage1, bundle_sha, domain, source, output, local)
                    colab.run_live(
                        command + ["--domain", *map(str, args)], env, log, input_text=key_input
                    )
                completed.append(domain)
            except Exception as error:
                failures[domain] = {"error": str(error), "log": log.name if log.is_file() else None}
                print(f"[{domain}] 미완료: {error}", flush=True)
        if not failures:
            print("비교표:", write_comparison(destination), flush=True)
        write_progress(
            destination / "summary.json",
            {
                "completed_domains": completed,
                "failures": failures,
                "comparison_written": not failures,
                "source_test_evaluated": False,
            },
        )
        if failures:
            raise RuntimeError(f"미완료 분야가 있습니다. 로그와 완료 기록: {destination}")
        print(f"{stage}차 세 분야 완료", flush=True)
        return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", nargs=5)
    mode.add_argument("--domain", nargs=7)
    mode.add_argument("--check-target", nargs=3)
    parser.add_argument("--stage", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    key = sys.stdin.readline().strip() if args.stage == 3 else None
    if args.check_target:
        if args.stage != 3:
            raise ValueError("Target key validation requires Stage 3")
        from mediflow_datasets.adaptation_stage3 import target_inputs

        runtime, stage1, sha = args.check_target
        config, plan = starting_plan(Path(runtime), Path(stage1), sha)
        for domain, model in plan.items():
            target = target_inputs(
                Path(runtime), domain, model["spec"], config["tasks"][domain], key
            )
            print(f"[{domain}] 암호화 촬영 입력 {len(target['rows'])}장 확인", flush=True)
        return
    if args.launch:
        runtime, work, project, sha, my_drive = args.launch
        launch(
            Path(runtime), Path(work), Path(project), sha, Path(my_drive), stage=args.stage, key=key
        )
        return
    runtime, stage1, sha, domain, source, output, local = args.domain
    run_domain(
        Path(runtime),
        Path(stage1),
        sha,
        domain,
        Path(source),
        Path(output),
        Path(local),
        stage=args.stage,
        key=key,
    )


if __name__ == "__main__":
    sys.exit(main())
