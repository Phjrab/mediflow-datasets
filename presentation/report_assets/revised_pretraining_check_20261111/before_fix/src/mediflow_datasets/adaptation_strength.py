"""Validation-only follow-up with frozen, stronger measured camera transformations."""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
import uuid
from pathlib import Path

from mediflow_datasets import adaptation_stage1_colab as colab
from mediflow_datasets import adaptation_stage2 as previous
from mediflow_datasets import domain_adaptation_batch as batch
from mediflow_datasets.domain_adaptation_io import write_progress


def run_domain(runtime, stage1, stage2, study_path, domain, archive, output, local):
    import cv2
    import keras
    import numpy as np
    import tensorflow as tf

    from mediflow_datasets import adaptation_stage1 as train
    from mediflow_datasets import camera_extended

    study = previous.read_json(study_path)
    config, plan = previous.starting_plan(runtime, stage1, study["bundle_sha256"])
    previous.require_stage2(stage2, study["bundle_sha256"])
    task, spec = config["tasks"][domain], plan[domain]["spec"]
    entry = study["domains"][domain]
    reference = previous.read_json(runtime / "references" / f"{domain}.json")
    profile, weak = entry["profile"], reference["extended_profile"]
    record = previous.read_json(stage2 / domain / "complete.json")
    prior_configuration = previous.read_json(stage2 / domain / "run_configuration.json")
    manifest = stage1 / domain / "source_manifest.json"
    initial = train.data.local_path(runtime, spec["model"])
    old_model = stage2 / domain / "best.keras"
    if (
        train.object_hash(config) != study["configuration_sha256"]
        or train.object_hash(reference) != entry["previous_reference_sha256"]
        or batch.digest(manifest) != entry["source_manifest_sha256"]
        or batch.digest(stage2 / domain / "source_manifest.json") != entry["source_manifest_sha256"]
        or batch.digest(initial) != spec["model_sha256"]
        or batch.digest(old_model) != record["best_model"]["sha256"]
        or record["best_model"]["sha256"] != entry["previous_best_sha256"]
        or record["model_config"] != spec
        or record["stage1_best"] != plan[domain]["stage1_best"]
        or prior_configuration["profile"] != weak
        or prior_configuration["training_configuration"] != config
    ):
        raise ValueError(f"{domain}: frozen inputs or comparison model changed")
    identity = {
        "study_sha256": batch.digest(study_path),
        "runner_sha256": batch.digest(__file__),
        "training_sha256": batch.digest(train.__file__),
        "camera_transform_sha256": batch.digest(camera_extended.__file__),
        "source_manifest_sha256": entry["source_manifest_sha256"],
    }
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "identity.json"
    if marker.exists() and previous.read_json(marker) != identity:
        raise ValueError("Output belongs to another study or code version")
    write_progress(marker, identity)
    complete = output / "complete.json"
    if complete.exists():
        saved = previous.read_json(complete)
        if batch.digest(output / "best.keras") != saved["best_model"]["sha256"]:
            raise ValueError("Completed checkpoint does not match its report")
        print(f"[{domain}] 완료 분야 재사용", flush=True)
        return
    cv2.setNumThreads(1)
    rows, val, pairs = train.prepare_source(domain, archive, task, reference, output)
    if previous.read_json(output / "source_manifest.json") != previous.read_json(manifest):
        raise ValueError("Original Train/Validation/Test split changed")
    write_progress(
        output / "run_configuration.json",
        {
            "identity": identity,
            "study": study,
            "model": spec,
            "training_configuration": config,
            "profile": profile,
            "previous_profile": weak,
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "gpu": [
                tf.config.experimental.get_device_details(d)
                for d in tf.config.list_physical_devices("GPU")
            ],
            "source_test_evaluated": False,
            "prior_source_test_inspected": True,
        },
    )
    size, batch_size = spec["size"], task["batch_size"]
    cache, shifted, pair_indexes = train.image_cache(
        local, archive, rows, val, pairs, task, size, profile["lighting"], expanded=profile
    )
    old_validation = np.lib.format.open_memmap(
        local / "previous_validation.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(val), size, size, 3),
    )
    for index, row in enumerate(val):
        old_validation[index] = camera_extended.transform(
            cache[len(rows) + index], weak, "expanded", train.exp.item_seed(row, 1707)
        )
    old_validation.flush()
    shifted["weak"] = old_validation
    keras.utils.set_random_seed(config["seed"])
    model = keras.models.load_model(initial, compile=False)
    train.exp.configure_partial(model)
    best = train.phase(
        model,
        (config, task, rows, val, profile, cache, shifted, pair_indexes),
        output / "expanded",
        local / "saving",
        spec,
        "expanded",
        config["epochs"],
        config["learning_rate"],
    )
    saved = train.export_model(model, output, "best", local / "saving")
    datasets = train.validation_pipelines(cache, shifted, len(rows), val, batch_size, size)
    metrics = {"stronger_expanded": {condition: best[condition] for condition in datasets}}
    del model
    for name, path in (("initial", initial), ("previous_expanded", old_model)):
        keras.backend.clear_session()
        model = keras.models.load_model(path, compile=False)
        metrics[name] = train.validate(
            train.predictor(model, spec),
            datasets,
            val,
            len(task["classes"]),
            f"[{domain} · {name}]",
        )
        del model
    write_progress(
        complete,
        {
            "best_model": saved,
            "best_epoch": best["epoch"],
            "model_config": spec,
            "validation": metrics,
            "source_test_evaluated": False,
            "target_accuracy": None,
            "historical_checkpoint_independence_verified": False,
            "epoch_selection": "max min(clean,strong expanded) Macro-F1, then clean",
        },
    )
    print(f"[{domain}] 후속 학습 완료 · 최적 {best['epoch']}회차", flush=True)


def comparison(output):
    records = {
        domain: previous.read_json(output / domain / "complete.json") for domain in colab.DOMAINS
    }
    selected = {}
    arms = ("initial", "previous_expanded", "stronger_expanded")
    with (output / "validation_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "model", "condition", "accuracy", "macro_f1"])
        for domain, record in records.items():
            metrics = record["validation"]
            selected[domain] = max(
                arms,
                key=lambda arm: (
                    min(metrics[arm][c]["macro_f1"] for c in ("clean", "weak", "expanded")),
                    metrics[arm]["clean"]["macro_f1"],
                    -arms.index(arm),
                ),
            )
            for arm in arms:
                for condition, result in metrics[arm].items():
                    writer.writerow(
                        [domain, arm, condition, result["accuracy"], result["macro_f1"]]
                    )
    write_progress(
        output / "selection.json",
        {
            "selected_on_validation": selected,
            "criteria": (
                "max worst(clean, previous expanded, stronger expanded) F1; clean; earlier arm"
            ),
            "source_test_evaluated": False,
            "prior_source_test_inspected": True,
        },
    )


def plots(output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    try:
        for row, domain in enumerate(colab.DOMAINS):
            history = previous.read_json(output / domain / "expanded/history.json")
            axes[row, 0].plot(
                [r["epoch"] for r in history],
                [r["train_loss_mean_per_update"] for r in history],
                marker="o",
            )
            axes[row, 0].set_title(f"{domain} · training loss")
            for condition in ("clean", "weak", "expanded"):
                axes[row, 1].plot(
                    [r["epoch"] for r in history],
                    [r[condition]["macro_f1"] for r in history],
                    marker="o",
                    label=condition,
                )
            axes[row, 1].set_title(f"{domain} · Validation Macro-F1")
            axes[row, 1].legend()
            for axis in axes[row]:
                axis.set_xlabel("Epoch")
                axis.grid(alpha=0.2)
        fig.suptitle("Measured stronger camera adaptation · Validation only", fontsize=16)
        (output / "plots").mkdir(exist_ok=True)
        fig.savefig(output / "plots/learning_curves.png", dpi=180)
    finally:
        plt.close(fig)


def launch(runtime, work, project, study_path, my_drive):
    import fcntl

    study = previous.read_json(study_path)
    study_sha = batch.digest(study_path)
    root = my_drive / "MediFlow_adaptation_runs" / study["bundle_sha256"][:16]
    destination = root / ("measured_strength_" + study_sha[:12])
    destination.mkdir(exist_ok=True)
    with (work.parent / f"mediflow_strength_{study_sha}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        python, versions = batch.prepare_python(work, executor=colab.run_live)
        if not versions["gpu"]:
            raise RuntimeError("Colab에서 GPU 런타임을 선택하세요.")
        env = batch.python_environment(python, runtime)
        env.update(MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        if subprocess.run(
            [str(python), "-c", "import matplotlib"], env=env, capture_output=True
        ).returncode:
            colab.run_live(
                [str(python), "-m", "pip", "install", "matplotlib>=3.8,<4"],
                env,
                work / "matplotlib_install.log",
            )
        write_progress(
            destination / "study.json",
            {"study_sha256": study_sha, "study": study, "environment": versions},
        )
        code = destination / "code"
        code.mkdir(exist_ok=True)
        for name in ("adaptation_strength.py", "adaptation_stage1.py", "camera_extended.py"):
            (code / name).write_bytes((runtime / "src/mediflow_datasets" / name).read_bytes())
        failures = {}
        command = [str(python), "-u", "-m", "mediflow_datasets.adaptation_strength"]
        for domain in colab.DOMAINS:
            log = destination / f"{domain}_{uuid.uuid4().hex[:12]}.log"
            try:
                with tempfile.TemporaryDirectory(prefix=f"{domain}_", dir=work) as temp:
                    local = Path(temp)
                    archive = local / "source.zip"
                    if not (destination / domain / "complete.json").exists():
                        config = previous.read_json(runtime / "configs/adaptation_stage1.json")
                        source = batch.find_source(project / "datasets", config["tasks"][domain])
                        colab.copy_source(source, archive)
                    colab.run_live(
                        command
                        + [
                            "--domain",
                            domain,
                            "--runtime",
                            str(runtime),
                            "--stage1",
                            str(root / "stage1"),
                            "--stage2",
                            str(root / "stage2"),
                            "--study",
                            str(study_path),
                            "--archive",
                            str(archive),
                            "--output",
                            str(destination / domain),
                            "--local",
                            str(local),
                        ],
                        env,
                        log,
                    )
            except Exception as error:
                failures[domain] = {"error": str(error), "log": log.name}
                print(f"[{domain}] 미완료 · {error}", flush=True)
        write_progress(
            destination / "summary.json",
            {
                "completed_domains": [d for d in colab.DOMAINS if d not in failures],
                "failures": failures,
                "source_test_evaluated": False,
            },
        )
        if failures:
            raise RuntimeError(f"미완료 분야가 있습니다. {destination}")
        comparison(destination)
        colab.run_live(command + ["--plots", str(destination)], env, work / "plots.log")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", nargs=5)
    parser.add_argument("--plots", type=Path)
    parser.add_argument("--domain", choices=colab.DOMAINS)
    for name in ("runtime", "stage1", "stage2", "study", "archive", "output", "local"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.launch:
        launch(*(Path(arg) for arg in args.launch))
    elif args.plots:
        plots(args.plots)
    else:
        run_domain(
            args.runtime,
            args.stage1,
            args.stage2,
            args.study,
            args.domain,
            args.archive,
            args.output,
            args.local,
        )


if __name__ == "__main__":
    main()
