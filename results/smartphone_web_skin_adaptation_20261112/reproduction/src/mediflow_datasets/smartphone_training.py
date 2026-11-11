"""Sequential web_skin trials, duplicate-condition reuse and resumable epochs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import math
import shutil
import sys
import time
import zipfile
from copy import deepcopy
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from mediflow_datasets import adaptation_stage1 as train
from mediflow_datasets import adaptation_stage3 as aligned
from mediflow_datasets import camera_extended as camera
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp
from mediflow_datasets import domain_alignment
from mediflow_datasets.domain_adaptation_io import write_progress

CONDITIONS = ("clean", "phone_lighting", "phone_expanded", "phone_refit")


def trial_model(destination, trial):
    record = data.read_json(destination / trial / "complete.json")
    if "alias_of" in record:
        if trial != "stage2" or record["alias_of"] not in ("stage1_basic", "stage1_selected"):
            raise ValueError("Invalid repeated-condition alias")
        path = destination / record["alias_of"] / "best.keras"
    else:
        path = destination / trial / "best.keras"
    if data.sha256(path) != record["best_model"]["sha256"]:
        raise ValueError("Trial model changed: " + trial)
    return path


def reuse_stage2(destination, first, role, spec):
    alias = f"stage1_{role}"
    path = trial_model(destination, alias)
    folder = destination / "stage2"
    folder.mkdir(exist_ok=True)
    write_progress(
        folder / "complete.json",
        {
            "stage": 2,
            "alias_of": alias,
            "new_training_executed": False,
            "reason": "Stage2 equals Stage1; user chose to reuse this identical condition",
            "model_spec": spec,
            "best_epoch": first["best_epoch"],
            "epochs_completed": 0,
            "epochs_reused": first["epochs_completed"],
            "best_model": {"file": f"../{alias}/best.keras", "sha256": data.sha256(path)},
            "validation": {
                "clean": first["validation"]["clean"],
                "phone_expanded": first["validation"]["phone_lighting"],
            },
            "source_test_evaluated": False,
            "target_accuracy": None,
        },
    )
    print("[stage2] 1차와 같은 변환: 선택된 1차 모델 재사용, 중복 학습 생략", flush=True)


def verify_runtime(runtime):
    manifest = data.read_json(runtime / "runtime_manifest.json")
    if manifest["schema"] != "mediflow_phone_runtime_manifest_v1":
        raise ValueError("Unknown smartphone runtime")
    for name, expected in manifest["files"].items():
        if data.sha256(data.local_path(runtime, name)) != expected:
            raise ValueError("Runtime file changed: " + name)
    study = data.read_json(runtime / "study.json")
    if study["domain"] != "web_skin" or study["task"]["domain"] != "web_skin":
        raise ValueError("This runtime accepts web_skin only")
    if study["task"]["classes"] != ["건선", "아토피", "여드름", "정상", "주사"]:
        raise ValueError("web_skin class order changed")
    if data.sha256(runtime / "source_manifest.json") != study["source_manifest_sha256"]:
        raise ValueError("Frozen source split changed")
    return study


def decrypt_target(runtime, role, size, key_text):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if role not in ("calibration", "holdout"):
        raise ValueError("Unknown Target role")
    try:
        key = base64.b64decode(key_text.strip(), validate=True)
        if len(key) != 32:
            raise ValueError("Wrong key size")
        member = f"target_encrypted/{role}_{size}.bin"
        metadata = data.read_json(runtime / member.replace(".bin", ".json"))
        cipher = (runtime / member).read_bytes()
        plaintext = AESGCM(key).decrypt(cipher[:12], cipher[12:], member.encode())
    except Exception:
        raise ValueError("스마트폰 전용 복호화 키 또는 암호화 입력을 확인하세요.") from None
    if (
        metadata["aad_utf8"] != member
        or metadata["input_size"] != size
        or metadata["cipher"] != "AES-256-GCM"
        or metadata["input_policy"] != "exif_transpose_icc_srgb_then_center_square"
        or hashlib.sha256(plaintext).hexdigest() != metadata["plaintext_sha256"]
    ):
        raise ValueError("Target metadata contract mismatch")
    rows = metadata["rows"]
    expected = {"P01", "P02", "P03", "P04"} if role == "calibration" else {"P05"}
    if (
        not rows
        or {r["person"] for r in rows} != expected
        or any(r["split"] != role or r["label"] is not None for r in rows)
        or len({r["id"] for r in rows}) != len(rows)
    ):
        raise ValueError("Target split, participant or label contract mismatch")
    with np.load(io.BytesIO(plaintext), allow_pickle=False) as arrays:
        images = arrays["images"]
    if (
        images.shape != (len(rows), size, size, 3)
        or images.dtype != np.float32
        or not np.isfinite(images).all()
        or images.min() < 0
        or images.max() > 255
    ):
        raise ValueError("Invalid Target input array")
    return {"rows": rows, "images": images, "sha256": metadata["plaintext_sha256"]}


def input_cache(source_zip, rows, val, task, folder, size, profiles):
    """Read Drive once, decode source images once per resolution; never train on Test."""
    folder.mkdir(exist_ok=True)
    identity = train.object_hash(
        {
            "rows": [r.path for r in [*rows, *val]],
            "size": size,
            "input_policy": task["input_policy"],
            "profiles": profiles,
            "source": task["source_sha256"],
        }
    )
    marker = folder / "complete.json"
    if marker.exists():
        if data.read_json(marker)["identity"] != identity:
            raise ValueError("Local cache identity changed; choose a new work directory")
        cache = np.load(folder / "inputs.npy", mmap_mode="r")
        views = {name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in CONDITIONS[1:]}
        return cache, views
    needed = (len(rows) + 4 * len(val)) * size * size * 3 * 4
    if shutil.disk_usage(folder).free < needed + 2 * 1024**3:
        raise RuntimeError("Colab 로컬 입력 캐시 공간이 부족합니다.")
    cache = np.lib.format.open_memmap(
        folder / "inputs.npy",
        mode="w+",
        dtype="float32",
        shape=(len(rows) + len(val), size, size, 3),
    )
    with zipfile.ZipFile(source_zip) as archive:
        for index, row in enumerate([*rows, *val]):
            with tf.device("/CPU:0"):
                cache[index] = exp.image_tensor(archive.read(row.path), size, task["input_policy"])
            if index == 0 or (index + 1) % 256 == 0 or index + 1 == len(cache):
                train.show(f"[{size}px] 로컬 SSD 입력 준비", index + 1, len(cache))
    cache.flush()
    views = {}
    for condition, stage in zip(CONDITIONS[1:], ("stage1", "stage2", "stage4"), strict=True):
        view = np.lib.format.open_memmap(
            folder / f"{condition}.npy", mode="w+", dtype="float32", shape=(len(val), size, size, 3)
        )
        p = deepcopy(profiles[stage])
        p["blur_input_scale"] *= size / 256
        for i, row in enumerate(val):
            view[i] = camera.transform(
                cache[len(rows) + i], p, "expanded", exp.item_seed(row, 1707)
            )
        view.flush()
        views[condition] = view
    write_progress(marker, {"identity": identity, "images": len(cache), "bytes": needed})
    return cache, views


def datasets(cache, views, rows, val, spec, batch):
    return train.validation_pipelines(cache, views, len(rows), val, batch, spec["size"])


def checkpoint_score(entry, condition):
    return min(entry["clean"]["macro_f1"], entry[condition]["macro_f1"]), entry["clean"]["macro_f1"]


def prune_snapshots(folder, keep):
    """Only this trial's generated optimizer snapshots; original model files stay untouched."""
    root = folder.resolve()
    for path in folder.glob("snapshot_*"):
        if path.name == keep:
            continue
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_relative_to(root) or resolved.parent != root:
            raise ValueError("Unsafe temporary snapshot path")
        if path.is_dir():
            shutil.rmtree(resolved)


def run_trial(runtime, destination, local, study, spec, stage, rows, val, cache, views, target):
    name = f"stage1_{spec['role']}" if stage == 1 else f"stage{stage}"
    folder = destination / name
    folder.mkdir(exist_ok=True)
    identity = {
        "study_sha256": data.sha256(runtime / "study.json"),
        "stage": stage,
        "model": spec,
        "code_sha256": data.sha256(Path(__file__)),
        "training_core_sha256": data.sha256(Path(train.__file__)),
    }
    marker = folder / "identity.json"
    if marker.exists() and data.read_json(marker) != identity:
        raise ValueError("Existing trial identity differs from this experiment")
    write_progress(marker, identity)
    if (folder / "complete.json").exists():
        complete = data.read_json(folder / "complete.json")
        if data.sha256(folder / "best.keras") != complete["best_model"]["sha256"]:
            raise ValueError("Completed model hash mismatch")
        print(f"[{name}] 완료 결과 재사용", flush=True)
        return complete
    keras.backend.clear_session()
    keras.utils.set_random_seed(study["seed"])
    model = keras.models.load_model(
        data.local_path(runtime, spec["model"]), compile=False, safe_mode=True
    )
    exp.configure_partial(model)
    profile = deepcopy(
        study["profiles"]["stage1" if stage == 1 else "stage4" if stage == 4 else "stage2"]
    )
    profile["blur_input_scale"] *= spec["size"] / 256
    alignment = None
    if stage == 3:
        setup_path = folder / "alignment_setup.json"
        if not setup_path.exists():
            write_progress(
                setup_path, aligned.calibrate(model, cache, rows, study["task"], spec, target)
            )
        setup = data.read_json(setup_path)
        if setup["target_sha256"] != target["sha256"]:
            raise ValueError("Alignment calibration changed")
        alignment = {**target, **setup}
    optimizer = keras.optimizers.Adam(study["learning_rate"])
    step, predict, *extra = train.steps(
        model, spec, optimizer, len(study["task"]["classes"]), alignment=alignment
    )
    state = tf.train.Checkpoint(model=model, optimizer=optimizer)
    history = train.epoch_records(folder, study["epochs"])
    if history:
        train.restore_epoch(state, folder, history[-1])
        print(f"[{name}] {len(history)}/{study['epochs']} epoch까지 복원", flush=True)
    condition = {1: "phone_lighting", 2: "phone_expanded", 3: "phone_expanded", 4: "phone_refit"}[
        stage
    ]
    data_sets = datasets(
        cache, {condition: views[condition]}, rows, val, spec, study["task"]["batch_size"]
    )
    write_progress(
        folder / "run_configuration.json",
        {
            "identity": identity,
            "study": study,
            "applied_profile": profile,
            "alignment": alignment is not None,
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "gpu": [
                tf.config.experimental.get_device_details(d)
                for d in tf.config.list_physical_devices("GPU")
            ],
            "source_test_evaluated": False,
            "target_labels_used": False,
        },
    )
    batch = study["task"]["batch_size"]
    n = len(rows)
    total = math.ceil(n / batch)
    for epoch in range(len(history), study["epochs"]):
        keras.utils.set_random_seed(study["seed"] + epoch)
        order = np.random.default_rng(study["seed"] + epoch).permutation(n)
        seeds = [exp.item_seed(row, study["seed"], epoch) for row in rows]
        apply = [exp.augmentation_applies(s, study["augmentation_probability"]) for s in seeds]

        def transform(i, image, seeds=seeds, apply=apply):
            return camera.transform(image, profile, "expanded", seeds[i]) if apply[i] else image

        eligible = list(range(total - int(n % batch == 1)))
        schedule = (
            domain_alignment.balanced_schedule(target["rows"], eligible, study["seed"], epoch)
            if alignment
            else {}
        )
        training = train.pipeline(
            cache, order, [rows[i].label for i in order], batch, spec["size"], transform
        )
        started = time.monotonic()
        losses = []
        terms = []
        print(
            f"\n[{name}] Epoch {epoch + 1}/{study['epochs']} · {n:,}장 · {total}배치"
            " · 첫 배치 그래프 준비",
            flush=True,
        )
        for i, (images, labels) in enumerate(training):
            if i in schedule:
                loss, ce, mmd, coral = map(
                    float, extra[0](images, labels, target["images"][schedule[i]])
                )
                terms.append((ce, mmd, coral))
                losses.append(loss)
            else:
                losses.append(float(step(images, labels)))
            if i == 0 or i + 1 == total or (i + 1) % max(5, total // 15) == 0:
                elapsed = time.monotonic() - started
                train.show(
                    f"[{name}] Epoch {epoch + 1}/{study['epochs']}",
                    i + 1,
                    total,
                    f"loss={np.mean(losses):.5f} · "
                    f"남은 학습 약 {elapsed * (total / (i + 1) - 1):.0f}초",
                )
        seconds = time.monotonic() - started
        metrics = train.validate(
            predict, data_sets, val, len(study["task"]["classes"]), f"[{name}]"
        )
        entry = {
            "epoch": epoch + 1,
            "train_loss_mean_per_update": float(np.mean(losses)),
            "train_seconds": seconds,
            "training_batches": total,
            "optimizer_updates": total * (4 if spec["architecture"] == "pmg" else 1),
            "alignment_steps": len(terms),
            "target_images_used": sum(map(len, schedule.values())),
            **metrics,
        }
        if terms:
            ce, mmd, coral = np.mean(terms, axis=0).tolist()
            entry["alignment"] = {"classification": ce, "mmd": mmd, "coral": coral}
        previous_best = (
            max(history, key=lambda r: checkpoint_score(r, condition)) if history else None
        )
        if previous_best is None or checkpoint_score(entry, condition) > checkpoint_score(
            previous_best, condition
        ):
            entry["best_model"] = train.export_model(
                model, folder, f"best_epoch_{epoch + 1:03}", local
            )
        else:
            entry["best_model"] = previous_best["best_model"]
        committed = train.save_epoch(state, folder, local, entry)
        history.append(committed)
        write_progress(folder / "history.json", history)
        prune_snapshots(folder, committed["snapshot"])
        # Keep epoch model references; retain only the latest optimizer snapshot.
        curves(folder, name, history, condition)
        print(
            f"[{name}] Epoch {epoch + 1} 저장 완료 · "
            f"원본 Val F1={metrics['clean']['macro_f1']:.5f} · "
            f"변환 Val F1={metrics[condition]['macro_f1']:.5f}",
            flush=True,
        )
    best = max(history, key=lambda r: checkpoint_score(r, condition))
    original_best = folder / best["best_model"]["file"]
    if data.sha256(original_best) != best["best_model"]["sha256"]:
        raise ValueError("Best epoch model is missing or changed")
    saved = train.publish_model(original_best, folder, "best")
    result = {
        "stage": stage,
        "model_spec": spec,
        "best_epoch": best["epoch"],
        "best_model": saved,
        "selection_condition": condition,
        "validation": {k: best[k] for k in ("clean", condition)},
        "epochs_completed": len(history),
        "source_test_evaluated": False,
        "target_accuracy": None,
    }
    write_progress(folder / "complete.json", result)
    del model, state, optimizer, step, predict, extra
    keras.backend.clear_session()
    return result


def curves(folder, name, history, condition):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout="constrained")
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_loss_mean_per_update"] for r in history], "o-")
    axes[0].set_title(name + " · train loss")
    for key in ("clean", condition):
        axes[1].plot(epochs, [r[key]["macro_f1"] for r in history], "o-", label=key)
    axes[1].set_title("Validation Macro-F1")
    axes[1].legend()
    for key in ("mmd", "coral"):
        if any("alignment" in r for r in history):
            axes[2].plot(
                epochs, [r.get("alignment", {}).get(key, np.nan) for r in history], "o-", label=key
            )
    axes[2].set_title("Alignment losses (Stage3)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    fig.savefig(folder / "learning_curves.png", dpi=160)
    plt.close(fig)


def final_reports(runtime, destination, local, study, spec, rows, val, cache, views, key):
    """Freeze Source-Validation choice before opening the reserved Target participant."""
    task = study["task"]
    count = len(task["classes"])
    models = {
        "initial": data.local_path(runtime, spec["model"]),
        "stage1": destination / f"stage1_{spec['role']}/best.keras",
        **{f"stage{s}": trial_model(destination, f"stage{s}") for s in (2, 3, 4)},
    }
    validation = {}
    for name, path in models.items():
        keras.backend.clear_session()
        model = keras.models.load_model(path, compile=False)
        validation[name] = train.validate(
            train.predictor(model, spec),
            datasets(cache, views, rows, val, spec, task["batch_size"]),
            val,
            count,
            f"[최종 비교 · {name}]",
        )
        del model
    winner = max(
        models,
        key=lambda name: (
            min(v["macro_f1"] for v in validation[name].values()),
            validation[name]["clean"]["macro_f1"],
            -list(models).index(name),
        ),
    )
    selection = {
        "selected_on": "Source Validation only",
        "model": winner,
        "path": str(models[winner].relative_to(destination))
        if winner != "initial"
        else spec["model"],
        "sha256": data.sha256(models[winner]),
        "conditions": list(CONDITIONS),
        "target_holdout_used": False,
        "source_test_evaluated": False,
    }
    write_progress(destination / "validation_comparison.json", validation)
    write_progress(destination / "selection.json", selection)
    target = decrypt_target(runtime, "calibration", spec["size"], key)
    holdout = decrypt_target(runtime, "holdout", spec["size"], key)
    reference = data.read_json(destination / "stage3/alignment_setup.json")["reference"]
    indices = aligned.sampled_indices(rows, count)
    source_images = cache[indices]
    diagnostics, baseline_predictions = {}, {}
    for name, path in models.items():
        keras.backend.clear_session()
        model = keras.models.load_model(path, compile=False)
        _, source_features = aligned.feature_outputs(model, source_images, spec, task["batch_size"])
        diagnostics[name] = {}
        for role, packet in (("calibration", target), ("holdout", holdout)):
            scores, features = aligned.feature_outputs(
                model, packet["images"], spec, task["batch_size"]
            )
            mmd, coral = map(float, domain_alignment.losses(source_features, features, reference))
            predictions = scores.argmax(1)
            if name == "initial":
                baseline_predictions[role] = predictions
            diagnostics[name][role] = {
                "mmd": mmd,
                "coral": coral,
                "target_accuracy": None,
                "predicted_class_changes_vs_initial": int(
                    np.sum(predictions != baseline_predictions[role])
                ),
                "predictions": [
                    {"id": r["id"], "predicted_index": int(p), "score": float(s[p])}
                    for r, p, s in zip(packet["rows"], predictions, scores, strict=True)
                ],
            }
        del model
    write_progress(
        destination / "feature_and_prediction_diagnostics.json",
        {
            "target_disease_labels_available": False,
            "target_accuracy": None,
            "score_meaning": "uncalibrated_model_score_not_probability_of_correctness",
            "holdout_opened_after_selection": True,
            "fixed_reference": reference,
            "source_train_diagnostic_count": len(indices),
            "models": diagnostics,
        },
    )
    comparison_plot(destination, validation)
    write_progress(
        destination / "summary.json",
        {
            "status": "completed",
            "training_trials": sum(
                data.read_json(destination / name / "complete.json").get(
                    "new_training_executed", True
                )
                for name in ("stage1_basic", "stage1_selected", "stage2", "stage3", "stage4")
            ),
            "aliases": {"stage2": "stage1"} if study.get("reuse_identical_stage2") else {},
            "validation_condition_aliases": study.get("validation_condition_aliases", {}),
            "chosen_starting_role": spec["role"],
            "selected_model": selection,
            "source_test_evaluated": False,
            "target_accuracy": None,
        },
    )


def comparison_plot(destination, validation):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for ax, metric in zip(axes, ("accuracy", "macro_f1"), strict=True):
        for i, condition in enumerate(CONDITIONS):
            ax.bar(
                np.arange(len(validation)) + i * 0.18,
                [r[condition][metric] for r in validation.values()],
                width=0.18,
                label=condition,
            )
        ax.set_xticks(np.arange(len(validation)) + 0.27, list(validation))
        ax.set_title("Validation " + metric)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(destination / "validation_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(
        len(validation), len(CONDITIONS), figsize=(13, 3 * len(validation)), layout="constrained"
    )
    for row, (name, values) in enumerate(validation.items()):
        for column, condition in enumerate(CONDITIONS):
            axis = axes[row, column]
            matrix = np.asarray(values[condition]["confusion_matrix"])
            axis.imshow(matrix, cmap="Blues")
            axis.set_title(name + " / " + condition, fontsize=8)
            axis.set_xticks(range(len(matrix)))
            axis.set_yticks(range(len(matrix)))
            for (y, x), value in np.ndenumerate(matrix):
                axis.text(x, y, str(value), ha="center", va="center", fontsize=7)
    fig.suptitle("Validation only · rows=true, columns=predicted · class order in study")
    fig.savefig(destination / "validation_confusion_matrices.png", dpi=160)
    plt.close(fig)


def results_zip(destination):
    for trial in ("stage1_basic", "stage1_selected", "stage2", "stage3", "stage4"):
        trial_model(destination, trial)
    path = destination / "MediFlow_Smartphone_WebSkin_Results_20261111.zip"
    temporary = path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for item in sorted(destination.rglob("*")):
            relative = item.relative_to(destination)
            if (
                item.is_file()
                and item != path
                and item != temporary
                and not any(p.startswith("snapshot_") for p in relative.parts)
                and not item.name.startswith("best_epoch_")
                and item.suffix in {".json", ".csv", ".png", ".md", ".keras"}
            ):
                archive.write(item, relative.as_posix())
    temporary.replace(path)
    return path


def run(runtime, source, destination, local, key, *, require_gpu=True):
    runtime, source, destination, local = map(Path, (runtime, source, destination, local))
    study = verify_runtime(runtime)
    if require_gpu and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Colab 런타임에서 GPU를 선택한 뒤 다시 실행하세요.")
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
    destination.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    marker = destination / "runtime_identity.json"
    identity = {
        "runtime_manifest_sha256": data.sha256(runtime / "runtime_manifest.json"),
        "study_sha256": data.sha256(runtime / "study.json"),
    }
    if marker.exists() and data.read_json(marker) != identity:
        raise ValueError("Run identity mismatch")
    write_progress(marker, identity)
    if (destination / "summary.json").exists():
        if data.read_json(destination / "summary.json").get("status") == "completed":
            return results_zip(destination)
    # Verify the key before copying/scanning several thousand Source images.
    for size in {s["size"] for s in study["task"]["models"].values()}:
        decrypt_target(runtime, "calibration", size, key)
    archive = local / "source.zip"
    if not archive.exists() or data.sha256(archive) != study["task"]["source_sha256"]:
        print("Drive Source ZIP → Colab 로컬 SSD 복사", flush=True)
        shutil.copyfile(source, archive)
    rows, val, _ = train.prepare_source(
        "web_skin", archive, study["task"], data.read_json(runtime / "reference.json"), destination
    )
    if data.sha256(destination / "source_manifest.json") != study["source_manifest_sha256"]:
        raise ValueError("Source Train/Validation/Test membership changed")
    caches = {}

    def cached(spec):
        size = spec["size"]
        if size not in caches:
            caches[size] = input_cache(
                archive, rows, val, study["task"], local / f"cache_{size}", size, study["profiles"]
            )
        return caches[size]

    first = {}
    for role, raw_spec in study["task"]["models"].items():
        spec = {**raw_spec, "role": role}
        cache, views = cached(spec)
        first[role] = run_trial(
            runtime,
            destination,
            local / f"saving_{role}",
            study,
            spec,
            1,
            rows,
            val,
            cache,
            views,
            None,
        )
    role = max(
        first,
        key=lambda k: (
            checkpoint_score(first[k]["validation"], "phone_lighting"),
            -list(first).index(k),
        ),
    )
    spec = {**study["task"]["models"][role], "role": role}
    cache, views = cached(spec)
    write_progress(
        destination / "stage1_selection.json",
        {
            "selected_role": role,
            "validation": first,
            "later_stages_start_from_original": True,
            "source_test_evaluated": False,
        },
    )
    target = decrypt_target(runtime, "calibration", spec["size"], key)
    for stage in (2, 3, 4):
        if (
            stage == 2
            and study.get("reuse_identical_stage2")
            and study["profiles"]["stage1"] == study["profiles"]["stage2"]
        ):
            reuse_stage2(destination, first[role], role, spec)
            continue
        run_trial(
            runtime,
            destination,
            local / f"saving_stage{stage}",
            study,
            spec,
            stage,
            rows,
            val,
            cache,
            views,
            target,
        )
    final_reports(runtime, destination, local, study, spec, rows, val, cache, views, key)
    archive = results_zip(destination)
    actual = data.read_json(destination / "summary.json")["training_trials"]
    print(f"전체 {actual}회 학습 완료 · 결과 ZIP:", archive, flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "source", "output", "local"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    key = sys.stdin.readline().strip()
    run(args.runtime, args.source, args.output, args.local, key)


if __name__ == "__main__":
    main()
