"""One independent joint-profile trial with frozen prior Validation comparisons."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from mediflow_datasets import adaptation_stage1 as train
from mediflow_datasets import adaptation_stage3 as features
from mediflow_datasets import camera_extended as camera
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as experiment
from mediflow_datasets import domain_alignment
from mediflow_datasets import smartphone_training as previous
from mediflow_datasets.domain_adaptation_io import write_progress

CONDITIONS = (*previous.CONDITIONS, "phone_joint")
RESULT_ZIP = "MediFlow_Smartphone_Joint_Results_20261112.zip"


def verify_previous(directory, study):
    directory = Path(directory)
    recipe = study["joint_experiment"]
    for name, expected in recipe["previous_result_files"].items():
        if data.sha256(data.local_path(directory, name)) != expected:
            raise ValueError("이전 결과 파일이 다릅니다: " + name)
    identity = data.read_json(directory / "runtime_identity.json")
    if identity != recipe["previous_runtime_identity"]:
        raise ValueError("이전 휴대폰 실행 결과와 설정이 다릅니다.")
    if data.read_json(directory / "summary.json")["status"] != "completed":
        raise ValueError("이전 1~4차 학습이 완료되지 않았습니다.")
    paths = {}
    for name, info in recipe["previous_models"].items():
        path = data.local_path(directory, info["file"])
        if data.sha256(path) != info["sha256"]:
            raise ValueError("이전 비교 모델 해시가 다릅니다: " + name)
        paths[name] = path
    return paths


def verify_study(runtime):
    study = previous.verify_runtime(runtime)
    recipe = study.get("joint_experiment", {})
    if (
        recipe.get("schema") != "mediflow_phone_joint_single_trial_v1"
        or recipe.get("candidate_id") != "f10926b662a4a43c"
        or recipe.get("training_trials") != 1
        or recipe.get("alignment_loss") is not False
        or recipe.get("source_test_used") is not False
        or recipe.get("target_holdout_used") is not False
        or recipe.get("authorization") != "User approved exploratory training of reviewed candidate"
        or study["joint_profile"] != data.read_json(runtime / "joint_candidate.json")["profile"]
        or tuple(recipe["validation_conditions"]) != CONDITIONS
        or list(study["task"]["models"]) != ["selected"]
    ):
        raise ValueError("공동 후보 학습 설정 계약이 다릅니다.")
    spec = study["task"]["models"]["selected"]
    if spec["architecture"] != "pmg" or spec["size"] != 256:
        raise ValueError("출발 모델은 적응 전 선정 PMG·B0·256이어야 합니다.")
    return study


def joint_view(cache, rows, val, study, size, folder):
    folder.mkdir(parents=True, exist_ok=True)
    identity = train.object_hash(
        {
            "profile": study["joint_profile"],
            "val": [r.path for r in val],
            "size": size,
            "seed": 1707,
            "input_policy": study["task"]["input_policy"],
            "source_sha256": study["task"]["source_sha256"],
        }
    )
    marker = folder / "complete.json"
    if marker.exists():
        if data.read_json(marker)["identity"] != identity:
            raise ValueError("새 후보 Validation 캐시 설정이 다릅니다.")
        return np.load(folder / "phone_joint.npy", mmap_mode="r")
    profile = deepcopy(study["joint_profile"])
    profile["blur_input_scale"] *= size / 256
    view = np.lib.format.open_memmap(
        folder / "phone_joint.npy", mode="w+", dtype="float32", shape=(len(val), size, size, 3)
    )
    for i, row in enumerate(val):
        view[i] = camera.transform(
            cache[len(rows) + i], profile, "expanded", experiment.item_seed(row, 1707)
        )
    view.flush()
    write_progress(marker, {"identity": identity, "count": len(val)})
    return view


def score(metrics):
    return min(m["macro_f1"] for m in metrics.values()), metrics["clean"]["macro_f1"]


def evaluate(path, spec, cache, views, rows, val, task, label):
    keras.backend.clear_session()
    model = keras.models.load_model(path, compile=False, safe_mode=True)
    result = train.validate(
        train.predictor(model, spec),
        previous.datasets(cache, views, rows, val, spec, task["batch_size"]),
        val,
        len(task["classes"]),
        label,
    )
    del model
    keras.backend.clear_session()
    return result


def final_comparison(
    runtime, destination, prior_paths, study, spec, rows, val, cache, views, target
):
    validation = deepcopy(data.read_json(runtime / "previous_validation.json"))
    if list(validation) != ["initial", "stage1", "stage2", "stage3", "stage4"]:
        raise ValueError("이전 공통 Validation 비교표가 다릅니다.")
    models = {"initial": data.local_path(runtime, spec["model"]), **prior_paths}
    for name, path in models.items():
        existing = validation[name]
        if tuple(existing) != previous.CONDITIONS:
            raise ValueError("이전 검증 조건이 다릅니다.")
        new = evaluate(
            path,
            spec,
            cache,
            {"phone_joint": views["phone_joint"]},
            rows,
            val,
            study["task"],
            f"[이전 모델 · 새 후보 검증 · {name}]",
        )
        validation[name]["phone_joint"] = new["phone_joint"]
    stage5 = destination / "stage5/best.keras"
    validation["stage5"] = evaluate(
        stage5, spec, cache, views, rows, val, study["task"], "[5차 · 공통 5조건 검증]"
    )
    models["stage5"] = stage5
    write_progress(destination / "validation_comparison.json", validation)
    winner = max(validation, key=lambda n: (*score(validation[n]), -list(validation).index(n)))
    selection = {
        "selected_on": "Source Validation only; five common conditions",
        "model": winner,
        "sha256": data.sha256(models[winner]),
        "conditions": list(CONDITIONS),
        "target_holdout_used": False,
        "source_test_evaluated": False,
        "device_similarity_selection": False,
        "official_candidate_changed": False,
    }
    write_progress(destination / "selection.json", selection)
    # Reuse old scalar metrics exactly; evaluate only Stage5 on the 42 calibration photographs.
    prior = data.read_json(runtime / "previous_calibration_diagnostics.json")
    reference = data.read_json(runtime / "feature_reference.json")
    indices = features.sampled_indices(rows, len(study["task"]["classes"]))
    keras.backend.clear_session()
    model = keras.models.load_model(stage5, compile=False, safe_mode=True)
    _, source_features = features.feature_outputs(
        model, cache[indices], spec, study["task"]["batch_size"]
    )
    scores, target_features = features.feature_outputs(
        model, target["images"], spec, study["task"]["batch_size"]
    )
    mmd, coral = map(float, domain_alignment.losses(source_features, target_features, reference))
    predictions = scores.argmax(1)
    baseline = prior["initial"]["predictions"]
    if [p["id"] for p in baseline] != [r["id"] for r in target["rows"]]:
        raise ValueError("휴대폰 설정용 사진 순서가 바뀌었습니다.")
    prior["stage5"] = {
        "mmd": mmd,
        "coral": coral,
        "target_accuracy": None,
        "predicted_class_changes_vs_initial": int(
            sum(int(p) != b["predicted_index"] for p, b in zip(predictions, baseline, strict=True))
        ),
        "predictions": [
            {"id": r["id"], "predicted_index": int(p), "score": float(s[p])}
            for r, p, s in zip(target["rows"], predictions, scores, strict=True)
        ],
    }
    del model
    keras.backend.clear_session()
    write_progress(
        destination / "calibration_diagnostics.json",
        {
            "models": prior,
            "prior_metrics_reused": ["initial", "stage1", "stage2", "stage3", "stage4"],
            "fixed_reference": reference,
            "target_count": 42,
            "source_count": len(indices),
            "target_labels_available": False,
            "target_accuracy": None,
            "target_holdout_used": False,
            "used_for_model_selection": False,
        },
    )
    plots(destination, validation)
    write_progress(
        destination / "summary.json",
        {
            "status": "completed",
            "training_trials": 1,
            "stage": 5,
            "candidate_id": study["joint_experiment"]["candidate_id"],
            "validation_selected_model": selection,
            "epochs_completed": study["epochs"],
            "reused_validation_results": 20,
            "new_validation_results": 10,
            "total_validation_results": 30,
            "source_test_evaluated": False,
            "target_holdout_used": False,
            "target_accuracy": None,
            "interpretation": "Exploratory reviewed-camera profile; device equivalence unverified",
        },
    )


def plots(destination, validation):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), layout="constrained")
    for ax, metric in zip(axes, ("accuracy", "macro_f1"), strict=True):
        for i, condition in enumerate(CONDITIONS):
            ax.bar(
                np.arange(len(validation)) + i * 0.15,
                [v[condition][metric] for v in validation.values()],
                width=0.15,
                label=condition,
            )
        ax.set_xticks(np.arange(len(validation)) + 0.3, list(validation))
        ax.set_title("Source Validation " + metric)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(destination / "validation_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(16, 3.6), layout="constrained")
    for ax, condition in zip(axes, CONDITIONS, strict=True):
        matrix = np.asarray(validation["stage5"][condition]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(condition, fontsize=9)
        ax.set_xticks(range(len(matrix)))
        ax.set_yticks(range(len(matrix)))
        for (y, x), value in np.ndenumerate(matrix):
            ax.text(x, y, str(value), ha="center", va="center", fontsize=8)
    fig.suptitle("Stage5 Validation; row=true / column=prediction; class order in study")
    fig.savefig(destination / "stage5_confusion_matrices.png", dpi=160)
    plt.close(fig)


def results_zip(destination):
    path = destination / RESULT_ZIP
    temporary = path.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(destination.rglob("*")):
            relative = p.relative_to(destination)
            if not p.is_file() or any(x.startswith("snapshot_") for x in relative.parts):
                continue
            if p.name.startswith("best_epoch_") or p.suffix not in (
                ".json",
                ".png",
                ".keras",
                ".md",
                ".py",
            ):
                continue
            z.write(p, relative.as_posix())
    temporary.replace(path)
    return path


def run(runtime, source, prior_directory, destination, local, key, *, require_gpu=True):
    runtime, source, destination, local = map(Path, (runtime, source, destination, local))
    study = verify_study(runtime)
    prior_paths = verify_previous(prior_directory, study)
    spec = {**study["task"]["models"]["selected"], "role": "selected"}
    if require_gpu and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Colab에서 GPU 런타임을 선택하세요.")
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
    destination.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    identity = {
        "study_sha256": data.sha256(runtime / "study.json"),
        "runtime_manifest_sha256": data.sha256(runtime / "runtime_manifest.json"),
    }
    marker = destination / "runtime_identity.json"
    if marker.exists() and data.read_json(marker) != identity:
        raise ValueError("5차 실행 식별값이 다릅니다.")
    write_progress(marker, identity)
    if (destination / "summary.json").exists():
        if data.read_json(destination / "summary.json")["status"] == "completed":
            previous.trial_model(destination, "stage5")
            return results_zip(destination)
    size = spec["size"]
    target = previous.decrypt_target(runtime, "calibration", size, key)
    old_local = Path("/content") / (
        "mediflow_phone_work_" + study["joint_experiment"]["previous_runtime_zip_sha256"][:12]
    )
    archive = old_local / "source.zip"
    if not archive.is_file() or data.sha256(archive) != study["task"]["source_sha256"]:
        archive = local / "source.zip"
        if not archive.is_file() or data.sha256(archive) != study["task"]["source_sha256"]:
            print("Drive Source ZIP → Colab 로컬 SSD 복사", flush=True)
            shutil.copyfile(source, archive)
        if data.sha256(archive) != study["task"]["source_sha256"]:
            raise ValueError("Source ZIP 해시 불일치")
    rows, val, _ = train.prepare_source(
        "web_skin", archive, study["task"], data.read_json(runtime / "reference.json"), destination
    )
    if data.sha256(destination / "source_manifest.json") != study["source_manifest_sha256"]:
        raise ValueError("원본 Train/Validation/Test 목록이 달라졌습니다.")
    old_cache = old_local / f"cache_{size}"
    folder = old_cache if (old_cache / "complete.json").exists() else local / f"cache_{size}"
    cache, views = previous.input_cache(
        archive, rows, val, study["task"], folder, size, study["profiles"]
    )
    views["phone_joint"] = joint_view(cache, rows, val, study, size, local / "joint_validation")
    previous.run_trial(
        runtime, destination, local / "saving_stage5", study, spec, 5, rows, val, cache, views, None
    )
    repro = destination / "reproduction"
    repro.mkdir(exist_ok=True)
    for name in ("study.json", "joint_candidate.json", "runtime_manifest.json"):
        shutil.copyfile(runtime / name, repro / name)
    for name in (
        "smartphone_joint_training.py",
        "smartphone_training.py",
        "adaptation_stage1.py",
        "camera_extended.py",
        "domain_adaptation_experiment.py",
    ):
        shutil.copyfile(runtime / "src/mediflow_datasets" / name, repro / name)
    final_comparison(
        runtime, destination, prior_paths, study, spec, rows, val, cache, views, target
    )
    archive = results_zip(destination)
    print("추가 1회 학습 완료 · 결과 ZIP:", archive, flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "source", "previous", "output", "local"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    key = sys.stdin.readline().strip()
    run(args.runtime, args.source, args.previous, args.output, args.local, key)


if __name__ == "__main__":
    main()
