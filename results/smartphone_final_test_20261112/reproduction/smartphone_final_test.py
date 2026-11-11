"""Frozen six-model smartphone comparison on the original Source Test split only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import shutil
import zipfile
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from mediflow_datasets import adaptation_final_test as original
from mediflow_datasets import adaptation_stage1 as training
from mediflow_datasets import camera_extended as camera
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as experiment
from mediflow_datasets.domain_adaptation_io import write_progress

MODELS = ("initial", "stage1", "stage2", "stage3", "stage4", "stage5")
CONDITIONS = ("clean", "phone_lighting", "phone_expanded", "phone_refit", "phone_joint")
RESULT_ZIP = "MediFlow_Smartphone_Final_Test_Results_20261112.zip"


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_runtime(runtime):
    manifest = data.read_json(runtime / "runtime_manifest.json")
    for name, expected in manifest["files"].items():
        if data.sha256(data.local_path(runtime, name)) != expected:
            raise ValueError("Test 실행 파일 해시 불일치: " + name)
    envelope = data.read_json(runtime / "evaluation_plan.json")
    plan = envelope["plan"]
    if object_hash(plan) != envelope["plan_sha256"]:
        raise ValueError("고정 Test 계획 해시 불일치")
    if (
        plan["schema"] != "mediflow_phone_final_test_v1"
        or tuple(plan["models"]) != MODELS
        or tuple(plan["conditions"]) != CONDITIONS
        or plan["selected_on_validation"] != "stage4"
        or plan["task"]["domain"] != "web_skin"
        or plan["test_used_for_selection"] is not False
        or plan["target_photos_used"] is not False
    ):
        raise ValueError("스마트폰 최종 Test 계약이 다릅니다.")
    return envelope


def verify_models(plan, roots):
    for record in plan["input_records"]:
        path = data.local_path(roots[record["root"]], record["file"])
        if data.sha256(path) != record["sha256"]:
            raise ValueError("기존 학습 기록이 다릅니다: " + record["file"])
    paths = {}
    for name, info in plan["models"].items():
        path = data.local_path(roots[info["root"]], info["file"])
        if data.sha256(path) != info["sha256"]:
            raise ValueError("Test 모델 해시 불일치: " + name)
        paths[name] = path
    return paths


def input_views(source, rows, plan, folder):
    """Decode the 400 Test inputs once; all checkpoints receive identical cached arrays."""
    folder.mkdir(parents=True, exist_ok=True)
    identity = object_hash({"plan": plan, "rows": [row.path for row in rows]})
    size = plan["model_spec"]["size"]
    marker = folder / "complete.json"
    shape = (len(rows), size, size, 3)
    if marker.exists():
        saved = data.read_json(marker)
        if saved["identity"] != identity:
            raise ValueError("Test 입력 캐시 설정이 다릅니다.")
        for name, expected in saved["sha256"].items():
            if data.sha256(folder / name) != expected:
                raise ValueError("Test 입력 캐시 손상: " + name)
        views = {c: np.load(folder / f"{c}.npy", mmap_mode="r") for c in CONDITIONS}
        if any(a.shape != shape or a.dtype != np.float32 for a in views.values()):
            raise ValueError("Test 입력 캐시 크기·형식이 다릅니다.")
        return views
    needed = len(CONDITIONS) * len(rows) * size * size * 3 * 4
    if shutil.disk_usage(folder).free < needed + 1024**3:
        raise RuntimeError("Colab 로컬 SSD의 Test 입력 캐시 공간이 부족합니다.")
    views = {
        c: np.lib.format.open_memmap(folder / f"{c}.npy", mode="w+", dtype="float32", shape=shape)
        for c in CONDITIONS
    }
    with zipfile.ZipFile(source) as archive, tf.device("/CPU:0"):
        for i, row in enumerate(rows):
            image = experiment.image_tensor(
                archive.read(row.path), size, plan["task"]["input_policy"]
            )
            views["clean"][i] = image
            seed = experiment.item_seed(row, plan["transform_seed"])
            for condition in CONDITIONS[1:]:
                views[condition][i] = camera.transform(
                    image, plan["profiles"][condition], "expanded", seed
                )
            if i == 0 or (i + 1) % 40 == 0 or i + 1 == len(rows):
                training.show("Test 원본·변환 입력 준비", i + 1, len(rows))
    for array in views.values():
        array.flush()
    write_progress(
        marker,
        {
            "identity": identity,
            "sha256": {f"{c}.npy": data.sha256(folder / f"{c}.npy") for c in CONDITIONS},
        },
    )
    return views


def checked_scores(values, count, classes):
    values = np.asarray(values)
    if (
        values.shape != (count, classes)
        or values.dtype != np.float32
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
        or not np.allclose(values.sum(-1), 1, atol=1e-5)
    ):
        raise ValueError("Test 예측 점수의 크기·유한성·합이 잘못되었습니다.")
    return values


def pair_result(output, model, condition, identity, rows, classes, predict=None, inputs=None):
    """Resume completed model/condition pairs only after verifying saved scores and metrics."""
    name = model + "__" + condition
    score_path = output / "scores" / (name + ".npz")
    metric_path = output / "metrics" / (name + ".json")
    if metric_path.exists():
        record = data.read_json(metric_path)
        if record["identity"] != identity or data.sha256(score_path) != record["scores_sha256"]:
            raise ValueError("저장된 Test 실행·예측 파일이 다릅니다: " + name)
        with np.load(score_path, allow_pickle=False) as saved:
            scores = checked_scores(saved["scores"], len(rows), classes)
        metrics = experiment.classification(rows, scores, classes)
        if metrics != record["metrics"]:
            raise ValueError("저장된 Test 지표와 예측이 다릅니다: " + name)
        print("완료 결과 재사용:", name, flush=True)
        return metrics, scores
    if predict is None:
        return None
    batches = []
    batch = identity["batch_size"]
    total = (len(rows) + batch - 1) // batch
    for step, start in enumerate(range(0, len(rows), batch), 1):
        values = tf.convert_to_tensor(np.asarray(inputs[start : start + batch]), tf.float32)
        batches.append(np.asarray(predict(values).numpy(), dtype=np.float32))
        if step == 1 or step % 10 == 0 or step == total:
            training.show(f"[{model} · {condition}] Test 배치", step, total)
    scores = checked_scores(np.concatenate(batches), len(rows), classes)
    metrics = experiment.classification(rows, scores, classes)
    score_path.parent.mkdir(exist_ok=True)
    metric_path.parent.mkdir(exist_ok=True)
    temporary = score_path.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, scores=scores)
    temporary.replace(score_path)
    write_progress(
        metric_path,
        {"identity": identity, "scores_sha256": data.sha256(score_path), "metrics": metrics},
    )
    print(
        f"[{model} · {condition}] 완료 · Accuracy={metrics['accuracy']}"
        f" · Macro-F1={metrics['macro_f1']}",
        flush=True,
    )
    return metrics, scores


def plots(output, result):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    folder = output / "plots"
    folder.mkdir(exist_ok=True)
    labels = ["Original", "Lighting", "Weak appearance", "Strong appearance", "Joint profile"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for axis, metric in zip(axes, ("accuracy", "macro_f1"), strict=True):
        for model in MODELS:
            label = model + (" (fixed on Validation)" if model == "stage4" else "")
            axis.plot(
                range(5),
                [result["metrics"][model][c][metric] for c in CONDITIONS],
                marker="o",
                label=label,
                linewidth=2 if model == "stage4" else 1,
            )
        axis.set_xticks(range(5), labels, rotation=15)
        axis.set_ylim(0, 1.01)
        axis.set_title("Source Test · " + metric)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.savefig(folder / "test_comparison.png", dpi=160)
    plt.close(fig)
    for model in MODELS:
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), layout="constrained")
        for axis, condition, label in zip(axes, CONDITIONS, labels, strict=True):
            matrix = np.asarray(result["metrics"][model][condition]["confusion_matrix"])
            axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(result["test_support"]))
            axis.set_title(label, fontsize=9)
            axis.set_xticks(range(len(matrix)))
            axis.set_yticks(range(len(matrix)))
            for (y, x), value in np.ndenumerate(matrix):
                axis.text(x, y, str(value), ha="center", va="center", fontsize=8)
        fig.suptitle(f"{model} · Source Test; row=true / column=predicted; class order in plan")
        fig.savefig(folder / (model + "_confusion_matrices.png"), dpi=160)
        plt.close(fig)


def result_zip(output):
    path = output / RESULT_ZIP
    partial = path.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.rglob("*")):
            if file.is_file() and file.suffix in (".json", ".csv", ".png", ".npz", ".py", ".md"):
                archive.write(file, file.relative_to(output).as_posix())
    partial.replace(path)
    return path


def run(runtime, source, prior, joint, output, local, *, require_gpu=True):
    runtime, source, prior, joint, output, local = map(
        lambda p: Path(p).resolve(), (runtime, source, prior, joint, output, local)
    )
    for write_root in (output, local):
        for protected in (runtime, prior, joint):
            if write_root.is_relative_to(protected) or protected.is_relative_to(write_root):
                raise ValueError("Test 출력/작업 폴더는 기존 입력 폴더와 분리해야 합니다.")
        if source.is_relative_to(write_root):
            raise ValueError("Source ZIP은 Test 출력/작업 폴더 밖에 있어야 합니다.")
    envelope = verify_runtime(runtime)
    plan = envelope["plan"]
    model_paths = verify_models(plan, {"runtime": runtime, "prior": prior, "joint": joint})
    if require_gpu and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Colab에서 GPU 런타임을 선택하세요.")
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
    output.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    marker = output / "evaluation_plan.json"
    if marker.exists() and data.read_json(marker) != envelope:
        raise ValueError("이미 저장된 최종 Test 계획이 다릅니다.")
    write_progress(marker, envelope)
    archive = local / "source.zip"
    if not archive.is_file() or data.sha256(archive) != plan["task"]["source_sha256"]:
        print("Drive Source ZIP → Colab 로컬 SSD 복사", flush=True)
        shutil.copyfile(source, archive)
    rows = original.prepare_test(archive, plan["task"], plan, runtime / "source_manifest.json")
    if [sum(r.label == i for r in rows) for i in range(len(plan["task"]["classes"]))] != plan[
        "test_support"
    ]:
        raise ValueError("Test 클래스별 수량이 다릅니다.")
    views = input_views(archive, rows, plan, local / "inputs")
    metrics, scores = {}, {}
    for number, name in enumerate(MODELS, 1):
        print(f"모델 {number}/{len(MODELS)} · {name}", flush=True)
        metrics[name], scores[name] = {}, {}
        identities = {
            c: {
                "plan_sha256": envelope["plan_sha256"],
                "model_sha256": plan["models"][name]["sha256"],
                "condition": c,
                "batch_size": plan["task"]["batch_size"],
                "runtime_manifest_sha256": data.sha256(runtime / "runtime_manifest.json"),
            }
            for c in CONDITIONS
        }
        completed = {
            c: pair_result(output, name, c, identities[c], rows, len(plan["task"]["classes"]))
            for c in CONDITIONS
        }
        model = predict = None
        if any(value is None for value in completed.values()):
            keras.backend.clear_session()
            checkpoint = local / (name + ".keras")
            if not checkpoint.exists() or data.sha256(checkpoint) != plan["models"][name]["sha256"]:
                shutil.copyfile(model_paths[name], checkpoint)
            if data.sha256(checkpoint) != plan["models"][name]["sha256"]:
                raise ValueError("로컬 모델 복사 해시 불일치: " + name)
            model = keras.models.load_model(checkpoint, compile=False, safe_mode=True)
            size = plan["model_spec"]["size"]
            if model.input_shape != (None, size, size, 3) or len(model.outputs) != 4:
                raise ValueError("PMG 입력/출력 계약이 다릅니다.")
            predict = training.predictor(model, plan["model_spec"])
        for condition in CONDITIONS:
            pair = completed[condition] or pair_result(
                output,
                name,
                condition,
                identities[condition],
                rows,
                len(plan["task"]["classes"]),
                predict,
                views[condition],
            )
            metrics[name][condition], scores[name][condition] = pair
        del model, predict
        keras.backend.clear_session()
        gc.collect()
    result = {
        "domain": "web_skin",
        "classes": plan["task"]["classes"],
        "test_count": len(rows),
        "test_support": plan["test_support"],
        "selected_on_validation": plan["selected_on_validation"],
        "metrics": metrics,
        "test_used_for_selection": False,
        "source_sha256": plan["task"]["source_sha256"],
        "target_accuracy": None,
    }
    write_progress(output / "test_results.json", result)
    original.write_comparison(output, [result])
    original.write_class_metrics(output, [result])
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        import csv

        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "condition",
                "source_path",
                "true_index",
                "predicted_index",
                *["score_" + str(i) for i in range(len(result["classes"]))],
            ]
        )
        for name in MODELS:
            for condition in CONDITIONS:
                for row, values in zip(rows, scores[name][condition], strict=True):
                    writer.writerow(
                        [
                            name,
                            condition,
                            row.path,
                            row.label,
                            int(np.argmax(values)),
                            *values.tolist(),
                        ]
                    )
    plots(output, result)
    repro = output / "reproduction"
    repro.mkdir(exist_ok=True)
    for name in ("runtime_manifest.json", "source_manifest.json"):
        shutil.copyfile(runtime / name, repro / name)
    for file in (runtime / "src/mediflow_datasets").glob("*.py"):
        shutil.copyfile(file, repro / file.name)
    verify_models(plan, {"runtime": runtime, "prior": prior, "joint": joint})
    write_progress(
        output / "summary.json",
        {
            "status": "completed",
            "comparisons": len(MODELS) * len(CONDITIONS),
            "test_images": len(rows),
            "selected_before_test": plan["selected_on_validation"],
            "plan_sha256": envelope["plan_sha256"],
            "source_test_evaluated": True,
            "training_executed": False,
            "target_photos_used": False,
            "target_accuracy": None,
            "test_used_for_selection": False,
            "official_candidate_changed": False,
            "environment": {
                "tensorflow": tf.__version__,
                "keras": keras.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "gpu": [
                    tf.config.experimental.get_device_details(g)
                    for g in tf.config.list_physical_devices("GPU")
                ],
            },
        },
    )
    destination = result_zip(output)
    print("스마트폰 최종 Test 완료 · ZIP:", destination, flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "source", "previous", "joint", "output", "local"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.runtime, args.source, args.previous, args.joint, args.output, args.local)


if __name__ == "__main__":
    main()
