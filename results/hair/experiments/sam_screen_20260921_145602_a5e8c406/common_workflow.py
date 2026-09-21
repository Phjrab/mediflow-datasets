"""Standalone Colab orchestration; shared contracts, audit gates and artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import stat
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import keras
import numpy as np
import tensorflow as tf

from mediflow_datasets import common_engine as engine

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def extract_zip(source, destination):
    """Validate every member before creating any output; no overwrite."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(source) as archive:
        seen = set()
        for item in archive.infolist():
            name = item.orig_filename
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or ":" in name
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError("Unsafe ZIP member: " + name)
            key = name.rstrip("/").casefold()
            if key in seen:
                raise ValueError("Duplicate ZIP destination: " + name)
            seen.add(key)
        destination.mkdir(parents=True)
        archive.extractall(destination)


def roots_for(extracted):
    roots = {}
    for kind in ("original", "augmented"):
        matches = [
            p
            for p in Path(extracted).rglob("*")
            if p.is_dir()
            and p.name.lower() == kind
            and all((p / s).is_dir() for s in ("train", "val", "test"))
        ]
        if len(matches) != 1:
            raise ValueError(f"{kind}/train,val,test 구조를 하나로 확인하세요: {matches}")
        roots[kind] = matches[0]
    return roots


def verify_audit(directory, domain, classes, digest):
    directory = Path(directory)
    manifest = engine.read_json(directory / "audit_manifest.json")
    for name in ("audit_summary.json", "image_inventory.csv", "audit_issues.csv"):
        if engine.file_hash(directory / name) != manifest[name]:
            raise ValueError("검증 보고서가 변경됐습니다: " + name)
    audit = engine.read_json(directory / "audit_summary.json")
    if (
        audit["domain"] != domain
        or audit["class_names"] != classes
        or audit["data_sha256"] != digest
        or audit.get("protocol") != "common_audit_v1"
        or audit["status"] != "mechanical_checks_passed_with_limitations"
    ):
        raise ValueError("대상/클래스/데이터가 다르거나 검증 문제가 있습니다. 공통 ①을 확인하세요.")
    return audit


def prepare(config, profiles, sources, commit, local_parent="/content"):
    c = dict(config)
    c.setdefault("expected_data_sha256", "")
    c.setdefault("seeds", [c["seed"]])
    if c["domain"] not in profiles or c["mode"] not in (
        "audit",
        "comparison",
        "suite",
        "baseline3",
        "paper_suite",
        "supcon_compare",
        "supcon_repeat",
        "paper_screen",
        "sam_screen",
    ):
        raise ValueError("DOMAIN/MODE 설정을 확인하세요.")
    if c["train_variant"] not in ("original", "augmented"):
        raise ValueError("TRAIN_VARIANT는 original 또는 augmented입니다.")
    for key in ("batch_size", "epochs1", "epochs2", "extension_epochs"):
        if not isinstance(c[key], int) or c[key] <= 0:
            raise ValueError(key + "는 양의 정수여야 합니다.")
    if (
        not isinstance(c["seeds"], list)
        or not c["seeds"]
        or any(not isinstance(value, int) or value < 0 for value in c["seeds"])
        or len(set(c["seeds"])) != len(c["seeds"])
    ):
        raise ValueError("SEEDS는 서로 다른 0 이상의 정수 목록이어야 합니다.")
    if c["mode"] == "baseline3" and len(c["seeds"]) != 3:
        raise ValueError("baseline3는 정확히 3개의 seed가 필요합니다.")
    if c["mode"] == "paper_suite" and len(c["seeds"]) != 3:
        raise ValueError("paper_suite는 정확히 3개의 seed가 필요합니다.")
    if c["mode"] == "supcon_compare" and c["seeds"] != [42]:
        raise ValueError("supcon_compare의 선별 seed는 [42]여야 합니다.")
    if c["mode"] == "supcon_repeat" and c["seeds"] != [43, 44]:
        raise ValueError("supcon_repeat의 확인 seed는 [43, 44]여야 합니다.")
    if c["mode"] == "paper_screen" and c["seeds"] != [42]:
        raise ValueError("paper_screen의 선별 seed는 [42]여야 합니다.")
    if c["mode"] == "sam_screen" and c["seeds"] != [42]:
        raise ValueError("sam_screen의 선별 seed는 [42]여야 합니다.")
    project = Path(c["project_root"])
    if not project.is_dir():
        raise FileNotFoundError("PROJECT_ROOT 폴더를 확인하세요: " + str(project))
    if c["mode"] != "audit" and not (
        c["audit_dir"] or c["expected_data_sha256"]
    ):
        raise ValueError("AUDIT_DIR 또는 확인된 EXPECTED_DATA_SHA256을 입력하세요.")
    if c["data_zip"]:
        candidates = [Path(c["data_zip"])]
    else:
        candidates = sorted(
            p
            for p in (project / "datasets").rglob(c["domain"] + "*")
            if p.is_file() and zipfile.is_zipfile(p)
        )
    if len(candidates) != 1 or not zipfile.is_zipfile(candidates[0]):
        raise ValueError(f"DATA_ZIP으로 ZIP 하나를 지정하세요: {candidates}")
    source = candidates[0]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    local = Path(local_parent) / ("mediflow_" + run_id)
    local.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(source) as archive:
        required = (
            source.stat().st_size + sum(m.file_size for m in archive.infolist()) + 2 * 1024**3
        )
    if required > shutil.disk_usage(local).free:
        raise RuntimeError("Colab 압축 해제 공간이 부족합니다.")
    copied = local / "input.zip"
    shutil.copyfile(source, copied)
    digest = engine.file_hash(copied)
    if digest != engine.file_hash(source):
        raise OSError("Drive ZIP 복사 내용 불일치")
    classes = profiles[c["domain"]]
    audit = None
    if c["mode"] != "audit":
        if c["audit_dir"]:
            audit = verify_audit(c["audit_dir"], c["domain"], classes, digest)
        else:
            expected = c["expected_data_sha256"].strip().lower()
            if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
                raise ValueError("EXPECTED_DATA_SHA256은 64자리 SHA-256이어야 합니다.")
            if digest != expected:
                raise ValueError("DATA_ZIP이 확인된 SHA-256과 다릅니다.")
            audit = {
                "domain": c["domain"],
                "class_names": classes,
                "data_sha256": digest,
                "protocol": "expected_sha256_v1",
                "status": "independent_audit_skipped",
                "limitations": [
                    "Independent common audit was skipped by the project owner",
                    "Person, lesion and capture-session leakage remains unverified",
                    "Perceptual near-duplicate and clinical label checks were not performed",
                ],
            }
    extract_zip(copied, local / "dataset")
    settings = {
        k: c[k]
        for k in (
            "domain",
            "mode",
            "seed",
            "seeds",
            "batch_size",
            "epochs1",
            "epochs2",
            "extension_epochs",
            "train_variant",
        )
    }
    settings.update(
        classes=classes,
        data_sha256=digest,
        source_hashes={k: signature(v) for k, v in sources.items()},
        protocol="common_v1",
        audit=signature(audit),
        environment=dict(
            tensorflow=tf.__version__,
            keras=keras.__version__,
            numpy=np.__version__,
            python=platform.python_version(),
        ),
    )
    if c["mode"] in (
        "paper_suite",
        "supcon_compare",
        "supcon_repeat",
        "paper_screen",
        "sam_screen",
    ):
        settings["experiments"] = c.get("experiments", [])
    if c["mode"] == "sam_screen":
        settings.update(
            sam_rho=c.get("sam_rho"),
            parent_run_dir=c.get("parent_run_dir"),
            parent_stage1_sha256=c.get("parent_stage1_sha256"),
        )
    sig = signature(settings)
    if c["resume_dir"]:
        output = Path(c["resume_dir"])
        if c["mode"] == "audit":
            raise ValueError("검사는 새 실행으로 시작하세요. RESUME_DIR을 비우세요.")
        if engine.read_json(output / "run_config.json")["signature"] != sig:
            raise ValueError(
                "코드/설정/환경/데이터/검증이 다른 실행입니다. 새 결과 폴더를 사용하세요."
            )
    else:
        output = project / "2_results" / c["domain"] / (c["mode"] + "_" + run_id)
        output.mkdir(parents=True, exist_ok=False)
        engine.write_json(
            output / "run_config.json",
            {
                "signature": sig,
                "settings": settings,
                "code_commit_at_generation": commit,
                "code_state": "embedded sources include uncommitted changes; exact sources saved",
                "source_zip": str(source),
                "audit_source": c["audit_dir"] or "expected_data_sha256_only",
                "baseline": "ImageNet pretrained EfficientNet; historical metrics not reused",
                "gpu": [str(d) for d in tf.config.list_physical_devices("GPU")],
            },
        )
        for name, code in sources.items():
            (output / (name + ".py")).write_text(code, encoding="utf-8")
        engine.write_json(output / "class_names.json", classes)
        if c["audit_dir"]:
            shutil.copyfile(
                Path(c["audit_dir"]) / "audit_summary.json", output / "audit_summary.json"
            )
            shutil.copyfile(
                Path(c["audit_dir"]) / "image_inventory.csv", output / "image_inventory.csv"
            )
        elif audit:
            engine.write_json(output / "audit_summary.json", audit)
    context = dict(
        config=c,
        classes=classes,
        signature=sig,
        output=output,
        local=local,
        data_hash=digest,
        audit=audit,
        extracted=local / "dataset",
        project=project,
    )
    if c["mode"] != "audit":
        context["roots"] = roots_for(context["extracted"])
        # Dataset ZIP is immutable and matches the audit; verify class folders again.
        for root in context["roots"].values():
            for split in ("train", "val", "test"):
                found = sorted(p.name for p in (root / split).iterdir() if p.is_dir())
                if found != sorted(classes):
                    raise ValueError(f"클래스 불일치: {root / split}")
    print("실행 결과:", output)
    return context


def archive_results(context):
    output = context["output"]
    destination = output.parent / (output.name + "_results_" + uuid.uuid4().hex[:8] + ".zip")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob("*")):
            if p.is_file() and p.suffix not in (".keras", ".tmp"):
                archive.write(p, output.name + "/" + p.relative_to(output).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise OSError("보고서 ZIP 검사 실패")
    print("로컬로 내려받을 결과 ZIP:", destination)
    return destination


def audit_run(context):
    from mediflow_datasets.common_audit import audit_dataset

    summary = audit_dataset(
        context["extracted"],
        context["output"],
        context["config"]["domain"],
        context["classes"],
        context["data_hash"],
    )
    engine.write_json(
        context["output"] / "audit_manifest.json",
        {
            name: engine.file_hash(context["output"] / name)
            for name in ("audit_summary.json", "image_inventory.csv", "audit_issues.csv")
        },
    )
    archive_results(context)
    print("검사 상태:", summary["status"], "\n학습 AUDIT_DIR:", context["output"])
    return summary


def factory(context, variant, seed=None):
    shuffle_seed = context["config"]["seed"] if seed is None else seed

    def load(split, size, shuffle):
        # All trials use exactly the same ORIGINAL validation and test images.
        root = context["roots"][variant if split == "train" else "original"]
        ds = keras.utils.image_dataset_from_directory(
            root / split,
            class_names=context["classes"],
            label_mode="categorical",
            image_size=(size, size),
            interpolation="bilinear",
            batch_size=context["config"]["batch_size"],
            shuffle=shuffle,
            seed=shuffle_seed if shuffle else None,
        )
        paths = [Path(p).relative_to(context["extracted"]).as_posix() for p in ds.file_paths]
        return ds.prefetch(tf.data.AUTOTUNE), paths

    return load


def run(context):
    c, output = context["config"], context["output"]
    if c["mode"] == "comparison":
        trials = [
            dict(id=kind, backbone="B0", size=224, loss="ce", variant=kind)
            for kind in ("original", "augmented")
        ]
    else:
        trials = [dict(spec, variant=c["train_variant"]) for spec in engine.TRIALS]
    records = []
    try:
        for spec in trials:
            spec["class_count"] = len(context["classes"])
            spec["class_names"] = context["classes"]
            load = factory(context, spec["variant"])
            record = engine.run_trial(
                spec,
                load,
                output,
                context["signature"],
                c["seed"],
                c["epochs1"],
                0 if c["mode"] == "comparison" else c["epochs2"],
            )
            records.append(record)
            engine.write_json(output / "progress.json", {"completed": [r["id"] for r in records]})
        if c["mode"] == "suite":
            parent = records[-1]
            records.append(
                engine.extend_b1(
                    parent,
                    factory(context, c["train_variant"]),
                    output,
                    context["signature"],
                    c["seed"],
                    c["extension_epochs"],
                )
            )
        engine.write_json(output / "all_validation_results.json", records)
        return records
    except Exception as exc:
        engine.write_json(
            output / ("failure_" + uuid.uuid4().hex[:8] + ".json"),
            {
                "error": repr(exc),
                "completed": [r["id"] for r in records],
                "resume_dir": str(output),
            },
        )
        print("중단. 완료된 실험을 유지합니다. RESUME_DIR:", output)
        raise


def confusion(ax, metrics, title):
    cm = np.asarray(metrics["confusion_matrix"])
    ax.imshow(cm, cmap="Blues")
    codes = [f"C{i}" for i in range(len(cm))]
    ax.set(
        title=title,
        xlabel="Predicted",
        ylabel="True",
        xticks=range(len(cm)),
        yticks=range(len(cm)),
        xticklabels=codes,
        yticklabels=codes,
    )
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )


def errors(context, csv_path, destination):
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image

    frame = pd.read_csv(csv_path)
    wrong = frame[frame.true_index != frame.pred_index].head(8)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, wrong.iterrows(), strict=False):
        path = (context["extracted"] / row["path"]).resolve()
        if not path.is_relative_to(context["extracted"].resolve()):
            raise ValueError("Prediction path escapes dataset")
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"True C{row.true_index} / Pred C{row.pred_index}")
    if wrong.empty:
        fig.suptitle("No misclassifications")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def overview(context, records):
    import matplotlib.pyplot as plt
    import pandas as pd

    output = context["output"]
    fig, axes = plt.subplots(
        (len(records) + 1) // 2, 4, figsize=(24, 4.5 * ((len(records) + 1) // 2)), squeeze=False
    )
    for index, record in enumerate(records):
        row, col = divmod(index, 2)
        for offset, metric in enumerate(("accuracy", "loss")):
            ax = axes[row, col * 2 + offset]
            h = record["history"]
            x = np.arange(1, len(h[metric]) + 1)
            ax.plot(x, h[metric], label="Train")
            ax.plot(x, h["val_" + metric], label="Validation")
            if record["stage_boundary"] < len(x):
                ax.axvline(record["stage_boundary"] + 0.5, ls="--", color="gray")
            if "extension_boundary" in record:
                ax.axvline(record["extension_boundary"] + 0.5, ls=":", color="green")
            ax.set(title=record["id"] + " / " + metric, xlabel="Epoch", ylabel=metric)
            if metric == "accuracy":
                ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            ax.legend()
        directory = output / record["id"] / record["attempt"]
        errors(
            context, directory / "validation_predictions.csv", directory / "validation_errors.png"
        )
    fig.suptitle(context["config"]["domain"] + " / Loss definitions differ across CE, LS, Focal")
    fig.tight_layout()
    fig.savefig(output / "all_training_curves.png", dpi=180)
    fig.savefig(output / "all_training_curves.pdf")
    plt.show()
    plt.close(fig)
    table = pd.DataFrame(
        [
            dict(
                experiment=r["id"],
                selected_stage=r["selected_stage"],
                validation_accuracy=r["validation"]["accuracy"],
                validation_macro_f1=r["validation"]["macro_f1"],
                parameters=r["parameters"],
                seconds_this_trial=r["training_seconds"],
                epochs=len(r["history"]["accuracy"]),
                model_bytes=r["model_bytes"],
            )
            for r in records
        ]
    )
    table.to_csv(output / "experiment_comparison.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    x = np.arange(len(records))
    axes[0].bar(x - 0.2, table.validation_accuracy, 0.4, label="Validation Accuracy")
    axes[0].bar(x + 0.2, table.validation_macro_f1, 0.4, label="Validation Macro F1")
    axes[0].set(xticks=x, xticklabels=table.experiment, ylim=(0, 1))
    axes[0].tick_params(axis="x", labelrotation=15)
    axes[0].legend()
    matrix = np.array([r["validation"]["class_f1"] for r in records])
    axes[1].imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[1].set(
        xticks=range(len(context["classes"])),
        xticklabels=[f"C{i}" for i in range(len(context["classes"]))],
        yticks=x,
        yticklabels=table.experiment,
        title="Validation class F1",
    )
    for i in range(len(records)):
        for j in range(len(context["classes"])):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "validation_performance_dashboard.png", dpi=180)
    plt.show()
    plt.close(fig)
    fig, axes = plt.subplots(
        (len(records) + 2) // 3, 3, figsize=(18, 6 * ((len(records) + 2) // 3)), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, records, strict=False):
        ax.axis("on")
        confusion(ax, r["validation"], r["id"])
    fig.tight_layout()
    fig.savefig(output / "all_validation_confusion_matrices.png", dpi=180)
    plt.show()
    plt.close(fig)


def finish(context, records):
    import matplotlib.pyplot as plt

    output = context["output"]
    overview(context, records)
    winner = engine.select_winner(records)
    model_path = engine.selected_model_path(output, winner)
    selection = dict(
        winner=winner["id"],
        model_sha256=engine.file_hash(model_path),
        signature=context["signature"],
        validation=winner["validation"],
    )
    selected_file = output / "selection_before_test.json"
    if selected_file.exists() and engine.read_json(selected_file) != selection:
        raise ValueError("이미 고정한 선택 모델이 다릅니다.")
    if not selected_file.exists():
        engine.write_json(selected_file, selection)
    marker = output / "test_completed.json"
    if marker.exists():
        tested = engine.read_json(marker)
        if tested["selection"] != selection:
            raise ValueError("기존 Test 모델과 다릅니다.")
        for name, digest in tested["hashes"].items():
            if engine.file_hash(output / name) != digest:
                raise ValueError("Test 파일이 변경됐습니다.")
        metrics = tested["metrics"]
    else:
        keras.backend.clear_session()
        model = keras.models.load_model(model_path, compile=False)
        ds, paths = factory(context, winner["spec"]["variant"])(
            "test", winner["spec"]["size"], False
        )
        metrics = engine.evaluate_to_files(model, ds, paths, output, "final_test")
        engine.write_json(
            marker,
            dict(
                selection=selection,
                metrics=metrics,
                hashes={
                    name: engine.file_hash(output / name)
                    for name in ("final_test_metrics.json", "final_test_predictions.csv")
                },
            ),
        )
        del model
    fig, ax = plt.subplots(figsize=(8, 8))
    confusion(ax, metrics, winner["id"] + " / Final Test")
    fig.tight_layout()
    fig.savefig(output / "final_test_confusion_matrix.png", dpi=180)
    plt.show()
    plt.close(fig)
    errors(context, output / "final_test_predictions.csv", output / "final_test_errors.png")
    card = dict(
        domain=context["config"]["domain"],
        class_names=context["classes"],
        normal_included="정상" in context["classes"],
        validation=winner["validation"],
        test=metrics,
        selected_model=winner["id"],
        model_sha256=selection["model_sha256"],
        input_size=winner["spec"]["size"],
        data_sha256=context["data_hash"],
        status="public_data_candidate_not_device_validated",
        limitations=context["audit"]["limitations"]
        + [
            "Single seed; small differences are not established as robust gains",
            "Out-of-scope rejection absent; scores are not calibrated correctness",
        ],
    )
    engine.write_json(output / "model_card.json", card)
    package_parent = (
        context["project"] / "2_results" / context["config"]["domain"] / "selected_models"
    )
    package = package_parent / (output.name + "_" + uuid.uuid4().hex[:8])
    package.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(model_path, package / "model.keras")
    if engine.file_hash(package / "model.keras") != selection["model_sha256"]:
        raise OSError("모델 복사 불일치")
    for name in (
        "class_names.json",
        "model_card.json",
        "run_config.json",
        "selection_before_test.json",
        "audit_summary.json",
        "final_test_metrics.json",
        "common_engine.py",
        "common_audit.py",
        "common_workflow.py",
    ):
        shutil.copyfile(output / name, package / name)
    engine.write_json(
        package / "preprocessing.json",
        dict(
            input_shape=[winner["spec"]["size"], winner["spec"]["size"], 3],
            color="RGB",
            dtype="float32",
            pixel_range=[0, 255],
            external_normalization=False,
            internal_rescaling="1/255",
            resize="TensorFlow bilinear; no crop/pad; antialias=False",
            exif_transpose=False,
            output="softmax scores in class_names.json order",
        ),
    )
    engine.write_json(
        package / "manifest.json",
        {p.name: engine.file_hash(p) for p in package.iterdir() if p.is_file()},
    )
    destination = Path(shutil.make_archive(str(package), "zip", package.parent, package.name))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise OSError("모델 ZIP 손상")
        manifest = engine.read_json(package / "manifest.json")
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read(package.name + "/" + name)).hexdigest() != digest:
                raise OSError("ZIP 내용 불일치: " + name)
    destination.with_suffix(".zip.sha256").write_text(
        engine.file_hash(destination), encoding="ascii"
    )
    archive_results(context)
    print("선정 모델:", winner["id"], "\nTest:", metrics, "\n후보 ZIP:", destination)
    return card
