"""Frozen MedSigLIP-448 linear-probe screening for Web Skin.

Only augmented Train and original Validation are read.  Test remains sealed until a
candidate is selected.  Image embeddings are written in deterministic shards so a
Colab interruption can resume without repeating completed feature extraction.
"""

from __future__ import annotations

import csv
import platform
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np

from mediflow_datasets import common_engine as engine

PROTOCOL = "web_skin_medsiglip_448_linear_probe_v1"
TRIAL_ID = "medsiglip_448_frozen_linear_seed_42"
EXPERIMENTS = [TRIAL_ID]
MODEL_ID = "google/medsiglip-448"
EXPECTED_CLASSES = ["건선", "아토피", "여드름", "정상", "주사"]
EXPECTED_COUNTS = {"train": 7200, "val": 500}
MODEL_CARD_URL = "https://huggingface.co/google/medsiglip-448"
OFFICIAL_GUIDE_URL = (
    "https://developers.google.com/health-ai-developer-foundations/medsiglip/get-started"
)
BASELINE = {
    "id": "pmg_b0_256_ce_seed_42",
    "validation_accuracy": 0.85,
    "validation_macro_f1": 0.8473279632397033,
    "validation_count": 500,
    "data_sha256": "f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d",
    "source": "web_skin_paper_suite_20260922_124758_717b465e",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def validate(context):
    c = context["config"]
    if c["domain"] != "web_skin" or c["mode"] != "web_skin_medsiglip_linear":
        raise ValueError("Web Skin MedSigLIP 모드가 아닙니다.")
    if context["classes"] != EXPECTED_CLASSES:
        raise ValueError("Web Skin 클래스 순서가 기존 계약과 다릅니다.")
    if context["data_hash"] != BASELINE["data_sha256"]:
        raise ValueError("Web Skin clean ZIP SHA-256이 기존 실험과 다릅니다.")
    if c["train_variant"] != "augmented":
        raise ValueError("학습 데이터는 기존과 같은 augmented여야 합니다.")
    if c["seed"] != 42 or c["seeds"] != [42]:
        raise ValueError("선별 실험 seed는 42 하나로 고정합니다.")
    if c.get("experiments") != EXPERIMENTS:
        raise ValueError("MedSigLIP Linear 실험 하나만 실행합니다.")
    fixed = {
        "model_id": MODEL_ID,
        "batch_size": 8,
        "epochs1": 50,
        "epochs2": 1,
        "extension_epochs": 1,
        "embedding_shard_size": 128,
        "linear_batch_size": 256,
        "linear_learning_rate": 1e-3,
        "linear_weight_decay": 1e-4,
    }
    for key, expected in fixed.items():
        if c.get(key) != expected:
            raise ValueError(f"{key}는 {expected!r}로 고정합니다.")


def _images(context, split):
    if split not in EXPECTED_COUNTS:
        raise ValueError("이 실험은 Train과 Validation만 사용합니다.")
    root = context["roots"]["augmented" if split == "train" else "original"] / split
    paths, labels = [], []
    for label, class_name in enumerate(context["classes"]):
        folder = root / class_name
        found = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        paths.extend(found)
        labels.extend([label] * len(found))
    if len(paths) != EXPECTED_COUNTS[split]:
        raise ValueError(f"{split} 이미지 수가 {EXPECTED_COUNTS[split]}장이 아닙니다: {len(paths)}")
    return paths, np.asarray(labels, dtype=np.int64)


def _shard_paths(directory, split, index):
    root = Path(directory) / "embedding_cache" / split
    root.mkdir(parents=True, exist_ok=True)
    return root / f"shard_{index:04d}.npz"


def _read_shard(path, expected_paths, expected_labels):
    try:
        with np.load(path, allow_pickle=False) as value:
            embeddings = value["embeddings"]
            labels = value["labels"]
            paths = value["paths"].tolist()
    except (OSError, ValueError, KeyError):
        return None
    if paths != expected_paths or not np.array_equal(labels, expected_labels):
        return None
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(paths)
        or not np.isfinite(embeddings).all()
    ):
        return None
    return embeddings.astype(np.float32, copy=False)


def _save_shard(path, embeddings, labels, paths):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex[:8] + ".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        paths=np.asarray(paths, dtype=np.str_),
    )
    temporary.replace(path)


def _load_encoder(token, model_id):
    import torch
    from transformers import AutoModel, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("MedSigLIP 특징 추출에는 Colab GPU가 필요합니다.")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = AutoModel.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    if not hasattr(model, "get_image_features"):
        raise TypeError("불러온 모델에 get_image_features가 없습니다.")
    return model, processor


def _extract_split(context, split, model, processor):
    import torch
    from PIL import Image

    paths, labels = _images(context, split)
    relative = [path.relative_to(context["extracted"]).as_posix() for path in paths]
    shard_size = context["config"]["embedding_shard_size"]
    batch_size = context["config"]["batch_size"]
    all_embeddings = []
    for shard_index, start in enumerate(range(0, len(paths), shard_size)):
        stop = min(start + shard_size, len(paths))
        shard_path = _shard_paths(context["output"], split, shard_index)
        cached = _read_shard(shard_path, relative[start:stop], labels[start:stop])
        if cached is not None:
            all_embeddings.append(cached)
            print(f"{split} embedding cache: {stop} / {len(paths)}")
            continue
        pieces = []
        for batch_start in range(start, stop, batch_size):
            batch_stop = min(batch_start + batch_size, stop)
            images = []
            for path in paths[batch_start:batch_stop]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.float16)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                features = model.get_image_features(pixel_values=pixel_values)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
            pieces.append(features.cpu().numpy())
        embeddings = np.concatenate(pieces).astype(np.float32, copy=False)
        _save_shard(shard_path, embeddings, labels[start:stop], relative[start:stop])
        all_embeddings.append(embeddings)
        print(f"{split} embedding: {stop} / {len(paths)}")
    return np.concatenate(all_embeddings), labels, relative


def _metrics(truth, probabilities):
    return engine.classification_metrics(truth, probabilities, len(EXPECTED_CLASSES))


def _linear_probe(context, train_x, train_y, val_x, val_y):
    import torch

    torch.manual_seed(context["config"]["seed"])
    torch.cuda.manual_seed_all(context["config"]["seed"])
    device = "cuda"
    head = torch.nn.Linear(train_x.shape[1], len(context["classes"])).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=context["config"]["linear_learning_rate"],
        weight_decay=context["config"]["linear_weight_decay"],
    )
    criterion = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(context["config"]["seed"])
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=context["config"]["linear_batch_size"],
        shuffle=True,
        generator=generator,
    )
    val_tensor = torch.from_numpy(val_x).to(device)
    best, history = None, []
    for epoch in range(1, context["config"]["epochs1"] + 1):
        head.train()
        total_loss, total = 0.0, 0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            total += len(targets)
        head.eval()
        with torch.inference_mode():
            probabilities = torch.softmax(head(val_tensor), dim=1).cpu().numpy()
        metrics = _metrics(val_y, probabilities)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
        }
        history.append(row)
        key = (metrics["accuracy"], metrics["macro_f1"], -epoch)
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "epoch": epoch,
                "state": {name: value.detach().cpu() for name, value in head.state_dict().items()},
                "probabilities": probabilities,
                "metrics": metrics,
            }
        print(
            f"Linear epoch {epoch:02d}/50 - loss {row['train_loss']:.4f} - "
            f"val_acc {row['val_accuracy']:.4f} - val_macro_f1 {row['val_macro_f1']:.4f}"
        )
    return best, history


def _save_predictions(path, paths, truth, probabilities):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "true_index", "pred_index", *[f"prob_C{i}" for i in range(5)]])
        for name, target, probs in zip(paths, truth, probabilities, strict=True):
            writer.writerow([name, int(target), int(probs.argmax()), *map(float, probs)])


def run(context, hf_token):
    validate(context)
    if not hf_token or not hf_token.startswith("hf_"):
        raise ValueError("Colab Secrets의 HF_TOKEN 읽기 토큰을 확인하세요.")
    output = context["output"]
    completed = output / "medsiglip_linear_completed.json"
    if completed.exists():
        record = engine.read_json(completed)
        if record["signature"] != context["signature"]:
            raise ValueError("완료 기록의 설정이 현재 실행과 다릅니다.")
        for relative, digest in record["artifact_hashes"].items():
            if engine.file_hash(output / relative) != digest:
                raise ValueError("완료 산출물이 변경됐습니다: " + relative)
        return record
    started = time.monotonic()
    model, processor = _load_encoder(hf_token, context["config"]["model_id"])
    train_x, train_y, _ = _extract_split(context, "train", model, processor)
    val_x, val_y, val_paths = _extract_split(context, "val", model, processor)
    del model, processor
    import torch

    torch.cuda.empty_cache()
    best, history = _linear_probe(context, train_x, train_y, val_x, val_y)
    head_path = output / "medsiglip_linear_head.pt"
    torch.save(
        {
            "state_dict": best["state"],
            "input_dim": int(train_x.shape[1]),
            "class_names": context["classes"],
            "model_id": MODEL_ID,
            "protocol": PROTOCOL,
        },
        head_path,
    )
    engine.write_json(output / "validation_metrics.json", best["metrics"])
    engine.write_json(output / "linear_history.json", history)
    _save_predictions(
        output / "validation_predictions.csv", val_paths, val_y, best["probabilities"]
    )
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    try:
        import transformers

        environment["transformers"] = transformers.__version__
    except ImportError:
        pass
    engine.write_json(output / "medsiglip_environment.json", environment)
    artifacts = [
        "medsiglip_linear_head.pt",
        "validation_metrics.json",
        "linear_history.json",
        "validation_predictions.csv",
        "medsiglip_environment.json",
    ]
    record = {
        "id": TRIAL_ID,
        "protocol": PROTOCOL,
        "signature": context["signature"],
        "model_id": MODEL_ID,
        "encoder_frozen": True,
        "head": "single Linear layer",
        "best_epoch": best["epoch"],
        "validation": best["metrics"],
        "elapsed_seconds": time.monotonic() - started,
        "test_evaluated": False,
        "artifact_hashes": {name: engine.file_hash(output / name) for name in artifacts},
    }
    engine.write_json(completed, record)
    return record


def _selected(record):
    new = record["validation"]
    return (
        new["macro_f1"] > BASELINE["validation_macro_f1"]
        or (
            new["macro_f1"] == BASELINE["validation_macro_f1"]
            and new["accuracy"] > BASELINE["validation_accuracy"]
        )
    )


def _plots(context, record):
    import matplotlib.pyplot as plt

    output = context["output"]
    history = engine.read_json(output / "linear_history.json")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    axes[0].set(title="MedSigLIP Linear Head Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Validation Accuracy")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="Validation Macro F1")
    axes[1].set(title="Validation Metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "medsiglip_training_curves.png", dpi=180)
    plt.close(figure)

    matrix = np.asarray(record["validation"]["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(5):
        for col in range(5):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    axis.set_xticks(range(5), context["classes"], rotation=35, ha="right")
    axis.set_yticks(range(5), context["classes"])
    axis.set(xlabel="Predicted", ylabel="True", title="MedSigLIP Validation Confusion Matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output / "medsiglip_validation_confusion_matrix.png", dpi=180)
    plt.close(figure)

    labels = ["PMG B0·256 v2", "MedSigLIP-448 Linear"]
    acc = [BASELINE["validation_accuracy"], record["validation"]["accuracy"]]
    f1 = [BASELINE["validation_macro_f1"], record["validation"]["macro_f1"]]
    x = np.arange(2)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - 0.18, acc, 0.36, label="Validation Accuracy")
    axis.bar(x + 0.18, f1, 0.36, label="Validation Macro F1")
    axis.set_xticks(x, labels)
    axis.set_ylim(max(0, min(acc + f1) - 0.08), 1)
    axis.set_title("Web Skin Validation Comparison")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    for index, value in enumerate(acc):
        axis.text(index - 0.18, value + 0.006, f"{value:.4f}", ha="center")
    for index, value in enumerate(f1):
        axis.text(index + 0.18, value + 0.006, f"{value:.4f}", ha="center")
    figure.tight_layout()
    figure.savefig(output / "medsiglip_vs_pmg_validation.png", dpi=180)
    plt.close(figure)


def _report_archive(context):
    output = context["output"]
    destination = output.parent / (output.name + "_reports_" + uuid.uuid4().hex[:8] + ".zip")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or "embedding_cache" in path.parts:
                continue
            archive.write(path, output.name + "/" + path.relative_to(output).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise OSError("보고서 ZIP 검사 실패")
    return destination


def summarize(context, record):
    validate(context)
    _plots(context, record)
    eligible = _selected(record)
    summary = {
        "protocol": PROTOCOL,
        "hypothesis": "Frozen medical image embeddings improve Web Skin linear separability.",
        "fixed": {
            "data_sha256": context["data_hash"],
            "class_names": context["classes"],
            "train_variant": "augmented",
            "train_count": EXPECTED_COUNTS["train"],
            "validation_count": EXPECTED_COUNTS["val"],
            "seed": 42,
        },
        "changed_variable": "Frozen MedSigLIP-448 image encoder plus a single Linear head",
        "necessary_accompanying_change": (
            "Input preprocessing follows the official MedSigLIP processor at 448 pixels."
        ),
        "baseline": BASELINE,
        "medsiglip_validation": record["validation"],
        "selection_rule": "Higher Validation Macro F1; Accuracy breaks an exact Macro F1 tie.",
        "eligible_for_separate_final_test": eligible,
        "current_v2_replaced": False,
        "test_evaluated": False,
        "model_card": MODEL_CARD_URL,
        "official_guide": OFFICIAL_GUIDE_URL,
    }
    engine.write_json(context["output"] / "medsiglip_validation_summary.json", summary)
    archive = _report_archive(context)
    print("Validation 비교 완료. Test는 실행하지 않았습니다.")
    print("결과 ZIP:", archive)
    return summary, archive
