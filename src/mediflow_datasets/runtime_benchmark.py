"""Read-only Colab benchmark of saved MediFlow classifier checkpoints."""

from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import platform
import statistics
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

COHORTS = ("skin", "web_skin", "hair5", "hair6")
EXPECTED_COUNTS = {"skin": 4, "web_skin": 12, "hair5": 14, "hair6": 4}
DIRECT_RUNS = (
    ("skin", "comparison_20260909_075056_72d865bf"),
    ("web_skin", "suite_20260908_014452_72768a42"),
    ("web_skin", "web_skin_paper_suite_20260922_124758_717b465e"),
    ("web_skin", "web_skin_pmg_b1_384_20260922_144631_ac7e5c8b"),
    ("hair", "supcon_compare_20260917_120447_38d75491"),
    ("hair", "paper_screen_20260917_141641_874f03f1"),
    ("hair", "sam_screen_20260921_145602_a5e8c406"),
    ("hair", "hair_six_class_four_20260928_011824_ad21e8ae"),
)
MEDSIGLIP_RUNS = (
    ("web_skin", "web_skin_medsiglip_linear_20260924_072442_f7176e07"),
    ("hair", "hair_medsiglip_linear_20260924_081708_9479fbb4"),
)
HAIR_CLEAN_RUNS = (
    "clean_two_stage_20260907_061005",
    "clean_256_two_stage_20260907_064214",
    "clean_256_label_smoothing_005_20260907_103757",
    "clean_256_focal_gamma_15_20260907_105525",
    "clean_256_efficientnetb1_ls005_20260907_111711",
)
LEGACY_ARCHIVES = (
    ("skin", "skin_dataset_results", 10),
    ("web_skin", "web_skin_dataset_results", 5),
    ("hair5", "hair_dataset_results", 5),
)
# Existing six checkpoints verified in results/EVALUATION_RECORD_AUDIT_20260927.json.
LEGACY_MODEL_SHA256 = {
    ("skin", "original"): "1b146a6a53742226ec677865aa71bd842a28c9db9c3174b352dd8bc87270770b",
    ("skin", "augmented"): "6d0c3d60fceebcd9299e6d79eef3266c4521906e92058b28f1841ec3535942bd",
    ("web_skin", "original"): "11c5a160ddeb406c4e80802e38a372e90a0222c1dddf208e270676f795f177ae",
    ("web_skin", "augmented"): "3b05c59500c272600438026b758842728b21d2ad2ce376cb435f436d4167310f",
    ("hair5", "original"): "d1695bf35701dee5b5a7b4623d105009053afc800d3900e265de4607550e5771",
    ("hair5", "augmented"): "324107bb1773e2f288b88985ebd81d07d1a8527a7e0963b3b964602d89042e68",
}


def cohort_for(source: Path, record: dict) -> str:
    """Use metadata first so five- and six-class Hair never mix."""
    parts = source.parts
    if "web_skin" in parts:
        return "web_skin"
    if "skin" in parts:
        return "skin"
    if "hair" not in parts:
        raise ValueError(f"알 수 없는 도메인: {source}")
    spec = record.get("spec", {})
    count = spec.get("class_count") or len(spec.get("class_names", []))
    if not count:
        count = len(record.get("validation", {}).get("class_f1", []))
    if count not in (5, 6):
        raise ValueError(f"Hair 클래스 수 확인 실패: {source}")
    return f"hair{count}"


def _legacy_rows(project_root: Path, cache_root: Path) -> list[dict]:
    """Unpack only the two selected legacy models per source ZIP into local cache."""

    def matches_variant(name: str, variant: str) -> bool:
        path = Path(name.replace("\\", "/").replace("::", "/").lower())
        return variant in path.parts or variant in path.stem

    def assets(source: zipfile.ZipFile, prefix: str = "", depth: int = 0):
        for item in source.infolist():
            if item.is_dir():
                continue
            name = prefix + item.filename
            lower = item.filename.lower()
            if lower.endswith(".zip") and depth < 2:
                with zipfile.ZipFile(io.BytesIO(source.read(item))) as nested:
                    yield from assets(nested, name + "::", depth + 1)
            elif lower.endswith(".keras") or Path(lower).name in (
                "results.json",
                "training_config.json",
            ):
                yield name, source.read(item)

    rows = []
    for cohort, archive_name, class_count in LEGACY_ARCHIVES:
        archive = project_root / "1_results" / archive_name
        if not archive.is_file():
            archive = archive.with_suffix(".zip")
        try:
            valid_archive = zipfile.is_zipfile(archive)
        except OSError:
            valid_archive = False
        found = []
        if valid_archive:
            with zipfile.ZipFile(archive) as source:
                found = list(assets(source))
        for variant in ("original", "augmented"):
            model = cache_root / cohort / f"{variant}.keras"
            row = {
                "cohort": cohort,
                "experiment": f"initial_b0_224_{variant}",
                "source_record": f"{archive}::{variant}",
                "model_path": str(model),
                "model_kind": "keras",
                "input_size": 224,
                "validation_accuracy": None,
                "validation_macro_f1": None,
                "training_seconds": None,
                "class_count": class_count,
                "crop_threshold": None,
                "status": "archive_missing" if not valid_archive else "archive_model_missing",
            }
            if valid_archive:
                candidates = [
                    (name, content, hashlib.sha256(content).hexdigest())
                    for name, content in found
                    if name.lower().endswith(".keras")
                ]
                expected = LEGACY_MODEL_SHA256[(cohort, variant)]
                matches = sorted(
                    (item for item in candidates if item[2] == expected), key=lambda item: item[0]
                )
                if matches:
                    selected_name, content, digest = matches[0]
                    model.parent.mkdir(parents=True, exist_ok=True)
                    temporary = model.with_suffix(".part")
                    temporary.write_bytes(content)
                    temporary.replace(model)
                    row["status"] = "ready"
                    row["source_record"] += f"::{selected_name}"
                    row["model_sha256"] = digest
                    row["archive_identical_copies"] = len(matches)
                else:
                    row["status"] = "archive_model_hash_missing"
                    row["error"] = (
                        f"Expected {expected}; available: "
                        + "; ".join(f"{name}={digest}" for name, _, digest in candidates[:12])
                    )[:1500]
                reports = []
                for name, content in found:
                    if Path(name).name.lower() not in ("results.json", "training_config.json"):
                        continue
                    report = json.loads(content.decode("utf-8-sig"))
                    if (
                        matches_variant(name, variant)
                        or str(report.get("dataset", "")).lower() == variant
                        or str(report.get("model", "")).lower() == variant
                    ):
                        reports.append(report)
                if len(reports) == 1:
                    report = reports[0]
                    row["validation_accuracy"] = report.get("best_val_accuracy")
                    minutes = report.get("training_time_minutes")
                    if minutes is not None:
                        row["training_seconds"] = float(minutes) * 60
            rows.append(row)
    return rows


def _hair_clean_rows(project_root: Path) -> list[dict]:
    rows = []
    for name in HAIR_CLEAN_RUNS:
        folder = project_root / "2_results" / "hair" / name
        model = folder / "best_model.keras"
        config_path = folder / "training_config.json"
        comparison_path = folder / "stage_comparison.csv"
        config = (
            json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        )
        accuracy = None
        if comparison_path.is_file():
            with comparison_path.open(encoding="utf-8-sig", newline="") as stream:
                for record in csv.DictReader(stream):
                    if record.get("stage") == "stage2_partial_finetuning":
                        accuracy = float(record["best_val_accuracy"])
        size = config.get("image_size", [None])[0]
        rows.append(
            {
                "cohort": "hair5",
                "experiment": name,
                "source_record": str(config_path),
                "model_path": str(model),
                "model_kind": "keras",
                "input_size": size,
                "validation_accuracy": accuracy,
                "validation_macro_f1": None,
                "training_seconds": None,
                "class_count": 5,
                "crop_threshold": None,
                "status": "ready" if model.is_file() else "checkpoint_missing",
            }
        )
    return rows


def discover(project_root: Path, cache_root: Path | None = None) -> list[dict]:
    """Return the fixed 34 primary training conditions, keeping extensions separate."""
    results = project_root / "2_results"
    rows = []
    for domain, run_name in DIRECT_RUNS:
        for path in sorted((results / domain / run_name).rglob("completed.json")):
            if path.parent.name == "b1_256_ls005_extend5":
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if not all(key in record for key in ("attempt", "selected_model", "validation")):
                    continue
                cohort = cohort_for(path, record)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            model = path.parent / record["attempt"] / record["selected_model"]
            spec = record.get("spec", {})
            rows.append(
                {
                    "cohort": cohort,
                    "experiment": record.get("id", path.parent.name),
                    "source_record": str(path),
                    "model_path": str(model),
                    "model_kind": "keras",
                    "input_size": spec.get("input_size") or spec.get("size"),
                    "validation_accuracy": record["validation"].get("accuracy"),
                    "validation_macro_f1": record["validation"].get("macro_f1"),
                    "training_seconds": record.get("training_seconds"),
                    "class_count": spec.get("class_count")
                    or len(spec.get("class_names", []))
                    or len(record["validation"].get("class_f1", [])),
                    "crop_threshold": spec.get("crop_threshold"),
                    "status": "ready" if model.is_file() else "checkpoint_missing",
                }
            )
    for domain, run_name in MEDSIGLIP_RUNS:
        path = results / domain / run_name / "medsiglip_linear_completed.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        model = path.parent / "medsiglip_linear_head.pt"
        rows.append(
            {
                "cohort": domain if domain == "web_skin" else "hair5",
                "experiment": "medsiglip_448_frozen_linear_seed_42",
                "source_record": str(path),
                "model_path": str(model),
                "model_kind": "medsiglip",
                "input_size": 448,
                "validation_accuracy": record.get("validation", {}).get("accuracy"),
                "validation_macro_f1": record.get("validation", {}).get("macro_f1"),
                "training_seconds": record.get("elapsed_seconds"),
                "class_count": len(record.get("validation", {}).get("class_f1", [])),
                "crop_threshold": None,
                "status": "ready" if model.is_file() else "checkpoint_missing",
            }
        )
    rows.extend(_hair_clean_rows(project_root))
    rows.extend(_legacy_rows(project_root, cache_root or Path("/content/mediflow_benchmark_cache")))
    return rows


def _keras_custom_objects():
    import keras

    @keras.saving.register_keras_serializable(package="MediFlow")
    class BilinearAttentionPooling(keras.layers.Layer):
        def call(self, inputs):
            features, attention = inputs
            pooled = keras.ops.einsum("bhwc,bhwm->bmc", features, attention)
            normalizer = keras.ops.sum(attention, axis=(1, 2))
            pooled = pooled / (keras.ops.expand_dims(normalizer, -1) + 1e-6)
            pooled = keras.ops.reshape(pooled, (keras.ops.shape(pooled)[0], -1))
            pooled = keras.ops.sign(pooled) * keras.ops.sqrt(keras.ops.abs(pooled) + 1e-8)
            norm = keras.ops.sqrt(
                keras.ops.sum(keras.ops.square(pooled), axis=-1, keepdims=True) + 1e-8
            )
            return pooled / norm

        def compute_output_shape(self, input_shape):
            feature_shape, attention_shape = input_shape
            return (feature_shape[0], feature_shape[-1] * attention_shape[-1])

    @keras.saving.register_keras_serializable(package="MediFlow")
    class MixStyle(keras.layers.Layer):
        def __init__(self, probability=0.5, alpha=0.1, **kwargs):
            super().__init__(**kwargs)
            self.probability = probability
            self.alpha = alpha

        def call(self, inputs, training=None):
            if training:
                raise ValueError("벤치마크에서 MixStyle 학습 모드를 사용할 수 없습니다.")
            return inputs

        def get_config(self):
            return {**super().get_config(), "probability": self.probability, "alpha": self.alpha}

    return {"BilinearAttentionPooling": BilinearAttentionPooling, "MixStyle": MixStyle}


def _attention_crop(images, maps, size, threshold):
    import tensorflow as tf

    def crop_one(item):
        image, attention = item
        score = tf.reduce_mean(attention, axis=-1, keepdims=True)
        score = tf.image.resize(score, (size, size), method="bilinear")
        score = score / (tf.reduce_max(score) + 1e-6)
        coordinates = tf.cast(tf.where(score[..., 0] >= threshold), tf.int32)

        def crop_region():
            top_left = tf.reduce_min(coordinates, axis=0)
            bottom_right = tf.reduce_max(coordinates, axis=0) + 1
            crop = tf.image.crop_to_bounding_box(
                image,
                top_left[0],
                top_left[1],
                tf.maximum(bottom_right[0] - top_left[0], 1),
                tf.maximum(bottom_right[1] - top_left[1], 1),
            )
            return tf.image.resize(crop, (size, size), method="bilinear")

        return tf.cond(tf.shape(coordinates)[0] > 0, crop_region, lambda: image)

    return tf.map_fn(
        crop_one,
        (images, maps),
        fn_output_signature=tf.TensorSpec((size, size, 3), images.dtype),
    )


def _load_keras(row):
    import keras
    import tensorflow as tf

    keras.backend.clear_session()
    model = keras.models.load_model(
        row["model_path"], compile=False, custom_objects=_keras_custom_objects()
    )
    size = row["input_size"] or model.input_shape[1]
    if isinstance(size, (tuple, list)):
        size = size[0]
    size = int(size)
    if model.input_shape[1] != size or model.input_shape[2] != size:
        raise ValueError(f"기록과 모델 입력 크기가 다릅니다: {row['experiment']}")
    if not isinstance(model.output_shape, list) and model.output_shape[-1] != row["class_count"]:
        raise ValueError(f"클래스 수가 다릅니다: {row['experiment']}")

    if "wsdan" in row["experiment"].lower():
        if row["crop_threshold"] is None:
            raise ValueError("WS-DAN crop threshold 기록이 없습니다.")
        probe = keras.Model(model.input, [model.output, model.get_layer("attention_maps").output])

        def infer(x):
            raw, maps = probe(x, training=False)
            crop = _attention_crop(x, maps, size, row["crop_threshold"])
            return (raw + model(crop, training=False)) / 2.0

    elif "pmg" in row["experiment"].lower():

        def infer(x):
            outputs = model(x, training=False)
            return tf.nn.softmax(tf.add_n(outputs), axis=-1)

    else:

        def infer(x):
            return model(x, training=False)

    return model, infer, size


def _load_medsiglip(row, token):
    import torch
    from transformers import AutoModel, AutoProcessor

    if not token:
        raise ValueError("Colab Secrets의 HF_TOKEN이 필요합니다.")
    if not torch.cuda.is_available():
        raise RuntimeError("MedSigLIP 측정에는 GPU가 필요합니다.")
    checkpoint = torch.load(row["model_path"], map_location="cpu", weights_only=True)
    model_id = checkpoint["model_id"]
    if model_id != "google/medsiglip-448":
        raise ValueError(f"예상과 다른 MedSigLIP 기본 모델: {model_id}")
    classes = checkpoint["class_names"]
    if len(classes) != row["class_count"]:
        raise ValueError("MedSigLIP 분류층 클래스 수 불일치")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    encoder = (
        AutoModel.from_pretrained(
            model_id, token=token, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        .eval()
        .to("cuda")
    )
    head = torch.nn.Linear(checkpoint["input_dim"], len(classes)).to("cuda").eval()
    head.load_state_dict(checkpoint["state_dict"])

    def infer(pixel_values):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = encoder.get_image_features(pixel_values=pixel_values)
            features = torch.nn.functional.normalize(features.float(), dim=-1)
            return torch.softmax(head(features), dim=-1)

    return (encoder, head, processor), infer


def _latency(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    return (
        statistics.median(ordered),
        ordered[min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)],
        statistics.mean(ordered),
    )


def benchmark_one(row: dict, image: Image.Image, token: str | None, warmup=10, repeats=30):
    """Batch-one model-only and image-to-prediction latency on one runtime."""
    import tensorflow as tf

    output = dict(row)
    if row["status"] != "ready":
        return output
    started = time.perf_counter()
    if row["model_kind"] == "medsiglip":
        import torch

        loaded, infer = _load_medsiglip(row, token)
        encoder, head, processor = loaded
        output["parameter_count"] = sum(p.numel() for p in encoder.parameters()) + sum(
            p.numel() for p in head.parameters()
        )
        output["parameter_bytes_loaded"] = sum(
            p.numel() * p.element_size() for p in encoder.parameters()
        ) + sum(p.numel() * p.element_size() for p in head.parameters())

        def prepare():
            return processor(images=[image], return_tensors="pt")["pixel_values"].to(
                "cuda", dtype=torch.float16
            )

        def sync(result):
            _ = result.cpu().numpy()
            torch.cuda.synchronize()

    else:
        model, infer, size = _load_keras(row)
        output["parameter_count"] = model.count_params()
        output["parameter_bytes_loaded"] = sum(
            int(np.prod(weight.shape)) * np.dtype(weight.dtype).itemsize for weight in model.weights
        )

        def prepare():
            rgb = image.convert("RGB").resize((size, size))
            return tf.convert_to_tensor(np.asarray(rgb, dtype=np.float32)[None, ...])

        def sync(result):
            values = result.numpy()
            if values.shape[-1] != row["class_count"]:
                raise ValueError("추론 출력 클래스 수 불일치")

    output["load_seconds"] = time.perf_counter() - started
    value = prepare()
    for _ in range(warmup):
        sync(infer(value))
    model_times, end_to_end_times = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        sync(infer(value))
        model_times.append(1000 * (time.perf_counter() - start))
        start = time.perf_counter()
        prepared = prepare()
        sync(infer(prepared))
        end_to_end_times.append(1000 * (time.perf_counter() - start))
    for prefix, samples in (("model", model_times), ("end_to_end", end_to_end_times)):
        output[f"{prefix}_p50_ms"], output[f"{prefix}_p95_ms"], output[f"{prefix}_mean_ms"] = (
            _latency(samples)
        )
    output["checkpoint_bytes"] = Path(row["model_path"]).stat().st_size
    output["status"] = "measured"
    if row["model_kind"] == "medsiglip":
        del loaded
        torch.cuda.empty_cache()
    gc.collect()
    tf.keras.backend.clear_session()
    return output


def run(
    project_root: Path,
    output_root: Path,
    token: str | None,
    warmup=10,
    repeats=30,
    planned_rows: list[dict] | None = None,
):
    """Write new reports only; source checkpoints and records stay untouched."""
    import tensorflow as tf

    if warmup < 1 or repeats < 5:
        raise ValueError("warmup >= 1, repeats >= 5가 필요합니다.")
    devices = [device.name for device in tf.config.list_physical_devices("GPU")]
    if not devices:
        raise RuntimeError("비교 측정에는 Colab GPU 런타임이 필요합니다.")
    rows = planned_rows if planned_rows is not None else discover(project_root)
    counts = {cohort: sum(row["cohort"] == cohort for row in rows) for cohort in COHORTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"34개 실험 목록이 불완전합니다: {counts}; 예상 {EXPECTED_COUNTS}")
    missing = [row["experiment"] for row in rows if row["status"] != "ready"]
    if missing:
        raise FileNotFoundError("모델 파일 사전 점검 실패: " + ", ".join(missing))
    previous = {}
    environment_path = output_root / "environment.json"
    if output_root.exists():
        if not environment_path.is_file():
            raise ValueError("재개 폴더에 environment.json이 없습니다.")
        old = json.loads(environment_path.read_text(encoding="utf-8"))
        if (old["warmup"], old["repeats_per_metric"], old["gpu"]) != (warmup, repeats, devices):
            raise ValueError("재개 폴더의 측정 설정 또는 GPU가 현재와 다릅니다.")
        previous_path = output_root / "benchmark.json"
        if previous_path.is_file():
            previous = {
                row["source_record"]: row
                for row in json.loads(previous_path.read_text(encoding="utf-8"))
            }
    else:
        output_root.mkdir(parents=True)
    environment = {
        "measured_at": datetime.now().isoformat(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "gpu": devices,
        "warmup": warmup,
        "repeats_per_metric": repeats,
        "batch_size": 1,
        "sample": "constant RGB 512x512; no dataset or labels used",
        "training_time_source": "original run metadata, not remeasured",
        "latency_definition": "model and image-to-prediction, synchronous, excluding model load",
    }
    if not environment_path.exists():
        environment_path.write_text(
            json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    image = Image.fromarray(np.full((512, 512, 3), 127, dtype=np.uint8), "RGB")
    results = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {row['cohort']} / {row['experiment']}")
        cached = previous.get(row["source_record"])
        if (
            cached
            and cached.get("model_path") == row["model_path"]
            and cached.get("status") == "measured"
        ):
            measured = cached
        else:
            try:
                measured = benchmark_one(row, image, token, warmup, repeats)
            except Exception as exc:
                measured = {**row, "status": "failed", "error": str(exc)[:500]}
        results.append(measured)
        _write_results(output_root, results)
        gc.collect()
        tf.keras.backend.clear_session()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    render_figures(results, output_root)
    return results


def _write_results(folder: Path, rows: list[dict]):
    (folder / "benchmark.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    keys = sorted({key for row in rows for key in row})
    with (folder / "benchmark.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render_figures(rows: list[dict], output: Path):
    import matplotlib.pyplot as plt

    for cohort in COHORTS:
        data = [row for row in rows if row["cohort"] == cohort and row["status"] == "measured"]
        if not data:
            continue
        labels = [row["experiment"].replace("_seed_42", "") for row in data]
        positions = np.arange(len(data))
        fig, axes = plt.subplots(2, 2, figsize=(max(13, len(data) * 1.5), 10))
        metrics = (
            ("validation_accuracy", "Validation accuracy", "#2563eb"),
            ("end_to_end_p50_ms", "End-to-end p50 (ms)", "#f97316"),
            ("end_to_end_p95_ms", "End-to-end p95 (ms)", "#ea580c"),
            ("training_seconds", "Recorded training time (s)", "#16a34a"),
        )
        for axis, (key, title, color) in zip(axes.flat, metrics, strict=True):
            values = [row.get(key) if row.get(key) is not None else np.nan for row in data]
            axis.bar(positions, values, color=color)
            axis.set_title(title)
            axis.set_xticks(positions, labels, rotation=35, ha="right")
            axis.grid(axis="y", alpha=0.2)
        fig.suptitle(f"MediFlow {cohort}: saved validation + measured runtime", fontsize=16)
        fig.tight_layout()
        fig.savefig(output / f"{cohort}_dashboard.png", dpi=180)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(10, 7))
        for row in data:
            if row.get("validation_accuracy") is None:
                continue
            axis.scatter(row["end_to_end_p50_ms"], row["validation_accuracy"], s=90)
            axis.annotate(
                row["experiment"],
                (row["end_to_end_p50_ms"], row["validation_accuracy"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("End-to-end p50 (ms, batch 1)")
        axis.set_ylabel("Saved validation accuracy")
        axis.set_title(f"{cohort}: accuracy / latency trade-off")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f"{cohort}_tradeoff.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(data) * 0.55)))
        heatmap_keys = (
            "validation_accuracy",
            "validation_macro_f1",
            "model_p50_ms",
            "end_to_end_p50_ms",
            "end_to_end_p95_ms",
            "training_seconds",
        )
        matrix = np.asarray(
            [
                [row.get(key) if row.get(key) is not None else np.nan for key in heatmap_keys]
                for row in data
            ],
            dtype=float,
        )
        normalized = np.full_like(matrix, np.nan)
        for column in range(matrix.shape[1]):
            values = matrix[:, column]
            finite = np.isfinite(values)
            if not finite.any():
                continue
            low, high = np.nanmin(values), np.nanmax(values)
            normalized[finite, column] = (
                (values[finite] - low) / (high - low) if high > low else 0.5
            )
        axes[0].imshow(normalized, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        axes[0].set_xticks(range(len(heatmap_keys)), heatmap_keys, rotation=45, ha="right")
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].set_title("Column-normalized metrics (color is not a score)")
        axes[1].boxplot(
            [
                [row["model_p50_ms"] for row in data],
                [row["end_to_end_p50_ms"] for row in data],
                [row["end_to_end_p95_ms"] for row in data],
            ],
            tick_labels=["model p50", "total p50", "total p95"],
        )
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("Summary distribution across models")
        fig.suptitle(f"{cohort}: metric overview")
        fig.tight_layout()
        fig.savefig(output / f"{cohort}_metrics_overview.png", dpi=180)
        plt.close(fig)
