"""Four validation-only paper experiments for the Web Skin classifier.

Each method is trained independently.  Historical B0/256/CE validation metrics
are a read-only reference and Test is deliberately never loaded.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import uuid
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from mediflow_datasets import common_engine as engine
from mediflow_datasets import common_workflow as flow
from mediflow_datasets import web_skin_wsdan as wsdan

PROTOCOL = "web_skin_four_papers_v1"
EXPERIMENTS = [
    wsdan.TRIAL_ID,
    "pmg_b0_256_ce_seed_42",
    "mixstyle_b0_256_ce_seed_42",
    "derm_foundation_mlp_seed_42",
]
PAPERS = {
    "wsdan": "https://arxiv.org/abs/1901.09891",
    "pmg": "https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650154.pdf",
    "mixstyle": "https://openreview.net/forum?id=6xHJ37MVxxp",
    "derm_foundation": (
        "https://developers.google.com/health-ai-developer-foundations/"
        "derm-foundation/model-card"
    ),
}


def validate(context):
    config = context["config"]
    if config["domain"] != "web_skin" or config["mode"] != "web_skin_paper_suite":
        raise ValueError("이 코드는 Web Skin 4개 논문 실험 전용입니다.")
    if config["train_variant"] != "augmented":
        raise ValueError("기존 기준선과 같은 augmented Train을 사용해야 합니다.")
    if config["seed"] != 42 or config["seeds"] != [42]:
        raise ValueError("논문 방법 선별 seed는 42로 고정합니다.")
    if config.get("experiments") != EXPERIMENTS:
        raise ValueError("WS-DAN, PMG, MixStyle, Derm Foundation 네 조건이 필요합니다.")
    if context["classes"] != wsdan.EXPECTED_CLASSES:
        raise ValueError("Web Skin 클래스 순서가 기존 모델 계약과 다릅니다.")
    if context["data_hash"] != wsdan.BASELINE["data_sha256"]:
        raise ValueError("저장된 기준선과 데이터 SHA-256이 다릅니다.")
    if config.get("pmg_jigsaw_grids") != [8, 4, 2]:
        raise ValueError("PMG jigsaw 순서는 원 논문의 8, 4, 2로 고정합니다.")
    if config.get("derm_model_id") != "google/derm-foundation":
        raise ValueError("Derm Foundation 공식 Hugging Face 모델 ID가 아닙니다.")
    if not isinstance(config.get("derm_embedding_batch_size"), int) or config[
        "derm_embedding_batch_size"
    ] <= 0:
        raise ValueError("DERM_EMBEDDING_BATCH_SIZE는 양의 정수여야 합니다.")
    for key in ("mixstyle_alpha", "mixstyle_probability"):
        value = config.get(key)
        if not isinstance(value, (float, int)) or not 0 < value <= 1:
            raise ValueError(key + "는 0보다 크고 1 이하여야 합니다.")


def _trial_signature(context, spec):
    return hashlib.sha256(
        (context["signature"] + json.dumps(spec, sort_keys=True)).encode()
    ).hexdigest()


def _attempt(context, trial_id):
    directory = (
        Path(context["output"]) / trial_id / ("attempt_" + uuid.uuid4().hex[:12])
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _cache_or_none(context, trial_id, signature):
    return engine.cached_record(context["output"], trial_id, signature)


def _run_wsdan(context):
    return wsdan.run(context)


def _jigsaw(images, grid):
    shape = tf.shape(images)
    batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    patch_height, patch_width = height // grid, width // grid
    patches = tf.reshape(
        images,
        (batch, grid, patch_height, grid, patch_width, channels),
    )
    patches = tf.transpose(patches, (0, 1, 3, 2, 4, 5))
    patches = tf.reshape(
        patches, (batch, grid * grid, patch_height, patch_width, channels)
    )
    patches = tf.gather(patches, tf.random.shuffle(tf.range(grid * grid)), axis=1)
    patches = tf.reshape(
        patches, (batch, grid, grid, patch_height, patch_width, channels)
    )
    patches = tf.transpose(patches, (0, 1, 3, 2, 4, 5))
    return tf.reshape(patches, (batch, height, width, channels))


def build_pmg_model(class_count, size=256, weights="imagenet"):
    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(size, size, 3)
    )
    feature_model = keras.Model(
        backbone.input,
        [
            backbone.get_layer("block3a_expand_activation").output,
            backbone.get_layer("block5a_expand_activation").output,
            backbone.get_layer("top_activation").output,
        ],
        name="pmg_backbone",
    )
    feature_model.trainable = False
    inputs = keras.Input((size, size, 3), name="image")
    features = feature_model(inputs, training=False)
    embeddings, logits = [], []
    for index, feature in enumerate(features, 1):
        value = keras.layers.Conv2D(
            256, 1, activation="relu", name=f"pmg_branch{index}_reduce"
        )(feature)
        value = keras.layers.Conv2D(
            512, 3, padding="same", activation="relu", name=f"pmg_branch{index}_conv"
        )(value)
        value = keras.layers.GlobalAveragePooling2D(
            name=f"pmg_branch{index}_pool"
        )(value)
        value = keras.layers.Dense(
            256, activation="elu", name=f"pmg_branch{index}_embedding"
        )(value)
        embeddings.append(value)
        logits.append(
            keras.layers.Dense(class_count, name=f"pmg_branch{index}_logits")(value)
        )
    combined = keras.layers.Concatenate(name="pmg_concat")(embeddings)
    combined = keras.layers.Dense(512, activation="elu", name="pmg_fusion")(combined)
    logits.append(keras.layers.Dense(class_count, name="pmg_concat_logits")(combined))
    return keras.Model(inputs, logits, name="web_skin_pmg_b0")


def _configure_pmg_partial(model):
    backbone = model.get_layer("pmg_backbone")
    backbone.trainable = True
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= len(backbone.layers) - 30 and not isinstance(
            layer, keras.layers.BatchNormalization
        )
    return [layer.name for layer in backbone.layers if layer.trainable]


def _pmg_predictor(model):
    total = keras.layers.Add(name="pmg_logit_sum")(model.outputs)
    probabilities = keras.layers.Activation("softmax", name="predictions")(total)
    return keras.Model(model.input, probabilities, name="web_skin_pmg_inference")


def _fit_pmg_stage(model, train, val, directory, name, epochs, learning_rate):
    optimizer = keras.optimizers.Adam(learning_rate)
    optimizer.build(model.trainable_variables)
    loss_function = keras.losses.CategoricalCrossentropy(from_logits=True)
    history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}
    best_score, best = -1.0, directory / f"{name}_best.keras"

    @tf.function
    def branch_step(images, labels, branch, grid):
        with tf.GradientTape() as tape:
            outputs = model(_jigsaw(images, grid), training=True)
            loss = loss_function(labels, outputs[branch])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            [
                (g, v)
                for g, v in zip(gradients, model.trainable_variables, strict=True)
                if g is not None
            ]
        )
        return loss

    @tf.function
    def fusion_step(images, labels):
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            loss = 2.0 * loss_function(labels, outputs[3])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            [
                (g, v)
                for g, v in zip(gradients, model.trainable_variables, strict=True)
                if g is not None
            ]
        )
        return loss, tf.nn.softmax(tf.add_n(outputs), axis=-1)

    predictor = _pmg_predictor(model)
    for epoch in range(epochs):
        train_loss = keras.metrics.Mean()
        train_accuracy = keras.metrics.CategoricalAccuracy()
        for images, labels in train:
            losses = [
                branch_step(images, labels, 0, 8),
                branch_step(images, labels, 1, 4),
                branch_step(images, labels, 2, 2),
            ]
            fusion_loss, predictions = fusion_step(images, labels)
            train_loss.update_state(tf.add_n(losses) + fusion_loss)
            train_accuracy.update_state(labels, predictions)
        val_loss = keras.metrics.Mean()
        val_accuracy = keras.metrics.CategoricalAccuracy()
        for images, labels in val:
            outputs = model(images, training=False)
            val_loss.update_state(loss_function(labels, outputs[3]))
            val_accuracy.update_state(labels, predictor(images, training=False))
        values = {
            "accuracy": float(train_accuracy.result()),
            "loss": float(train_loss.result()),
            "val_accuracy": float(val_accuracy.result()),
            "val_loss": float(val_loss.result()),
        }
        for key, value in values.items():
            history[key].append(value)
        engine.write_json(directory / f"{name}_history.json", history)
        print(name, "Epoch", epoch + 1, "/", epochs, values)
        if values["val_accuracy"] > best_score:
            best_score = values["val_accuracy"]
            model.save(best)
    model.save(directory / f"{name}_last.keras")
    return history, best


def _run_pmg(context):
    config, trial_id = context["config"], EXPERIMENTS[1]
    spec = {
        "id": trial_id,
        "method": "efficientnet_b0_pmg_adaptation",
        "paper": PAPERS["pmg"],
        "jigsaw_grids": [8, 4, 2],
        "input_size": 256,
        "seed": 42,
        "test_evaluated": False,
    }
    signature = _trial_signature(context, spec)
    cached = _cache_or_none(context, trial_id, signature)
    if cached:
        return cached
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    directory = _attempt(context, trial_id)
    engine.write_json(directory / "spec.json", spec)
    train, _ = flow.factory(context, "augmented", seed=42)("train", 256, True)
    val, paths = flow.factory(context, "augmented", seed=42)("val", 256, False)
    started = time.monotonic()
    model = build_pmg_model(len(context["classes"]))
    h1, best1 = _fit_pmg_stage(
        model, train, val, directory, "stage1", config["epochs1"], 1e-4
    )
    del model
    keras.backend.clear_session()
    model = keras.models.load_model(best1, compile=False)
    trainable = _configure_pmg_partial(model)
    h2, best2 = _fit_pmg_stage(
        model, train, val, directory, "stage2", config["epochs2"], 1e-5
    )
    selected_stage = (
        "stage2"
        if engine.checkpoint_choice(max(h1["val_accuracy"]), max(h2["val_accuracy"]))
        else "stage1"
    )
    selected = best2 if selected_stage == "stage2" else best1
    del model
    keras.backend.clear_session()
    model = keras.models.load_model(selected, compile=False)
    predictor = _pmg_predictor(model)
    metrics = engine.evaluate_to_files(predictor, val, paths, directory, "validation")
    record = {
        "id": trial_id,
        "spec": spec,
        "signature": signature,
        "attempt": directory.name,
        "selected_model": selected.name,
        "selected_stage": selected_stage,
        "validation": metrics,
        "stage1_best_val": max(h1["val_accuracy"]),
        "stage2_best_val": max(h2["val_accuracy"]),
        "training_seconds": time.monotonic() - started,
        "parameters": model.count_params(),
        "model_bytes": selected.stat().st_size,
        "trainable_backbone_layers": trainable,
        "history": {key: h1[key] + h2[key] for key in h1},
        "stage_boundary": len(h1["accuracy"]),
        "inference": "sum of three branch logits and fused logits on one original image",
        "test_evaluated": False,
    }
    engine.finish_record(directory, record)
    return record


@keras.saving.register_keras_serializable(package="MediFlow")
class MixStyle(keras.layers.Layer):
    def __init__(self, probability=0.5, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.alpha = alpha

    def call(self, inputs, training=None):
        if training is not True:
            return inputs
        mean = tf.reduce_mean(inputs, axis=(1, 2), keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=(1, 2), keepdims=True)
        deviation = tf.sqrt(variance + 1e-6)
        normalized = (inputs - tf.stop_gradient(mean)) / tf.stop_gradient(deviation)
        permutation = tf.random.shuffle(tf.range(tf.shape(inputs)[0]))
        mixed_mean = tf.gather(mean, permutation)
        mixed_deviation = tf.gather(deviation, permutation)
        first = tf.random.gamma((tf.shape(inputs)[0], 1, 1, 1), self.alpha)
        second = tf.random.gamma((tf.shape(inputs)[0], 1, 1, 1), self.alpha)
        weight = first / (first + second + 1e-6)
        target_mean = weight * mean + (1.0 - weight) * mixed_mean
        target_deviation = weight * deviation + (1.0 - weight) * mixed_deviation
        styled = normalized * tf.stop_gradient(target_deviation) + tf.stop_gradient(
            target_mean
        )
        return tf.cond(
            tf.random.uniform(()) < self.probability, lambda: styled, lambda: inputs
        )

    def get_config(self):
        return {
            **super().get_config(),
            "probability": self.probability,
            "alpha": self.alpha,
        }


def build_mixstyle_model(
    class_count, size=256, probability=0.5, alpha=0.1, weights="imagenet"
):
    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(size, size, 3)
    )
    split = backbone.get_layer("block2b_add").output
    early = keras.Model(backbone.input, split, name="mixstyle_early")
    late = keras.Model(split, backbone.output, name="mixstyle_late")
    early.trainable = False
    late.trainable = False
    inputs = keras.Input((size, size, 3), name="image")
    features = early(inputs, training=False)
    features = MixStyle(probability, alpha, name="mixstyle")(features)
    features = late(features, training=False)
    features = keras.layers.GlobalAveragePooling2D()(features)
    features = keras.layers.Dropout(0.3)(features)
    outputs = keras.layers.Dense(class_count, activation="softmax", name="predictions")(
        features
    )
    model = keras.Model(inputs, outputs, name="web_skin_mixstyle_b0")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def _configure_mixstyle_partial(model):
    late = model.get_layer("mixstyle_late")
    late.trainable = True
    for index, layer in enumerate(late.layers):
        layer.trainable = index >= len(late.layers) - 30 and not isinstance(
            layer, keras.layers.BatchNormalization
        )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return [layer.name for layer in late.layers if layer.trainable]


def _run_mixstyle(context):
    config, trial_id = context["config"], EXPERIMENTS[2]
    spec = {
        "id": trial_id,
        "method": "mixstyle_after_efficientnet_block2b",
        "paper": PAPERS["mixstyle"],
        "alpha": float(config["mixstyle_alpha"]),
        "probability": float(config["mixstyle_probability"]),
        "input_size": 256,
        "seed": 42,
        "test_evaluated": False,
    }
    signature = _trial_signature(context, spec)
    cached = _cache_or_none(context, trial_id, signature)
    if cached:
        return cached
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    directory = _attempt(context, trial_id)
    engine.write_json(directory / "spec.json", spec)
    train, _ = flow.factory(context, "augmented", seed=42)("train", 256, True)
    val, paths = flow.factory(context, "augmented", seed=42)("val", 256, False)
    started = time.monotonic()
    model = build_mixstyle_model(
        len(context["classes"]),
        probability=config["mixstyle_probability"],
        alpha=config["mixstyle_alpha"],
    )
    h1, best1 = engine.fit_stage(
        model, train, val, directory, "stage1", config["epochs1"]
    )
    del model
    keras.backend.clear_session()
    model = keras.models.load_model(best1, compile=False)
    trainable = _configure_mixstyle_partial(model)
    h2, best2 = engine.fit_stage(
        model, train, val, directory, "stage2", config["epochs2"]
    )
    selected_stage = (
        "stage2"
        if engine.checkpoint_choice(max(h1["val_accuracy"]), max(h2["val_accuracy"]))
        else "stage1"
    )
    selected = best2 if selected_stage == "stage2" else best1
    del model
    keras.backend.clear_session()
    model = keras.models.load_model(selected, compile=False)
    metrics = engine.evaluate_to_files(model, val, paths, directory, "validation")
    record = {
        "id": trial_id,
        "spec": spec,
        "signature": signature,
        "attempt": directory.name,
        "selected_model": selected.name,
        "selected_stage": selected_stage,
        "validation": metrics,
        "stage1_best_val": max(h1["val_accuracy"]),
        "stage2_best_val": max(h2["val_accuracy"]),
        "training_seconds": time.monotonic() - started,
        "parameters": model.count_params(),
        "model_bytes": selected.stat().st_size,
        "trainable_backbone_layers": trainable,
        "history": {key: h1[key] + h2[key] for key in h1},
        "stage_boundary": len(h1["accuracy"]),
        "inference": "MixStyle disabled automatically; one original image",
        "test_evaluated": False,
    }
    engine.finish_record(directory, record)
    return record


def _paths_and_labels(context, split):
    root = context["roots"]["augmented" if split == "train" else "original"] / split
    paths, labels = [], []
    for index, name in enumerate(context["classes"]):
        found = sorted(
            path
            for path in (root / name).rglob("*")
            if path.is_file() and path.suffix.lower() in flow.EXTENSIONS
        )
        paths.extend(found)
        labels.extend([index] * len(found))
    return paths, np.asarray(labels, dtype=np.int64)


def _derm_input(path):
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB").resize((448, 448), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/encoded": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[buffer.getvalue()])
                )
            }
        )
    ).SerializeToString()


def _extract_derm_embeddings(infer, paths, batch_size, cache_path):
    if cache_path.is_file():
        values = np.load(cache_path)
        if values.shape == (len(paths), 6144) and np.isfinite(values).all():
            return values
        raise ValueError("Derm embedding cache shape 또는 값이 잘못됐습니다.")
    values = []
    for start in range(0, len(paths), batch_size):
        batch = [_derm_input(path) for path in paths[start : start + batch_size]]
        with tf.device("/CPU:0"):
            output = infer(inputs=tf.constant(batch))["embedding"].numpy()
        output = np.asarray(output).reshape((len(batch), -1))
        if output.shape[1] != 6144:
            raise ValueError("Derm Foundation embedding 출력이 6144차원이 아닙니다.")
        values.append(output)
        print("Derm embedding:", min(start + batch_size, len(paths)), "/", len(paths))
    result = np.concatenate(values).astype("float32")
    if result.shape != (len(paths), 6144) or not np.isfinite(result).all():
        raise ValueError("Derm Foundation embedding 출력이 6144차원이 아닙니다.")
    temporary = cache_path.with_suffix(".tmp.npy")
    np.save(temporary, result)
    temporary.replace(cache_path)
    return result


def _run_derm_foundation(context):
    from huggingface_hub import snapshot_download

    config, trial_id = context["config"], EXPERIMENTS[3]
    spec = {
        "id": trial_id,
        "method": "derm_foundation_6144_embedding_mlp",
        "paper": PAPERS["derm_foundation"],
        "model_id": config["derm_model_id"],
        "input_size": 448,
        "embedding_size": 6144,
        "classifier": "Dense256-Dropout0.1-Dense128-Dropout0.1-Softmax",
        "seed": 42,
        "test_evaluated": False,
    }
    signature = _trial_signature(context, spec)
    cached = _cache_or_none(context, trial_id, signature)
    if cached:
        return cached
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    directory = _attempt(context, trial_id)
    engine.write_json(directory / "spec.json", spec)
    train_paths, train_labels = _paths_and_labels(context, "train")
    val_paths, val_labels = _paths_and_labels(context, "val")
    cache = (
        Path(context["project"])
        / "embedding_cache"
        / "derm_foundation"
        / context["data_hash"]
    )
    cache.mkdir(parents=True, exist_ok=True)
    model_directory = snapshot_download(
        repo_id=config["derm_model_id"],
        allow_patterns=["saved_model.pb", "fingerprint.pb", "variables/*"],
    )
    loaded = tf.saved_model.load(model_directory)
    infer = loaded.signatures["serving_default"]
    started = time.monotonic()
    train_embeddings = _extract_derm_embeddings(
        infer,
        train_paths,
        config["derm_embedding_batch_size"],
        cache / "augmented_train.npy",
    )
    val_embeddings = _extract_derm_embeddings(
        infer,
        val_paths,
        config["derm_embedding_batch_size"],
        cache / "original_val.npy",
    )
    del loaded, infer
    inputs = keras.Input((6144,), name="derm_embedding")
    hidden = keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        bias_regularizer=keras.regularizers.l2(1e-4),
    )(inputs)
    hidden = keras.layers.Dropout(0.1)(hidden)
    hidden = keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        bias_regularizer=keras.regularizers.l2(1e-4),
    )(hidden)
    hidden = keras.layers.Dropout(0.1)(hidden)
    outputs = keras.layers.Dense(
        len(context["classes"]), activation="softmax", name="predictions"
    )(hidden)
    model = keras.Model(inputs, outputs, name="web_skin_derm_foundation_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    checkpoint = directory / "classifier_best.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_accuracy", mode="max", save_best_only=True
        ),
        keras.callbacks.CSVLogger(directory / "classifier_log.csv"),
    ]
    history_object = model.fit(
        train_embeddings,
        train_labels,
        validation_data=(val_embeddings, val_labels),
        batch_size=config["batch_size"],
        epochs=config["epochs1"],
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    history = {
        key: [float(value) for value in values]
        for key, values in history_object.history.items()
    }
    model = keras.models.load_model(checkpoint, compile=False)
    probabilities = model.predict(
        val_embeddings, batch_size=config["batch_size"], verbose=0
    )
    metrics = engine.classification_metrics(
        val_labels, probabilities, len(context["classes"])
    )
    engine.write_json(directory / "validation_metrics.json", metrics)
    relative_paths = [
        path.relative_to(context["extracted"]).as_posix() for path in val_paths
    ]
    engine.save_predictions(
        directory / "validation_predictions.csv",
        relative_paths,
        val_labels,
        probabilities,
    )
    engine.write_json(directory / "classifier_history.json", history)
    record = {
        "id": trial_id,
        "spec": spec,
        "signature": signature,
        "attempt": directory.name,
        "selected_model": checkpoint.name,
        "selected_stage": "embedding_mlp",
        "validation": metrics,
        "stage1_best_val": max(history["val_accuracy"]),
        "stage2_best_val": None,
        "training_seconds": time.monotonic() - started,
        "parameters": model.count_params(),
        "model_bytes": checkpoint.stat().st_size,
        "history": history,
        "stage_boundary": len(history["accuracy"]),
        "embedding_cache": str(cache),
        "inference": (
            "448 PNG -> frozen Derm Foundation SavedModel on CPU -> "
            "6144 embedding -> MLP"
        ),
        "test_evaluated": False,
    }
    engine.finish_record(directory, record)
    return record


def run(context):
    validate(context)
    records = []
    runners = (_run_wsdan, _run_pmg, _run_mixstyle, _run_derm_foundation)
    try:
        for runner in runners:
            record = runner(context)
            records.append(record)
            engine.write_json(
                Path(context["output"]) / "progress.json",
                {"completed": [item["id"] for item in records]},
            )
        engine.write_json(
            Path(context["output"]) / "all_validation_results.json", records
        )
        return records
    except Exception as exc:
        engine.write_json(
            Path(context["output"]) / ("failure_" + uuid.uuid4().hex[:8] + ".json"),
            {
                "error": repr(exc),
                "completed": [item["id"] for item in records],
                "resume_dir": str(context["output"]),
            },
        )
        print("중단 전 완료 실험은 보존됐습니다. RESUME_DIR:", context["output"])
        raise


def _confusion(axis, matrix, title):
    matrix = np.asarray(matrix, dtype=int)
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axis.set(
        title=title,
        xlabel="Predicted",
        ylabel="True",
        xticks=range(len(matrix)),
        yticks=range(len(matrix)),
        xticklabels=[f"C{i}" for i in range(len(matrix))],
        yticklabels=[f"C{i}" for i in range(len(matrix))],
    )
    return image


def summarize(context, records):
    import matplotlib.pyplot as plt
    import pandas as pd

    validate(context)
    output = Path(context["output"])
    baseline = {
        "experiment": "saved_b0_256_ce_baseline",
        "validation_accuracy": wsdan.BASELINE["validation_accuracy"],
        "validation_macro_f1": wsdan.BASELINE["validation_macro_f1"],
        "trained_in_this_run": False,
    }
    rows = [baseline]
    for record in records:
        rows.append(
            {
                "experiment": record["id"],
                "validation_accuracy": record["validation"]["accuracy"],
                "validation_macro_f1": record["validation"]["macro_f1"],
                "trained_in_this_run": True,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "paper_method_comparison.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))

    labels = ["Baseline", "WS-DAN", "PMG", "MixStyle", "Derm Foundation"]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(15, 6))
    axis.bar(x - 0.2, table.validation_accuracy, 0.4, label="Validation Accuracy")
    axis.bar(x + 0.2, table.validation_macro_f1, 0.4, label="Validation Macro F1")
    for index, value in enumerate(table.validation_accuracy):
        axis.text(index - 0.2, value + 0.005, f"{value:.4f}", ha="center")
    for index, value in enumerate(table.validation_macro_f1):
        axis.text(index + 0.2, value + 0.005, f"{value:.4f}", ha="center")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_title("Web Skin Paper-based Method Comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "paper_method_performance_dashboard.png", dpi=180)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for axis, record, label in zip(axes.flat, records, labels[1:], strict=True):
        _confusion(axis, record["validation"]["confusion_matrix"], label)
    fig.suptitle("Web Skin Validation Confusion Matrices (C0-C4)")
    fig.tight_layout()
    fig.savefig(output / "paper_method_confusion_matrices.png", dpi=180)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for axis, record, label in zip(axes.flat, records, labels[1:], strict=True):
        history = record["history"]
        epochs = np.arange(1, len(history["accuracy"]) + 1)
        axis.plot(epochs, history["accuracy"], label="Train accuracy")
        axis.plot(epochs, history["val_accuracy"], label="Validation accuracy")
        if record["stage_boundary"] < len(epochs):
            axis.axvline(record["stage_boundary"] + 0.5, ls="--", color="gray")
        axis.set(title=label, xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "paper_method_training_curves.png", dpi=180)
    plt.show()
    plt.close(fig)

    class_rows = [wsdan.BASELINE["class_f1"]] + [
        record["validation"]["class_f1"] for record in records
    ]
    frame = pd.DataFrame(class_rows, index=labels, columns=context["classes"])
    frame.to_csv(output / "paper_method_class_f1.csv", encoding="utf-8-sig")
    summary = {
        "protocol": PROTOCOL,
        "data_sha256": context["data_hash"],
        "classes": context["classes"],
        "baseline_retrained": False,
        "test_evaluated": False,
        "papers": PAPERS,
        "comparison": rows,
        "winner_by_validation_macro_f1": table.iloc[
            int(table.validation_macro_f1.argmax())
        ].to_dict(),
        "derm_foundation_note": (
            "Google marks Derm Foundation as legacy; this requested comparison uses the "
            "official gated Hugging Face model and a downstream MLP."
        ),
    }
    engine.write_json(output / "paper_method_summary.json", summary)
    archive = flow.archive_results(context)
    return summary, archive
