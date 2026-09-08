

# ---- cell ----

from google.colab import drive
drive.mount('/content/drive')
%pip -q install tensorflow==2.20.0 keras==3.13.2 pandas matplotlib pillow

import gc
import hashlib
import json
import math
import platform
import shutil
import stat
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

if tf.__version__ != '2.20.0' or keras.__version__ != '3.13.2':
    raise RuntimeError('런타임을 재시작한 뒤 처음부터 실행하세요.')
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('Colab 런타임 유형에서 GPU를 선택하세요.')
print(tf.__version__, keras.__version__, tf.config.list_physical_devices('GPU'))

# ---- cell ----

ENGINE_SOURCE = '"""Sequential Web Skin experiments; validation selection precedes any test inference.\n\nThe Colab notebook embeds an exact copy of this module so no repository checkout\nis needed in Colab. Completed trials are reused only after artifact verification.\nInterrupted attempts are preserved and restarted, not resumed mid-epoch.\n"""\n\nfrom __future__ import annotations\n\nimport csv\nimport hashlib\nimport json\nimport time\nimport uuid\nfrom pathlib import Path\n\nimport keras\nimport numpy as np\n\nPROTOCOL = "web_skin_sweep_v1"\nCLASSES = ["건선", "아토피", "여드름", "정상", "주사"]\nTRIALS = [\n    {"id": "b0_224_ce", "backbone": "B0", "size": 224, "loss": "ce"},\n    {"id": "b0_256_ce", "backbone": "B0", "size": 256, "loss": "ce"},\n    {"id": "b0_256_ls005", "backbone": "B0", "size": 256, "loss": "ls005"},\n    {"id": "b0_256_focal15", "backbone": "B0", "size": 256, "loss": "focal15"},\n    {"id": "b1_256_ls005", "backbone": "B1", "size": 256, "loss": "ls005"},\n]\n\n\ndef file_hash(path):\n    digest = hashlib.sha256()\n    with Path(path).open("rb") as stream:\n        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):\n            digest.update(chunk)\n    return digest.hexdigest()\n\n\ndef write_json(path, value):\n    path = Path(path)\n    temporary = path.with_name(".json-" + uuid.uuid4().hex[:12] + ".tmp")\n    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")\n    temporary.replace(path)\n\n\ndef read_json(path):\n    return json.loads(Path(path).read_text(encoding="utf-8"))\n\n\ndef classification_metrics(truth, probabilities, count=5):\n    truth = np.asarray(truth, dtype=np.int64)\n    probabilities = np.asarray(probabilities)\n    if (\n        truth.ndim != 1\n        or not len(truth)\n        or probabilities.shape != (len(truth), count)\n        or not np.isfinite(probabilities).all()\n        or np.any(truth < 0)\n        or np.any(truth >= count)\n    ):\n        raise ValueError("Invalid evaluation arrays")\n    predictions = probabilities.argmax(axis=1)\n    cm = np.bincount(count * truth + predictions, minlength=count * count).reshape(count, count)\n    tp = np.diag(cm).astype(float)\n    precision = np.divide(tp, cm.sum(0), out=np.zeros(count), where=cm.sum(0) != 0)\n    recall = np.divide(tp, cm.sum(1), out=np.zeros(count), where=cm.sum(1) != 0)\n    f1 = np.divide(\n        2 * precision * recall,\n        precision + recall,\n        out=np.zeros(count),\n        where=precision + recall != 0,\n    )\n    return {\n        "accuracy": float(np.mean(truth == predictions)),\n        "macro_f1": float(f1.mean()),\n        "class_f1": f1.tolist(),\n        "precision": precision.tolist(),\n        "recall": recall.tolist(),\n        "support": cm.sum(1).tolist(),\n        "confusion_matrix": cm.tolist(),\n        "count": len(truth),\n    }\n\n\ndef predict_dataset(model, dataset):\n    truth, probabilities = [], []\n    for images, labels in dataset:\n        probabilities.extend(model(images, training=False).numpy())\n        truth.extend(np.argmax(labels.numpy(), axis=1))\n    return np.asarray(truth, dtype=np.int64), np.asarray(probabilities)\n\n\ndef save_predictions(path, paths, truth, probabilities):\n    if len(paths) != len(truth):\n        raise ValueError("File order and prediction count differ")\n    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:\n        writer = csv.writer(stream)\n        writer.writerow(["path", "true_index", "pred_index", *[f"prob_C{i}" for i in range(5)]])\n        for name, target, probs in zip(paths, truth, probabilities, strict=True):\n            writer.writerow([name, int(target), int(probs.argmax()), *map(float, probs)])\n\n\ndef loss_function(name):\n    if name == "ce":\n        return keras.losses.CategoricalCrossentropy()\n    if name == "ls005":\n        return keras.losses.CategoricalCrossentropy(label_smoothing=0.05)\n    if name == "focal15":\n        return keras.losses.CategoricalFocalCrossentropy(alpha=1.0, gamma=1.5)\n    raise ValueError(name)\n\n\ndef build_model(spec):\n    builder = {"B0": keras.applications.EfficientNetB0, "B1": keras.applications.EfficientNetB1}\n    size = spec["size"]\n    backbone = builder[spec["backbone"]](\n        include_top=False, weights="imagenet", input_shape=(size, size, 3)\n    )\n    backbone.trainable = False\n    inputs = keras.Input((size, size, 3))\n    features = backbone(inputs, training=False)\n    features = keras.layers.GlobalAveragePooling2D()(features)\n    features = keras.layers.Dropout(0.3)(features)\n    outputs = keras.layers.Dense(5, activation="softmax")(features)\n    model = keras.Model(inputs, outputs)\n    model.compile(\n        optimizer=keras.optimizers.Adam(1e-4),\n        loss=loss_function(spec["loss"]),\n        metrics=["accuracy"],\n    )\n    return model\n\n\ndef configure_partial(model, loss_name):\n    backbones = [\n        layer\n        for layer in model.layers\n        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower()\n    ]\n    if len(backbones) != 1:\n        raise ValueError("Expected one EfficientNet backbone")\n    backbone = backbones[0]\n    backbone.trainable = True\n    for index, layer in enumerate(backbone.layers):\n        layer.trainable = index >= len(backbone.layers) - 30 and not isinstance(\n            layer, keras.layers.BatchNormalization\n        )\n    model.compile(\n        optimizer=keras.optimizers.Adam(1e-5), loss=loss_function(loss_name), metrics=["accuracy"]\n    )\n    return [layer.name for layer in backbone.layers if layer.trainable]\n\n\nclass HistoryBackup(keras.callbacks.Callback):\n    def __init__(self, path):\n        super().__init__()\n        self.path = path\n        self.values = {}\n\n    def on_epoch_end(self, epoch, logs=None):\n        for key, value in (logs or {}).items():\n            self.values.setdefault(key, []).append(float(value))\n        write_json(self.path, self.values)\n\n\ndef fit_stage(model, train, val, directory, name, epochs):\n    best = directory / (name + "_best.keras")\n    callbacks = [\n        keras.callbacks.ModelCheckpoint(\n            str(best), monitor="val_accuracy", mode="max", save_best_only=True\n        ),\n        keras.callbacks.CSVLogger(str(directory / (name + "_log.csv"))),\n        HistoryBackup(directory / (name + "_history.json")),\n        keras.callbacks.TerminateOnNaN(),\n    ]\n    history = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks, verbose=2)\n    values = {key: [float(v) for v in seq] for key, seq in history.history.items()}\n    if len(values.get("val_accuracy", [])) != epochs or not all(\n        np.isfinite(seq).all() for seq in values.values()\n    ):\n        raise RuntimeError("Incomplete or non-finite training; attempt retained")\n    model.save(directory / (name + "_last.keras"))\n    return values, best\n\n\ndef checkpoint_choice(baseline_score, new_score):\n    """Keep the earlier/simpler checkpoint on ties."""\n    return new_score > baseline_score\n\n\ndef cached_record(root, trial_id, signature):\n    trial_dir = Path(root) / trial_id\n    marker = trial_dir / "completed.json"\n    if not marker.exists():\n        return None\n    record = read_json(marker)\n    if record["signature"] != signature:\n        raise ValueError("Resume settings differ; use a new suite directory")\n    for relative, digest in record["artifact_hashes"].items():\n        target = (trial_dir / relative).resolve()\n        if not target.is_relative_to(trial_dir.resolve()) or file_hash(target) != digest:\n            raise ValueError("Completed artifact changed or corrupted: " + relative)\n    return record\n\n\ndef evaluate_to_files(model, dataset, paths, directory, prefix):\n    truth, probabilities = predict_dataset(model, dataset)\n    metrics = classification_metrics(truth, probabilities)\n    write_json(directory / (prefix + "_metrics.json"), metrics)\n    save_predictions(directory / (prefix + "_predictions.csv"), paths, truth, probabilities)\n    return metrics\n\n\ndef run_trial(spec, dataset_factory, root, signature, seed=42, epochs1=15, epochs2=10):\n    cached = cached_record(root, spec["id"], signature)\n    if cached:\n        return cached\n    keras.backend.clear_session()\n    keras.utils.set_random_seed(seed)\n    directory = Path(root) / spec["id"] / ("attempt_" + uuid.uuid4().hex[:12])\n    directory.mkdir(parents=True, exist_ok=False)\n    write_json(directory / "spec.json", spec)\n    train, _ = dataset_factory("train", spec["size"], True)\n    val, paths = dataset_factory("val", spec["size"], False)\n    started = time.monotonic()\n    model = build_model(spec)\n    h1, best1 = fit_stage(model, train, val, directory, "stage1", epochs1)\n    del model\n    keras.backend.clear_session()\n    model = keras.models.load_model(best1, compile=False)\n    trainable = configure_partial(model, spec["loss"])\n    h2, best2 = fit_stage(model, train, val, directory, "stage2", epochs2)\n    del model\n    selected_stage = (\n        "stage2"\n        if checkpoint_choice(max(h1["val_accuracy"]), max(h2["val_accuracy"]))\n        else "stage1"\n    )\n    selected = best2 if selected_stage == "stage2" else best1\n    model = keras.models.load_model(selected, compile=False)\n    metrics = evaluate_to_files(model, val, paths, directory, "validation")\n    record = {\n        "id": spec["id"],\n        "spec": spec,\n        "signature": signature,\n        "attempt": directory.name,\n        "selected_model": selected.name,\n        "selected_stage": selected_stage,\n        "validation": metrics,\n        "stage1_best_val": max(h1["val_accuracy"]),\n        "stage2_best_val": max(h2["val_accuracy"]),\n        "training_seconds": time.monotonic() - started,\n        "parameters": model.count_params(),\n        "model_bytes": selected.stat().st_size,\n        "trainable_backbone_layers": trainable,\n        "history": {key: h1[key] + h2[key] for key in h1},\n        "stage_boundary": len(h1["accuracy"]),\n    }\n    finish_record(directory, record)\n    return record\n\n\ndef finish_record(directory, record):\n    write_json(directory / "record.json", record)\n    record["artifact_hashes"] = {\n        str(path.relative_to(directory.parent)): file_hash(path)\n        for path in directory.iterdir()\n        if path.is_file()\n    }\n    write_json(directory.parent / "completed.json", record)\n\n\ndef selected_model_path(root, record):\n    return Path(root) / record["id"] / record["attempt"] / record["selected_model"]\n\n\ndef extend_b1(parent, dataset_factory, root, signature, seed=42, epochs=5):\n    trial_id = "b1_256_ls005_extend5"\n    cached = cached_record(root, trial_id, signature)\n    if cached:\n        return cached\n    keras.backend.clear_session()\n    keras.utils.set_random_seed(seed)\n    directory = Path(root) / trial_id / ("attempt_" + uuid.uuid4().hex[:12])\n    directory.mkdir(parents=True, exist_ok=False)\n    parent_dir = Path(root) / parent["id"] / parent["attempt"]\n    # Continue from epoch 10\'s LAST checkpoint, including optimizer state.\n    source = parent_dir / "stage2_last.keras"\n    model = keras.models.load_model(source)\n    if model.optimizer is None:\n        raise ValueError("Extension requires saved optimizer")\n    train, _ = dataset_factory("train", 256, True)\n    val, paths = dataset_factory("val", 256, False)\n    started = time.monotonic()\n    history, best = fit_stage(model, train, val, directory, "extension", epochs)\n    del model\n    parent_best = selected_model_path(root, parent)\n    parent_score = max(parent["stage1_best_val"], parent["stage2_best_val"])\n    keep_extension = checkpoint_choice(parent_score, max(history["val_accuracy"]))\n    selected = directory / "selected.keras"\n    import shutil\n\n    shutil.copyfile(best if keep_extension else parent_best, selected)\n    model = keras.models.load_model(selected, compile=False)\n    metrics = evaluate_to_files(model, val, paths, directory, "validation")\n    record = {\n        "id": trial_id,\n        "spec": {**parent["spec"], "id": trial_id},\n        "signature": signature,\n        "attempt": directory.name,\n        "selected_model": selected.name,\n        "selected_stage": "extension" if keep_extension else "parent_" + parent["selected_stage"],\n        "validation": metrics,\n        "parameters": model.count_params(),\n        "model_bytes": selected.stat().st_size,\n        "training_seconds": time.monotonic() - started,\n        "history": {key: parent["history"][key] + history[key] for key in history},\n        "stage_boundary": parent["stage_boundary"],\n        "extension_boundary": len(parent["history"]["accuracy"]),\n        "parent_last_sha256": file_hash(source),\n        "optimizer_restored": True,\n        "extension_best_val": max(history["val_accuracy"]),\n        "stage1_best_val": parent["stage1_best_val"],\n        "stage2_best_val": parent["stage2_best_val"],\n    }\n    finish_record(directory, record)\n    return record\n\n\ndef select_winner(records):\n    # Stable order preserves earlier experiments on exact ties.\n    return max(records, key=lambda record: record["validation"]["accuracy"])\n'
exec(compile(ENGINE_SOURCE, 'experiment_suite.py', 'exec'))
ENGINE_SHA256 = hashlib.sha256(ENGINE_SOURCE.encode('utf-8')).hexdigest()

# ---- cell ----

MY_DRIVE = Path('/content/drive/MyDrive')
DATA_ZIP_OVERRIDE = ''
RESUME_SUITE_DIR = ''
EXPECTED_DATA_SHA256 = 'f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d'
SEED = 42
BATCH_SIZE = 32
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 10
EXTENSION_EPOCHS = 5
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
LOCAL_ROOT = Path('/content') / ('web_skin_suite_' + RUN_ID)
LOCAL_ROOT.mkdir(parents=True, exist_ok=False)
settings = {
    'protocol': PROTOCOL, 'engine_sha256': ENGINE_SHA256,
    'notebook_revision': 'review_20260908',
    'data_sha256': EXPECTED_DATA_SHA256, 'class_names': CLASSES, 'trials': TRIALS,
    'seed': SEED, 'batch_size': BATCH_SIZE, 'stage1_epochs': STAGE1_EPOCHS,
    'stage2_epochs': STAGE2_EPOCHS, 'extension_epochs': EXTENSION_EPOCHS,
    'selection': 'validation_accuracy; ties keep earlier trial/stage',
    'test_policy': 'final winner only',
    'preprocessing': 'TensorFlow directory loader, RGB float32 0..255, bilinear resize; no external /255',
}
SIGNATURE = hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
if RESUME_SUITE_DIR:
    SUITE_DIR = Path(RESUME_SUITE_DIR)
    if not (SUITE_DIR / 'suite_config.json').exists():
        raise FileNotFoundError('재개할 suite_config.json이 없습니다.')
    if read_json(SUITE_DIR / 'suite_config.json')['signature'] != SIGNATURE:
        raise ValueError('기존 실험과 설정/코드가 다릅니다. 새 실험 폴더로 실행하세요.')
else:
    SUITE_DIR = MY_DRIVE / 'mediflow_experiments' / 'web_skin' / ('suite_' + RUN_ID)
    SUITE_DIR.mkdir(parents=True, exist_ok=False)
    write_json(SUITE_DIR / 'suite_config.json', {'signature': SIGNATURE, 'settings': settings,
        'code_commit_at_creation': '60970a4389c32768fed2644cd3bdc02c668940d3',
        'code_state': 'new uncommitted module and notebook; engine + notebook snapshot saved',
        'existing_reported_augmented': {'validation_accuracy': 0.6880000233650208,
            'test_accuracy': 0.8199999928474426, 'macro_f1': 0.8195818185502844},
        'limitations': ['person/lesion/session leakage not verified',
            'augmentation lineage/modified near-duplicates not verified',
            'original training data hash unavailable', 'no real webcam validation',
            'one seed only; small differences not established as robust gains',
            'normal included; out-of-scope rejection not implemented']})
(SUITE_DIR / 'experiment_suite.py').write_text(ENGINE_SOURCE, encoding='utf-8')
write_json(SUITE_DIR / ('environment_' + RUN_ID + '.json'), {
    'python': platform.python_version(), 'tensorflow': tf.__version__, 'keras': keras.__version__,
    'numpy': np.__version__, 'gpu': [str(d) for d in tf.config.list_physical_devices('GPU')]})
print('결과 폴더 / 재개 시 사용할 경로:', SUITE_DIR)

# ---- cell ----

if DATA_ZIP_OVERRIDE:
    candidates = [Path(DATA_ZIP_OVERRIDE)]
else:
    candidates = [p for p in MY_DRIVE.rglob('web_skin_processed*')
                  if p.is_file() and zipfile.is_zipfile(p)]
candidates = sorted({p.resolve() for p in candidates if p.is_file() and zipfile.is_zipfile(p)})
if len(candidates) != 1:
    raise ValueError(f'DATA_ZIP_OVERRIDE로 입력 ZIP을 하나 지정하세요: {candidates}')
source_zip = candidates[0]
with zipfile.ZipFile(source_zip) as archive:
    members = [m for m in archive.infolist()
               if 'augmented' in [part.lower() for part in Path(m.filename).parts]]
    required = sum(m.file_size for m in members) + source_zip.stat().st_size + 2 * 1024**3
if required > shutil.disk_usage(LOCAL_ROOT).free:
    raise RuntimeError('Colab 디스크 공간이 부족합니다.')
local_zip = LOCAL_ROOT / 'data.zip'
with source_zip.open('rb') as src, local_zip.open('xb') as dst:
    digest = hashlib.sha256()
    for chunk in iter(lambda: src.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
        dst.write(chunk)
if digest.hexdigest() != EXPECTED_DATA_SHA256 or file_hash(local_zip) != EXPECTED_DATA_SHA256:
    raise ValueError('기존 검증받은 ZIP과 다릅니다. 이 데이터로 학습하지 않습니다.')
DATA_DIR = LOCAL_ROOT / 'data'
DATA_DIR.mkdir()
with zipfile.ZipFile(local_zip) as archive:
    for member in members:
        target = (DATA_DIR / member.filename).resolve()
        if not target.is_relative_to(DATA_DIR.resolve()) or stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError('잘못된 ZIP 경로')
        archive.extract(member, DATA_DIR)
roots = [p for p in DATA_DIR.rglob('*') if p.is_dir() and p.name.lower() == 'augmented'
         and all((p / s).is_dir() for s in ('train', 'val', 'test'))]
if len(roots) != 1:
    raise ValueError(f'Augmented 폴더 구조 확인 필요: {roots}')
DATA_ROOT = roots[0]
write_json(SUITE_DIR / 'data_source.json', {
    'source_zip': str(source_zip), 'sha256': EXPECTED_DATA_SHA256,
    'prior_audit': 'dataset_audit_20260907_174038_67c0377d.zip',
    'audit_policy': 'verified same ZIP; no repeated image duplicate audit',
    'train': 7200, 'validation': 500, 'test': 400,
})
def dataset_factory(split, size, shuffle):
    ds = keras.utils.image_dataset_from_directory(
        DATA_ROOT / split, class_names=CLASSES, labels='inferred', label_mode='categorical',
        image_size=(size, size), batch_size=BATCH_SIZE, shuffle=shuffle,
        seed=SEED if shuffle else None, interpolation='bilinear')
    paths = [str(Path(p).relative_to(DATA_ROOT)) for p in ds.file_paths]
    return ds.prefetch(tf.data.AUTOTUNE), paths
print('동일 ZIP 확인 완료. 검증된 데이터로 학습합니다.')

# ---- cell ----

CODES = [f'C{i}' for i in range(5)]
write_json(SUITE_DIR / 'class_mapping.json', dict(zip(CODES, CLASSES)))

def save_error_examples(predictions_path, destination, title, limit=8):
    frame = pd.read_csv(predictions_path)
    wrong = frame[frame.true_index != frame.pred_index].copy()
    if wrong.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, 'No misclassifications in this split', ha='center')
        ax.axis('off')
    else:
        wrong['score'] = wrong[[f'prob_C{i}' for i in range(5)]].max(axis=1)
        sample = wrong.sort_values('score', ascending=False).head(limit)
        fig, axes = plt.subplots(math.ceil(len(sample)/4), 4, figsize=(14, 4*math.ceil(len(sample)/4)), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, (_, row) in zip(axes.flat, sample.iterrows()):
            path = (DATA_ROOT / row['path']).resolve()
            if not path.is_relative_to(DATA_ROOT.resolve()):
                raise ValueError('잘못된 이미지 경로')
            with Image.open(path) as image:
                ax.imshow(image.convert('RGB'))
            ax.set_title(f"True C{int(row.true_index)} / Pred C{int(row.pred_index)}", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)

def draw_individual(record):
    directory = SUITE_DIR / record['id'] / record['attempt']
    hist = record['history']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric in zip(axes, ('accuracy', 'loss')):
        epochs = np.arange(1, len(hist[metric])+1)
        ax.plot(epochs, hist[metric], label='Train')
        ax.plot(epochs, hist['val_' + metric], label='Validation')
        ax.axvline(record['stage_boundary'] + 0.5, color='gray', ls='--')
        if 'extension_boundary' in record:
            ax.axvline(record['extension_boundary'] + 0.5, color='green', ls=':')
        ax.set(title=record['id']+' / '+metric, xlabel='Epoch', ylabel=metric)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / 'training_curves.png', dpi=180)
    plt.close(fig)
    save_error_examples(directory / 'validation_predictions.csv',
                        directory / 'validation_errors.png', record['id']+' / Validation errors')

def overview(records):
    fig, axes = plt.subplots(3, 4, figsize=(24, 14))
    max_loss = max(max(r['history'][k]) for r in records for k in ('loss', 'val_loss'))
    for index, record in enumerate(records):
        row, col = divmod(index, 2)
        for offset, metric in enumerate(('accuracy', 'loss')):
            ax = axes[row, col*2+offset]
            history = record['history']
            x = np.arange(1, len(history[metric])+1)
            ax.plot(x, history[metric], label='Train')
            ax.plot(x, history['val_'+metric], label='Validation')
            ax.axvline(record['stage_boundary']+0.5, color='gray', ls='--')
            if 'extension_boundary' in record:
                ax.axvspan(record['extension_boundary']+0.5, len(x)+0.5, color='#dcfce7', alpha=0.6)
            ax.set(title=record['id']+' / '+metric, xlabel='Epoch', ylabel=metric,
                   ylim=(0, 1) if metric=='accuracy' else (0, max_loss*1.05))
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
    fig.suptitle('Web Skin / All training curves', fontsize=20)
    fig.text(0.5, 0.008, 'Loss definitions differ: compare trends within each experiment, not absolute loss between CE / LS / Focal.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(SUITE_DIR / 'all_training_curves.png', dpi=180)
    plt.show()
    plt.close(fig)

    labels = [r['id'] for r in records]
    table = pd.DataFrame([{
        'experiment': r['id'], 'selected_stage': r['selected_stage'],
        'validation_accuracy': r['validation']['accuracy'],
        'validation_macro_f1': r['validation']['macro_f1'],
        'elapsed_seconds_this_trial': r['training_seconds'],
        'parameters': r['parameters'], 'model_bytes': r['model_bytes'],
        'stage1_best_val': r['stage1_best_val'], 'stage2_best_val': r['stage2_best_val'],
    } for r in records])
    table.to_csv(SUITE_DIR / 'experiment_comparison.csv', index=False, encoding='utf-8-sig')
    display(table)
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={'height_ratios':[1, 1.2]})
    x = np.arange(len(records))
    axes[0].bar(x-0.18, table.validation_accuracy, width=0.36, label='Validation Accuracy')
    axes[0].bar(x+0.18, table.validation_macro_f1, width=0.36, label='Validation Macro F1')
    axes[0].axhline(0.6880000233650208, color='gray', ls='--', label='Historical baseline Val (reported)')
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 1), title='Validation comparison / selected checkpoints')
    axes[0].tick_params(axis='x', labelrotation=15)
    axes[0].legend(fontsize=9)
    matrix = np.array([r['validation']['class_f1'] for r in records])
    axes[1].imshow(matrix, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    axes[1].set(xticks=range(5), xticklabels=CODES, yticks=x, yticklabels=labels, title='Validation class F1')
    for i in range(len(records)):
        for j in range(5):
            axes[1].text(j, i, str(matrix[i,j]), ha='center', va='center', fontsize=8,
                         color='white' if matrix[i,j]>0.55 else 'black')
    fig.tight_layout()
    fig.savefig(SUITE_DIR / 'validation_performance_dashboard.png', dpi=180)
    plt.show()
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, r in zip(axes.flat, records):
        cm = np.asarray(r['validation']['confusion_matrix'])
        ax.imshow(cm, cmap='Blues')
        ax.set(title=r['id'], xticks=range(5), yticks=range(5),
               xticklabels=CODES, yticklabels=CODES, xlabel='Predicted', ylabel='True')
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.suptitle('All Validation confusion matrices')
    fig.tight_layout()
    fig.savefig(SUITE_DIR / 'all_validation_confusion_matrices.png', dpi=180)
    plt.show()
    plt.close(fig)

# ---- cell ----

records = []
try:
    for spec in TRIALS:
        print('실험 시작:', spec['id'])
        record = run_trial(spec, dataset_factory, SUITE_DIR, SIGNATURE, SEED,
                           STAGE1_EPOCHS, STAGE2_EPOCHS)
        records.append(record)
        draw_individual(record)
        write_json(SUITE_DIR / 'progress.json', {'completed': [r['id'] for r in records]})
        gc.collect()
    parent = next(r for r in records if r['id']=='b1_256_ls005')
    extended = extend_b1(parent, dataset_factory, SUITE_DIR, SIGNATURE, SEED, EXTENSION_EPOCHS)
    records.append(extended)
    draw_individual(extended)
    write_json(SUITE_DIR / 'all_validation_results.json', records)
    overview(records)
except Exception as exc:
    write_json(SUITE_DIR / ('failure_' + RUN_ID + '.json'), {
        'error': repr(exc), 'completed': [r['id'] for r in records],
        'resume_suite_dir': str(SUITE_DIR),
        'policy': 'completed trials reused; interrupted trial restarted in a new attempt'})
    print('완료 모델/로그는 Drive에 남아 있습니다. 재개 경로:', SUITE_DIR)
    raise

# ---- cell ----

winner = select_winner(records)
winner_path = selected_model_path(SUITE_DIR, winner)
selection = {
    'winner': winner['id'], 'model_sha256': file_hash(winner_path),
    'validation': winner['validation'], 'signature': SIGNATURE,
    'historical_baseline_validation': 0.6880000233650208,
    'above_historical_val': winner['validation']['accuracy'] > 0.6880000233650208,
    'status': 'public_data_candidate_not_device_validated',
    'selection_policy': 'Validation Accuracy, exact ties retain earlier experiment',
}
selection_path = SUITE_DIR / 'selection_before_test.json'
if selection_path.exists():
    if read_json(selection_path) != selection:
        raise ValueError('이미 Test 선택을 고정한 suite와 후보가 다릅니다. 중단합니다.')
else:
    write_json(selection_path, selection)
test_marker = SUITE_DIR / 'test_completed.json'
if test_marker.exists():
    test_record = read_json(test_marker)
    if test_record['model_sha256'] != selection['model_sha256']:
        raise ValueError('Test 결과와 선택 모델 불일치')
    for name, digest in test_record['artifact_hashes'].items():
        if file_hash(SUITE_DIR / name) != digest:
            raise ValueError('기존 Test 결과 파일이 변경됐습니다.')
    print('기존 Test 평가 결과를 재사용합니다.')
else:
    keras.backend.clear_session()
    model = keras.models.load_model(winner_path, compile=False)
    test_ds, paths = dataset_factory('test', winner['spec']['size'], False)
    metrics = evaluate_to_files(model, test_ds, paths, SUITE_DIR, 'final_test')
    test_record = {
        'model_sha256': selection['model_sha256'], 'metrics': metrics,
        'artifact_hashes': {name: file_hash(SUITE_DIR/name)
            for name in ('final_test_metrics.json', 'final_test_predictions.csv')}}
    write_json(test_marker, test_record)
    del model
save_error_examples(SUITE_DIR / 'final_test_predictions.csv',
                    SUITE_DIR / 'final_test_errors.png', 'Final candidate / Test errors')
cm = np.asarray(test_record['metrics']['confusion_matrix'])
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(cm, cmap='Blues')
ax.set(xticks=range(5), yticks=range(5), xticklabels=CODES, yticklabels=CODES,
       xlabel='Predicted', ylabel='True', title=winner['id']+' / Final Test')
for i in range(5):
    for j in range(5):
        ax.text(j,i,str(cm[i,j]),ha='center',va='center',
                color='white' if cm[i,j]>cm.max()/2 else 'black')
fig.tight_layout()
fig.savefig(SUITE_DIR / 'final_test_confusion_matrix.png', dpi=180)
plt.show()
plt.close(fig)
print('선택 모델:', winner['id'])
print('선택 모델 Test:', test_record['metrics'])
if not selection['above_historical_val']:
    print('기존 보고 Validation보다 높지 않습니다. 기존 모델 교체를 권장하지 않습니다.')

# ---- cell ----

(SUITE_DIR / ('notebook_snapshot_' + RUN_ID + '.py')).write_text(
    '\n\n# ---- cell ----\n\n'.join(get_ipython().history_manager.input_hist_raw), encoding='utf-8')
card = {
    'domain': 'web_skin', 'class_names': CLASSES, 'normal_included': True,
    'model': winner['id'], 'model_relative_path': str(winner_path.relative_to(SUITE_DIR)),
    'model_sha256': selection['model_sha256'], 'input_size': winner['spec']['size'],
    'input': 'RGB float32 0..255; internal rescaling; no external normalization',
    'output': '5 softmax scores in class_names order, not calibrated correctness probabilities',
    'validation': winner['validation'], 'test': test_record['metrics'],
    'limitations': read_json(SUITE_DIR/'suite_config.json')['limitations'],
}
write_json(SUITE_DIR / 'model_card.json', card)
for full in (False, True):
    label = 'full_models' if full else 'reports'
    local_output = LOCAL_ROOT / (SUITE_DIR.name + '_' + label + '_' + RUN_ID + '.zip')
    with zipfile.ZipFile(local_output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(SUITE_DIR.rglob('*')):
            if p.is_file() and p.suffix != '.tmp' and (full or p.suffix != '.keras'):
                archive.write(p, arcname=SUITE_DIR.name + '/' + str(p.relative_to(SUITE_DIR)))
    destination = SUITE_DIR.parent / local_output.name
    with local_output.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    if file_hash(destination) != file_hash(local_output):
        raise IOError('결과 ZIP 복사 검증 실패')
    print(label, 'ZIP:', destination)
print('완료. 우선 reports ZIP을 공유하세요.')