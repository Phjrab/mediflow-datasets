"""Reference inference for packaged candidates; not a clinical evaluation or web API."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = {
    "hair": ["비듬_0006.jpg", "미세각질_0012.jpg"],
    "web_skin": ["정상_000002.png", "여드름_000034.png"],
    "skin": ["광선각화증_0001.png", "보웬병_0001.png"],
}
PROTOCOL = "candidate_reproduction_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess(path: Path, size: int):
    """Match Keras directory loader: decode RGB, no EXIF, TF bilinear, no crop/pad."""
    import tensorflow as tf

    image = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    return tf.expand_dims(
        tf.image.resize(image, (size, size), method="bilinear", antialias=False), 0
    )


def compare(reference: dict, actual: dict, reference_dir: Path, actual_dir: Path) -> list[str]:
    """Fixed engineering tolerances; failures must be investigated, not auto-relaxed."""
    failures = []
    if reference["protocol"] != actual["protocol"]:
        return ["protocol mismatch"]
    if set(reference["cases"]) != set(actual["cases"]):
        return ["case set mismatch"]
    for case_id, expected in reference["cases"].items():
        observed = actual["cases"][case_id]
        for key in (
            "domain",
            "image",
            "image_sha256",
            "model_sha256",
            "candidate_id",
            "class_names",
            "normal_class_included",
            "input_shape",
            "predicted_index",
        ):
            if expected[key] != observed[key]:
                failures.append(f"{case_id}: {key} mismatch")
        for label, left, right, atol, rtol in (
            (
                "input",
                np.load(reference_dir / expected["tensor"], allow_pickle=False),
                np.load(actual_dir / observed["tensor"], allow_pickle=False),
                1e-4,
                0,
            ),
            ("scores", np.asarray(expected["scores"]), np.asarray(observed["scores"]), 1e-5, 1e-4),
        ):
            if (
                left.shape != right.shape
                or not np.isfinite(right).all()
                or not np.allclose(left, right, atol=atol, rtol=rtol)
            ):
                failures.append(f"{case_id}: {label} mismatch")
    return failures


def run(output: Path, reference: Path | None = None) -> dict:
    import keras
    import tensorflow as tf

    index = json.loads((ROOT / "results/CANDIDATE_INDEX.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": PROTOCOL,
        "purpose": "execution parity only; sample names are not verified ground truth",
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "devices": [str(d) for d in tf.config.list_physical_devices()],
        },
        "code_sha256": sha256(Path(__file__)),
        "cases": {},
    }
    for args, key in (
        (["rev-parse", "HEAD"], "commit"),
        (["status", "--short"], "working_tree_status"),
    ):
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        report[key] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    for domain, filenames in SAMPLES.items():
        spec = index["candidates"][domain]
        model_path = ROOT / "results" / spec["model"]
        digest = sha256(model_path)
        if digest != spec["model_sha256"]:
            raise ValueError(f"{domain}: model SHA-256 mismatch")
        classes = json.loads(model_path.with_name("class_names.json").read_text(encoding="utf-8"))
        if not isinstance(classes, list) or len(set(classes)) != len(classes):
            raise ValueError(f"{domain}: invalid class array")
        model = tf.keras.models.load_model(model_path, compile=False)
        size = spec["input_size"]
        if tuple(model.input_shape[1:]) != (size, size, 3):
            raise ValueError(f"{domain}: input shape mismatch")
        if model.output_shape[-1] != len(classes):
            raise ValueError(f"{domain}: class count mismatch")
        for number, filename in enumerate(filenames):
            image = ROOT / "data_examples" / domain / filename
            inputs = preprocess(image, size)
            scores = np.asarray(model(inputs, training=False))[0]
            if (
                scores.shape != (len(classes),)
                or not np.isfinite(scores).all()
                or np.any(scores < 0)
                or np.any(scores > 1)
                or not np.isclose(scores.sum(), 1, atol=1e-5)
            ):
                raise ValueError(f"{domain}: invalid softmax output")
            case_id = f"{domain}_{number}"
            tensor = case_id + ".npy"
            np.save(output / tensor, inputs.numpy(), allow_pickle=False)
            predicted = int(scores.argmax())
            report["cases"][case_id] = {
                "domain": domain,
                "candidate_id": spec["candidate_id"],
                "model_sha256": digest,
                "class_names": classes,
                "normal_class_included": spec["normal_class_included"],
                "image": image.relative_to(ROOT).as_posix(),
                "image_sha256": sha256(image),
                "input_shape": list(inputs.shape),
                "tensor": tensor,
                "input_dtype": str(inputs.dtype.name),
                "scores": scores.tolist(),
                "predicted_index": predicted,
                "predicted_label": classes[predicted],
                "normal_assessment": "not_supported"
                if domain != "web_skin"
                else "normal_class_present_not_clinically_validated",
                "calibration_status": "not_calibrated",
            }
        del model
        keras.backend.clear_session()
    if reference:
        baseline = json.loads((reference / "reference.json").read_text(encoding="utf-8"))
        failures = compare(baseline, report, reference, output)
        report["comparison"] = {"passed": not failures, "failures": failures}
    (output / "reference.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--reference", type=Path, help="Existing reference directory to compare")
    args = parser.parse_args()
    report = run(args.output, args.reference)
    print(json.dumps(report.get("comparison", {"reference_created": True}), ensure_ascii=False))
    if report.get("comparison", {}).get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
