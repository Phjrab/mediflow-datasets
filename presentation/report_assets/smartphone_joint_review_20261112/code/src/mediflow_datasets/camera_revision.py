"""Reconsider chroma and focus in all three domains using a frozen Train/Target replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mediflow_datasets import camera_extended as camera
from mediflow_datasets import camera_similarity as audit
from mediflow_datasets import domain_adaptation as runner
from mediflow_datasets import domain_adaptation_analysis as analysis
from mediflow_datasets import domain_adaptation_data as data
from mediflow_datasets import domain_adaptation_experiment as exp


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def add_normalized_detail(metrics):
    result = dict(metrics)
    if "detail" in result and "contrast" in result:
        result["normalized_detail"] = result["detail"] / max(result["contrast"], 1e-3)
    return result


def color_direction(source, target, key, measured_changes):
    """Require stable residual direction, with a budget from observed within-person changes."""
    if any(row.get("split", "train") != "train" for row in source):
        raise ValueError("Direction calibration requires Train only")
    if any(row.get("label") is not None or "group" not in row for row in target):
        raise ValueError("Target direction requires unlabeled calibration groups")
    classes, groups = defaultdict(list), defaultdict(list)
    for row in source:
        if key in row["metrics"]:
            classes[row["label"]].append(row["metrics"][key])
    for row in target:
        if key in row["metrics"]:
            groups[row["group"]].append(row["metrics"][key])
    if not classes or len(groups) != 8:
        raise ValueError("Comparable class and person/condition metrics are required")
    anchor = float(np.median([np.median(v) for v in classes.values()]))
    medians = {key: float(np.median(values)) for key, values in groups.items()}
    people = sorted({key.split("/")[0] for key in groups})
    if len(people) != 4:
        raise ValueError("The current calibration requires four participants")

    def direction(values):
        low, high = np.quantile(np.asarray(values) - anchor, [0.25, 0.75])
        return 1 if low > 0 else -1 if high < 0 else 0

    sign = direction(list(medians.values()))
    held_out = {
        person: direction([v for key, v in medians.items() if not key.startswith(person + "/")])
        for person in people
    }
    stable = sign != 0 and all(value == sign for value in held_out.values())
    changes = np.asarray(measured_changes, dtype=np.float64)
    if changes.shape != (4,) or not np.isfinite(changes).all():
        raise ValueError("Four finite within-person changes are required")
    budget = min(3.0, float(np.quantile(np.abs(changes), 0.75)))
    bounds = sorted([0.0, sign * budget]) if stable else [0.0, 0.0]
    return bounds, {
        "source_after_lighting_class_balanced_anchor": anchor,
        "target_equal_group_medians": medians,
        "full_direction": sign,
        "leave_one_person_out_directions": held_out,
        "direction_stable": stable,
        "within_person_absolute_q75_limit": budget,
        "proposed_range": bounds,
        "policy": "Zero-inclusive bounded shift; direction stable across four omissions",
    }


def does_not_worsen(candidate, baseline, target, keys):
    evidence = {}
    accepted = True
    improved = False
    for key in keys:
        before = audit.distribution_gap(baseline, target, key)
        after = audit.distribution_gap(candidate, target, key)
        if before is None or after is None:
            raise ValueError("Missing metrics in the fixed comparison")
        evidence[key] = {"before": before, "after": after}
        accepted &= after <= before + 1e-12
        improved |= after < before - 1e-12
    return bool(accepted and improved), evidence


def fast_measure(image, domain, detector, box):
    if domain == "web_skin" and box is None:
        return {}
    quantized = np.rint(image).clip(0, 255).astype(np.uint8)
    ok, png = cv2.imencode(".png", quantized[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise ValueError("Failed to encode the measurement input")
    return add_normalized_detail(analysis.measure(png.tobytes(), domain, len(image), detector, box))


def evaluate(images, records, profile, domain, detector):
    rows, per_seed = [], {seed: [] for seed in audit.SEEDS}
    endpoint_max = 0.0
    for original, row in zip(images, records, strict=True):
        box = audit.box_from_metrics(row["metrics"], len(original))
        image_row = data.ImageRow(row["path"], "train", row["label"])
        metrics = []
        for seed_base in audit.SEEDS:
            seed = exp.item_seed(image_row, seed_base)
            transformed = camera.transform(original, profile, "expanded", seed)
            metric = fast_measure(transformed, domain, detector, box)
            metrics.append(metric)
            per_seed[seed_base].append({**row, "metrics": metric})
            endpoint_max = max(
                endpoint_max,
                float(
                    np.mean(
                        ((transformed <= 0) | (transformed >= 255))
                        & (original > 0)
                        & (original < 255)
                    )
                ),
            )
        rows.append({**row, "metrics": audit.median_repetitions(metrics)})
    return {"rows": rows, "per_seed": per_seed, "new_endpoint_max": endpoint_max}


def approve(candidate, baseline, target, keys):
    accepted, evidence = does_not_worsen(candidate["rows"], baseline["rows"], target, keys)
    evidence = {"per_image_median": evidence, "per_seed": {}}
    for seed in audit.SEEDS:
        ok, distance = does_not_worsen(
            candidate["per_seed"][seed], baseline["per_seed"][seed], target, keys
        )
        evidence["per_seed"][seed] = {"accepted": ok, "distances": distance}
        accepted &= ok
    return bool(accepted), evidence


def parameter_key(profile):
    return tuple(
        tuple(profile[key]) for key in ("color_a", "color_b", "texture_gain", "blur_sigma_roi64")
    )


def rebuild(root, output):
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    detector = analysis.face_detector()
    study_path = root / "configs/camera_strength_20261110.json"
    study = data.read_json(study_path)
    old_audit = root / "presentation/report_assets/camera_similarity_check_20261110"
    verified = data.read_json(old_audit / "verification.json")
    if (
        data.sha256(study_path) != verified["study_sha256"]
        or data.sha256(old_audit / "summary.json") != verified["summary_sha256"]
        or data.sha256(old_audit / "measurements.json") != verified["measurement_sha256"]
    ):
        raise ValueError("Frozen replay inputs changed")
    previous = data.read_json(old_audit / "measurements.json")
    summary = data.read_json(old_audit / "summary.json")
    study = deepcopy(study)
    study["parent_study_sha256"] = verified["study_sha256"]
    study["hypothesis"] = "Direction-audited bounded camera profile versus previous Stage 2"
    study["changed_condition"] = "camera transform profile"
    study["accompanying_changes"] = (
        "Lighting unchanged; revise all domains' chroma directions and retain focus/texture "
        "only when absolute and contrast-normalized detail gaps do not increase"
    )
    study["profile_builder_sha256"] = data.sha256(Path(__file__))
    study["revision_schema"] = "camera_direction_and_focus_audit_v1"
    decisions, measured = {}, {}
    for domain in audit.DOMAINS:
        entry = study["domains"][domain]
        old_profile = entry["profile"]
        packet_path = root / entry["measurement_provenance"]["target"]
        if data.sha256(packet_path) != entry["measurement_provenance"]["target_sha256"]:
            raise ValueError("Target packet changed")
        packet = data.read_json(packet_path)
        target_image_rows = [data.ImageRow(**r) for r in packet["rows"]]
        if (
            runner.target_fingerprint(root / "camera_capture", target_image_rows)
            != packet["fingerprint"]
        ):
            raise ValueError("Actual target captures changed")
        manifest_path = (
            root / f"results/camera_adaptation_stage1_20261110/{domain}/source_manifest.json"
        )
        if data.sha256(manifest_path) != entry["source_manifest_sha256"]:
            raise ValueError("Retained Train split changed")
        keep = {r["path"]: r["label"] for r in data.read_json(manifest_path)["original"]["train"]}
        source = previous[domain]["groups"]["original"]
        target = [
            {**r, "metrics": add_normalized_detail(r["metrics"])}
            for r in previous[domain]["groups"]["target"]
        ]
        if len(source) != summary["domains"][domain]["source_unique_count"]:
            raise ValueError("Source preview count changed")
        paths = {r["path"]: r for r in summary["domains"][domain]["source_png_provenance"]}
        images = []
        for row in source:
            if keep.get(row["path"]) != row["label"]:
                raise ValueError("Only retained Train may enter calibration")
            path = root / paths[row["path"]]["preview"]
            if data.sha256(path) != paths[row["path"]]["sha256"]:
                raise ValueError("Source preview PNG changed")
            with Image.open(path) as photo:
                images.append(np.asarray(photo.convert("RGB"), dtype=np.float32))
        neutral = deepcopy(old_profile)
        neutral.update(
            color_a=[0.0, 0.0],
            color_b=[0.0, 0.0],
            texture_gain=[1.0, 1.0],
            blur_sigma_roi64=[0.0, 0.0],
        )
        cache = {}

        def get(profile, cache=cache, images=images, source=source, domain=domain):
            key = parameter_key(profile)
            if key not in cache:
                cache[key] = evaluate(images, source, profile, domain, detector)
            return cache[key]

        lighting = get(neutral)
        proposed, direction_evidence = deepcopy(neutral), {}
        for axis in ("a", "b"):
            bounds, evidence = color_direction(
                lighting["rows"],
                target,
                f"lab_{axis}",
                old_profile["evidence"]["chroma"][f"lab_{axis}"]["participant_changes"],
            )
            proposed[f"color_{axis}"] = bounds
            direction_evidence[axis] = evidence
        colors, color_audit = {}, {}
        for name, axes in (
            ("both", ("a", "b")),
            ("b_only", ("b",)),
            ("a_only", ("a",)),
            ("none", ()),
        ):
            candidate = deepcopy(neutral)
            for axis in axes:
                candidate[f"color_{axis}"] = proposed[f"color_{axis}"]
            result = get(candidate)
            ok, evidence = approve(result, lighting, target, ("lab_a", "lab_b"))
            colors[name] = candidate
            color_audit[name] = {"accepted": ok, "distances": evidence}
        color_choice = next(
            (name for name in ("both", "b_only", "a_only") if color_audit[name]["accepted"]), "none"
        )
        color_base = colors[color_choice]
        color_result = get(color_base)
        focus_profiles, focus_audit = {}, {}
        focus_keys = (
            ("detail", "normalized_detail", "dark_ridge")
            if domain == "hair"
            else ("detail", "normalized_detail")
        )
        for name, texture, blur in (
            ("both", True, True),
            ("texture_only", True, False),
            ("blur_only", False, True),
            ("none", False, False),
        ):
            candidate = deepcopy(color_base)
            if texture:
                candidate["texture_gain"] = old_profile["texture_gain"]
            if blur:
                candidate["blur_sigma_roi64"] = old_profile["blur_sigma_roi64"]
            result = get(candidate)
            ok, evidence = approve(result, color_result, target, focus_keys)
            focus_profiles[name] = candidate
            focus_audit[name] = {"accepted": ok, "distances": evidence}
        focus_choice = next(
            (
                name
                for name in ("both", "texture_only", "blur_only")
                if focus_audit[name]["accepted"]
            ),
            "none",
        )
        selected = focus_profiles[focus_choice]
        selected["calibration_version"] = "direction_and_focus_audited_v1"
        selected["lighting"]["calibration_version"] = "direction_and_focus_audited_v1"
        selected["lighting"]["train_visual_review"] = "revised_train_replay_not_clinical_validation"
        selected["lighting"]["review_note"] = (
            "Same retained Train previews and actual calibration captures remeasured. "
            "Unlabeled descriptive audit; disease-feature preservation and accuracy unverified."
        )
        for axis in ("a", "b"):
            selected["evidence"]["chroma"][f"lab_{axis}"]["range"] = selected[f"color_{axis}"]
        selected["revision_evidence"] = {
            "color_direction": direction_evidence,
            "color_choice": color_choice,
            "focus_choice": focus_choice,
            "source_preview_count": len(source),
            "target_count": len(target),
            "seeds": list(audit.SEEDS),
            "selection_basis": "Unlabeled Train/calibration proxy checks, not classifier scores",
        }
        selected["previous_engineering_bounds"] = deepcopy(selected["engineering_bounds"])
        selected["engineering_bounds"] = {
            "chroma_limit_lab": 3,
            "chroma_budget_policy": "q75 absolute within-person change; includes zero",
            "texture_gain_limits": selected["texture_gain"],
            "blur_max_roi64": selected["blur_sigma_roi64"][1],
            "lighting": deepcopy(selected["lighting"]["bounds"]),
        }
        selected["calibration_evidence"].pop("color_texture_focus_measured_fraction", None)
        selected["calibration_evidence"]["revision_policy"] = "direction_and_focus_audited_v1"
        selected["limitations"] += [
            "Revised chroma direction remains confounded with disease, person and body region",
            "Consistency across omissions is a sensitivity check, not independent validation",
            "Focus gates are content-dependent proxies; no physical camera equivalence established",
        ]
        entry["profile"] = selected
        decisions[domain] = {
            "color_proposals": direction_evidence,
            "color_candidates": color_audit,
            "focus_candidates": focus_audit,
            "selected_color": color_choice,
            "selected_focus": focus_choice,
            "unique_profiles_measured": len(cache),
            "new_endpoint_max": get(selected)["new_endpoint_max"],
        }
        measured[domain] = {
            "original": [{**r, "metrics": add_normalized_detail(r["metrics"])} for r in source],
            "lighting": lighting["rows"],
            "color_only": color_result["rows"],
            "revised": get(selected)["rows"],
            "target": target,
        }
        old_repetitions = defaultdict(list)
        for row in previous[domain]["repetitions"]:
            if row["condition"] == "stronger":
                old_repetitions[row["path"]].append(add_normalized_detail(row["metrics"]))
        measured[domain]["rejected_expanded"] = [
            {**row, "metrics": audit.median_repetitions(old_repetitions[row["path"]])}
            for row in source
        ]
        print(
            domain,
            "colors",
            color_choice,
            "focus",
            focus_choice,
            "unique fixed candidates",
            len(cache),
            flush=True,
        )
    study["revision_decision_sha256"] = hashlib.sha256(
        json.dumps(decisions, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    write_new(output / "study.json", study)
    write_new(output / "decisions.json", decisions)
    write_new(output / "measurements.json", measured)
    write_new(
        output / "summary.json",
        {
            "groups": {
                d: {name: analysis.statistics(rows) for name, rows in groups.items()}
                for d, groups in measured.items()
            },
            "gaps": {
                d: {
                    metric: {
                        name: audit.distribution_gap(rows, groups["target"], metric)
                        for name, rows in groups.items()
                        if name != "target"
                    }
                    for metric in (*audit.METRICS, "normalized_detail")
                }
                for d, groups in measured.items()
            },
            "training_executed": False,
            "target_accuracy": None,
            "parent_study_sha256": verified["study_sha256"],
            "analysis_code_sha256": data.sha256(Path(__file__)),
        },
    )
    figures(root, output, study, summary)
    return study


def figures(root, output, study, prior_summary):
    sheet = Image.new("RGB", (1620, 1230), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 23)
    draw.text(
        (20, 12), "세 분야 재검토 · 조명 / 이전 확대 / 수정본 / 실제 촬영", fill="black", font=font
    )
    for column, title in enumerate(
        ("조명만 적용", "이전 확대 후보", "재검토 수정본", "실제 P01 기준", "실제 P01 변경")
    ):
        draw.text((20 + column * 320, 60), title, fill="black", font=font)
    old = data.read_json(root / "configs/camera_strength_20261110.json")
    for row_index, domain in enumerate(audit.DOMAINS):
        info = prior_summary["domains"][domain]
        path = info["visual_paths"][0]
        preview = next(p for p in info["source_png_provenance"] if p["path"] == path)
        with Image.open(root / preview["preview"]) as photo:
            original = np.asarray(photo.convert("RGB"), dtype=np.float32)
        selected = study["domains"][domain]["profile"]
        old_profile = old["domains"][domain]["profile"]
        row = data.ImageRow(path, "train")
        seed = exp.item_seed(row, 42)
        panels = [
            camera.transform(original, selected, "lighting", seed),
            camera.transform(original, old_profile, "expanded", seed),
            camera.transform(original, selected, "expanded", seed),
        ]
        cache = root / f"camera_capture/_review/nine_experiments_20261109/{domain}"
        meta = data.read_json(cache / "target_inputs.json")
        if data.sha256(cache / "target_inputs.npz") != meta["array_sha256"]:
            raise ValueError("Target illustration cache changed")
        with np.load(cache / "target_inputs.npz", allow_pickle=False) as archive:
            inputs = archive["images"]
        for target_path in info["visual_paths"][-2:]:
            panels.append(
                inputs[next(i for i, r in enumerate(meta["rows"]) if r["path"] == target_path)]
            )
        y = 105 + row_index * 370
        for column, array in enumerate(panels):
            photo = Image.fromarray(np.rint(array).clip(0, 255).astype(np.uint8))
            sheet.paste(photo.resize((300, 300), Image.Resampling.BILINEAR), (20 + column * 320, y))
        draw.text(
            (20, y + 308), f"{domain} · 실제 촬영의 질환 정답 미확인", fill="black", font=font
        )
    draw.text(
        (20, 1200),
        "변환 방향 점검 · 장비와의 동일함·질환 분류 성능 향상은 미검증",
        fill="black",
        font=font,
    )
    sheet.save(output / "revised_examples.png")
    result = data.read_json(output / "summary.json")
    distribution_figure(result, output)


def distribution_figure(result, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    keys = ("y_p50", "contrast", "lab_a", "lab_b", "detail", "normalized_detail")
    names = ("original", "lighting", "rejected_expanded", "revised", "target")
    colors = ("#777777", "#3882a3", "#b697b5", "#cf6f20", "#279441")
    fig, axes = plt.subplots(3, len(keys), figsize=(19, 11), layout="constrained")
    for row_index, domain in enumerate(audit.DOMAINS):
        groups = result["groups"][domain]
        for column, metric in enumerate(keys):
            axis = axes[row_index, column]
            for y, (name, color) in enumerate(zip(names, colors, strict=True)):
                values = groups[name].get(metric)
                if not values:
                    continue
                axis.plot([values["q05"], values["q95"]], [y, y], color=color, alpha=0.5)
                axis.plot([values["q25"], values["q75"]], [y, y], color=color, linewidth=6)
                axis.scatter([values["q50"]], [y], color=color, s=35, zorder=4)
            axis.set_yticks(range(len(names)), names)
            axis.set_title(f"{domain} / {metric}")
            axis.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "All-domain camera revision · retained Train / unlabeled captures\n"
        "Three-seed per-image medians · dots: median, bars: IQR and 5-95%"
    )
    fig.savefig(output / "distributions.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
