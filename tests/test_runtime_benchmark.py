import hashlib
import io
import json
import zipfile

import mediflow_datasets.runtime_benchmark as benchmark
from mediflow_datasets.runtime_benchmark import (
    EXPECTED_COUNTS,
    _latency,
    _legacy_rows,
    cohort_for,
    discover,
)


def test_hair_class_versions_remain_separate(tmp_path):
    source = tmp_path / "2_results" / "hair" / "run" / "completed.json"
    assert cohort_for(source, {"spec": {"class_count": 5}}) == "hair5"
    assert cohort_for(source, {"spec": {"class_count": 6}}) == "hair6"


def test_discover_reports_missing_checkpoint_without_fabricating_latency(tmp_path):
    folder = tmp_path / "2_results" / "skin" / "comparison_20260909_075056_72d865bf" / "original"
    folder.mkdir(parents=True)
    (folder / "completed.json").write_text(
        json.dumps(
            {
                "id": "original",
                "attempt": "attempt_1",
                "selected_model": "stage1_best.keras",
                "spec": {"size": 224, "class_count": 10},
                "validation": {"accuracy": 0.9, "macro_f1": 0.8},
                "training_seconds": 100,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        row for row in discover(tmp_path, tmp_path / "cache") if row["experiment"] == "original"
    ]
    assert len(rows) == 1
    assert rows[0]["cohort"] == "skin"
    assert rows[0]["status"] == "checkpoint_missing"
    assert "end_to_end_p50_ms" not in rows[0]


def test_legacy_zip_extracts_only_two_best_models_and_reported_metrics(tmp_path, monkeypatch):
    digests = {
        ("skin", variant): hashlib.sha256(variant.encode()).hexdigest()
        for variant in ("original", "augmented")
    }
    monkeypatch.setattr(benchmark, "LEGACY_MODEL_SHA256", digests)
    archive = tmp_path / "1_results" / "skin_dataset_results"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as stream:
        for variant in ("original", "augmented"):
            stream.writestr(f"skin/{variant}/best_model.keras", variant.encode())
            stream.writestr(
                f"skin/{variant}/results.json",
                json.dumps({"best_val_accuracy": 0.9}),
            )
        stream.writestr("skin/original/unused.txt", "not a model")
    rows = _legacy_rows(tmp_path, tmp_path / "cache")
    skin = [row for row in rows if row["cohort"] == "skin"]
    assert len(skin) == 2
    assert all(row["status"] == "ready" for row in skin)
    assert all(row["validation_accuracy"] == 0.9 for row in skin)
    assert {
        (tmp_path / "cache" / "skin" / f"{name}.keras").read_bytes()
        for name in ("original", "augmented")
    } == {b"original", b"augmented"}


def test_latency_quantiles_use_recorded_samples():
    assert _latency([1, 2, 3, 4, 5]) == (3, 5, 3)


def test_legacy_zip_can_contain_another_results_zip(tmp_path, monkeypatch):
    digests = {
        ("skin", variant): hashlib.sha256(variant.encode()).hexdigest()
        for variant in ("original", "augmented")
    }
    monkeypatch.setattr(benchmark, "LEGACY_MODEL_SHA256", digests)
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as stream:
        stream.writestr("result/Original/best_model.keras", b"original")
        stream.writestr("result/Augmented/best_model.keras", b"augmented")
    outer = tmp_path / "1_results" / "skin_dataset_results"
    outer.parent.mkdir()
    with zipfile.ZipFile(outer, "w") as stream:
        stream.writestr("skin_model_results.zip", inner.getvalue())
    rows = _legacy_rows(tmp_path, tmp_path / "cache")
    assert [row["status"] for row in rows[:2]] == ["ready", "ready"]


def test_legacy_zip_uses_hash_when_names_are_missing_or_ambiguous(tmp_path, monkeypatch):
    digests = {
        ("skin", variant): hashlib.sha256(f"skin:{variant}".encode()).hexdigest()
        for variant in ("original", "augmented")
    }
    digests.update(
        {
            ("web_skin", variant): hashlib.sha256(f"web_skin:{variant}".encode()).hexdigest()
            for variant in ("original", "augmented")
        }
    )
    monkeypatch.setattr(benchmark, "LEGACY_MODEL_SHA256", digests)
    archive_folder = tmp_path / "1_results"
    archive_folder.mkdir()
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as stream:
        for variant in ("original", "augmented"):
            stream.writestr(f"unnamed/model_{variant[0]}.keras", f"skin:{variant}")
            stream.writestr(
                f"unnamed/report_{variant[0]}/results.json",
                json.dumps({"dataset": variant, "best_val_accuracy": 0.8}),
            )
    with zipfile.ZipFile(archive_folder / "skin_dataset_results", "w") as stream:
        stream.writestr("skin_model_results.zip", inner.getvalue())
    with zipfile.ZipFile(archive_folder / "web_skin_dataset_results", "w") as stream:
        for variant in ("original", "augmented"):
            stream.writestr(f"web_skin/{variant}/best_model.keras", f"web_skin:{variant}")
            stream.writestr(f"duplicate/{variant}/best_model.keras", b"wrong model")
    rows = _legacy_rows(tmp_path, tmp_path / "cache")
    selected = [row for row in rows if row["cohort"] in ("skin", "web_skin")]
    assert len(selected) == 4
    assert all(row["status"] == "ready" for row in selected)
    assert [row["validation_accuracy"] for row in selected[:2]] == [0.8, 0.8]
    for row in selected:
        variant = row["experiment"].rsplit("_", 1)[-1]
        assert (tmp_path / "cache" / row["cohort"] / f"{variant}.keras").read_bytes() == (
            f"{row['cohort']}:{variant}".encode()
        )


def test_known_drive_layout_has_exactly_34_primary_conditions(tmp_path, monkeypatch):
    digests = {
        (cohort, variant): hashlib.sha256(f"{cohort}:{variant}".encode()).hexdigest()
        for cohort in ("skin", "web_skin", "hair5")
        for variant in ("original", "augmented")
    }
    monkeypatch.setattr(benchmark, "LEGACY_MODEL_SHA256", digests)
    direct = {
        ("skin", "comparison_20260909_075056_72d865bf"): ["original", "augmented"],
        ("web_skin", "suite_20260908_014452_72768a42"): [
            "b0_224_ce",
            "b0_256_ce",
            "b0_256_ls005",
            "b0_256_focal15",
            "b1_256_ls005",
        ],
        ("web_skin", "web_skin_paper_suite_20260922_124758_717b465e"): [
            "wsdan_b0_256_ce_seed_42",
            "pmg_b0_256_ce_seed_42",
            "mixstyle_b0_256_ce_seed_42",
        ],
        ("web_skin", "web_skin_pmg_b1_384_20260922_144631_ac7e5c8b"): ["pmg_b1_384_ce_seed_42"],
        ("hair", "supcon_compare_20260917_120447_38d75491"): [
            "baseline_b1_256_seed_42",
            "supcon_b1_256_seed_42",
        ],
        ("hair", "paper_screen_20260917_141641_874f03f1"): [
            "dinov2_small_224_seed_42",
            "efficientnetv2s_256_seed_42",
            "multires_b1_384_seed_42",
        ],
        ("hair", "sam_screen_20260921_145602_a5e8c406"): ["sam_b1_384_seed_42"],
        ("hair", "hair_six_class_four_20260928_011824_ad21e8ae"): [
            "b0_256_original",
            "b0_256_augmented",
            "b1_384_original",
            "b1_384_augmented",
        ],
    }
    for (domain, run), experiments in direct.items():
        for name in experiments:
            folder = tmp_path / "2_results" / domain / run / name
            attempt = folder / "attempt_test"
            attempt.mkdir(parents=True)
            (attempt / "best.keras").write_bytes(b"checkpoint")
            count = 6 if run.startswith("hair_six_class") else 10 if domain == "skin" else 5
            (folder / "completed.json").write_text(
                json.dumps(
                    {
                        "id": name,
                        "attempt": "attempt_test",
                        "selected_model": "best.keras",
                        "spec": {"class_count": count, "size": 224},
                        "validation": {"accuracy": 0.5, "macro_f1": 0.5},
                    }
                ),
                encoding="utf-8",
            )
    for domain, run in (
        ("web_skin", "web_skin_medsiglip_linear_20260924_072442_f7176e07"),
        ("hair", "hair_medsiglip_linear_20260924_081708_9479fbb4"),
    ):
        folder = tmp_path / "2_results" / domain / run
        folder.mkdir(parents=True)
        (folder / "medsiglip_linear_head.pt").write_bytes(b"head")
        (folder / "medsiglip_linear_completed.json").write_text(
            json.dumps({"validation": {"class_f1": [0.5] * 5}}), encoding="utf-8"
        )
    for name in (
        "clean_two_stage_20260907_061005",
        "clean_256_two_stage_20260907_064214",
        "clean_256_label_smoothing_005_20260907_103757",
        "clean_256_focal_gamma_15_20260907_105525",
        "clean_256_efficientnetb1_ls005_20260907_111711",
    ):
        folder = tmp_path / "2_results" / "hair" / name
        folder.mkdir(parents=True)
        (folder / "best_model.keras").write_bytes(b"checkpoint")
        (folder / "training_config.json").write_text(
            json.dumps({"image_size": [256, 256]}), encoding="utf-8"
        )
    archive_folder = tmp_path / "1_results"
    archive_folder.mkdir()
    for domain in ("skin", "web_skin", "hair"):
        with zipfile.ZipFile(archive_folder / f"{domain}_dataset_results", "w") as stream:
            for variant in ("original", "augmented"):
                cohort = "hair5" if domain == "hair" else domain
                stream.writestr(
                    f"{domain}/{variant}/best_model.keras", f"{cohort}:{variant}".encode()
                )
    rows = discover(tmp_path, tmp_path / "cache")
    counts = {cohort: sum(row["cohort"] == cohort for row in rows) for cohort in EXPECTED_COUNTS}
    assert counts == EXPECTED_COUNTS
    assert len(rows) == 34
    assert all(row["status"] == "ready" for row in rows)
