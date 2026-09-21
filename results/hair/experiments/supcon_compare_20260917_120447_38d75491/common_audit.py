"""Original Web Skin mechanical audit generalized to three class contracts.

Byte/pixel hashes do not establish person, lesion, session or augmentation lineage.
"""

import hashlib
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from PIL import Image
from tqdm.auto import tqdm

from mediflow_datasets.common_engine import file_hash, write_json


def audit_dataset(extract_root, report_dir, domain, classes, data_hash):
    EXTRACT_ROOT, REPORT_DIR = Path(extract_root), Path(report_dir)
    CLASS_NAMES = classes
    CLASS_CODES = {name: f"C{i}" for i, name in enumerate(classes)}
    DATA_SHA256 = data_hash
    SPLITS = ("train", "val", "test")
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
    TRAIN_LOADER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
    sha256 = file_hash

    def save_json(name, value):
        write_json(REPORT_DIR / name, value)

    def save_csv(name, rows, columns):
        pd.DataFrame(rows, columns=columns).to_csv(
            REPORT_DIR / name, index=False, encoding="utf-8-sig"
        )

    roots, inventory, issues, metadata_files = {}, [], [], []
    for path in EXTRACT_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".csv"}:
            metadata_files.append(str(path.relative_to(EXTRACT_ROOT)))
    for kind in ("original", "augmented"):
        matches = [
            p
            for p in EXTRACT_ROOT.rglob("*")
            if p.is_dir()
            and p.name.lower() == kind
            and all((p / split).is_dir() for split in SPLITS)
        ]
        if len(matches) != 1:
            issues.append(
                {"type": "root_structure", "detail": f"{kind}: {[str(p) for p in matches]}"}
            )
            continue
        root = roots[kind] = matches[0]
        for split in SPLITS:
            classes = sorted(p.name for p in (root / split).iterdir() if p.is_dir())
            if classes != sorted(CLASS_NAMES):
                issues.append({"type": "class_structure", "detail": f"{kind}/{split}: {classes}"})
            stray = [
                p
                for p in (root / split).iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if stray:
                issues.append(
                    {"type": "image_outside_class", "detail": f"{kind}/{split}: {len(stray)}"}
                )
            for cls in sorted(set(classes) | set(CLASS_NAMES)):
                paths = sorted(
                    p
                    for p in (root / split / cls).rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
                if not paths:
                    issues.append({"type": "empty_class", "detail": f"{kind}/{split}/{cls}"})
                for path in tqdm(paths, desc=f"{kind}/{split}/{CLASS_CODES.get(cls, cls)}"):
                    relative_path = str(path.relative_to(EXTRACT_ROOT))
                    row = {"kind": kind, "split": split, "class_name": cls, "path": relative_path}
                    try:
                        row["sha256"] = sha256(path)
                        with Image.open(path) as image:
                            row.update(
                                width=image.width,
                                height=image.height,
                                mode=image.mode,
                                frames=getattr(image, "n_frames", 1),
                            )
                            rgb = image.convert("RGB")
                            row["pixel_sha256"] = hashlib.sha256(
                                str(rgb.size).encode("ascii") + b":" + rgb.tobytes()
                            ).hexdigest()
                        row["readable"] = True
                        if row["frames"] != 1:
                            issues.append({"type": "multi_frame_image", "detail": relative_path})
                    except Exception as exc:
                        row["readable"] = False
                        issues.append(
                            {"type": "unreadable_image", "detail": f"{relative_path}: {exc}"}
                        )
                    if path.suffix.lower() not in TRAIN_LOADER_EXTENSIONS:
                        issues.append(
                            {"type": "training_loader_unsupported", "detail": relative_path}
                        )
                    inventory.append(row)
    df = pd.DataFrame(
        inventory,
        columns=[
            "kind",
            "split",
            "class_name",
            "path",
            "sha256",
            "pixel_sha256",
            "width",
            "height",
            "mode",
            "frames",
            "readable",
        ],
    )
    df.to_csv(REPORT_DIR / "image_inventory.csv", index=False, encoding="utf-8-sig")
    save_json(
        "metadata_inventory.json",
        {
            "files": sorted(metadata_files),
            "status": "파일 목록만 수집. 사람·병변·촬영 세션·증강 출처 대응 관계는 미검증",
        },
    )
    print("조사한 이미지 파일 수:", len(df))

    cross_split_rows, conflict_rows, within_rows, pair_rows, compare_rows = [], [], [], [], []
    for key in ("sha256", "pixel_sha256"):
        valid = df.dropna(subset=[key])
        for digest, group in valid.groupby(key):
            if group["split"].nunique() > 1:
                cross_split_rows.append(
                    {
                        "hash_type": key,
                        "hash": digest,
                        "file_count": len(group),
                        "splits": "|".join(sorted(group["split"].unique())),
                        "paths": "|".join(group["path"]),
                    }
                )
            if group["class_name"].nunique() > 1:
                conflict_rows.append(
                    {
                        "hash_type": key,
                        "hash": digest,
                        "file_count": len(group),
                        "classes": "|".join(sorted(group["class_name"].unique())),
                        "paths": "|".join(group["path"]),
                    }
                )
        for (kind, split, digest), group in valid.groupby(["kind", "split", key]):
            if len(group) > 1:
                within_rows.append(
                    {
                        "kind": kind,
                        "split": split,
                        "hash_type": key,
                        "hash": digest,
                        "file_count": len(group),
                        "paths": "|".join(group["path"]),
                    }
                )
        for kind in ("original", "augmented"):
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                a = set(valid[(valid.kind == kind) & (valid.split == left)][key])
                b = set(valid[(valid.kind == kind) & (valid.split == right)][key])
                pair_rows.append(
                    {
                        "kind": kind,
                        "hash_type": key,
                        "pair": left + "_" + right,
                        "shared_hash_groups": len(a & b),
                    }
                )

    def counted(kind, split, key="sha256"):
        part = df[(df.kind == kind) & (df.split == split)].dropna(subset=[key])
        return Counter(zip(part["class_name"], part[key], strict=True))

    if len(roots) == 2:
        for split in ("val", "test"):
            a, b = counted("original", split), counted("augmented", split)
            compare_rows.append(
                {
                    "check": "evaluation_multiset_equal",
                    "split": split,
                    "passed": a == b,
                    "missing": sum((a - b).values()),
                    "extra": sum((b - a).values()),
                }
            )
            if a != b:
                issues.append({"type": "evaluation_mismatch", "detail": split})
        missing = counted("original", "train") - counted("augmented", "train")
        compare_rows.append(
            {
                "check": "original_train_contained",
                "split": "train",
                "passed": not missing,
                "missing": sum(missing.values()),
                "extra": None,
            }
        )
        if missing:
            issues.append({"type": "missing_train_original", "detail": str(sum(missing.values()))})
    if cross_split_rows:
        issues.append(
            {"type": "cross_split_duplicates", "detail": "cross_split_duplicates.csv 참조"}
        )
    if conflict_rows:
        issues.append({"type": "label_conflicts", "detail": "label_conflicts.csv 참조"})

    save_csv(
        "cross_split_duplicates.csv",
        cross_split_rows,
        ["hash_type", "hash", "file_count", "splits", "paths"],
    )
    save_csv(
        "label_conflicts.csv",
        conflict_rows,
        ["hash_type", "hash", "file_count", "classes", "paths"],
    )
    save_csv(
        "within_split_duplicates.csv",
        within_rows,
        ["kind", "split", "hash_type", "hash", "file_count", "paths"],
    )
    save_csv(
        "split_overlap_counts.csv", pair_rows, ["kind", "hash_type", "pair", "shared_hash_groups"]
    )
    save_csv(
        "dataset_consistency.csv", compare_rows, ["check", "split", "passed", "missing", "extra"]
    )
    save_csv("audit_issues.csv", issues, ["type", "detail"])
    counts = df.groupby(["kind", "split", "class_name"]).size().rename("count").reset_index()
    counts.to_csv(REPORT_DIR / "dataset_counts.csv", index=False, encoding="utf-8-sig")
    resolution = (
        df.groupby(["kind", "split", "width", "height"]).size().rename("count").reset_index()
    )
    resolution.to_csv(REPORT_DIR / "resolution_counts.csv", index=False)
    summary = {
        "domain": domain,
        "data_sha256": DATA_SHA256,
        "class_names": CLASS_NAMES,
        "status": "issues_found" if issues else "mechanical_checks_passed_with_limitations",
        "issue_count": len(issues),
        "image_files": len(df),
        "counts": counts.to_dict("records"),
        "cross_split_groups_by_hash_type": dict(Counter(r["hash_type"] for r in cross_split_rows)),
        "label_conflict_groups_by_hash_type": dict(Counter(r["hash_type"] for r in conflict_rows)),
        "within_split_duplicate_groups_by_hash_type": dict(
            Counter(r["hash_type"] for r in within_rows)
        ),
        "metadata_file_count": len(metadata_files),
        "limitations": [
            "사람·병변·촬영 세션의 이미지 대응 정보 미검증",
            "증강 출처 기록의 연결 미검증; 변환된 파생본은 해시가 다를 수 있음",
            "재압축·밝기·회전·유사 장면 탐지는 미수행",
            "폴더 라벨의 의미상 정확성 및 복수 상태 동시 존재 여부 미검증",
            "이전 학습 당시 데이터 해시가 없어 원 학습 분할 동일성 입증 불가",
            "실제 대상 장비 데이터 검증은 별도",
        ],
        "interpretation": (
            "SHA와 pixel 그룹은 같은 사례가 겹칠 수 있음. "
            "Original/Augmented 복사본도 포함하므로 합산 제거 수로 사용 금지."
        ),
        "protocol": "common_audit_v1",
        "next_step": "보고서 검토 후 정제 또는 기존 모델 재현 여부 결정. 자동 학습하지 않음.",
    }
    save_json("audit_summary.json", summary)
    display(counts)
    display(pd.DataFrame(pair_rows))
    display(pd.DataFrame(compare_rows))
    print("검사 상태:", summary["status"], "문제 항목:", len(issues))
    print("\n".join(summary["limitations"]))

    pairs, seen = [], set()
    for key in ("sha256", "pixel_sha256"):
        for _digest, group in df.dropna(subset=[key]).groupby(key):
            if group["split"].nunique() <= 1 and group["class_name"].nunique() <= 1:
                continue
            readable = group[group["readable"]]
            if readable.empty:
                continue
            left = readable.iloc[0]
            alternatives = readable[
                (readable["split"] != left["split"])
                | (readable["class_name"] != left["class_name"])
            ]
            if alternatives.empty:
                continue
            right = alternatives.iloc[0]
            identity = tuple(sorted([left["path"], right["path"]]))
            if identity in seen:
                continue
            seen.add(identity)
            pairs.append((key, left, right))
            if len(pairs) >= 8:
                break
        if len(pairs) >= 8:
            break
    example_rows = []
    if pairs:
        fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 3.6 * len(pairs)), squeeze=False)
        for index, (key, left, right) in enumerate(pairs):
            example_rows.append(
                {
                    "case": index + 1,
                    "hash_type": key,
                    "left_path": left["path"],
                    "right_path": right["path"],
                }
            )
            for side, row in enumerate((left, right)):
                with Image.open(EXTRACT_ROOT / row["path"]) as source:
                    axes[index, side].imshow(source.convert("RGB"))
                label = CLASS_CODES.get(row["class_name"], "unexpected class")
                axes[index, side].set_title(
                    f"Case {index + 1}: {row['kind']}/{row['split']}/{label}\n{key}", fontsize=9
                )
                axes[index, side].axis("off")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "duplicate_examples.png", dpi=140)
        plt.show()
        plt.close(fig)
    else:
        print("표시할 분할 간 중복/라벨 충돌 사진 쌍이 없습니다.")
    save_csv("example_pairs.csv", example_rows, ["case", "hash_type", "left_path", "right_path"])
    if not counts.empty:
        plot_counts = counts.copy()
        plot_counts["class_name"] = plot_counts["class_name"].map(
            lambda x: CLASS_CODES.get(x, "unexpected")
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, kind in zip(axes, ("original", "augmented"), strict=True):
            part = plot_counts[plot_counts.kind == kind]
            if not part.empty:
                part.pivot_table(
                    index="class_name", columns="split", values="count", aggfunc="sum", fill_value=0
                ).plot.bar(ax=ax)
            ax.set(title=domain + " " + kind, ylabel="Image count", xlabel="Class code")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "dataset_counts.png", dpi=180)
        plt.show()
        plt.close(fig)
    save_json("class_code_mapping.json", CLASS_CODES)
    print(CLASS_CODES)

    return summary
