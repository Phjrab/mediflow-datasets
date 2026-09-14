"""Rebuild Hair figures from the preserved comparison workbook and curve panels."""

import csv
import hashlib
import json

import openpyxl

from mediflow_datasets.presentation_charts import LABELS, ROOT, render


def main():
    workbook_path = ROOT / "results/hair/hair_experiment_comparison_20260907.xlsx"
    workbook = openpyxl.load_workbook(workbook_path, data_only=True)
    training = [r for r in workbook.worksheets[0].values if r[2] == "Training"]
    f1_rows = list(workbook.worksheets[1].values)[6:12]
    assert len(training) == len(f1_rows) == 6
    sources = {}

    def track(path):
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()

    track(workbook_path)
    rows = []
    for i, (key, row, f1) in enumerate(zip(list(LABELS)[:6], training, f1_rows, strict=True)):
        assert row[11] == f1[6]
        curve = ROOT / f"results/hair/_dashboard_parts/curve_{i + 1}.png"
        track(curve)
        rows.append(
            {
                "key": key,
                "histories": [],
                "curve_image": curve.relative_to(ROOT).as_posix(),
                "metrics": {
                    "accuracy": row[9],
                    "test_accuracy": row[10],
                    "macro_f1": row[11],
                    "class_f1": list(f1[1:6]),
                },
            }
        )
    index_path = ROOT / "results/CANDIDATE_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    candidate = (ROOT / "results" / index["candidates"]["hair"]["model"]).parent
    report_path = candidate / "source_classification_report.csv"
    track(report_path)
    with report_path.open(encoding="utf-8-sig", newline="") as stream:
        report = list(csv.reader(stream))
    assert [float(r[3]) for r in report[1:6]] == rows[-1]["metrics"]["class_f1"]
    final = {
        "accuracy": float(report[6][1]),
        "macro_f1": float(report[7][3]),
        "count": int(float(report[7][4])),
        "class_f1": rows[-1]["metrics"]["class_f1"],
    }
    assert final["accuracy"] == rows[-1]["metrics"]["test_accuracy"]
    assert final["macro_f1"] == rows[-1]["metrics"]["macro_f1"]
    render(
        "hair",
        "",
        {
            "experiments": rows,
            "sources": sources,
            "test": final,
            "selection": {"winner": "b1_256_ls005_extend5", "criterion": "validation_accuracy"},
        },
    )


if __name__ == "__main__":
    main()
