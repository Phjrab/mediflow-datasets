import csv
import json

import pytest

from mediflow_datasets.reproduction_audit import check_prediction_record


def test_prediction_audit_recomputes_metrics_and_detects_tampering(tmp_path):
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "true_index", "pred_index", "prob_C0", "prob_C1"])
        writer.writerow(["original/test/first/a.png", 0, 0, 0.9, 0.1])
        writer.writerow(["original/test/second/b.png", 1, 0, 0.8, 0.2])
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "accuracy": 0.5,
                "macro_f1": 1 / 3,
                "class_f1": [2 / 3, 0.0],
                "precision": [0.5, 0.0],
                "recall": [1.0, 0.0],
                "support": [1, 1],
                "confusion_matrix": [[1, 0], [1, 0]],
                "count": 2,
            }
        ),
        encoding="utf-8",
    )

    result = check_prediction_record(predictions, metrics, ["first", "second"])
    assert result["accuracy"] == 0.5
    assert result["confusion_matrix"] == [[1, 0], [1, 0]]

    metrics.write_text(metrics.read_text(encoding="utf-8").replace("0.5", "0.6", 1))
    with pytest.raises(ValueError, match="Reported accuracy differs"):
        check_prediction_record(predictions, metrics, ["first", "second"])
