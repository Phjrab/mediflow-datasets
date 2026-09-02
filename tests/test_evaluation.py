import numpy as np

from mediflow_datasets.evaluation import classification_metrics


def test_classification_metrics_uses_model_class_order():
    result = classification_metrics(
        labels=np.asarray([0, 0, 1, 1]),
        predictions=np.asarray([0, 1, 1, 1]),
        class_names=["first", "second"],
    )

    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["per_class"]["first"]["recall"] == 0.5
    assert result["per_class"]["second"]["recall"] == 1.0
