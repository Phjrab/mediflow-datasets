from __future__ import annotations

import pytest

from mediflow_datasets.inference import load_model
from mediflow_datasets.models import MODEL_VARIANTS

pytestmark = pytest.mark.model


@pytest.mark.parametrize(
    ("model_type", "variant"),
    [
        (model_type, variant)
        for model_type, variants in MODEL_VARIANTS.items()
        for variant in variants
    ],
)
def test_saved_model_loads_and_has_expected_output(model_type, variant):
    pytest.importorskip("tensorflow")
    spec = MODEL_VARIANTS[model_type][variant]
    model = load_model(spec)
    assert tuple(model.input_shape[1:]) == (224, 224, 3)
    assert int(model.output_shape[-1]) == len(spec.class_names())
