from __future__ import annotations

import json
import zipfile

import pytest

from mediflow_datasets.models import MODEL_SPECS, MODEL_VARIANTS

EXPECTED_CLASSES = {
    "skin": [
        "광선각화증",
        "기저세포암",
        "보웬병",
        "사마귀",
        "지루각화증",
        "편평세포암",
        "표피낭종",
        "피부섬유종",
        "혈관종",
        "흑색점",
    ],
    "web_skin": ["건선", "아토피", "여드름", "정상", "주사"],
    "hair": ["모낭사이홍반", "미세각질", "비듬", "탈모", "피지과다"],
}


def _keras_config(model_path):
    with zipfile.ZipFile(model_path) as archive:
        return json.loads(archive.read("config.json"))


@pytest.mark.parametrize("model_type", MODEL_SPECS)
def test_class_order_matches_result_json(model_type):
    assert MODEL_SPECS[model_type].class_names() == EXPECTED_CLASSES[model_type]


@pytest.mark.parametrize("model_type", MODEL_SPECS)
def test_model_archive_output_matches_class_count(model_type):
    spec = MODEL_SPECS[model_type]
    config = _keras_config(spec.model_path)
    layers = config["config"]["layers"]
    output_layer_name = config["config"]["output_layers"][0]
    output_layer = next(layer for layer in layers if layer["name"] == output_layer_name)
    assert output_layer["class_name"] == "Dense"
    assert output_layer["config"]["units"] == len(spec.class_names())


@pytest.mark.parametrize("model_type", MODEL_SPECS)
def test_saved_efficientnet_contains_internal_rescaling(model_type):
    text = json.dumps(_keras_config(MODEL_SPECS[model_type].model_path))
    assert '"class_name": "Rescaling"' in text
    assert '"scale": 0.00392156862745098' in text


@pytest.mark.parametrize(
    ("model_type", "variant"),
    [
        (model_type, variant)
        for model_type, variants in MODEL_VARIANTS.items()
        for variant in variants
    ],
)
def test_all_model_variants_have_matching_class_order(model_type, variant):
    assert MODEL_VARIANTS[model_type][variant].class_names() == EXPECTED_CLASSES[model_type]
