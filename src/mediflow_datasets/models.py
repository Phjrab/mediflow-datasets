from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "results"


@dataclass(frozen=True)
class ModelSpec:
    """Files and intended input domain for one deployed classifier."""

    name: str
    purpose: str
    model_path: Path
    classes_path: Path
    classes_key: str | None = "classes"
    image_size: tuple[int, int] = (224, 224)

    def class_names(self) -> list[str]:
        data = json.loads(self.classes_path.read_text(encoding="utf-8"))
        if self.classes_key is not None:
            values = data[self.classes_key]
            if not isinstance(values, list):
                raise ValueError(f"{self.classes_path}: '{self.classes_key}' must be a list")
            return [str(value) for value in values]

        # Hair class_names.json stores an index-to-name object.
        expected = [str(index) for index in range(len(data))]
        if sorted(data, key=int) != expected:
            raise ValueError(f"{self.classes_path}: class indices must be contiguous from 0")
        return [str(data[index]) for index in expected]


MODEL_VARIANTS: dict[str, dict[str, ModelSpec]] = {
    "skin": {
        variant: ModelSpec(
            name="skin",
            purpose=f"USB 현미경 피부 병변 10-class ({variant})",
            model_path=RESULTS_ROOT / f"skin/{variant}/best_model.keras",
            classes_path=RESULTS_ROOT / f"skin/{variant}/results.json",
        )
        for variant in ("original", "augmented")
    },
    "web_skin": {
        variant: ModelSpec(
            name="web_skin",
            purpose=f"웹캠 얼굴 피부 5-class ({variant})",
            model_path=RESULTS_ROOT / f"web_skin/{variant}/best_model.keras",
            classes_path=RESULTS_ROOT / f"web_skin/{variant}/results.json",
        )
        for variant in ("original", "augmented")
    },
    "hair": {
        variant: ModelSpec(
            name="hair",
            purpose=f"USB 현미경 두피 5-class ({variant})",
            model_path=RESULTS_ROOT / f"hair/{variant}/best_model.keras",
            classes_path=RESULTS_ROOT / f"hair/{variant}/class_names.json",
            classes_key=None,
        )
        for variant in ("original", "augmented")
    },
}

# Augmented models remain the default deployment candidates.
MODEL_SPECS: dict[str, ModelSpec] = {
    model_type: variants["augmented"] for model_type, variants in MODEL_VARIANTS.items()
}


def get_model_spec(model_type: str, variant: str = "augmented") -> ModelSpec:
    try:
        variants = MODEL_VARIANTS[model_type]
    except KeyError as error:
        choices = ", ".join(MODEL_VARIANTS)
        raise ValueError(f"Unknown model type '{model_type}'. Choose one of: {choices}") from error

    try:
        return variants[variant]
    except KeyError as error:
        choices = ", ".join(variants)
        raise ValueError(f"Unknown variant '{variant}'. Choose one of: {choices}") from error
