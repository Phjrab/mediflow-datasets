from __future__ import annotations

import numpy as np
from PIL import Image

from mediflow_datasets.inference import preprocess_image
from mediflow_datasets.models import MODEL_SPECS


def test_preprocess_uses_training_pixel_range(tmp_path):
    path = tmp_path / "white.png"
    Image.new("RGB", (20, 10), color=(255, 128, 0)).save(path)

    tensor = preprocess_image(path, MODEL_SPECS["skin"])

    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0].tolist() == [255.0, 128.0, 0.0]
    assert float(tensor.max()) == 255.0
