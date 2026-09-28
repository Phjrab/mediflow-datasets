"""MediFlow Web Skin PMG inference helper."""
from pathlib import Path

import keras
import tensorflow as tf


def load_predictor(model_path):
    model = keras.models.load_model(Path(model_path), compile=False)
    if not isinstance(model.outputs, list) or len(model.outputs) != 4:
        raise ValueError("Expected four PMG logit outputs")
    total = keras.layers.Add(name="pmg_logit_sum")(model.outputs)
    probabilities = keras.layers.Activation("softmax", name="predictions")(total)
    return keras.Model(model.input, probabilities, name="web_skin_pmg_inference")


def load_image(image_path):
    content = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(content, channels=3, expand_animations=False)
    image = tf.image.resize(tf.cast(image, tf.float32), (256, 256), method="bilinear")
    return image[None, ...]


def predict_file(predictor, image_path):
    return predictor(load_image(image_path), training=False).numpy()[0]
