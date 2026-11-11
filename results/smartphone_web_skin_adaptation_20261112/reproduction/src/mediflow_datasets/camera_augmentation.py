"""Draft combined camera transformations for controlled adaptation experiments."""

from __future__ import annotations

import cv2
import numpy as np

PROFILE_VERSION = "camera-bundle-preview-v2"
DOMAINS = ("web_skin", "skin", "hair")


def _luminance_contrast(image, factor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    shift = (gray - np.median(gray)) * (factor - 1)
    return image + shift[:, :, None]


def _web_skin(image, rng):
    side = image.shape[0]
    face_scale = rng.uniform(0.56, 0.72)
    width_scale = face_scale * (9 / 16)
    center = (side - 1) / 2
    transform = np.array(
        [[width_scale, 0, center * (1 - width_scale)], [0, face_scale, center * (1 - face_scale)]],
        dtype=np.float32,
    )
    corners = image[[0, 0, -1, -1], [0, -1, 0, -1]]
    border = tuple(float(value) for value in np.median(corners, axis=0))
    framed = cv2.warpAffine(image, transform, (side, side), borderValue=border)
    x = np.linspace(0, 1, side, dtype=np.float32)
    edge_distance = np.minimum(x, 1 - x)
    feather = np.clip(np.minimum(edge_distance[:, None], edge_distance[None, :]) / 0.12, 0, 1)
    alpha = cv2.warpAffine(feather, transform, (side, side))[:, :, None]
    framed = framed * alpha + np.asarray(border, dtype=np.float32) * (1 - alpha)
    light_gain = rng.uniform(0.58, 0.80)
    contrast = rng.uniform(1.05, 1.20)
    side_light = rng.uniform(8, 22)
    ramp = np.clip((0.62 - x) / 0.24, 0, 1)
    toned = _luminance_contrast(framed, contrast) * light_gain
    return toned + side_light * ramp[None, :, None], {
        "scale_y": face_scale,
        "scale_x": width_scale,
        "light_gain": light_gain,
        "luminance_contrast_factor": contrast,
        "left_light_max_offset": side_light,
    }


def _skin(image, rng):
    warmth = rng.uniform(0.5, 1.0)
    blur = rng.uniform(0.35, 0.75)
    contrast = rng.uniform(0.94, 0.99)
    colored = image + warmth * np.array([4, 1, -5], dtype=np.float32)
    toned = _luminance_contrast(colored, contrast)
    return cv2.GaussianBlur(toned, (0, 0), blur), {
        "warmth_strength": warmth,
        "blur_sigma_px": blur,
        "luminance_contrast_factor": contrast,
    }


def _hair(image, rng):
    side = image.shape[0]
    blur = rng.uniform(0.35, 0.7)
    peak = rng.uniform(5, 18)
    radius = rng.uniform(0.08, 0.15)
    center_x, center_y = rng.uniform(0.3, 0.7, size=2)
    coords = np.linspace(0, 1, side, dtype=np.float32)
    distance_sq = (coords[None, :] - center_x) ** 2 + (coords[:, None] - center_y) ** 2
    local_light = peak * np.exp(-distance_sq / (2 * radius**2))
    return cv2.GaussianBlur(image, (0, 0), blur) + local_light[:, :, None], {
        "blur_sigma_px": blur,
        "local_light_max_offset": peak,
        "local_light_radius_fraction": radius,
        "local_light_center_x": center_x,
        "local_light_center_y": center_y,
    }


def apply_camera_bundle(
    image: np.ndarray, domain: str, *, seed: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Transform one square RGB image in the saved models' 0–255 input range.

    Pass pixels after the model's existing resize, and keep the same resize in both
    arms. The caller controls which split and images receive this draft profile.
    """
    transforms = {"web_skin": _web_skin, "skin": _skin, "hair": _hair}
    if domain not in transforms:
        raise ValueError(f"Unknown camera domain: {domain}")
    pixels = np.asarray(image, dtype=np.float32)
    if (
        pixels.ndim != 3
        or pixels.shape[2] != 3
        or pixels.shape[0] != pixels.shape[1]
        or pixels.shape[0] < 32
    ):
        raise ValueError("Expected a square RGB image of at least 32 pixels per side")
    if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 255:
        raise ValueError("Expected finite RGB pixels in the 0–255 input range")
    transformed, settings = transforms[domain](pixels, np.random.default_rng(seed))
    return np.clip(transformed, 0, 255).astype(np.float32), {
        name: float(value) for name, value in settings.items()
    }
