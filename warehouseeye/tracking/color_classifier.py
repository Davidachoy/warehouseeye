"""Dominant clothing color classification using HSV + KMeans."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def _to_rgb_array(crop_image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(crop_image, Image.Image):
        return np.asarray(crop_image.convert("RGB"))
    if crop_image.ndim == 3 and crop_image.shape[2] == 3:
        return crop_image
    raise ValueError("Expected RGB PIL image or HxWx3 numpy array.")


def _tag_from_hsv_center(center: np.ndarray) -> str:
    """Map HSV centroid to clothing-style tag."""
    color_name = _hsv_to_name(float(center[0]), float(center[1]), float(center[2]))
    if color_name == "orange":
        return "orange_vest"
    if color_name == "black":
        return "black_hoodie"
    if color_name == "unknown":
        return "unknown"
    return f"{color_name}_top"


def _hsv_to_name(h: float, s: float, v: float) -> str:
    if v < 40:
        return "black"
    if s < 35 and v > 190:
        return "white"
    if h < 10 or h >= 170:
        return "red"
    if 10 <= h < 28:
        return "orange"
    if 28 <= h < 40:
        return "yellow"
    if 40 <= h < 85:
        return "green"
    if 85 <= h < 130:
        return "blue"
    return "unknown"


def classify_color(crop_image: Image.Image | np.ndarray) -> str:
    """Classify dominant torso color into a human-friendly clothing tag."""
    rgb = _to_rgb_array(crop_image)
    if rgb.size == 0:
        return "unknown"
    torso = rgb[: max(1, rgb.shape[0] // 2), :, :]
    hsv = cv2.cvtColor(torso, cv2.COLOR_RGB2HSV)
    pixels = hsv.reshape((-1, 3)).astype(np.float32)
    if len(pixels) < 3:
        return "unknown"

    # Solid-color crops: per-channel spread ~0 (axis=0); global ptp mixes H/S/V scales.
    if float(np.max(np.ptp(pixels, axis=0))) < 1.0:
        return _tag_from_hsv_center(pixels.mean(axis=0))

    model = KMeans(n_clusters=3, n_init="auto", random_state=42)
    labels = model.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant_center = model.cluster_centers_[int(np.argmax(counts))]
    return _tag_from_hsv_center(dominant_center)


def main() -> None:
    """Simple standalone test entrypoint."""
    import logging

    logging.basicConfig(level=logging.INFO)
    dummy = np.full((20, 20, 3), [255, 128, 0], dtype=np.uint8)
    logging.getLogger(__name__).info("classified_color", extra={"tag": classify_color(dummy)})


if __name__ == "__main__":
    main()

