"""Tests for dominant color classification."""

import numpy as np

from warehouseeye.tracking.color_classifier import classify_color


def test_classify_orange_vest() -> None:
    orange = np.full((40, 40, 3), [255, 140, 0], dtype=np.uint8)
    tag = classify_color(orange)
    assert tag == "orange_vest"

