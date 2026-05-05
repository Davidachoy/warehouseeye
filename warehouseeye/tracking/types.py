"""Shared tracking data models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Single person detection in xyxy format."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

