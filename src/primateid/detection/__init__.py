"""Detection backends for primate face/body detection in raw photos.

Supported backends:
- YOLO (YOLOv8 via ultralytics)
- SAM3 (Segment Anything Model 2.1 / SAM3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection:
    """A single detection result."""
    bbox: tuple  # (x1, y1, x2, y2) ints
    confidence: float = 1.0
    label: str = "primate"
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # HxW binary mask


def get_detector(backend: str = "yolo", **kwargs):
    """Factory function to get a detector by name.

    Args:
        backend: 'yolo' or 'sam3'
        **kwargs: passed to detector constructor

    Returns:
        A detector instance with a .detect(image_path) method.
    """
    if backend == "yolo":
        from primateid.detection.yolo import YOLODetector
        return YOLODetector(**kwargs)
    elif backend == "sam3":
        from primateid.detection.sam3 import SAM3Detector
        return SAM3Detector(**kwargs)
    else:
        raise ValueError(f"Unknown detection backend: {backend!r}. Choose 'yolo' or 'sam3'.")
