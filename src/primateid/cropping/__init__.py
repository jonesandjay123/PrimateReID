"""Cropping module for extracting primate regions from raw photos.

Supports two modes:
- box: Simple bounding box crop
- mask: SAM3 mask-based crop (background removed)
"""

from __future__ import annotations

from primateid.cropping.cropper import BoxCropper, MaskCropper, get_cropper

__all__ = ["BoxCropper", "MaskCropper", "get_cropper"]
