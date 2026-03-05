"""Cropping implementations for detected primate regions.

Two modes:
- BoxCropper: Simple rectangular bounding box extraction.
- MaskCropper: Uses SAM3 segmentation mask to isolate the subject,
  removing background (filled with black or transparent).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from primateid.detection import Detection

logger = logging.getLogger(__name__)


class BoxCropper:
    """Crop detected regions using bounding boxes.

    Args:
        padding: Fractional padding around bbox. 0.1 = 10% on each side.
        min_size: Minimum crop dimension in pixels. Smaller crops are skipped.
    """

    def __init__(self, padding: float = 0.1, min_size: int = 20):
        self.padding = padding
        self.min_size = min_size

    def crop(
        self,
        image_path: str | Path,
        detections: List[Detection],
        output_dir: Path,
        prefix: str = "",
    ) -> List[Path]:
        """Crop all detections from an image and save to output_dir.

        Args:
            image_path: Source image path.
            detections: List of Detection objects.
            output_dir: Directory to save cropped images.
            prefix: Filename prefix for crops.

        Returns:
            List of paths to saved crop files.
        """
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            bw, bh = x2 - x1, y2 - y1

            if bw < self.min_size or bh < self.min_size:
                continue

            # Apply padding
            pad_x = int(bw * self.padding)
            pad_y = int(bh * self.padding)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            crop = image.crop((x1, y1, x2, y2))
            stem = Path(image_path).stem
            fname = f"{prefix}{stem}_crop{i:03d}.jpg"
            out_path = output_dir / fname
            crop.save(out_path, quality=95)
            saved.append(out_path)

        logger.info(f"BoxCrop: saved {len(saved)} crops from {Path(image_path).name}")
        return saved


class MaskCropper:
    """Crop detected regions using segmentation masks.

    Applies the mask to isolate the subject, removing background.
    The crop is then tightly bounded to the masked region.

    Args:
        padding: Fractional padding around mask bbox. Default 0.05.
        bg_color: Background fill color (R, G, B). Default (0, 0, 0) black.
        min_size: Minimum crop dimension in pixels. Default 20.
    """

    def __init__(
        self,
        padding: float = 0.05,
        bg_color: tuple = (0, 0, 0),
        min_size: int = 20,
    ):
        self.padding = padding
        self.bg_color = bg_color
        self.min_size = min_size

    def crop(
        self,
        image_path: str | Path,
        detections: List[Detection],
        output_dir: Path,
        prefix: str = "",
    ) -> List[Path]:
        """Crop detections using masks. Falls back to box crop if no mask.

        Args:
            image_path: Source image path.
            detections: List of Detection objects (should have .mask set).
            output_dir: Directory to save cropped images.
            prefix: Filename prefix for crops.

        Returns:
            List of paths to saved crop files.
        """
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)
        w, h = image.size
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        box_fallback = BoxCropper(padding=self.padding, min_size=self.min_size)

        for i, det in enumerate(detections):
            if det.mask is None:
                # Fall back to box crop for detections without masks
                logger.debug(f"No mask for detection {i}, falling back to box crop")
                fb = box_fallback.crop(image_path, [det], output_dir, prefix=f"{prefix}nomask_")
                saved.extend(fb)
                continue

            mask = det.mask  # HxW binary
            # Apply mask: keep subject, fill background
            masked = img_array.copy()
            masked[mask == 0] = self.bg_color

            # Find tight bbox around mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                continue
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            bw, bh = x2 - x1, y2 - y1
            if bw < self.min_size or bh < self.min_size:
                continue

            # Add padding
            pad_x = int(bw * self.padding)
            pad_y = int(bh * self.padding)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x + 1)
            y2 = min(h, y2 + pad_y + 1)

            crop = Image.fromarray(masked[y1:y2, x1:x2])
            stem = Path(image_path).stem
            fname = f"{prefix}{stem}_mask{i:03d}.jpg"
            out_path = output_dir / fname
            crop.save(out_path, quality=95)
            saved.append(out_path)

        logger.info(f"MaskCrop: saved {len(saved)} crops from {Path(image_path).name}")
        return saved


def get_cropper(mode: str = "box", **kwargs):
    """Factory function to get a cropper by mode.

    Args:
        mode: 'box' or 'mask'
        **kwargs: passed to cropper constructor

    Returns:
        A cropper instance with a .crop() method.
    """
    if mode == "box":
        return BoxCropper(**kwargs)
    elif mode == "mask":
        return MaskCropper(**kwargs)
    else:
        raise ValueError(f"Unknown crop mode: {mode!r}. Choose 'box' or 'mask'.")
