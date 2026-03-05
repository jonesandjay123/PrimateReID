"""SAM3 (Segment Anything Model 2.1) detection backend for primate detection.

SAM3 is Meta's Segment Anything Model — it generates high-quality masks
for objects in images. We use automatic mask generation mode to find
all segments, then filter by size/score for likely primate regions.

Requires: pip install sam-2
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from primateid.detection import Detection

logger = logging.getLogger(__name__)

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False


class SAM3Detector:
    """Detect primate regions using SAM 2.1 automatic mask generation.

    SAM doesn't produce class labels — it segments *everything*. We filter
    masks by area and stability score to find likely primate face/body regions.

    Args:
        model_cfg: SAM2 model config name. Default 'sam2.1_hiera_small'.
        checkpoint: Path to SAM2 checkpoint file. Required if SAM3 is available.
        device: Torch device string. Default 'cpu'.
        min_area_ratio: Minimum mask area as fraction of image. Default 0.01.
        max_area_ratio: Maximum mask area as fraction of image. Default 0.8.
        stability_threshold: Minimum stability score. Default 0.9.
        points_per_side: Grid density for auto mask gen. Default 32.
    """

    def __init__(
        self,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.8,
        stability_threshold: float = 0.9,
        points_per_side: int = 32,
    ):
        if not SAM3_AVAILABLE:
            raise ImportError(
                "sam-2 is not installed. Install with:\n"
                "  pip install sam-2\n"
                "Or install all detection deps:\n"
                "  pip install -r requirements-detection.txt"
            )
        if checkpoint is None:
            raise ValueError(
                "SAM3 requires a checkpoint path. Download from:\n"
                "  https://github.com/facebookresearch/sam2#checkpoints\n"
                "Then pass --sam3-checkpoint <path>"
            )
        import torch
        self.device = device
        self.sam_model = build_sam2(model_cfg, checkpoint, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=points_per_side,
            stability_score_thresh=stability_threshold,
        )
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def detect(self, image_path: str | Path) -> List[Detection]:
        """Run SAM3 automatic mask generation on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of Detection objects with bboxes, confidence, and masks.
        """
        image_path = Path(image_path)
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        total_area = h * w

        masks = self.mask_generator.generate(image)

        detections = []
        for mask_data in masks:
            area = mask_data["area"]
            ratio = area / total_area
            if ratio < self.min_area_ratio or ratio > self.max_area_ratio:
                continue

            # bbox from SAM is [x, y, w, h] — convert to (x1, y1, x2, y2)
            bx, by, bw, bh = mask_data["bbox"]
            bbox = (int(bx), int(by), int(bx + bw), int(by + bh))
            score = mask_data.get("predicted_iou", mask_data.get("stability_score", 1.0))
            binary_mask = mask_data["segmentation"].astype(np.uint8)  # HxW

            detections.append(Detection(
                bbox=bbox,
                confidence=float(score),
                label="segment",
                mask=binary_mask,
            ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.info(f"SAM3: {len(detections)} segments in {image_path.name}")
        return detections

    def detect_batch(self, image_paths: list[str | Path]) -> dict[str, List[Detection]]:
        """Run detection on multiple images.

        Args:
            image_paths: List of image file paths.

        Returns:
            Dict mapping filename to list of Detection objects.
        """
        all_detections = {}
        for p in image_paths:
            p = Path(p)
            all_detections[p.name] = self.detect(p)
        return all_detections
