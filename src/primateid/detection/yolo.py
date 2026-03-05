"""YOLOv8 detection backend for primate face/body detection.

Requires: pip install ultralytics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from primateid.detection import Detection

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class YOLODetector:
    """Detect primate faces/bodies using YOLOv8.

    Uses a general-purpose YOLOv8 model. For best results on primates,
    fine-tune on a primate face dataset and pass the weights path.

    Args:
        model_path: Path to YOLO weights. Default uses yolov8n.pt (nano).
        conf_threshold: Minimum confidence score. Default 0.25.
        device: Torch device string. Default 'cpu'.
        target_classes: COCO class IDs to keep. Default None keeps all.
            For animals in COCO: [14]=bird, [15]=cat, [16]=dog, [17]=horse,
            [18]=sheep, [19]=cow, [20]=elephant, [21]=bear, [22]=zebra, [23]=giraffe.
            There's no 'primate' class in COCO — use a fine-tuned model or keep all.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        device: str = "cpu",
        target_classes: list[int] | None = None,
    ):
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. Install with:\n"
                "  pip install ultralytics\n"
                "Or install all detection deps:\n"
                "  pip install -r requirements-detection.txt"
            )
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.target_classes = target_classes

    def detect(self, image_path: str | Path) -> List[Detection]:
        """Run detection on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of Detection objects with bboxes and confidence scores.
        """
        image_path = Path(image_path)
        results = self.model(
            str(image_path),
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                if self.target_classes and cls_id not in self.target_classes:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i])
                label = result.names.get(cls_id, "unknown")
                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    label=label,
                ))

        logger.info(f"YOLO: {len(detections)} detections in {image_path.name}")
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
