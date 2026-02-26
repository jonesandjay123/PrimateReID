"""Pluggable multi-backbone embedder for primate re-identification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

BackboneName = Literal["resnet50", "facenet", "arcface", "dinov2"]


class MultiEmbedder:
    """Unified embedding interface supporting multiple backbones.

    Args:
        backbone: Model backbone name ("resnet50" or "facenet").
        device: Torch device string.
    """

    def __init__(self, backbone: str = "resnet50", device: str = "cpu") -> None:
        self.backbone_name = backbone
        self.device = torch.device(device)
        self.model: torch.nn.Module
        self.transform: transforms.Compose

        if backbone == "resnet50":
            self._init_resnet50()
        elif backbone == "facenet":
            self._init_facenet()
        elif backbone == "arcface":
            self._init_arcface()
        elif backbone == "dinov2":
            self._init_dinov2()
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone!r}. "
                "Use 'resnet50', 'facenet', 'arcface', or 'dinov2'."
            )

        self.model.eval()
        self.model.to(self.device)
        logger.info("Loaded backbone=%s on device=%s", backbone, device)

    # ------------------------------------------------------------------
    # Backbone initialisers
    # ------------------------------------------------------------------

    def _init_resnet50(self) -> None:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        # Remove final FC → output is avgpool 2048-d
        model.fc = torch.nn.Identity()
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _init_facenet(self) -> None:
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise ImportError(
                "facenet-pytorch is required for the 'facenet' backbone. "
                "Install it with: pip install facenet-pytorch"
            )
        self.model = InceptionResnetV1(pretrained="vggface2")
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            # facenet_pytorch expects [0,1] float tensors; no extra norm needed
        ])

    def _init_arcface(self) -> None:
        """ArcFace via insightface — direct recognition model only.

        Skips face detection/landmark entirely since our inputs are
        already cropped faces. Feeds directly into w600k_r50.onnx
        recognition head for fairer cross-species comparison.
        """
        try:
            from insightface.model_zoo import get_model
            from insightface.utils.storage import ensure_available
        except ImportError:
            raise ImportError(
                "insightface is required for the 'arcface' backbone. "
                "Install it with: pip install insightface onnxruntime"
            )
        import onnxruntime  # noqa: F401 — ensure available

        # Download buffalo_l if needed, then load ONLY the recognition model
        model_dir = ensure_available("buffalo_l")
        import os
        rec_path = os.path.join(model_dir, "w600k_r50.onnx")
        self._arcface_rec = get_model(rec_path, providers=["CPUExecutionProvider"])
        self._arcface_rec.prepare(ctx_id=-1)

        # Lightweight wrapper so .eval()/.to() don't fail
        self.model = torch.nn.Identity()
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])

    def _init_dinov2(self) -> None:
        """DINOv2 ViT-S/14 from Meta — self-supervised, 384-d embeddings."""
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, image_path: Path) -> np.ndarray:
        """Return L2-normalised embedding vector for a single image."""
        if self.backbone_name == "arcface":
            return self._embed_arcface(image_path)

        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model(tensor).squeeze(0).cpu().numpy()
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec

    def embed_batch(self, image_paths: list[Path]) -> np.ndarray:
        """Return (N, dim) L2-normalised embeddings for a list of images."""
        if self.backbone_name == "arcface":
            return np.stack([self._embed_arcface(p) for p in image_paths])

        tensors = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(self.transform(img))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            vecs = self.model(batch).cpu().numpy()
        # L2 normalise each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms

    # ------------------------------------------------------------------
    # ArcFace helpers
    # ------------------------------------------------------------------

    def _embed_arcface(self, image_path: Path) -> np.ndarray:
        """Get ArcFace embedding directly from recognition model.

        No face detection or landmark alignment — our inputs are already
        cropped faces. This gives a fairer comparison: the recognition
        model sees the same crop quality as other backbones.
        """
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Resize to 112x112 as expected by ArcFace recognition model
        img_resized = cv2.resize(img, (112, 112))

        # insightface rec model's .get() expects a face object with bbox,
        # but .get_feat() accepts a pre-aligned (112,112,3) BGR numpy array
        vec = self._arcface_rec.get_feat(img_resized).flatten()

        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec
