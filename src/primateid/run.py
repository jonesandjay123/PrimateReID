"""CLI entry point for PrimateReID evaluation pipeline.

Usage:
    # Pre-cropped images (existing workflow):
    python -m primateid.run --crops data/sample_crops --backbone resnet50

    # Full pipeline from raw photos:
    python -m primateid.run --input data/raw_photos --detector yolo --crop-mode box --backbone dinov2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def run_detection_and_cropping(
    input_dir: Path,
    detector_name: str,
    crop_mode: str,
    output_crops_dir: Path,
    device: str = "cpu",
    **detector_kwargs,
) -> Path:
    """Run detection + cropping pipeline on raw photos.

    Args:
        input_dir: Directory with raw photos (flat or nested).
        detector_name: 'yolo' or 'sam3'.
        crop_mode: 'box' or 'mask'.
        output_crops_dir: Where to save cropped images (organized by source).
        device: Torch device.
        **detector_kwargs: Extra kwargs for the detector.

    Returns:
        Path to the crops directory (ready for embedding).
    """
    from primateid.detection import get_detector
    from primateid.cropping import get_cropper

    logging.info(f"Detection backend: {detector_name}")
    logging.info(f"Crop mode: {crop_mode}")

    detector = get_detector(detector_name, device=device, **detector_kwargs)
    cropper = get_cropper(crop_mode)

    # Collect all images
    image_files = sorted(
        p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_files:
        print(f"Error: no images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    logging.info(f"Found {len(image_files)} images in {input_dir}")

    # For each image, detect and crop. Use parent folder name as identity
    # if images are organized in subfolders; otherwise use 'unknown'.
    total_crops = 0
    for img_path in image_files:
        rel = img_path.relative_to(input_dir)
        if len(rel.parts) > 1:
            identity = rel.parts[0]
        else:
            identity = "unknown"

        identity_dir = output_crops_dir / identity
        detections = detector.detect(img_path)
        if not detections:
            logging.warning(f"No detections in {img_path.name}")
            continue

        crops = cropper.crop(img_path, detections, identity_dir)
        total_crops += len(crops)

    logging.info(f"Total crops generated: {total_crops}")
    if total_crops == 0:
        print("Error: no crops were generated. Check your detector settings.", file=sys.stderr)
        sys.exit(1)

    return output_crops_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PrimateReID — primate face detection, cropping, and re-identification pipeline"
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--crops", type=Path,
        help="Path to pre-cropped images directory (skip detection/cropping)"
    )
    input_group.add_argument(
        "--input", type=Path,
        help="Path to raw photos directory (full pipeline: detect → crop → embed → eval)"
    )

    # Detection options
    parser.add_argument(
        "--detector", type=str, default="yolo", choices=["yolo", "sam3"],
        help="Detection backend (default: yolo). Only used with --input."
    )
    parser.add_argument(
        "--crop-mode", type=str, default="box", choices=["box", "mask"],
        help="Cropping mode (default: box). 'mask' uses SAM3 segmentation masks."
    )
    parser.add_argument(
        "--sam3-checkpoint", type=str, default=None,
        help="Path to SAM3 checkpoint file (required when --detector sam3)"
    )
    parser.add_argument(
        "--sam3-config", type=str, default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM3 model config (default: sam2.1_hiera_s)"
    )

    # Embedding / evaluation
    parser.add_argument(
        "--backbone", type=str, default="resnet50",
        choices=["resnet50", "facenet", "arcface", "dinov2", "clip"],
        help="Embedding backbone (default: resnet50)"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Determine crops directory
    if args.crops:
        crops_dir = args.crops
        if not crops_dir.is_dir():
            print(f"Error: crops directory not found: {crops_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        # Full pipeline: detect + crop first
        input_dir = args.input
        if not input_dir.is_dir():
            print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
            sys.exit(1)

        # Build detector kwargs
        detector_kwargs = {}
        if args.detector == "sam3":
            detector_kwargs["checkpoint"] = args.sam3_checkpoint
            detector_kwargs["model_cfg"] = args.sam3_config

        # Crops go into a temp dir under results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        crops_dir = Path("results") / f"crops_{args.detector}_{ts}"

        crops_dir = run_detection_and_cropping(
            input_dir=input_dir,
            detector_name=args.detector,
            crop_mode=args.crop_mode,
            output_crops_dir=crops_dir,
            device=args.device,
            **detector_kwargs,
        )

    # Output dir
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"{args.backbone}_{ts}"
    else:
        output_dir = args.output

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "backbone": args.backbone,
        "crops": str(crops_dir),
        "device": args.device,
        "output": str(output_dir),
    }
    if args.input:
        config["input"] = str(args.input)
        config["detector"] = args.detector
        config["crop_mode"] = args.crop_mode

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    from primateid.embedding.multi_embedder import MultiEmbedder
    from primateid.evaluation.evaluator import ReIDEvaluator

    embedder = MultiEmbedder(backbone=args.backbone, device=args.device)
    evaluator = ReIDEvaluator(embedder)
    summary = evaluator.evaluate(crops_dir, output_dir)

    print("\n" + "=" * 50)
    print("PrimateReID Evaluation Complete")
    print("=" * 50)
    print(f"  Backbone:       {args.backbone}")
    if args.input:
        print(f"  Detector:       {args.detector}")
        print(f"  Crop Mode:      {args.crop_mode}")
    print(f"  AUC:            {summary['auc']:.4f}")
    print(f"  EER:            {summary['eer_pct']:.1f}%")
    print(f"  Decidability:   {summary['decidability']:.4f}")
    print(f"  Best Threshold: {summary['best_threshold']:.4f}")
    print(f"  Results:        {output_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
