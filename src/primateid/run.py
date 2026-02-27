"""CLI entry point for PrimateReID evaluation pipeline.

Usage:
    python -m primateid.run --crops data/sample_crops --backbone resnet50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="PrimateReID evaluation pipeline")
    parser.add_argument("--crops", type=Path, required=True, help="Path to crops directory")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "facenet", "arcface", "dinov2", "clip"], help="Embedding backbone")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.crops.is_dir():
        print(f"Error: crops directory not found: {args.crops}", file=sys.stderr)
        sys.exit(1)

    # Output dir
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"{args.backbone}_{ts}"
    else:
        output_dir = args.output

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {"backbone": args.backbone, "crops": str(args.crops),
              "device": args.device, "output": str(output_dir)}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    from primateid.embedding.multi_embedder import MultiEmbedder
    from primateid.evaluation.evaluator import ReIDEvaluator

    embedder = MultiEmbedder(backbone=args.backbone, device=args.device)
    evaluator = ReIDEvaluator(embedder)
    summary = evaluator.evaluate(args.crops, output_dir)

    print("\n" + "=" * 50)
    print("PrimateReID Evaluation Complete")
    print("=" * 50)
    print(f"  Backbone:       {args.backbone}")
    print(f"  AUC:            {summary['auc']:.4f}")
    print(f"  EER:            {summary['eer_pct']:.1f}%")
    print(f"  Decidability:   {summary['decidability']:.4f}")
    print(f"  Best Threshold: {summary['best_threshold']:.4f}")
    print(f"  Results:        {output_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
