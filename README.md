# PrimateReID

End-to-end pipeline for primate face detection, cropping, and individual re-identification.

> **[繁體中文版 README](README.zh-TW.md)**

## Architecture

```
Raw Photo → Detection (YOLO/SAM3) → Crop (box/mask) → Embedding (FaceNet/ArcFace/DINOv2) → ReID Evaluation
```

**PrimateReID** handles the full pipeline from raw field photos to individual identification, with built-in evaluation metrics (AUC, EER, decidability) and visualisation.

## Quick Start

```bash
git clone https://github.com/jonesandjay123/PrimateReID.git
cd PrimateReID
pip install -r requirements.txt
```

### Option A: Run with included demo data (real chimpanzee faces)

The repo includes `data/demo_chimp_crops/` — 10 chimpanzees × 30 cropped face photos (300 images, ~22MB) from the CTai/CZoo dataset, ready to test out of the box:

```bash
# DINOv2 — Self-supervised visual features (384-d) ⭐ Best performer
PYTHONPATH=src python3 -m primateid.run --crops data/demo_chimp_crops --backbone dinov2

# ResNet50 — ImageNet general-purpose (2048-d)
PYTHONPATH=src python3 -m primateid.run --crops data/demo_chimp_crops --backbone resnet50

# FaceNet — Human face specialized (512-d)
PYTHONPATH=src python3 -m primateid.run --crops data/demo_chimp_crops --backbone facenet

# ArcFace — SOTA human face recognition via InsightFace (512-d)
PYTHONPATH=src python3 -m primateid.run --crops data/demo_chimp_crops --backbone arcface
```

### Option B: Full pipeline from raw photos (detect → crop → embed → eval)

Install detection dependencies first:

```bash
pip install -r requirements-detection.txt
```

Then run the full pipeline:

```bash
# YOLO detection + box crop
PYTHONPATH=src python3 -m primateid.run \
    --input data/raw_photos \
    --detector yolo \
    --crop-mode box \
    --backbone dinov2

# SAM3 detection + mask crop (requires SAM3 checkpoint)
PYTHONPATH=src python3 -m primateid.run \
    --input data/raw_photos \
    --detector sam3 \
    --sam3-checkpoint /path/to/sam2.1_hiera_small.pt \
    --crop-mode mask \
    --backbone dinov2
```

### Option C: Run with synthetic sample data

```bash
python3 scripts/generate_sample_data.py
PYTHONPATH=src python3 -m primateid.run --crops data/sample_crops --backbone resnet50
```

### Option D: Run with your own data

Organise your cropped images into `data/your_dataset/<individual_name>/` folders (see [Data Format](#data-format) below), then:

```bash
PYTHONPATH=src python3 -m primateid.run --crops data/your_dataset --backbone resnet50
```

### CLI Options

```
Input (mutually exclusive, one required):
  --crops PATH          Path to pre-cropped images directory (skip detection)
  --input PATH          Path to raw photos directory (full pipeline)

Detection (only used with --input):
  --detector STR        Detection backend: yolo | sam3 (default: yolo)
  --crop-mode STR       Cropping mode: box | mask (default: box)
  --sam3-checkpoint PATH  Path to SAM3 model checkpoint
  --sam3-config STR     SAM3 model config (default: sam2.1_hiera_s)

Embedding & Evaluation:
  --backbone STR        Embedding backbone: resnet50 | facenet | arcface | dinov2 (default: resnet50)
  --output PATH         Output directory (default: results/<backbone>_<timestamp>/)
  --device STR          Torch device (default: cpu)
```

### Output Structure

```
results/resnet50_20260225_173000/
├── config.json              # Run parameters
├── pairs.csv                # Pairs used for evaluation
├── embeddings.npz           # All embeddings
├── scores.csv               # img1, img2, label, similarity
├── summary.json             # AUC, EER, d', threshold
├── figures/
│   ├── roc_curve.png        # ROC curve with EER point
│   └── score_distribution.png  # Genuine vs impostor score histograms
└── report.md                # Human-readable summary report
```

### Data Format

Organise crops into sub-folders by individual identity:

```
data/crops/
├── monkey_A/
│   ├── 001.jpg
│   └── 002.jpg
├── monkey_B/
│   └── 001.jpg
```

Pairs are automatically generated from the folder structure (genuine = same folder, impostor = different folders). To use custom pairs, place a `pairs.csv` in the crops directory.

## Pipeline Components

### Detection

Front-end face/body detection supporting two backends:

| Backend | Model | Approach | Masks? |
|---------|-------|----------|--------|
| **YOLO** | YOLOv8 (ultralytics) | Object detection with bounding boxes | ❌ |
| **SAM3** | SAM 2.1 (Meta) | Automatic mask generation → filter by size/score | ✅ |

**YOLO** is faster and produces class-labeled detections. Best with a fine-tuned primate model.
**SAM3** produces high-quality segmentation masks for mask-based cropping, but is class-agnostic (segments everything, then we filter).

### Cropping

Extracts individual primate regions from detected areas:

| Mode | Description | Best With |
|------|-------------|-----------|
| **box** | Simple bounding box crop with padding | YOLO or SAM3 |
| **mask** | Mask-based crop, background removed | SAM3 (has masks) |

### Embedding

Generates identity-discriminative feature vectors using multiple backbones:

- **DINOv2** ⭐ — Meta's self-supervised ViT-S/14, 384-d embeddings (best performer)
- **ResNet50** — ImageNet-pretrained, 2048-d embeddings
- **ArcFace** — InsightFace buffalo_l (MS1MV2), 512-d embeddings with angular margin loss
- **FaceNet** — VGGFace2-pretrained InceptionResNetV1, 512-d embeddings

All embeddings are L2-normalised so cosine similarity = dot product.

### Evaluation

Built-in evaluation engine computing:
- **AUC** — Area Under the ROC Curve
- **EER** — Equal Error Rate
- **d' (decidability)** — separation between genuine and impostor distributions
- **Best threshold** — optimal operating point (Youden's J)

## Installation

### Core dependencies (embedding + evaluation)

```bash
pip install -r requirements.txt
```

### Detection dependencies (optional)

```bash
# YOLO
pip install ultralytics

# SAM3 (Segment Anything Model 2.1)
pip install sam-2
# Download checkpoint from https://github.com/facebookresearch/sam2#checkpoints

# Or install all detection deps:
pip install -r requirements-detection.txt
```

## Project Structure

```
PrimateReID/
├── src/primateid/
│   ├── detection/        # YOLO, SAM3 detection backends
│   │   ├── __init__.py   # Detection dataclass + factory
│   │   ├── yolo.py       # YOLOv8 backend
│   │   └── sam3.py       # SAM 2.1 backend
│   ├── cropping/         # Box crop, mask crop
│   │   ├── __init__.py
│   │   └── cropper.py    # BoxCropper, MaskCropper
│   ├── embedding/        # Multi-backbone embedder
│   ├── evaluation/       # Pairs generation + metrics + plotting
│   ├── utils/
│   └── run.py            # CLI entry point
├── scripts/              # Utility scripts
├── configs/              # Experiment configuration (YAML)
├── data/                 # Test data
├── results/              # Experiment outputs
└── tests/
```

## Demo Data

| Dataset | Path | Description |
|---------|------|-------------|
| **Demo chimpanzees** | `data/demo_chimp_crops/` | 10 individuals × 30 real face crops from CTai/CZoo (~22MB) |
| **Synthetic samples** | `data/sample_crops/` | Generated via `scripts/generate_sample_data.py` (random noise, for CI/smoke tests) |

## Baseline Results (2026-02-25)

Tested on `data/demo_chimp_crops/` — 10 chimpanzees × 30 face crops (300 images):

| Backbone | Training | AUC | EER | d' | Verdict |
|----------|----------|-----|-----|----|---------|
| **DINOv2** ⭐ | Self-supervised (LVD-142M) | **0.725** | **34.7%** | **0.80** | Best — no labels needed |
| ResNet50 | Supervised (ImageNet) | 0.688 | 36.3% | 0.67 | Generic features, decent |
| ArcFace | Human faces (MS1MV2) | 0.640 | 40.2% | 0.51 | Metric learning helps, still human-biased |
| FaceNet | Human faces (VGGFace2) | 0.614 | 42.2% | 0.41 | Over-specialized for humans |

**Key finding**: Human face models (FaceNet, ArcFace) perform *worse* than general-purpose models on primate faces. Self-supervised learning (DINOv2) generalizes best across species.

See [docs/baseline-results.md](docs/baseline-results.md) for full analysis.

## Roadmap

- [x] Multi-backbone embedding pipeline (ResNet50, FaceNet, ArcFace, DINOv2)
- [x] Evaluation engine (AUC, EER, d', ROC curves)
- [x] Demo dataset (10 chimpanzees × 30 crops)
- [x] YOLO detection backend
- [x] SAM3 (SAM 2.1) detection backend
- [x] Box crop and mask crop modules
- [x] Full pipeline CLI (raw photos → detection → crop → embed → eval)
- [ ] Fine-tuned YOLO model for primate faces
- [ ] Fine-tuned DINOv2 with metric learning (triplet/contrastive loss)
- [ ] Video tracking integration (SAM3 video mode)
- [ ] Web UI for interactive evaluation
- [ ] Multi-species support (macaques, gorillas, orangutans)
- [ ] Field deployment guide

## Related Repos

- [FaceThresholdLab](https://github.com/jonesandjay123/FaceThresholdLab) — Evaluation engine for face embedding analysis
- [FacialRecognitionTest](https://github.com/jonesandjay123/FacialRecognitionTest) — Earlier facial recognition experiments

## Contributors

- **Jones** — Project lead, pipeline architecture
- **Eleane (趙以琳)** — SAM3 detection research, field testing

## License

MIT
