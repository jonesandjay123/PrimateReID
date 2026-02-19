# Monkey Face Detection with SAM3

Detecting and distinguishing individual monkey faces in images using Meta's **Segment Anything Model 3 (SAM3)**, while filtering out human faces.

## Project Goal

Given an image containing multiple faces (both monkey and human):
1. **Detect all monkey faces** in the image
2. **Distinguish between different individual monkeys** (not just "there's a monkey")
3. **Exclude human faces** from results

## Background

This project explores whether SAM3's zero-shot segmentation capabilities can be adapted for **primate face detection** — specifically for research scenarios where multiple macaques need to be individually identified in field or lab photographs.

## Tech Stack

- **Model**: [SAM3 (Segment Anything Model 3)](https://huggingface.co/facebook/sam3) — Meta's latest segmentation model
- **Language**: Python 3.10+
- **Framework**: PyTorch + HuggingFace Transformers
- **Additional**: OpenCV for image processing

## Project Structure

```
facedetection/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── detector.py          # Core detection pipeline
│   ├── sam3_wrapper.py      # SAM3 model loading & inference
│   ├── face_filter.py       # Human vs monkey face classification
│   └── visualizer.py        # Result visualization & annotation
├── notebooks/
│   └── exploration.ipynb    # Interactive exploration & experiments
├── data/
│   ├── sample/              # Sample test images
│   └── output/              # Detection results
├── tests/
│   └── test_detector.py
└── docs/
    └── research_notes.md    # Findings, limitations, alternatives
```

## Getting Started

```bash
# Clone
git clone https://github.com/jonesandjay123/facedetection.git
cd facedetection

# Install dependencies
pip install -r requirements.txt

# Run detection on a sample image
python -m src.detector --input data/sample/test.jpg --output data/output/
```

## Research Questions

1. Can SAM3 segment monkey faces with zero-shot prompting (no fine-tuning)?
2. How well does it distinguish between individual monkeys vs. just detecting "monkey"?
3. What prompt engineering strategies work best for primate face segmentation?
4. How does it compare to traditional face detection (MTCNN, RetinaFace) + classification pipelines?

## Approach

### Phase 1: SAM3 Exploration
- Load SAM3 from HuggingFace
- Test zero-shot segmentation on monkey images
- Evaluate segmentation quality for faces specifically

### Phase 2: Face Filtering
- Integrate a lightweight classifier to separate monkey vs human faces
- Options: fine-tuned ResNet, CLIP zero-shot, or custom CNN

### Phase 3: Individual Identification
- Explore embedding-based approaches for individual monkey recognition
- Compare SAM3 embeddings vs dedicated face recognition models

## Alternative Models to Explore

| Model | Source | Strength |
|-------|--------|----------|
| SAM3 | Meta/HuggingFace | Zero-shot segmentation |
| SAM2 | Meta | Video segmentation |
| YOLOv8-face | Ultralytics | Fast face detection |
| CLIP | OpenAI | Zero-shot classification |
| DINOv2 | Meta | Visual feature extraction |

## License

MIT

## Contributors

- Jones — Project lead
- Eleane (趙以琳) — Research collaboration
