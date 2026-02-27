# Baseline Results & Backbone Research

> Date: 2026-02-25 | Dataset: demo_chimp_crops (10 chimpanzees × 30 crops)

## Baseline Results

| Backbone | Training Data | Dim | AUC | EER | d' (decidability) | Best Threshold |
|----------|--------------|-----|-----|-----|--------------------|----------------|
| **DINOv2 ViT-S/14** ⭐ | Self-supervised (LVD-142M) | 384 | **0.7251** | **34.7%** | **0.802** | 0.732 |
| **ResNet50** | Supervised (ImageNet) | 2048 | 0.6884 | 36.3% | 0.673 | 0.624 |
| **FaceNet** | Human faces (VGGFace2) | 512 | 0.6141 | 42.2% | 0.407 | 0.751 |
| **ArcFace** | Human faces (MS1MV2) | 512 | 0.6400 | 40.2% | 0.507 | 0.536 |
| **CLIP ViT-B/16** | Image-text (400M pairs) | 512 | 0.6521 | 39.4% | 0.505 | 0.900 |

### Interpretation

- **All models struggle** — even the best (DINOv2) has AUC 0.73, far from production-grade
- **Clear ranking**: Self-supervised > General supervised > Human face specialized
- **DINOv2 wins** despite having the smallest embedding dimension (384-d). Its self-supervised training on diverse visual data produces features that generalize across species better than any supervised model.
- **ArcFace beats FaceNet but not ResNet50**: With detection bypassed (direct crop → recognition model), ArcFace's metric learning (angular margin) helps vs FaceNet, suggesting the *training objective* matters. But it still can't beat generic ResNet50 features, confirming human-face bias hurts.
- **CLIP underperforms**: Despite being trained on 400M image-text pairs, CLIP (0.65) ranks below ResNet50 (0.69). CLIP's visual features are optimized for matching images to text descriptions, not for fine-grained individual discrimination within a species.
- **The "specialization penalty"**: More human-face-specific training → worse cross-species performance. FaceNet (0.61) < ArcFace (0.64) < CLIP (0.65) < ResNet50 (0.69) < DINOv2 (0.73). Note: ArcFace initially scored 0.55 when using its built-in face detector (which fails on chimps); bypassing detection improved it to 0.64.

### Key Insight

The gap between "category recognition" (is this a chimpanzee?) and "individual recognition" (is this Fredy or Victor?) requires identity-discriminative features that general-purpose models don't learn. However, **self-supervised learning shows the most promise** — DINOv2 learns fine-grained visual distinctions without any labels, making it the best starting point for cross-species transfer.

The "specialization penalty" finding is a strong research narrative: **human face expertise hurts primate face identification**. This validates the need for primate-specific fine-tuning rather than simply using bigger human face models.

---

## Backbone Candidates for Next Steps

### Tier 1: Ready to Integrate (pre-trained weights available)

| Model | Source | Embedding Dim | Training Data | Expected Benefit |
|-------|--------|---------------|---------------|------------------|
| **ArcFace (InsightFace)** | [deepinsight/insightface](https://github.com/deepinsight/insightface) | 512 | MS1MV2 (human faces) | Superior metric learning loss; better intra-class compactness |
| **SphereFace** | [clcarwin/sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch) | 512 | CASIA-WebFace | Angular margin; used in Deb et al. 2018 for lemur ReID |
| **DINOv2** | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | 768/1024 | LVD-142M (diverse) | Self-supervised; strong on fine-grained visual tasks without labels |
| **CLIP (ViT-B/16)** | [openai/clip](https://github.com/openai/CLIP) | 512 | 400M image-text pairs | Zero-shot generalization; vision-language alignment |

### Tier 2: Requires Fine-Tuning (most promising for primate ReID)

| Approach | Description | Key Paper |
|----------|-------------|-----------|
| **ArcFace fine-tuned on primate faces** | Start from human face weights, fine-tune on CTai/CZoo | Deb et al. "Face Recognition: Primates in the Wild" (2018) |
| **PrimNet** | Primate-specific network from Freytag et al. | "Chimpanzee Faces in the Wild" GCPR 2016 |
| **Log-Euclidean CNN** | Covariance-based features for primate identity | Freytag et al. GCPR 2016 |
| **Triplet Loss fine-tuning** | Any backbone + triplet/contrastive loss on primate data | Standard metric learning |

### Tier 3: Research Frontier

| Approach | Description |
|----------|-------------|
| **MegaDescriptor** | Animal re-identification foundation model (WildlifeDatasets project) |
| **SAM3 + embedding** | Open-vocabulary detection → crop → fine-tuned embedding |
| **Self-supervised pretraining on primate data** | DINO/MAE on unlabeled primate photos → fine-tune |

---

## Recommended Next Steps

### Phase 1: Quick Wins (no training required)

1. **Add ArcFace backbone** — InsightFace provides ONNX models, easy to load
2. **Add DINOv2 backbone** — Self-supervised features; likely better than supervised ImageNet for novel domains
3. **Add CLIP backbone** — Strong zero-shot generalization
4. Re-run evaluation on demo_chimp_crops → compare all 5 backbones

### Phase 2: Fine-Tuning (requires GPU)

4. **Fine-tune ArcFace on full CTai/CZoo** (7,187 images, 86 individuals)
   - Use the annotation file to split train/test
   - ArcFace loss + chimpanzee identity labels
   - Expected: dramatic improvement (literature reports >90% accuracy after fine-tuning)
5. **Triplet loss training** on any backbone as alternative

### Phase 3: Cross-Species Generalization

6. Test fine-tuned chimp model on other primate species (orangutan, macaque)
7. Investigate domain adaptation / few-shot learning for new species

---

## Dataset Source

The demo chimpanzee crops come from the **CTai/CZoo dataset** by Freytag et al.:

- **Paper**: "Chimpanzee Faces in the Wild: Log-Euclidean CNNs for Predicting Identities and Attributes of Primates" (GCPR 2016)
- **Repository**: [cvjena/chimpanzee_faces](https://github.com/cvjena/chimpanzee_faces)
- **Full dataset**: 7,187 cropped face images, 86 individuals, with identity/gender/age annotations
- **License**: Non-commercial research use

---

## References

1. Freytag, A. et al. "Chimpanzee Faces in the Wild: Log-Euclidean CNNs for Predicting Identities and Attributes of Primates." GCPR 2016.
2. Deb, D. et al. "Face Recognition: Primates in the Wild." IEEE BTAS 2018.
3. Guan, Y. et al. "Face recognition of a Lorisidae species based on computer vision." Global Ecology and Conservation, 2023.
4. Deng, J. et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR 2019.
5. Oquab, M. et al. "DINOv2: Learning Robust Visual Features without Supervision." 2023.
