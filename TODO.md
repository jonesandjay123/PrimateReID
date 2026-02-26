# TODO — PrimateReID

> Last updated: 2026-02-25

## Priority A: Data Alignment with E小姐

- [ ] **Ask E小姐: are her macaque crops organized by individual?** (`crops/<individual_name>/*.jpg`)
  - If yes → PrimateReID can run her baseline immediately (even on Mac Air)
  - If no → need to help her reorganize or write a label-mapping script
- [ ] **Define the data contract clearly**: folder-per-identity structure → auto pairs → report
- [ ] Share the demo run results with E小姐 as proof the pipeline works

## Priority A+: E小姐's Research Direction (2026-02-25)

E小姐's key insight from her ML engineer discussion:
> "model 本身有很多 layer，可以讓他用部分比較適合的 layer 剩下再去 train"
> → **Transfer learning with partial layer freezing** — use primate-pretrained front layers, fine-tune back layers on macaque data

Reference repos from E小姐:
- [ ] **[KordingLab/PrimateFace](https://github.com/KordingLab/PrimateFace)** — Cross-species primate facial analysis ecosystem (Parodi et al. 2025)
  - DINOv2-based features for primate faces (aligns with our best baseline!)
  - Includes: face detection, landmark tracking, pseudo-labeling GUI
  - Has pretrained models on HuggingFace: `fparodi/PrimateFace`
  - **Key value**: Primate-specific DINOv2 features — can serve as our embedding backbone
  - Requires GPU (Google Colab tutorials available)
- [ ] **[C-Poirier-Lab/MacqD](https://github.com/C-Poirier-Lab/MacqD)** — Macaque detection model
  - SWIN Transformer + Mask R-CNN for macaque detection in home-cage
  - Handles occlusions, glass reflections, overexposure
  - Three trained models (single/curriculum/combined)
  - **Key value**: Robust macaque detector → replaces YOLO in our detection stage

Suggested integration plan:
- [ ] Use **PrimateFace's DINOv2 features** as a new backbone in our embedder (likely beats our vanilla DINOv2 0.725 AUC)
- [ ] Use **MacqD** as detection front-end for E小姐's macaque data
- [ ] **Partial fine-tuning**: Freeze early layers of PrimateFace model, train final layers on E小姐's specific macaque population
- [ ] Compare: PrimateFace DINOv2 vs our vanilla DINOv2 vs fine-tuned version

## Priority B: Add Meaningful Backbones

Current baselines (both near random on primate ReID):
- ResNet50 (ImageNet): AUC 0.69, EER 36.3%
- FaceNet (VGGFace2): AUC 0.61, EER 42.2%

Completed baselines (4 backbones tested 2026-02-25):
- [x] **ResNet50** (ImageNet) — AUC 0.69 ✅
- [x] **FaceNet** (VGGFace2) — AUC 0.61 ✅
- [x] **ArcFace** (InsightFace) — AUC 0.55 ✅ (near random — specialization penalty!)
- [x] **DINOv2** (self-supervised) — AUC 0.73 ✅ ⭐ Best performer

Next backbones to explore:
- [ ] CLIP ViT-B/16 — zero-shot generalization baseline
- [ ] SphereFace — used in Deb et al. 2018 for lemur ReID
- [ ] DINOv2 ViT-B/14 or ViT-L/14 — larger DINOv2 variants may improve further

## Priority C: Engineering Hygiene

- [x] Demo data in repo kept small (~22MB, 300 images) ✅
- [ ] **Rule**: repo only holds demo data (<50MB); large datasets use download script + `.gitignore`
- [ ] Write `scripts/download_full_dataset.sh` for CTai/CZoo (7,187 images)
- [ ] Add `.gitignore` rules for `data/full_*` directories

## Future: Fine-Tuning (requires GPU)

- [ ] Fine-tune ArcFace on full CTai/CZoo (86 individuals, 7K images)
- [ ] Triplet loss training as alternative approach
- [ ] Cross-species evaluation: chimp model → macaque/orangutan

## Quality Assurance: Multi-AI Code Review

- [ ] **Let different AI agents review this repo independently** to catch blind spots
  - Gemini Pro 3.1 / Codex 5.3 / GPT / others
  - Check: code correctness, evaluation methodology, statistical validity, best practices
  - Avoid single-AI echo chamber (all code so far written by Claude Opus 4.6)

## Research Questions

1. **Does metric learning (ArcFace) help cross-species?** — Human face weights with angular margin vs generic features
2. **Does self-supervised (DINOv2) beat supervised for novel species?** — No labels needed, potentially more domain-agnostic
3. **How many fine-tuning samples per individual are needed?** — Few-shot threshold for practical field use
4. **Cross-species transfer**: Can a chimp-trained model identify macaques? At what accuracy drop?
