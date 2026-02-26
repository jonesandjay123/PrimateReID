# TODO — PrimateReID

> Last updated: 2026-02-25

## Priority A: Data Alignment with E小姐

- [ ] **Ask E小姐: are her macaque crops organized by individual?** (`crops/<individual_name>/*.jpg`)
  - If yes → PrimateReID can run her baseline immediately (even on Mac Air)
  - If no → need to help her reorganize or write a label-mapping script
- [ ] **Define the data contract clearly**: folder-per-identity structure → auto pairs → report
- [ ] Share the demo run results with E小姐 as proof the pipeline works

## Priority B: Add Meaningful Backbones

Current baselines (both near random on primate ReID):
- ResNet50 (ImageNet): AUC 0.69, EER 36.3%
- FaceNet (VGGFace2): AUC 0.61, EER 42.2%

Next two to add (tells the full story: supervised vs face-specialized vs self-supervised):
- [ ] **ArcFace (InsightFace)** — strongest human face ReID; tests if metric learning helps cross-species
- [ ] **DINOv2** — self-supervised; expected to generalize better to novel domains

Optional but valuable:
- [ ] CLIP ViT-B/16 — zero-shot generalization baseline
- [ ] SphereFace — used in Deb et al. 2018 for lemur ReID

## Priority C: Engineering Hygiene

- [x] Demo data in repo kept small (~22MB, 300 images) ✅
- [ ] **Rule**: repo only holds demo data (<50MB); large datasets use download script + `.gitignore`
- [ ] Write `scripts/download_full_dataset.sh` for CTai/CZoo (7,187 images)
- [ ] Add `.gitignore` rules for `data/full_*` directories

## Future: Fine-Tuning (requires GPU)

- [ ] Fine-tune ArcFace on full CTai/CZoo (86 individuals, 7K images)
- [ ] Triplet loss training as alternative approach
- [ ] Cross-species evaluation: chimp model → macaque/orangutan

## Research Questions

1. **Does metric learning (ArcFace) help cross-species?** — Human face weights with angular margin vs generic features
2. **Does self-supervised (DINOv2) beat supervised for novel species?** — No labels needed, potentially more domain-agnostic
3. **How many fine-tuning samples per individual are needed?** — Few-shot threshold for practical field use
4. **Cross-species transfer**: Can a chimp-trained model identify macaques? At what accuracy drop?
