# 基線結果與骨幹網路研究

> 日期：2026-02-25 | 資料集：demo_chimp_crops（10 隻黑猩猩 × 30 張裁切照）

## 基線結果

| 骨幹網路 | 訓練資料 | 維度 | AUC | EER | d'（可分辨度）| 最佳閾值 |
|----------|---------|------|-----|-----|-----------:|----------|
| **DINOv2 ViT-S/14** ⭐ | 自監督（LVD-142M）| 384 | **0.7251** | **34.7%** | **0.802** | 0.732 |
| **ResNet50** | 監督式（ImageNet）| 2048 | 0.6884 | 36.3% | 0.673 | 0.624 |
| **FaceNet** | 人臉（VGGFace2）| 512 | 0.6141 | 42.2% | 0.407 | 0.751 |
| **ArcFace** | 人臉（MS1MV2）| 512 | 0.6400 | 40.2% | 0.507 | 0.536 |
| **CLIP ViT-B/16** | 圖文配對（4 億對）| 512 | 0.6521 | 39.4% | 0.505 | 0.900 |

### 結果解讀

- **所有模型都不理想** — 最好的 DINOv2 也只有 AUC 0.73，離實用還很遠
- **排名清晰**：自監督 > 通用監督 > 人臉專精
- **DINOv2 勝出**，儘管嵌入維度最小（384 維）。自監督訓練在多樣視覺資料上產生的特徵，跨物種泛化能力最強
- **ArcFace 贏 FaceNet 但輸 ResNet50**：跳過人臉偵測（直接裁切 → 辨識模型）後，ArcFace 的度量學習（角度間距）比 FaceNet 有效，說明*訓練目標*有影響。但仍無法勝過通用 ResNet50 特徵，確認人臉偏差的負面影響
- **CLIP 表現不佳**：儘管在 4 億圖文配對上訓練，CLIP (0.65) 低於 ResNet50 (0.69)。CLIP 的視覺特徵針對「圖片配文字」最佳化，不擅長同物種內的細粒度個體區分
- **「專精化懲罰」**：越偏向人臉的訓練 → 跨物種表現越差。FaceNet (0.61) < ArcFace (0.64) < CLIP (0.65) < ResNet50 (0.69) < DINOv2 (0.73)。注意：ArcFace 最初使用內建人臉偵測器時只有 0.55（偵測器對猩猩失效），跳過偵測後提升至 0.64

### 核心洞察

「類別辨識」（這是不是黑猩猩？）和「個體辨識」（這是 Fredy 還是 Victor？）之間的鴻溝，需要通用模型學不到的「身份鑑別特徵」。然而，**自監督學習展現了最大的潛力** — DINOv2 不靠任何標籤就學到細粒度的視覺差異，是跨物種遷移的最佳起點。

「專精化懲罰」的發現是很好的研究敘事：**人臉辨識的專業知識反而傷害靈長類辨識**。這驗證了需要靈長類專用微調，而不是單純用更大的人臉模型。

---

## 骨幹網路候選方案

### 第一梯隊：可直接整合（有預訓練權重）

| 模型 | 來源 | 嵌入維度 | 訓練資料 | 預期效果 |
|------|------|---------|---------|---------|
| **ArcFace (InsightFace)** | [deepinsight/insightface](https://github.com/deepinsight/insightface) | 512 | MS1MV2（人臉）| 更優的度量學習損失；更好的類內緊湊度 |
| **SphereFace** | [clcarwin/sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch) | 512 | CASIA-WebFace | 角度間距；Deb 2018 曾用於狐猴 ReID |
| **DINOv2** | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | 768/1024 | LVD-142M（多樣）| 自監督學習；在無標籤的細粒度任務上表現強 |
| **CLIP (ViT-B/16)** | [openai/clip](https://github.com/openai/CLIP) | 512 | 4 億圖文配對 | 零樣本泛化；視覺-語言對齊 |

### 第二梯隊：需要微調（對靈長類 ReID 最有潛力）

| 方法 | 說明 | 關鍵論文 |
|------|------|---------|
| **ArcFace 微調靈長類臉** | 從人臉權重開始，用 CTai/CZoo 微調 | Deb et al. 2018 |
| **PrimNet** | 靈長類專用網路 | Freytag et al. GCPR 2016 |
| **Log-Euclidean CNN** | 協方差特徵用於靈長類身份 | Freytag et al. GCPR 2016 |
| **Triplet Loss 微調** | 任何骨幹 + 三元組/對比損失 | 標準度量學習 |

### 第三梯隊：研究前沿

| 方法 | 說明 |
|------|------|
| **MegaDescriptor** | 動物再辨識基礎模型（WildlifeDatasets 專案）|
| **SAM3 + embedding** | 開放詞彙偵測 → 裁切 → 微調嵌入 |
| **自監督預訓練** | DINO/MAE 在無標籤靈長類照片上預訓練 → 微調 |

---

## 建議下一步

### 階段一：快速驗證（不需訓練）

1. **加入 ArcFace** — InsightFace 提供 ONNX 模型，容易載入
2. **加入 DINOv2** — 自監督特徵；在新領域可能比監督式 ImageNet 更好
3. **加入 CLIP** — 強零樣本泛化能力
4. 在 demo_chimp_crops 重新跑評估 → 5 種骨幹橫向比較

### 階段二：微調（需要 GPU）

5. **用完整 CTai/CZoo 微調 ArcFace**（7,187 張，86 隻個體）
   - 使用 annotation 分 train/test
   - ArcFace loss + 黑猩猩身份標籤
   - 預期：大幅提升（文獻報告微調後 >90% 準確率）
6. **Triplet loss 訓練**作為替代方案

### 階段三：跨物種泛化

7. 用微調後的黑猩猩模型測試其他靈長類（紅毛猩猩、獼猴）
8. 研究 domain adaptation / few-shot learning 用於新物種

---

## 資料來源

Demo 黑猩猩裁切來自 **CTai/CZoo 資料集**（Freytag et al.）：

- **論文**：Chimpanzee Faces in the Wild: Log-Euclidean CNNs for Predicting Identities and Attributes of Primates (GCPR 2016)
- **倉庫**：[cvjena/chimpanzee_faces](https://github.com/cvjena/chimpanzee_faces)
- **完整資料集**：7,187 張裁切臉部照片、86 隻個體，含身份/性別/年齡標註
- **授權**：非商業研究用途

---

## 參考文獻

1. Freytag, A. et al. "Chimpanzee Faces in the Wild: Log-Euclidean CNNs for Predicting Identities and Attributes of Primates." GCPR 2016.
2. Deb, D. et al. "Face Recognition: Primates in the Wild." IEEE BTAS 2018.
3. Guan, Y. et al. "Face recognition of a Lorisidae species based on computer vision." Global Ecology and Conservation, 2023.
4. Deng, J. et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR 2019.
5. Oquab, M. et al. "DINOv2: Learning Robust Visual Features without Supervision." 2023.
