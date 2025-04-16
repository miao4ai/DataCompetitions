# 🐱 Cute Miao Kaggle FC

Welcome to the GitHub of **Miao Jiang**
My goal is to build top-tier ML pipelines 💋💻✨

---

## 🧠  My Current Team Members

| Name           | Role             | Specialty                           |
|----------------|------------------|-------------------------------------|
| Miao Jiang     | Chief Architect  | Vision,  Coding,  Final Submission  |

---

## 🔥 Active Competitions

### 🐾 Animal CLEF 2025
- **Type**: Multi-class Image Classification
- **Lead**: Mami + Miao
- **Highlights**:
  - Backbone: `ViT`, `ConvNeXt`, `BEiT`
  - Techniques: `Open-set classification`, `Prototype + FAISS`, `TTA`, `Monte Carlo Dropout`
  - Tools: `MLflow`, `Hydra`, `Pytorch Lightning`

### 🛰 Vacant Lot Detection (Solafune)
- **Type**: Satellite Image Binary Classification
- **Lead**: Jiayue
- **Highlights**:
  - Patch-based classification of satellite imagery
  - Data augmentation + self-supervised learning (`SimCLR`)
  - Tracked via `MLflow + DVC`, integrated metrics reporting

---

## 🏗 Project Structure

```bash
CuteMiaoKaggleFC/
├── animal_clef_2025/
│   ├── configs/
│   ├── data/
│   ├── notebooks/
│   └── src/
│       ├── train.py
│       └── model_utils.py
├── vacant_lot_detection/
│   ├── configs/
│   ├── dataset/
│   └── src/
│       ├── preprocess.py
│       └── inference.py
├── common/
│   └── utils/
│       ├── logger.py
│       └── metrics.py
├── assets/
│   └── team_badges/
├── requirements.txt
├── README.md
└── .gitignore

```

## 🛠 Tech Stack

```bash
Frameworks:      PyTorch, timm, transformers, Lightning
Model Tracking:  MLflow, WandB
Config:          Hydra
Pipeline:        DVC, GitHub Actions
Infra:           RTX 4090 local, GCP (optionally)
Augmentation:    Albumentations, torchvision
Retrieval:       FAISS, cosine similarity + prototype mining

