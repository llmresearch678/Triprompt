# TRIPROMPT

**Deformation-Aware Multimodal Prompting for Robust 3D Medical Image Segmentation**:  
A query-centric framework that jointly models anatomical structure, medical semantics, and population-level deformation for robust 3D segmentation.

---

## 🔍 1. Overview

**TRIPROMPT** is a deformation-aware, multimodal prompting framework for **3D medical image segmentation**.  
It addresses a key limitation of existing prompt-based segmentation methods.

Most existing approaches model **what** an organ is and **where** it is, but ignore **how it deforms** across patients, anatomies, and disease stages.  
To overcome this limitation, TRIPROMPT introduces a **Population-level Deformation Prompt (PDP)** and integrates it with **structural** and **textual** prompts using a **query-centric alignment mechanism**, enabling robust segmentation under large anatomical variability.

**Triprompt Framework**
![MedicalImaging1-upgraded](https://github.com/user-attachments/assets/433a67ab-38f7-4992-89e7-c6d511c90f6f)

---

## ✨ 2. Key Contributions

1. **Shared 3D Backbone**:  
   TRIPROMPT adopts **Swin-UNETR** as a strong volumetric encoder **without architectural modification**, ensuring reproducibility and fair comparison.

2. **Structural Prompt (Qa)**:  
   Encodes localized anatomical appearance and spatial structure directly from backbone feature maps via a co-trained lightweight 3D region-proposal head — fully automatic, no human input required at test time.

3. **Text Prompt (Qt)**:  
   Injects medical semantic priors using a pretrained clinical language model (**ClinicalBERT**), enabling semantic conditioning without manual spatial input.

4. **Population-Level Deformation Prompt (PDP / Qd)**:  
   Learns non-rigid anatomical deformation patterns from **binary shape masks across subjects**, capturing population-level physiological variability. PDP masks are sampled exclusively from training-split subjects and retrieved at inference via **class-conditional cosine nearest-neighbor** search (top-k=5) — no test-set masks are ever accessed.

5. **Query-Centric Multimodal Alignment**:  
   Segmentation queries attend jointly to **Qa**, **Qt**, and **Qd** via cross-attention and differentiable hard spatial routing (Gumbel-Softmax + STE), followed by residual refinement for stable fusion.

6. **Robust Training Objective**:  
   Combines **multi-label Dice loss** for segmentation with a **contrastive query–prompt alignment loss** to enforce multimodal consistency.

---

## 🧠 3. Method Overview

### Pipeline Summary

1. **Input 3D CT volume** → Swin-UNETR backbone  
2. **Backbone features** → Structural Prompt Encoder (**Qa**) via co-trained region-proposal head  
3. **Medical text descriptions** → Text Prompt Encoder (**Qt**)  
4. **Binary masks from other training subjects** → Deformation Prompt Encoder (**Qd**)  
5. **Segmentation queries** attend to **{Qa, Qt, Qd}** via **TriPromptAligner** with hard spatial routing  
6. **Refined queries** guide voxel-wise, multi-label segmentation

```text
Triprompt/
├── models/
│   ├── backbone.py                # Swin-UNETR backbone
│   ├── structural_prompt.py       # Structural Prompt (Qa)
│   ├── text_prompt.py             # Text Prompt (Qt)
│   ├── deformation_prompt.py      # Deformation Prompt (PDP / Qd)
│   ├── triprompt_aligner.py       # Query-centric alignment
│   └── triprompt_model.py         # Full TRIPROMPT model
│
├── datasets/
│   └── ct_dataset.py              # Unified 3D CT dataset loader
│
├── losses/
│   ├── dice_loss.py
│   └── contrastive_alignment.py
│
├── train.py                       # Training script
├── inference.py                   # Inference script
├── utils.py                       # Reproducibility & checkpointing
├── requirements.txt
└── README.md
```

---

## 📊 4. Supported Datasets

This implementation supports all datasets used in the **TRIPROMPT** paper, once harmonized into a unified **NIfTI** format:

| Dataset | Structures | Split |
|---|---|---|
| FLARE22 | 13 abdominal organs | Test |
| Medical Segmentation Decathlon (MSD) | Multi-organ, tumors | Train/Test |
| LiTS | Liver, liver tumors | Train |
| KiTS19 / KiTS21 | Kidney, kidney tumors | Train |
| AMOS / WORD | Multi-organ | Train |
| CT-ORG | Multi-organ | Train |
| Pancreas-CT | Pancreas | Train |
| AbdomenCT-1K | Abdominal organs | Train |
| Internal colorectal cohort | Colorectal tumors (Stage I–IV) | Test (IRB-exempt, de-identified) |

---

## ⚙️ 5. Installation

### 1️⃣ Create environment (recommended)

```bash
conda create -n triprompt python=3.9 -y
conda activate triprompt
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### Key libraries

- **PyTorch** >= 1.13
- **MONAI** >= 1.1
- **HuggingFace Transformers** >= 4.30
- **NumPy**
- **NiBabel**

---

## 📂 6. Dataset Preparation

### Expected directory structure

```text
data/
├── train/
│   ├── images/
│   │     ├── case_0001.nii.gz
│   │     └── ...
│   └── masks/
│         ├── case_0001.nii.gz
│         └── ...
│
├── val/
│   ├── images/
│   └── masks/
│
├── test/
│   ├── images/
│   └── masks/
│
└── mask_bank/           # PDP training-split mask bank (auto-generated)
    ├── class_00/
    ├── class_01/
    └── ...
```

✔ Images and masks must have matching filenames  
✔ Masks can be binary or multi-label  
✔ The `mask_bank/` is built **exclusively from training subjects** before any evaluation begins

### Preprocessing Pipeline

All datasets are preprocessed with a unified harmonization pipeline:

- **Intensity normalization**: CT Hounsfield unit clipping and z-score normalization per volume
- **Voxel resampling**: all volumes resampled to a common isotropic resolution (1.5 × 1.5 × 1.5 mm)
- **Region-aware cropping**: foreground-based bounding box cropping to reduce background dominance
- **Label harmonization**: label spaces across all 11 datasets mapped to a unified 13-class anatomical ontology following standard cross-dataset normalization practices

---

## 🚀 7. How to Run

### 🔹 Training

```bash
python train.py \
  --data_root ./data \
  --batch_size 2 \
  --input_size 96 96 96 \
  --lr 1e-4 \
  --lr_schedule cosine \
  --warmup_epochs 5 \
  --total_iterations 300000 \
  --augmentation True \
  --num_classes 13 \
  --gumbel_tau 0.5 \
  --topk_pdp 5 \
  --seed 42
```

**What happens:**
- Sets deterministic random seeds across Python, NumPy, and PyTorch
- Loads 3D CT volumes via `CTDataset` with multi-dataset sampling (proportional to dataset size)
- Optimizes: Dice loss + Cross-entropy loss + Contrastive alignment losses (λ₁, λ₂ via gradient-norm balancing, updated per training step)
- Saves model checkpoints every 10 epochs to `checkpoints/`

### Full Training Recipe

| Hyperparameter | Value |
|---|---|
| Backbone | Swin-UNETR (pretrained) |
| Input resolution | 96 × 96 × 96 |
| Batch size | 2 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR schedule | Cosine annealing |
| Warmup | 5 epochs |
| Total iterations | 300,000 |
| Gumbel-Softmax τ | 0.5 (fixed) |
| PDP retrieval top-k | 5 |
| Loss weights λ₁, λ₂ | Gradient-norm balanced (per step) |
| Augmentation | Random flip, rotation, intensity jitter |
| Multi-dataset sampling | Proportional to dataset size |

### 🔹 Resume Training

```python
load_checkpoint(
    checkpoint_path="checkpoints/epoch_50.pth",
    model=model,
    optimizer=optimizer,
    device=device
)
```

This allows training to resume exactly from the saved epoch, including the optimizer state.

### 🔹 Inference

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --data_root ./data/test \
  --mask_bank ./data/mask_bank \
  --topk_pdp 5 \
  --output_dir ./predictions
```

**This will:**
- Load a trained model checkpoint
- Retrieve top-k PDP masks per class from the frozen training-split mask bank via cosine nearest-neighbor
- Run voxel-wise, multi-label segmentation inference
- Save predictions as NIfTI (.nii.gz) files

### Some of our model predicted output Results:

![M3_New_Upgraded](https://github.com/user-attachments/assets/1e7c7209-2417-4c69-8c5c-d0f792300eba)
<img width="352" height="373" alt="Image_Medical_Paper2 (1)" src="https://github.com/user-attachments/assets/e59ae022-fa97-413a-8148-19568c9110f0" />
<img width="706" height="698" alt="Image_Medical_Paper_1 (1)" src="https://github.com/user-attachments/assets/650c0e03-7b74-4d16-8c7b-1cc366f1d443" />

---

## 📈 8. Evaluation

The output predictions are compatible with standard medical image segmentation metrics:

- **Dice Similarity Coefficient (DSC)**
- **95th Percentile Hausdorff Distance (HD95)**
- **Organ-wise / tumor-wise evaluation protocols**

Multi-label output channels are preserved to ensure fair and consistent comparison across datasets.

---

## 🔬 9. Reproducibility

We are committed to full reproducibility. The following details are provided to support independent replication.

### PDP Mask Retrieval Policy

At inference, PDP masks are retrieved from a **frozen mask bank built exclusively from training subjects**. No ground-truth masks from validation or test images are ever accessed.

**Retrieval algorithm:**
1. For each target class `c`, compute cosine similarity between the query image's backbone features and PDP latent embeddings of all training masks for class `c`
2. Retrieve the **top-k = 5** most similar masks
3. Pass retrieved masks through the PDP encoder to obtain deformation conditioning tokens

```python
# Pseudocode for PDP retrieval
def retrieve_pdp_masks(query_features, mask_bank_embeddings, class_c, k=5):
    similarities = cosine_similarity(query_features, mask_bank_embeddings[class_c])
    top_k_indices = torch.topk(similarities, k).indices
    return mask_bank_embeddings[class_c][top_k_indices]
```

Sensitivity to k is stable across k ∈ {1, 3, 5, 10} — full sensitivity analysis provided in supplementary.

### Structural Prompt Generation

The structural prompt `Qa` is generated **fully automatically** by a lightweight 3D region-proposal head co-trained end-to-end with the segmentation network. No human input is required at test time. The proposal head predicts class-specific bounding proposals from backbone features, which are then cropped and passed to the structural encoder.

### Baseline Reimplementation

All baselines (SAM, MedSAM, SAM-Med2D, SAM-Med3D, SegVol, CT-SAM3D, Universal, ZePT) are retrained from scratch under **identical conditions**:

| Setting | Value |
|---|---|
| Data splits | Same FLARE22 official splits |
| Preprocessing | Same harmonized pipeline |
| Backbone | Swin-UNETR (where applicable) |
| Augmentation | Same strategy |
| Training budget | 300,000 iterations |
| Hardware | Same GPU configuration |

### Uncertainty Quantification

Results in Tables 1 and 2 are reported as **mean ± std over 3 independent random seeds** (seeds: 42, 123, 2024) in the camera-ready version.

### Random Seeds

```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Label Space Harmonization

Label spaces across all 11 datasets are mapped to a unified 13-class anatomical ontology. Dataset-specific labels absent in a given volume are masked out during loss computation (ignored in Dice and CE losses) to avoid false negatives during multi-dataset joint training.

### Internal Dataset

The internal colorectal tumor dataset (80 3D CT scans, Stage I–IV) is **IRB-exempt** (de-identified, retrospective clinical data). Public splits are not available for this cohort; all reported results use a fixed 60/20 train/test split.

---

## 📜 License

This project is released for research use only. See `LICENSE` for details.

---

