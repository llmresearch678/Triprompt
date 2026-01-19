**TRIPROMPT**

**Deformation-Aware Multimodal Prompting for Robust 3D Medical Image Segmentation**:  
A query-centric framework that jointly models anatomical structure, medical semantics, and population-level deformation for robust 3D segmentation.

🔍 **1. Overview**

**TRIPROMPT** is a deformation-aware, multimodal prompting framework for **3D medical image segmentation**.
It addresses a key limitation of existing prompt-based segmentation methods.

Most existing approaches model **what** an organ is and **where** it is, but ignore **how it deforms** across patients, anatomies, and disease stages.
To overcome this limitation, TRIPROMPT introduces a **Population-level Deformation Prompt (PDP)** and integrates it with **structural** and **textual** prompts using a **query-centric alignment mechanism**, enabling robust segmentation under large anatomical variability.

**Triprompt Framework**
![MedicalImaging1-upgraded](https://github.com/user-attachments/assets/433a67ab-38f7-4992-89e7-c6d511c90f6f)

✨ **2. Key Contributions**

1. **Shared 3D Backbone**:  
   TRIPROMPT adopts **Swin-UNETR** as a strong volumetric encoder **without architectural modification**, ensuring reproducibility and fair comparison.

2. **Structural Prompt (Qa)**:  
   Encodes localized anatomical appearance and spatial structure directly from backbone feature maps.

3. **Text Prompt (Qt)**:  
   Injects medical semantic priors using a pretrained clinical language model (**ClinicalBERT**), enabling semantic conditioning without manual spatial input.

4. **Population-Level Deformation Prompt (PDP / Qd)**:  
   Learns non-rigid anatomical deformation patterns from **binary shape masks across subjects**, capturing population-level physiological variability.

5. **Query-Centric Multimodal Alignment**:  
   Segmentation queries attend jointly to **Qa**, **Qt**, and **Qd** via cross-attention, followed by residual refinement for stable fusion.

6. **Robust Training Objective**:  
   Combines **multi-label Dice loss** for segmentation with a **contrastive query–prompt alignment loss** to enforce multimodal consistency.

🧠 **3. Method Overview**

**Pipeline Summary**

1. **Input 3D CT volume** → Swin-UNETR backbone  
2. **Backbone features** → Structural Prompt Encoder (**Qa**)  
3. **Medical text descriptions** → Text Prompt Encoder (**Qt**)  
4. **Binary masks from other subjects** → Deformation Prompt Encoder (**Qd**)  
5. **Segmentation queries** attend to **{Qa, Qt, Qd}** via **TriPromptAligner**  
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

📊 **Supported Datasets**

This implementation supports all datasets used in the **TRIPROMPT** paper, once harmonized into a unified **NIfTI** format:

- **FLARE22**
- **Medical Segmentation Decathlon (MSD)**
- **LiTS**
- **KiTS19 / KiTS21**
- **AMOS / WORD**
- **CT-ORG**
- **Pancreas-CT**
- **AbdomenCT-1K**
- **De-identified internal clinical CT datasets**

---

⚙️ **Installation**

### **1️⃣ Create environment (recommended)**

```bash
conda create -n triprompt python=3.9 -y
conda activate triprompt
```

### **2️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

### **Key libraries**

- **PyTorch**
- **MONAI**
- **HuggingFace Transformers**
- **NumPy**
- **NiBabel**

📂 Dataset Preparation

Expected directory structure:

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
├── test/
│   ├── images/
│   └── masks/
```
✔ Images and masks must have matching filenames
✔ Masks can be binary or multi-label

🚀 **How to Run**

---

🔹 **Training**

python train.py

### **What happens:**

- **Sets deterministic random seeds** across Python, NumPy, and PyTorch
- **Loads 3D CT volumes via CTDataset**
- **Optimizes the following objectives:**
  - **Dice loss (segmentation)**
  - **Contrastive alignment loss (queries ↔ prompts)**
- **Saves model checkpoints every 10 epochs** to the **checkpoints/** directory

---

🔹 **Resume Training**

load_checkpoint(
    checkpoint_path="checkpoints/epoch_50.pth",
    model=model,
    optimizer=optimizer,
    device=device
)

**This allows training to resume exactly from the saved epoch**, including the optimizer state.

---

🔹 **Inference**

python inference.py

### **This will:**

- **Load a trained model checkpoint**
- **Run voxel-wise, multi-label segmentation inference**
- **Save predictions as NIfTI (.nii.gz) files**

### **Some of our model predicted output Results:**

![M3_New_Upgraded](https://github.com/user-attachments/assets/1e7c7209-2417-4c69-8c5c-d0f792300eba)
<img width="352" height="373" alt="Image_Medical_Paper2 (1)" src="https://github.com/user-attachments/assets/e59ae022-fa97-413a-8148-19568c9110f0" />
<img width="706" height="698" alt="Image_Medical_Paper_1 (1)" src="https://github.com/user-attachments/assets/650c0e03-7b74-4d16-8c7b-1cc366f1d443" />



---

📈 **Evaluation**

**The output predictions are compatible with standard medical image segmentation metrics, including:**

- **Dice score**
- **HD95**
- **Organ-wise / tumor-wise evaluation protocols**

**Multi-label output channels are preserved** to ensure **fair and consistent comparison** across datasets.

---

🔬 **Reproducibility Notes**

- **Fixed random seeds** across **Python**, **NumPy**, and **PyTorch**
- **Deterministic CuDNN behavior** enabled
- **No prompt generation inside dataset loaders**
- **Deformation prompts sampled from different subjects** to prevent data leakage
