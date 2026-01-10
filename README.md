TRIPROMPT
Deformation-Aware Multimodal Prompting for Robust 3D Medical Image Segmentation
<p align="center"> <b>A query-centric framework that jointly models anatomical structure, medical semantics, and population-level deformation for robust 3D segmentation.</b> </p>
🔍 Overview

TRIPROMPT is a deformation-aware, multimodal prompting framework for 3D medical image segmentation.
It addresses a key limitation of existing prompt-based segmentation methods:

they model what an organ is and where it is, but ignore how it deforms across patients and disease stages.

TRIPROMPT introduces a Population-level Deformation Prompt (PDP) and integrates it with structural and textual prompts using a query-centric alignment mechanism, enabling robust segmentation under large anatomical variability.

✨ Key Contributions

Shared 3D Backbone
Uses Swin-UNETR as a strong volumetric encoder without architectural modification, ensuring reproducibility.

Structural Prompt (Qa)
Encodes localized anatomical appearance and spatial structure from backbone features.

Text Prompt (Qt)
Injects medical semantic priors using a pretrained clinical language model (ClinicalBERT).

Population-Level Deformation Prompt (PDP / Qd)
Learns non-rigid anatomical deformation patterns from binary shape masks across subjects.

Query-Centric Multimodal Alignment
Segmentation queries attend jointly to Qa, Qt, and Qd via cross-attention, followed by residual refinement.

Robust Training Objective
Combines multi-label Dice loss with contrastive query–prompt alignment loss.

🧠 Method Overview

Pipeline summary:

Input 3D CT volume → Swin-UNETR backbone

Backbone features → Structural Prompt Encoder (Qa)

Medical text → Text Prompt Encoder (Qt)

Binary masks from other subjects → Deformation Prompt Encoder (Qd)

Segmentation queries attend to {Qa, Qt, Qd} via TriPromptAligner

Refined queries guide voxel-wise segmentation

📁 Repository Structure
Triprompt/
│
├── models/
│   ├── backbone.py              # Swin-UNETR backbone
│   ├── structural_prompt.py     # Structural Prompt (Qa)
│   ├── text_prompt.py           # Text Prompt (Qt)
│   ├── deformation_prompt.py    # Deformation Prompt (PDP / Qd)
│   ├── triprompt_aligner.py     # Query-centric alignment
│   └── triprompt_model.py       # Full TRIPROMPT model
│
├── datasets/
│   └── ct_dataset.py            # Unified 3D CT dataset loader
│
├── losses/
│   ├── dice_loss.py
│   └── contrastive_alignment.py
│
├── train.py                     # Training script
├── inference.py                 # Inference script
├── utils.py                     # Reproducibility & checkpointing
│
├── requirements.txt
└── README.md

📊 Supported Datasets

This implementation supports all datasets used in the TRIPROMPT paper, once harmonized into a unified NIfTI format:

FLARE22

Medical Segmentation Decathlon (MSD)

LiTS

KiTS19 / KiTS21

AMOS / WORD

CT-ORG

Pancreas-CT

AbdomenCT-1K

De-identified internal clinical CT datasets

⚙️ Installation
1️⃣ Create environment (recommended)
conda create -n triprompt python=3.9 -y
conda activate triprompt

2️⃣ Install dependencies
pip install -r requirements.txt


Key libraries:

PyTorch

MONAI

HuggingFace Transformers

NumPy

NiBabel

📂 Dataset Preparation

Expected directory structure:

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


✔ Images and masks must have matching filenames
✔ Masks can be binary or multi-label

🚀 How to Run
🔹 Training
python train.py


What happens:

Sets deterministic random seeds

Loads CT volumes via CTDataset

Optimizes:

Dice loss (segmentation)

Contrastive alignment loss (queries ↔ prompts)

Saves checkpoints every 10 epochs to checkpoints/

🔹 Resume Training
load_checkpoint(
    checkpoint_path="checkpoints/epoch_50.pth",
    model=model,
    optimizer=optimizer,
    device=device
)

🔹 Inference
python inference.py


This will:

Load a trained checkpoint

Run voxel-wise multi-label inference

Save predictions as NIfTI files

Output example:

output/case_001_pred.nii.gz

📈 Evaluation

The output predictions are compatible with:

Dice score

HD95

Organ-wise / tumor-wise evaluation protocols

Multi-label channels are preserved for fair comparison.

🔬 Reproducibility Notes

Fixed random seeds across Python, NumPy, and PyTorch

Deterministic CuDNN behavior

No prompt generation inside dataset loaders

Deformation prompts sampled from different subjects to prevent leakage
