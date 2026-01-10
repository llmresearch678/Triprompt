TRIPROMPT: Deformation-Aware Multimodal Prompting for Robust 3D Medical Image Segmentation

This repository provides the official PyTorch implementation of TRIPROMPT, a deformation-aware, query-centric multimodal prompting framework for robust 3D medical image segmentation.
TRIPROMPT is designed to address the limitations of existing prompt-based segmentation methods by jointly modeling what an anatomical structure is (semantics), where it is located (structure), and how it deforms across patients and pathologies (physiology).

The framework integrates a shared 3D backbone with three complementary prompt modalities—structural, textual, and deformation-aware—through a query-centric alignment mechanism, enabling stable and generalizable segmentation under severe anatomical variability.

Key Contributions and Design Principles

Shared 3D Backbone
A Swin-UNETR encoder is used as a common volumetric feature extractor without architectural modification, ensuring strong spatial representations and reproducibility across datasets.

Structural Prompt (Qa)
Encodes localized anatomical appearance and spatial cues derived from image sub-volumes, providing precise geometric grounding for segmentation queries.

Textual Prompt (Qt)
Incorporates medical semantic knowledge using pretrained clinical language models, enabling semantic conditioning and improved generalization to rare organs and tumor classes.

Population-Level Deformation Prompt (PDP / Qd)
Learns statistical, non-rigid deformation patterns from cross-subject shape masks, capturing physiological variations such as respiration-induced motion, tumor growth, and organ displacement without relying on explicit biomechanical models.

Query-Centric Multimodal Alignment
Structural, semantic, and deformation prompts are integrated at the query level, allowing the segmentation head to condition simultaneously on appearance, meaning, and deformation in a stable and interpretable manner.

Loss Design
The framework employs a multi-label Dice loss for voxel-wise segmentation under severe class imbalance, together with contrastive alignment losses to enforce consistency between segmentation queries and multimodal prompt representations.

Supported Datasets

This implementation is designed to support all datasets used in the TRIPROMPT paper, including:

FLARE22

Medical Segmentation Decathlon (MSD)

LiTS (Liver Tumor Segmentation)

KiTS (Kidney Tumor Segmentation)

AMOS / WORD

CT-ORG

Pancreas-CT

AbdomenCT-1K

De-identified internal clinical CT datasets

All datasets are assumed to be harmonized into a unified NIfTI-based format, following standard preprocessing steps such as intensity normalization, voxel resampling, and orientation standardization.

Repository Structure
models/        Backbone and prompt encoders (Qa, Qt, Qd)
datasets/      Unified 3D CT dataset loader
losses/        Dice loss and contrastive alignment losses
train.py       Training pipeline
inference.py   Inference and prediction export
utils.py       Utilities (checkpointing, reproducibility)
README.md      Documentation

Reproducibility and Design Notes

No prompt information is generated at the dataset level, ensuring strict separation between data loading and model conditioning.

The deformation prompt operates exclusively on binary shape masks sampled from training subjects, preventing data leakage.

The codebase is modular and supports clean ablation of individual prompt modalities.

The implementation is intended for research and reproducibility, and aligns with common evaluation protocols in IJCAI, MICCAI, and IEEE TMI.
