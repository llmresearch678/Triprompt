# TRIPROMPT: Deformation-Aware Multimodal Prompting for 3D Segmentation

This repository provides a PyTorch implementation of a TRIPROMPT-style
3D medical image segmentation framework. The system integrates a shared
3D backbone with prompt-based conditioning to achieve robust segmentation
under anatomical variability.

## Key Components
- **Backbone**: Swin-UNETR for volumetric feature extraction
- **Structural Prompt**: Local anatomical appearance encoding
- **Text Prompt**: Medical semantic conditioning
- **Deformation Prompt (PDP)**: Population-level shape variability prior
- **Loss**: Dice loss for class-imbalanced 3D segmentation

## Repository Structure
