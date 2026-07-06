# XAI Comparative Study: Post-Hoc vs Inherent Explainability

**Explainable Artificial Intelligence**  
**MSc in Artificial Intelligence, University of Florence**

## Overview
This repository contains the official codebase for the project work evaluating and comparing **Explainable AI (XAI)** techniques. The primary objective is to investigate the behavior of different neural architectures when trained on datasets containing explicit confounding factors (the *Clever Hans* effect).

We contrast **Post-Hoc Explainability** (SHAP, LIME) applied to black-box models against **Inherent Interpretability** (ProtoPNet). Additionally, we evaluate a **Hybrid Quantum Convolutional Neural Network (QCNN)** to assess the susceptibility of Quantum Machine Learning to confounding biases.

### Datasets
- **CLEVR-Hans3**: A visual reasoning dataset with specific class-confounder correlations injected into the background.
- **MNMath**: A mathematical visual reasoning dataset serving as an unconfounded baseline.

### Architectures
1. `ResNet-50` (Standard Convolutional Neural Network)
2. `ViT` (Vision Transformer)
3. `Hybrid QCNN` (Quantum Convolutional Neural Network - 8 Qubits)
4. `ProtoPNet` (Inherently Interpretable Prototypical Part Network)

## Results & Tracking
All experimental results, including training metrics, accuracy drops (Validation vs Test), generated SHAP/LIME attribution maps, infidelity scores, and ProtoPNet prototypes, are tracked and publicly available on Weights & Biases.

**📊 View the complete results on W&B:** [XAI_Comparative_Study Dashboard](https://wandb.ai/dylan-fouepe-university-of-florence/XAI_Comparative_Study)

## Reproducibility
This project can be fully containerized using Docker to ensure strict reproducibility. No local Python environment setup is required. Hardware acceleration is supported natively.

### 1. Build the Environment
Clone the repository and build the Docker image:
```bash
docker compose build
```

### 2. Exploratory Data Analysis (EDA)
Analyze the dataset distributions and visually inspect the confounding factors:
```bash
docker compose run --rm eda_clevr
docker compose run --rm eda_mnmath
```

### 3. Model Training
Train the architectures on the CLEVR-Hans3 dataset. You can specify GPU visibility using the standard `CUDA_VISIBLE_DEVICES` environment variable:
```bash
CUDA_VISIBLE_DEVICES=N docker compose run --rm train_resnet50_clevr
CUDA_VISIBLE_DEVICES=N docker compose run --rm train_vit_clevr
CUDA_VISIBLE_DEVICES=N docker compose run --rm train_hybrid_qcnn_clevr
CUDA_VISIBLE_DEVICES=N docker compose run --rm train_protopnet_clevr
```
*(To train on the MNMath baseline dataset, replace `_clevr` with `_mnmath` in the commands above).*

### 4. Explainability Evaluation
Generate SHAP and LIME explanations, compute their infidelity scores, and automatically log the attribution heatmaps to W&B:
```bash
CUDA_VISIBLE_DEVICES=N docker compose run --rm explain_resnet50_clevr
CUDA_VISIBLE_DEVICES=N docker compose run --rm explain_vit_clevr
CUDA_VISIBLE_DEVICES=N docker compose run --rm explain_hybrid_qcnn_clevr
```
*(Note: ProtoPNet does not require a post-hoc explain script as its interpretability is inherent and logged directly during the evaluation phase).*
