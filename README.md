# Comparative Study of Post-Hoc vs Inherent Explainability

**Explainable Artificial Intelligence**  
**MSc in Artificial Intelligence, University of Florence**  

## Overview
The objective of this project work is to test some explainability techniques (such as SHAP, LIME, and ProtoPNet) on 2 selected datasets, using 2 different neural architectures. The aim is to evaluate how these techniques behave when the models are trained on datasets containing explicit confounding factors.

The selected datasets are:
- **CLEVR-Hans3**: A visual reasoning dataset with specific class-confounder correlations.
- **MNMath**: A mathematical visual reasoning dataset.

## Implementation Details
We test and evaluate both post-hoc methods and inherently interpretable architectures.

- **2 Neural Architectures**: 
  - `ResNet-50` (Standard Convolutional Neural Network)
  - `ViT` (Vision Transformer)
- **Explainability Techniques**: 
  - Post-hoc: SHAP and LIME applied to the black-box models.
  - Inherent: `ProtoPNet` architecture, which explains its reasoning by dissecting images into prototypes.
- **Bonus Architecture**: A `Hybrid QCNN` is included as an experimental quantum-classical model.
- **Tracking**: Integrated with Weights & Biases for live monitoring of metrics and explanation maps.
- **Hardware**: Multi-GPU support via `torch.nn.parallel.DistributedDataParallel` (DDP).

## Logging Metrics
All experimental runs, including Exploratory Data Analysis (EDA) and generated XAI attribution maps, are publicly logged on Weights & Biases:
- **Project Tracking**: [WandB Project](https://wandb.ai/your-username/XAI_Comparative_Study)

## Reproducibility

You can use the provided Docker environment to ensure strict reproducibility of all experimental results.

### 1. Build Environment
```bash
docker compose build
```

### 2. Exploratory Data Analysis (EDA)
To analyze the dataset distributions and confounding factors:
```bash
docker compose run --rm eda_clevr
docker compose run --rm eda_mnmath
```

### 3. Model Training
To train the respective architectures:
```bash
docker compose run --rm train_resnet50_clevr
docker compose run --rm train_vit_clevr
docker compose run --rm train_protopnet_clevr
```

### 4. Explainability Evaluation
To generate SHAP and LIME explanations and compute their fidelity scores:
```bash
docker compose run --rm explain_resnet50_clevr
docker compose run --rm explain_vit_clevr
```

All experimental metrics, generated explanations, and training logs are automatically synchronized with Weights & Biases.
