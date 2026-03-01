# ProjectWorkExplainableAI

Explainable AI with Quantum-Classical Hybrid CNNs — a comparative study of explanation methods across multiple model architectures on the CLEVR-Hans and BDD-OIA datasets.

## Project Overview

This project investigates explainability techniques (GradCAM, Integrated Gradients, SHAP, LIME) applied to:

- **Quantum Kernel SVM** — quantum feature map + classical SVM with precomputed kernel
- **Hybrid QCNN** (`CLEVRQCNNClassifier`) — PennyLane quantum circuit + classical CNN
- **ResNet-18** — pretrained torchvision backbone with GradCAM support
- **Vision Transformer (ViT)** — transformer-based image classifier
- **TemporalQCNN** — quantum-classical model for video/temporal data (BDD-OIA)

## Repository Structure

```
├── configs/                  # YAML configuration files
├── data_loader/
│   ├── clevr_hans_loader.py  # CLEVR-Hans dataset loader
│   ├── bdd_oia_loader.py     # BDD-OIA dataset loader
│   └── utils.py              # Build transforms, load datasets by name
├── explainability/           # Explainability modules
│   ├── grad_explainer.py     # Integrated Gradients, GradCAM
│   ├── shap_explainer.py     # SHAP explanations
│   ├── lime_explainer.py     # LIME explanations
│   └── metrics.py            # Faithfulness, infidelity, sparsity metrics
├── models/
│   ├── clevr_qcnn.py         # Hybrid QCNN classifier
│   ├── temporal_qcnn.py      # Temporal QCNN for video
│   ├── resnet18_classifier.py# ResNet-18 with GradCAM hooks
│   ├── vit_classifier.py     # Vision Transformer classifier
│   ├── protopnet.py          # ProtoPNet interpretable model
│   └── svm_extension.py      # EnhancedSVM with precomputed kernel support
├── scripts/
│   └── pipeline_backends.py  # Quantum kernel computation engine
├── utils/
│   ├── checkpoint.py         # Save/load model checkpoints
│   ├── device.py             # Device selection utility
│   └── early_stopping.py     # Early stopping callback
├── tests/                    # Unit tests
├── train_all_models.py       # Unified training script
├── train_clevr_hans.py       # CLEVR-Hans training script
├── train_bdd_oia.py          # BDD-OIA training script
├── explain_clevr.py          # Generate CLEVR-Hans explanations
├── explain_bdd.py            # Generate BDD-OIA explanations
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU acceleration with quantum circuits:
```bash
pip install pennylane-lightning[gpu]
```

## Usage

### Training

Train all models on CLEVR-Hans:
```bash
python train_all_models.py --config configs/train_all_models.yaml
```

Train individual models:
```bash
python train_clevr_hans.py --config configs/clevr_hans_training.yaml
python train_bdd_oia.py --config configs/bdd_oia_training.yaml
```

### Explanation Generation

Generate explanations for CLEVR-Hans:
```bash
python explain_clevr.py --config configs/clevr_hans_explain.yaml
```

Generate explanations for BDD-OIA:
```bash
python explain_bdd.py --config configs/bdd_oia_explain.yaml
```

## Docker

```bash
docker-compose up --build
```

## Running Tests

```bash
pytest tests/
```

## Model Descriptions

| Model | Description |
|-------|-------------|
| `CLEVRQCNNClassifier` | Hybrid quantum-classical CNN using PennyLane circuits for feature extraction |
| `TemporalQCNN` | Extends QCNN to video sequences with temporal pooling |
| `ResNet18Classifier` | ResNet-18 with forward/backward hooks for GradCAM |
| `ViTClassifier` | Vision Transformer fine-tuned for CLEVR-Hans classification |
| `EnhancedSVM` | sklearn SVC wrapper with precomputed quantum kernel support, NaN safety, and threshold tuning |

## License

MIT License — see [LICENSE](LICENSE).
