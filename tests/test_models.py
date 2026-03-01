"""Basic forward pass tests for all 4 model architectures."""
import pytest
import torch


def test_clevr_qcnn_forward():
    """CLEVRQCNNClassifier forward pass."""
    from models.clevr_qcnn import CLEVRQCNNClassifier

    model = CLEVRQCNNClassifier(
        n_classes=3,
        input_channel=3,
        n_qubits=2,
        n_layers=1,
        backend="default.qubit",
        conv_channels=[8, 16],
        hidden_sizes=[32],
        dropout=0.0,
    )
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 3), f"Expected (2, 3), got {out.shape}"

def test_resnet18_forward():
    """ResNet18Classifier forward pass."""
    from models.resnet18_classifier import ResNet18Classifier

    model = ResNet18Classifier(n_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3), f"Expected (2, 3), got {out.shape}"


def test_resnet18_return_features():
    """ResNet18Classifier forward pass with return_features=True."""
    from models.resnet18_classifier import ResNet18Classifier

    model = ResNet18Classifier(n_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits, features = model(x, return_features=True)
    assert logits.shape == (2, 3)
    assert features.shape == (2, 512)


def test_vit_forward():
    """ViTClassifier forward pass."""
    pytest.importorskip("transformers")
    from models.vit_classifier import ViTClassifier

    model = ViTClassifier(n_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3), f"Expected (2, 3), got {out.shape}"
