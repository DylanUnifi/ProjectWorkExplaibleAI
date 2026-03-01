"""Tests for utility modules: EarlyStopping, get_device, save/load checkpoint."""
import os
import pytest
import torch
import tempfile


def test_early_stopping_patience():
    """EarlyStopping triggers after patience epochs with no improvement."""
    from utils.early_stopping import EarlyStopping

    es = EarlyStopping(patience=3, min_delta=0.0)
    assert not es(0.5)
    assert not es(0.4)  # No improvement
    assert not es(0.4)  # No improvement
    assert es(0.4)      # Patience exhausted


def test_early_stopping_improvement():
    """EarlyStopping resets counter on improvement."""
    from utils.early_stopping import EarlyStopping

    es = EarlyStopping(patience=2, min_delta=0.0)
    assert not es(0.5)
    assert not es(0.4)  # No improvement (counter=1)
    assert not es(0.6)  # Improvement -> counter reset
    assert not es(0.5)  # No improvement (counter=1)
    assert es(0.5)      # Patience exhausted (counter=2)


def test_get_device_returns_device():
    """get_device returns a valid torch.device."""
    from utils.device import get_device

    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda")


def test_save_and_load_checkpoint():
    """save_checkpoint and safe_load_checkpoint round-trip."""
    from utils.checkpoint import save_checkpoint, safe_load_checkpoint
    from models.resnet18_classifier import ResNet18Classifier

    model = ResNet18Classifier(n_classes=3, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(model, optimizer, epoch=5, save_dir=tmpdir, fold=0, metric=0.9)
        ckpt_path = os.path.join(tmpdir, "best_model.pth")
        assert os.path.exists(ckpt_path)

        # Load into a fresh model
        new_model = ResNet18Classifier(n_classes=3, pretrained=False)
        epoch = safe_load_checkpoint(ckpt_path, new_model, device="cpu")
        assert epoch == 5


def test_safe_load_checkpoint_missing_file():
    """safe_load_checkpoint returns 0 when file does not exist."""
    from utils.checkpoint import safe_load_checkpoint
    from models.resnet18_classifier import ResNet18Classifier

    model = ResNet18Classifier(n_classes=3, pretrained=False)
    epoch = safe_load_checkpoint("/nonexistent/path.pth", model, device="cpu")
    assert epoch == 0
