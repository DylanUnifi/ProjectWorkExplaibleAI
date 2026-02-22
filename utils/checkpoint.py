import os
import torch


def save_checkpoint(model, optimizer, epoch, save_dir, fold=0, metric=0.0):
    """Save model checkpoint."""
    path = os.path.join(save_dir, "best_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "fold": fold,
        "metric": metric,
    }, path)


def safe_load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load checkpoint safely."""
    if not os.path.exists(path):
        print(f"Warning: No checkpoint found at {path}")
        return 0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0)
