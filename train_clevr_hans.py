# train_clevr_hans.py

"""
Single-model training script for the Hybrid QCNN on CLEVR-Hans.

Usage:
    python train_clevr_hans.py --config configs/clevr_hans_training.yaml
"""

import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import wandb
from tqdm import tqdm

from data_loader.clevr_hans_loader import get_clevr_hans_loaders
from models.clevr_qcnn import CLEVRQCNNClassifier
from utils.device import get_device


def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy on a dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


def train_clevr_hans(config):
    """Train a single Hybrid QCNN model on CLEVR-Hans."""

    device = get_device()
    variant = config["dataset"]["variant"]
    n_classes = config.get("n_classes", 3 if variant == "clevr_hans3" else 7)

    # Load data
    print(f"Loading {variant} dataset...")
    train_loader, val_loader, test_loader = get_clevr_hans_loaders(
        root_dir=config["dataset"]["root"],
        variant=variant,
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"].get("num_workers", 4),
    )

    # Build model
    qcnn_cfg = config["qcnn"]
    model = CLEVRQCNNClassifier(
        n_classes=n_classes,
        input_channel=3,
        n_qubits=qcnn_cfg["n_qubits"],
        n_layers=qcnn_cfg["layers"],
        backend=qcnn_cfg["backend"],
        conv_channels=qcnn_cfg.get("conv_channels"),
        hidden_sizes=qcnn_cfg.get("hidden_sizes"),
        dropout=qcnn_cfg.get("dropout", 0.0),
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    # W&B
    wandb.init(
        project="xai_qcnn",
        name=config.get("experiment_name", "train_clevr_hans"),
        config=config,
    )

    # Training setup
    train_cfg = qcnn_cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    use_amp = train_cfg.get("use_amp", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    epochs = train_cfg["epochs"]
    best_val_acc = 0.0
    save_path = Path(config.get("checkpoint_path", "checkpoints/qcnn_best.pth"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * train_correct / train_total:.2f}%",
            })

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        val_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch + 1}: Train Acc={100 * train_acc:.2f}% | Val Acc={100 * val_acc:.2f}%")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict()
            torch.save({"model_state_dict": state, "epoch": epoch, "val_acc": val_acc}, save_path)
            print(f"  ✅ New best checkpoint saved (val_acc={100 * val_acc:.2f}%)")

        scheduler.step()

    # Final test evaluation
    test_acc = evaluate_model(model, test_loader, device)
    print(f"\nTest Accuracy: {100 * test_acc:.2f}%")
    wandb.log({"test_acc": test_acc})
    wandb.finish()

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_clevr_hans(config)
