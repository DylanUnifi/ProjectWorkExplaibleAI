# train_clevr_hans.py

# Similar structure à train_bdd_oia.py mais pour CLEVR
# Utilise CLEVRQCNNClassifier et get_clevr_hans_loaders

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from data_loader.clevr_hans_loader import get_clevr_hans_loaders
from models.clevr_qcnn import CLEVRQCNNClassifier
from utils.device import get_device
from utils.early_stopping import EarlyStopping


def train_clevr_hans(config):
    """Train QCNN on CLEVR-Hans dataset."""
    
    DEVICE = get_device()
    EXPERIMENT_NAME = f"clevr_hans_{config['experiment_name']}"
    SAVE_DIR = os.path.join("checkpoints", "clevr_hans", EXPERIMENT_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # W&B
    wandb.init(
        project="xai_qcnn",
        name=EXPERIMENT_NAME,
        config=config,
        group="clevr_hans"
    )
    
    # Data
    variant = config["dataset"]["variant"]  # "clevr_hans3" or "clevr_hans7"
    n_classes = 3 if variant == "clevr_hans3" else 7
    
    print(f"🧩 Loading {variant} dataset...")
    train_loader, val_loader, test_loader = get_clevr_hans_loaders(
        root_dir=config["dataset"]["root"],
        variant=variant,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    
    # Model
    model = CLEVRQCNNClassifier(
        n_classes=n_classes,
        input_channel=3,
        n_qubits=config["quantum"]["n_qubits"],
        n_layers=config["quantum"]["layers"],
        backend=config["quantum"]["backend"],
        conv_channels=config["model"]["conv_channels"],
        hidden_sizes=config["model"]["hidden_sizes"],
        dropout=config["model"]["dropout"],
    )
    
    # Multi-GPU
    if False:  # DataParallel disabled: incompatible with PennyLane quantum layers
        model = nn.DataParallel(model)
    
    model = model.to(DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # AMP
    scaler = torch.amp.GradScaler('cuda') if (DEVICE.type == "cuda" and config["training"].get("use_amp", True)) else None
    
    # Training loop (similaire à BDD-OIA)
    best_val_acc = 0
    EPOCHS = config["training"]["epochs"]
    early_stopping = EarlyStopping(patience=config["training"].get("early_stopping", 15))
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
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
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                logits = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc={100*train_acc:.2f}% | Val Acc={100*val_acc:.2f}%")
        
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        })
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  ✅ Best model saved: {100*val_acc:.2f}%")
        
        scheduler.step()
        
        if early_stopping(val_acc):
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break
    
    # Test
    print("\n🧪 Testing...")
    checkpoint = torch.load(os.path.join(SAVE_DIR, "best_model.pth"))
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            logits = model(images)
            preds = logits.argmax(dim=1)
            
            test_correct += (preds == labels).sum().item()
            test_total += len(labels)
    
    test_acc = test_correct / test_total
    print(f"📊 Test Accuracy: {100*test_acc:.2f}%")
    
    wandb.log({"test/accuracy": test_acc})
    wandb.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    train_clevr_hans(config)
