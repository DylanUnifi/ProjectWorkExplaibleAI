# train_bdd_oia.py

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from data_loader.bdd_oia_loader import get_bdd_oia_loaders
from models.temporal_qcnn import TemporalQCNN
from utils.checkpoint import save_checkpoint, safe_load_checkpoint
from utils.early_stopping import EarlyStopping
from utils.device import get_device


def train_bdd_oia(config):
    """Train Temporal QCNN on BDD-OIA dataset."""
    
    # Setup
    DEVICE = get_device()
    EXPERIMENT_NAME = f"bdd_oia_{config['experiment_name']}"
    SAVE_DIR = os.path.join("checkpoints", "bdd_oia", EXPERIMENT_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # W&B
    wandb.init(
        project="xai_qcnn",
        name=EXPERIMENT_NAME,
        config=config,
        group="bdd_oia"
    )
    
    # Data
    print("📹 Loading BDD-OIA dataset...")
    train_loader, val_loader, test_loader = get_bdd_oia_loaders(
        root_dir=config["dataset"]["root"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        n_frames=config["dataset"]["n_frames"],
    )
    
    # Model
    print("🔧 Building Temporal QCNN...")
    model = TemporalQCNN(
        in_channels=3,
        n_qubits=config["quantum"]["n_qubits"],
        n_layers=config["quantum"]["layers"],
        backend=config["quantum"]["backend"],
        conv_channels=config["model"]["conv_channels"],
        hidden_sizes=config["model"]["hidden_sizes"],
        n_actions=4,
        n_explanations=21,
        dropout=config["model"]["dropout"],
    )
    
    # Multi-GPU disabled — PennyLane quantum layers are incompatible with nn.DataParallel.
    model = model.to(DEVICE)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Loss functions (multi-task)
    criterion_action = nn.CrossEntropyLoss()
    criterion_explanation = nn.BCEWithLogitsLoss()
    
    # Mixed precision
    use_amp = config["training"].get("use_amp", True)
    scaler = torch.amp.GradScaler('cuda') if (DEVICE.type == "cuda" and use_amp) else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config["training"]["early_stopping"])
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, "logs"))
    
    # Training loop
    best_val_acc = 0
    EPOCHS = config["training"]["epochs"]
    
    for epoch in range(EPOCHS):
        # ──────────────────────────────────────────────────────
        # TRAIN
        # ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in pbar:
            videos = batch["video"].to(DEVICE)  # (B, C, T, H, W)
            actions = batch["action"].to(DEVICE)  # (B,)
            explanations = batch["explanations"].to(DEVICE)  # (B, 21)
            
            optimizer.zero_grad()
            
            # Forward with AMP
            if use_amp:
                with torch.amp.autocast('cuda'):
                    action_logits, explanation_logits = model(videos)
                    
                    # Multi-task loss
                    loss_action = criterion_action(action_logits, actions)
                    loss_explanation = criterion_explanation(explanation_logits, explanations)
                    
                    # Weighted combination
                    loss = loss_action + 0.5 * loss_explanation
                
                # Backward with scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                action_logits, explanation_logits = model(videos)
                loss_action = criterion_action(action_logits, actions)
                loss_explanation = criterion_explanation(explanation_logits, explanations)
                loss = loss_action + 0.5 * loss_explanation
                
                loss.backward()
                optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            preds = action_logits.argmax(dim=1)
            train_correct += (preds == actions).sum().item()
            train_total += len(actions)
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100*train_correct/train_total:.2f}%"
            })
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # ──────────────────────────────────────────────────────
        # VALIDATION
        # ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                videos = batch["video"].to(DEVICE)
                actions = batch["action"].to(DEVICE)
                explanations = batch["explanations"].to(DEVICE)
                
                action_logits, explanation_logits = model(videos)
                
                loss_action = criterion_action(action_logits, actions)
                loss_explanation = criterion_explanation(explanation_logits, explanations)
                loss = loss_action + 0.5 * loss_explanation
                
                val_loss += loss.item()
                preds = action_logits.argmax(dim=1)
                val_correct += (preds == actions).sum().item()
                val_total += len(actions)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Logging
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {100*val_acc:.2f}%")
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model.module if isinstance(model, nn.DataParallel) else model,
                optimizer,
                epoch,
                SAVE_DIR,
                fold=0,
                metric=val_acc
            )
            print(f"  ✅ New best model saved! Val Acc: {100*val_acc:.2f}%")
            
            wandb.run.summary["best_val_acc"] = val_acc
            wandb.run.summary["best_epoch"] = epoch
        
        # Early stopping
        if early_stopping(val_acc):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Scheduler step
        scheduler.step()
    
    # ──────────────────────────────────────────────────────
    # TEST EVALUATION
    # ──────────────────────────────────────────────────────
    print("\n🧪 Testing on test set...")
    
    # Load best model
    checkpoint_path = os.path.join(SAVE_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            videos = batch["video"].to(DEVICE)
            actions = batch["action"].to(DEVICE)
            
            action_logits, _ = model(videos)
            preds = action_logits.argmax(dim=1)
            
            test_correct += (preds == actions).sum().item()
            test_total += len(actions)
    
    test_acc = test_correct / test_total
    print(f"\n📊 Test Accuracy: {100*test_acc:.2f}%")
    
    wandb.log({"test/accuracy": test_acc})
    wandb.finish()
    writer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    train_bdd_oia(config)
