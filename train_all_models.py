# train_all_models.py

"""
Unified training script for all 5 models:
1. Quantum Kernel SVM
2. Hybrid QCNN
3. ResNet-18
4. Vision Transformer (ViT)
5. ProtoPNet
"""

import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import wandb
from tqdm import tqdm

from data_loader.clevr_hans_loader import get_clevr_hans_loaders
from models.resnet18_classifier import ResNet18Classifier
from models.vit_classifier import ViTClassifier
from models.protopnet import ProtoPNet, train_protopnet
from models.clevr_qcnn import CLEVRQCNNClassifier
from scripts.pipeline_backends import compute_kernel_matrix
from models.svm_extension import EnhancedSVM

def train_single_model(model_name, config, train_loader, val_loader, test_loader):
    """
    Train a single model based on model_name.
    
    Returns:
        model: Trained model
        metrics: Dict of metrics
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 3 if config["dataset"]["variant"] == "clevr_hans3" else 7
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}\n")
    
    # ═══════════════════════════════════════════════════════════
    # 1. QUANTUM KERNEL SVM
    # ═══════════════════════════════════════════════════════════
    if model_name == "quantum_kernel_svm":
        from data_loader.utils import extract_features_pca
        
        print("Extracting features with PCA...")
        X_train, y_train, pca = extract_features_pca(
            train_loader,
            n_components=config["quantum_kernel"]["n_qubits"],
        )
        X_val, y_val, _ = extract_features_pca(
            val_loader,
            n_components=config["quantum_kernel"]["n_qubits"],
            pca_model=pca,
        )
        X_test, y_test, _ = extract_features_pca(
            test_loader,
            n_components=config["quantum_kernel"]["n_qubits"],
            pca_model=pca,
        )
        
        # Initialize quantum weights
        import numpy as np
        n_qubits = config["quantum_kernel"]["n_qubits"]
        n_layers = config["quantum_kernel"]["layers"]
        quantum_weights = np.random.normal(0, 0.1, (n_layers, n_qubits)).astype(np.float32)
        
        print("Computing quantum kernels...")
        K_train = compute_kernel_matrix(
            X_train,
            weights=quantum_weights,
            device_name="lightning.gpu",
            symmetric=True,
            gram_backend="tensorcore",
            dtype="float32",
        )
        
        K_val = compute_kernel_matrix(
            X_val, Y=X_train,
            weights=quantum_weights,
            device_name="lightning.gpu",
            symmetric=False,
            gram_backend="tensorcore",
            dtype="float32",
        )
        
        K_test = compute_kernel_matrix(
            X_test, Y=X_train,
            weights=quantum_weights,
            device_name="lightning.gpu",
            symmetric=False,
            gram_backend="tensorcore",
            dtype="float32",
        )
        
        # Train SVM
        print("Training SVM...")
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        
        svm = SVC(kernel="precomputed", probability=True)
        param_grid = {"C": [0.1, 1.0, 10.0, 100.0]}
        grid = GridSearchCV(svm, param_grid, cv=3, scoring="accuracy")
        grid.fit(K_train, y_train)
        
        best_svm = grid.best_estimator_
        
        # Evaluate
        train_acc = best_svm.score(K_train, y_train)
        val_acc = best_svm.score(K_val, y_val)
        test_acc = best_svm.score(K_test, y_test)
        
        metrics = {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "best_C": grid.best_params_["C"],
        }
        
        # Save model
        model_data = {
            "model": best_svm,
            "quantum_weights": quantum_weights,
            "pca": pca,
            "K_train": K_train,
            "X_train": X_train,
        }
        
        return model_data, metrics
    
    # ═══════════════════════════════════════════════════════════
    # 2. HYBRID QCNN
    # ═══════════════════════════════════════════════════════════
    elif model_name == "hybrid_qcnn":
        model = CLEVRQCNNClassifier(
            n_classes=n_classes,
            input_channel=3,
            n_qubits=config["qcnn"]["n_qubits"],
            n_layers=config["qcnn"]["layers"],
            backend=config["qcnn"]["backend"],
            conv_channels=config["qcnn"]["conv_channels"],
            hidden_sizes=config["qcnn"]["hidden_sizes"],
            dropout=config["qcnn"]["dropout"],
        )
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        # Train
        metrics = train_neural_network(
            model, train_loader, val_loader, test_loader,
            config["qcnn"]["training"],
            model_name="qcnn"
        )
        
        return model, metrics
    
    # ═══════════════════════════════════════════════════════════
    # 3. RESNET-18
    # ═══════════════════════════════════════════════════════════
    elif model_name == "resnet18":
        model = ResNet18Classifier(
            n_classes=n_classes,
            pretrained=config["resnet18"]["pretrained"],
            freeze_backbone=config["resnet18"]["freeze_backbone"],
        )
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        metrics = train_neural_network(
            model, train_loader, val_loader, test_loader,
            config["resnet18"]["training"],
            model_name="resnet18"
        )
        
        return model, metrics
    
    # ═══════════════════════════════════════════════════════════
    # 4. VISION TRANSFORMER
    # ═══════════════════════════════════════════════════════════
    elif model_name == "vit":
        model = ViTClassifier(
            n_classes=n_classes,
            pretrained=config["vit"]["pretrained"],
            image_size=224,
        )
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        metrics = train_neural_network(
            model, train_loader, val_loader, test_loader,
            config["vit"]["training"],
            model_name="vit"
        )
        
        return model, metrics
    
    # ═══════════════════════════════════════════════════════════
    # 5. PROTOPNET
    # ═══════════════════════════════════════════════════════════
    elif model_name == "protopnet":
        model = ProtoPNet(
            n_classes=n_classes,
            n_prototypes_per_class=config["protopnet"]["n_prototypes_per_class"],
            prototype_shape=tuple(config["protopnet"]["prototype_shape"]),
            backbone=config["protopnet"]["backbone"],
            pretrained=config["protopnet"]["pretrained"],
        )
        
        model = model.to(device)
        
        # Special 3-phase training
        model = train_protopnet(model, train_loader, val_loader, config["protopnet"]["training"])
        
        # Evaluate
        test_acc = evaluate_model(model, test_loader, device)
        
        metrics = {
            "test_accuracy": test_acc,
        }
        
        return model, metrics
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_neural_network(model, train_loader, val_loader, test_loader, train_config, model_name):
    """Generic training loop for neural networks (QCNN, ResNet, ViT)."""
    
    device = next(model.parameters()).device
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # AMP
    use_amp = train_config.get("use_amp", True)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Training loop
    best_val_acc = 0
    epochs = train_config["epochs"]
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
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
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100*train_correct/train_total:.2f}%"
            })
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        val_acc = evaluate_model(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}: Train Acc={100*train_acc:.2f}% | Val Acc={100*val_acc:.2f}%")
        
        wandb.log({
            f"{model_name}/epoch": epoch,
            f"{model_name}/train_loss": train_loss,
            f"{model_name}/train_acc": train_acc,
            f"{model_name}/val_acc": val_acc,
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save checkpoint
            save_path = Path(f"checkpoints/{model_name}_best.pth")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, save_path)
        
        scheduler.step()
    
    # Test
    test_acc = evaluate_model(model, test_loader, device)
    
    metrics = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
    }
    
    return metrics


def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy."""
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
    
    return correct / total


def main(config_path):
    """Train all models sequentially."""
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # W&B
    wandb.init(
        project="xai_comparative_study",
        name=config["experiment_name"],
        config=config,
    )
    
    # Load data
    print("Loading CLEVR-Hans dataset...")
    train_loader, val_loader, test_loader = get_clevr_hans_loaders(
        root_dir=config["dataset"]["root"],
        variant=config["dataset"]["variant"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )
    
    # Models to train
    models_to_train = config.get("models_to_train", [
        "quantum_kernel_svm",
        "hybrid_qcnn",
        "resnet18",
        "vit",
        "protopnet",
    ])
    
    # Train each model
    results = {}
    
    for model_name in models_to_train:
        try:
            model, metrics = train_single_model(
                model_name, config, train_loader, val_loader, test_loader
            )
            
            results[model_name] = {
                "model": model,
                "metrics": metrics,
            }
            
            # Log to W&B
            wandb.log({
                f"final/{model_name}_test_acc": metrics.get("test_accuracy", metrics.get("test_acc", 0)),
            })
            
            print(f"\n✅ {model_name} trained successfully!")
            print(f"   Test Accuracy: {metrics.get('test_accuracy', metrics.get('test_acc', 0)):.2%}")
        
        except Exception as e:
            print(f"\n❌ {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for model_name, result in results.items():
        acc = result["metrics"].get("test_accuracy", result["metrics"].get("test_acc", 0))
        print(f"{model_name:20s}: {100*acc:.2f}%")
    
    wandb.finish()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    main(args.config)
