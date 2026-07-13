import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

# Imports from merged files
from dataset import get_cle4evr_loaders, get_clevr_hans_loaders, get_mnmath_loaders
from model import ResNet50Classifier, ViTClassifier, ProtoPNet, train_protopnet, HybridQCNNClassifier, HybridQViT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet50", "vit", "protopnet", "hybrid_qcnn", "hybrid_qvit"])
    parser.add_argument("--dataset", type=str, required=True, choices=["cle4evr", "mnmath"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Training"):
        images, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, tuple):
            logits = outputs[0]
            concept_logits = outputs[-1] if len(outputs) >= 2 and outputs[-1] is not None else None
        else:
            logits = outputs
            concept_logits = None
            
        if logits.dim() > 2:
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = criterion(logits, labels)
            
        if concept_logits is not None and "concepts" in batch:
            concepts = batch["concepts"].to(device)
            loss += criterion(concept_logits.view(-1, concept_logits.size(-1)), concepts.view(-1))
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
                concept_logits = outputs[-1] if len(outputs) >= 2 and outputs[-1] is not None else None
            else:
                logits = outputs
                concept_logits = None
                
            if logits.dim() > 2:
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                loss = criterion(logits, labels)
                
            if concept_logits is not None and "concepts" in batch:
                concepts = batch["concepts"].to(device)
                loss += criterion(concept_logits.view(-1, concept_logits.size(-1)), concepts.view(-1))
                
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return total_loss / len(loader), correct / total

def main():
    args = parse_args()
    wandb.init(project="XAI_Comparative_Study", name=f"{args.model}_{args.dataset}")
    wandb.config.update(args)

    # 1. Load Data
    print(f"Loading {args.dataset}...")
    num_equations = 1
    num_concepts = 0
    if args.dataset == "cle4evr":
        train_loader, val_loader, test_loader = get_cle4evr_loaders(root_dir="./CLEVR-Hans3", batch_size=args.batch_size)
        n_classes = 2
        in_channels = 3
    else:
        train_loader, val_loader, test_loader, num_equations, num_concepts = get_mnmath_loaders(batch_size=args.batch_size)
        n_classes = 19
        in_channels = 1

    # 2. Build Model
    print(f"Building {args.model}...")
    kwargs = {"num_equations": num_equations, "num_concepts": num_concepts}
    if args.model == "resnet50":
        model = ResNet50Classifier(n_classes=n_classes, input_channels=in_channels, **kwargs)
    elif args.model == "vit":
        model = ViTClassifier(n_classes=n_classes, pretrained=True, input_channels=in_channels, **kwargs)
    elif args.model == "protopnet":
        model = ProtoPNet(n_classes=n_classes, input_channels=in_channels, n_prototypes_per_class=10, **kwargs)
    elif args.model == "hybrid_qcnn":
        model = HybridQCNNClassifier(n_classes=n_classes, input_channel=in_channels, n_qubits=8, n_layers=1, backend="lightning.qubit", **kwargs)
    elif args.model == "hybrid_qvit":
        model = HybridQViT(n_classes=n_classes, input_channel=in_channels, n_qubits=8, img_size=64, patch_size=8, backend="default.qubit", **kwargs)

    model = model.to(args.device)
    
    # Use multiple GPUs if available (excluding ProtoPNet and QCNN which have custom routines)
    if args.device == "cuda" and torch.cuda.device_count() > 1 and args.model in ["resnet50", "vit"]:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)

    # 3. Train
    if args.model == "protopnet":
        print("Training ProtoPNet using specialized routine...")
        config = {
            "epochs_phase1": args.epochs,
            "epochs_phase3": max(1, args.epochs // 2),
            "lr_features": 1e-4,
            "lr_projection": 1e-4,
            "lr_prototypes": 3e-3,
            "lr_last": 1e-4,
            "lr_last_finetune": 1e-4
        }
        model = train_protopnet(model, train_loader, val_loader, config)
        
        # Save the final model manually since train_protopnet doesn't
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/{args.model}_{args.dataset}_best.pth")
        print("Saved final ProtoPNet model!")
        
        # Visualize and save the learned prototypes
        print("Generating and saving prototypes visualizations...")
        model.visualize_prototypes(train_loader, args.device, save_dir=f"prototypes_{args.dataset}")
        print(f"Prototypes saved to prototypes_{args.dataset}/ directory!")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, args.device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)
            
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs("checkpoints", exist_ok=True)
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, f"checkpoints/{args.model}_{args.dataset}_best.pth")
                print("Saved best model!")

    # 4. Test
    print("\nTesting...")
    # Load best model
    if os.path.exists(f"checkpoints/{args.model}_{args.dataset}_best.pth") and args.model != "protopnet":
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(torch.load(f"checkpoints/{args.model}_{args.dataset}_best.pth"))
    
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), args.device)
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    main()
