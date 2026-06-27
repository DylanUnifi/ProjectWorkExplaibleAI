import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

# Imports from merged files
from dataset import get_clevr_hans_loaders, get_mnmath_loaders
from model import ResNet50Classifier, ViTClassifier, ProtoPNet, train_protopnet, CLEVRQCNNClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet50", "vit", "protopnet", "hybrid_qcnn"])
    parser.add_argument("--dataset", type=str, required=True, choices=["clevr_hans3", "mnmath"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
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
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return total_loss / len(loader), correct / total

def main():
    args = parse_args()
    wandb.init(project="XAI_Comparative_Study", name=f"{args.model}_{args.dataset}")
    wandb.config.update(args)

    # 1. Load Data
    print(f"Loading {args.dataset}...")
    if args.dataset == "clevr_hans3":
        train_loader, val_loader, test_loader = get_clevr_hans_loaders(batch_size=args.batch_size)
        n_classes = 3
        in_channels = 3
    else:
        train_loader, val_loader, test_loader = get_mnmath_loaders(batch_size=args.batch_size)
        n_classes = 2
        in_channels = 1

    # 2. Build Model
    print(f"Building {args.model}...")
    if args.model == "resnet50":
        model = ResNet50Classifier(n_classes=n_classes, input_channels=in_channels)
    elif args.model == "vit":
        model = ViTClassifier(n_classes=n_classes, pretrained=True)
    elif args.model == "protopnet":
        model = ProtoPNet(num_classes=n_classes, num_prototypes=n_classes*10)
    elif args.model == "hybrid_qcnn":
        model = CLEVRQCNNClassifier(n_classes=n_classes, input_channel=in_channels, n_qubits=8, n_layers=1)

        
    model = model.to(args.device)

    # 3. Train
    if args.model == "protopnet":
        print("Training ProtoPNet using specialized routine...")
        model = train_protopnet(model, train_loader, val_loader, args.epochs, args.device, save_dir="checkpoints")
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
                torch.save(model.state_dict(), f"checkpoints/{args.model}_{args.dataset}_best.pth")
                print("Saved best model!")

    # 4. Test
    print("\nTesting...")
    # Load best model
    if os.path.exists(f"checkpoints/{args.model}_{args.dataset}_best.pth") and args.model != "protopnet":
        model.load_state_dict(torch.load(f"checkpoints/{args.model}_{args.dataset}_best.pth"))
    
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), args.device)
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    main()
