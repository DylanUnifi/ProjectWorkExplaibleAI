# models/protopnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm

class ProtoPNet(nn.Module):
    """
    Prototypical Part Network (Chen et al., 2019).
    
    Architecture:
    1. CNN feature extractor (ResNet-18 backbone)
    2. Prototype layer (learned prototypes)
    3. Distance computation (L2 distance)
    4. Fully connected layer (prototype → class)
    
    Interpretability:
    - Each prototype represents a visual pattern
    - Predictions = weighted similarity to prototypes
    - Visualizable: "This image is class X because it looks like prototype P"
    """
    
    def __init__(
        self,
        n_classes=3,
        n_prototypes_per_class=10,
        prototype_shape=(128, 1, 1),  # (channels, H, W)
        backbone="resnet18",
        pretrained=True,
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.n_prototypes = n_classes * n_prototypes_per_class
        self.prototype_shape = prototype_shape
        
        # ═══════════════════════════════════════════════════
        # 1. Feature Extractor (CNN backbone)
        # ═══════════════════════════════════════════════════
        if backbone == "resnet18":
            if pretrained:
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                resnet = models.resnet18(weights=None)
            # Remove FC and avgpool
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Projection to prototype dimension
        self.projection = nn.Conv2d(
            feature_dim,
            prototype_shape[0],
            kernel_size=1,
            bias=False
        )
        
        # ═══════════════════════════════════════════════════
        # 2. Prototype Layer (learned)
        # ═══════════════════════════════════════════════════
        # Shape: (n_prototypes, prototype_channels, 1, 1)
        self.prototypes = nn.Parameter(
            torch.randn(self.n_prototypes, *prototype_shape),
            requires_grad=True
        )
        
        # Initialize prototypes
        nn.init.kaiming_normal_(self.prototypes, mode='fan_out')
        
        # ═══════════════════════════════════════════════════
        # 3. Classification Layer
        # ═══════════════════════════════════════════════════
        # Each prototype votes for classes
        # Shape: (n_classes, n_prototypes)
        self.last_layer = nn.Linear(self.n_prototypes, n_classes, bias=False)
        
        # Initialize: each prototype connected to its class
        self._initialize_last_layer()
        
        # Epsilon for numerical stability
        self.epsilon = 1e-4
    
    def _initialize_last_layer(self):
        """
        Initialize last layer:
        - Prototypes of class i have positive weight for class i
        - Prototypes of other classes have small negative weight
        """
        n_prototypes_per_class = self.n_prototypes // self.n_classes
        
        with torch.no_grad():
            for j in range(self.n_prototypes):
                class_idx = j // n_prototypes_per_class
                
                for c in range(self.n_classes):
                    if c == class_idx:
                        self.last_layer.weight[c, j] = 1.0
                    else:
                        self.last_layer.weight[c, j] = -0.5
    
    def forward(self, x, return_distances=False):
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) input images
            return_distances: If True, return prototype distances
        
        Returns:
            logits: (B, n_classes)
            min_distances: (B, n_prototypes) [if return_distances]
        """
        batch_size = x.shape[0]
        
        # ═══════════════════════════════════════════════════
        # 1. Extract features
        # ═══════════════════════════════════════════════════
        features = self.features(x)  # (B, 512, H', W')
        features = self.projection(features)  # (B, prototype_dim, H', W')
        
        # ═══════════════════════════════════════════════════
        # 2. Compute distances to prototypes
        # ═══════════════════════════════════════════════════
        # For each spatial location, compute L2 distance to each prototype
        
        # Reshape features: (B, C, H', W') → (B, C, H'×W')
        B, C, H, W = features.shape
        features_flat = features.view(B, C, H * W)  # (B, C, H×W)
        
        # Prototypes: (n_prototypes, C, 1, 1) → (n_prototypes, C)
        prototypes_flat = self.prototypes.view(self.n_prototypes, C)
        
        # Compute pairwise squared L2 distances
        # distances[b, p, loc] = ||features[b, :, loc] - prototypes[p, :]||^2
        
        # Expand dimensions for broadcasting
        features_expanded = features_flat.unsqueeze(1)  # (B, 1, C, H×W)
        prototypes_expanded = prototypes_flat.unsqueeze(0).unsqueeze(-1)  # (1, n_prototypes, C, 1)
        
        # Squared L2 distance
        distances_sq = torch.sum(
            (features_expanded - prototypes_expanded) ** 2,
            dim=2
        )  # (B, n_prototypes, H×W)
        
        # Min distance per prototype (over all spatial locations)
        min_distances, _ = torch.min(distances_sq, dim=2)  # (B, n_prototypes)
        
        # ═══════════════════════════════════════════════════
        # 3. Convert distances to similarities (inverted)
        # ═══════════════════════════════════════════════════
        # similarity = log((distance + 1) / (distance + epsilon))
        # When distance is small → similarity is large
        
        similarities = torch.log((min_distances + 1) / (min_distances + self.epsilon))
        
        # ═══════════════════════════════════════════════════
        # 4. Classification
        # ═══════════════════════════════════════════════════
        logits = self.last_layer(similarities)  # (B, n_classes)
        
        if return_distances:
            return logits, min_distances
        
        return logits
    
    def push_prototypes(self, dataloader, device):
        """
        Prototype projection step (ProtoPNet training phase 2).
        
        For each prototype, find the nearest patch in training data
        and replace the prototype with that patch.
        
        This makes prototypes correspond to actual image patches.
        """
        self.eval()
        
        # Find nearest patches for each prototype
        nearest_patches = [None] * self.n_prototypes
        min_distances_global = [float('inf')] * self.n_prototypes
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Pushing prototypes"):
                images = batch["image"].to(device)
                
                # Extract features
                features = self.features(images)
                features = self.projection(features)  # (B, C, H', W')
                
                B, C, H, W = features.shape
                features_flat = features.view(B, C, H * W).permute(0, 2, 1)  # (B, H×W, C)
                
                # For each prototype
                for p in range(self.n_prototypes):
                    prototype = self.prototypes[p].view(C)  # (C,)
                    
                    # Compute distances to all patches in batch
                    distances = torch.sum(
                        (features_flat - prototype.unsqueeze(0).unsqueeze(0)) ** 2,
                        dim=2
                    )  # (B, H×W)
                    
                    # Find minimum
                    min_dist, min_idx = torch.min(distances.view(-1), dim=0)
                    
                    if min_dist < min_distances_global[p]:
                        min_distances_global[p] = min_dist.item()
                        
                        # Extract patch
                        batch_idx = min_idx // (H * W)
                        spatial_idx = min_idx % (H * W)
                        h_idx = spatial_idx // W
                        w_idx = spatial_idx % W
                        
                        nearest_patch = features[batch_idx, :, h_idx, w_idx]
                        nearest_patches[p] = nearest_patch.cpu()
        
        # Replace prototypes
        with torch.no_grad():
            for p in range(self.n_prototypes):
                if nearest_patches[p] is not None:
                    self.prototypes[p] = nearest_patches[p].to(device).view(self.prototype_shape)
        
        print(f"Prototypes pushed. Avg distance: {sum(min_distances_global) / len(min_distances_global):.4f}")
    
    def visualize_prototypes(self, dataloader, device, save_dir="prototypes"):
        """
        Visualize learned prototypes by finding their nearest patches.
        """
        import os
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.eval()
        
        # For each prototype, find nearest image patch
        with torch.no_grad():
            for p in range(self.n_prototypes):
                min_distance = float('inf')
                nearest_image = None
                nearest_location = None
                
                for batch in dataloader:
                    images = batch["image"].to(device)
                    
                    # Extract features
                    features = self.features(images)
                    features = self.projection(features)
                    
                    B, C, H, W = features.shape
                    
                    # Compute distances
                    prototype = self.prototypes[p].view(C, 1, 1)
                    distances = torch.sum((features - prototype) ** 2, dim=1)  # (B, H, W)
                    
                    # Find minimum
                    min_dist, min_idx = torch.min(distances.view(B, -1), dim=1)
                    batch_min_dist, batch_idx = torch.min(min_dist, dim=0)
                    
                    if batch_min_dist < min_distance:
                        min_distance = batch_min_dist.item()
                        
                        spatial_idx = min_idx[batch_idx]
                        h = spatial_idx // W
                        w = spatial_idx % W
                        
                        nearest_image = images[batch_idx].cpu()
                        nearest_location = (h.item(), w.item())
                
                # Plot
                if nearest_image is not None:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    
                    # Full image
                    img = nearest_image.permute(1, 2, 0).numpy()
                    img = (img - img.min()) / (img.max() - img.min())
                    axes[0].imshow(img)
                    axes[0].set_title(f"Prototype {p}")
                    axes[0].axis("off")
                    
                    # Heatmap
                    h, w = nearest_location
                    heatmap = torch.zeros_like(nearest_image[0])
                    receptive_field_size = 16  # Approximation
                    heatmap[
                        max(0, h-receptive_field_size):min(heatmap.shape[0], h+receptive_field_size),
                        max(0, w-receptive_field_size):min(heatmap.shape[1], w+receptive_field_size)
                    ] = 1.0
                    
                    axes[1].imshow(img)
                    axes[1].imshow(heatmap.numpy(), cmap="hot", alpha=0.5)
                    axes[1].set_title(f"Nearest patch (dist={min_distance:.2f})")
                    axes[1].axis("off")
                    
                    plt.tight_layout()
                    plt.savefig(save_dir / f"prototype_{p}.png", dpi=150)
                    plt.close()


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
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


# Special training procedure for ProtoPNet
def train_protopnet(model, train_loader, val_loader, config):
    """
    3-phase training for ProtoPNet:
    1. Joint training (all layers)
    2. Prototype pushing (find nearest patches)
    3. Last layer fine-tuning (freeze prototypes)
    """
    
    device = torch.device("cuda")
    model = model.to(device)
    
    # ═══════════════════════════════════════════════════
    # PHASE 1: Joint Training
    # ═══════════════════════════════════════════════════
    print("Phase 1: Joint training...")
    
    optimizer = torch.optim.Adam([
        {"params": model.features.parameters(), "lr": config["lr_features"]},
        {"params": model.projection.parameters(), "lr": config["lr_projection"]},
        {"params": model.prototypes, "lr": config["lr_prototypes"]},
        {"params": model.last_layer.parameters(), "lr": config["lr_last"]},
    ])
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config["epochs_phase1"]):
        model.train()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            logits, min_distances = model(images, return_distances=True)
            
            # Multi-component loss
            loss_ce = criterion(logits, labels)
            
            # Clustering loss: minimize distance to class prototypes
            n_prototypes_per_class = model.n_prototypes // model.n_classes
            cluster_loss = 0
            for c in range(model.n_classes):
                class_mask = (labels == c)
                if class_mask.any():
                    class_prototypes = range(c * n_prototypes_per_class, (c+1) * n_prototypes_per_class)
                    class_distances = min_distances[class_mask][:, class_prototypes]
                    cluster_loss += torch.mean(torch.min(class_distances, dim=1)[0])
            
            cluster_loss /= model.n_classes
            
            # Separation loss: maximize distance to other prototypes
            separation_loss = -torch.mean(min_distances)
            
            # Total loss
            loss = loss_ce + 0.8 * cluster_loss + 0.08 * separation_loss
            
            loss.backward()
            optimizer.step()
        
        # Validation
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Phase 1 Epoch {epoch+1}: Val Acc = {val_acc:.2%}")
    
    # ═══════════════════════════════════════════════════
    # PHASE 2: Prototype Pushing
    # ═══════════════════════════════════════════════════
    print("\nPhase 2: Pushing prototypes...")
    model.push_prototypes(train_loader, device)
    
    # ═══════════════════════════════════════════════════
    # PHASE 3: Last Layer Fine-tuning
    # ═══════════════════════════════════════════════════
    print("\nPhase 3: Fine-tuning last layer...")
    
    # Freeze all except last layer
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.projection.parameters():
        param.requires_grad = False
    model.prototypes.requires_grad = False
    
    optimizer_last = torch.optim.Adam(
        model.last_layer.parameters(),
        lr=config["lr_last_finetune"]
    )
    
    for epoch in range(config["epochs_phase3"]):
        model.train()
        
        for batch in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer_last.zero_grad()
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer_last.step()
        
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Phase 3 Epoch {epoch+1}: Val Acc = {val_acc:.2%}")
    
    return model
