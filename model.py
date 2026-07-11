# models/resnet50_classifier.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm


class NonInPlaceBottleneckBlock(nn.Module):
    """
    Wrapper for ResNet-50 Bottleneck blocks that converts in-place operations
    to out-of-place operations for SHAP/gradient compatibility.
    """
    def __init__(self, block):
        super().__init__()
        self.block = block
    
    def forward(self, x):
        identity = x
        
        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)
        
        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out = self.block.relu(out)
        
        out = self.block.conv3(out)
        out = self.block.bn3(out)
        
        if self.block.downsample is not None:
            identity = self.block.downsample(x)
        
        # Use non-inplace addition
        out = out + identity
        out = self.block.relu(out)
        
        return out


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 for CLEVR-Hans / BDD-OIA / MNMath.
    Optimized for AMP (Tensor Cores) and SHAP compatibility.
    """
    
    def __init__(self, n_classes=3, input_channels=3, pretrained=True, freeze_backbone=False, num_equations=1, num_concepts=0):
        super().__init__()
        
        self.num_equations = num_equations
        self.n_classes = n_classes
        self.num_concepts = num_concepts
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        if input_channels != 3 or n_classes == 19:
            self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()
        
        # Disable inplace ReLU operations for gradient computation compatibility
        # (required for SHAP and other gradient-based explainers)
        for module in self.backbone.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        
        # Wrap bottleneck blocks to convert inplace additions to non-inplace
        for layer in [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]:
            for i, block in enumerate(layer):
                layer[i] = NonInPlaceBottleneckBlock(block)
        
        # Freeze early layers si transfer learning
        if freeze_backbone:
            for param in list(self.backbone.parameters())[:-15]:  # Freeze all but last few params
                param.requires_grad = False
        
        # Replace final FC
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, n_classes * max(1, num_equations))
        
        if self.num_concepts > 0:
            self.concept_head = nn.Linear(num_features, self.num_concepts * 10)
        else:
            self.concept_head = None
        
        # Store intermediate activations for GradCAM
        self.activations = {}
        self.gradients = {}
        
        # Register hooks for layer4 (last conv block)
        self.backbone.layer4.register_forward_hook(self._save_activation)
        self.backbone.layer4.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations['layer4'] = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients['layer4'] = grad_output[0]
    
    def forward(self, x, return_features=False):
        # Forward through backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # (B, 2048, H', W')
        
        # Global average pooling
        x = self.backbone.avgpool(x)  # (B, 2048, 1, 1)
        features = torch.flatten(x, 1)  # (B, 2048)
        
        # Classification
        logits = self.backbone.fc(features)
        if self.num_equations > 1:
            logits = logits.view(features.size(0), self.num_equations, self.n_classes)
            
        concept_logits = None
        if self.concept_head is not None:
            concept_logits = self.concept_head(features).view(features.size(0), self.num_concepts, 10)
        
        if self.num_concepts > 0 or self.num_equations > 1:
            if return_features:
                return logits, features, concept_logits
            return logits, concept_logits
        else:
            if return_features:
                return logits, features
            return logits
    
    def get_gradcam_weights(self):
        if 'layer4' not in self.activations or 'layer4' not in self.gradients:
            raise ValueError("Forward and backward pass required first")
        gradients = self.gradients['layer4']
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        return weights
    
    def generate_gradcam(self, class_idx=None):
        weights = self.get_gradcam_weights()
        activations = self.activations['layer4']
        cam = torch.sum(weights * activations, dim=1)
        cam = torch.clamp(cam, min=0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


# models/vit_classifier.py

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class ViTClassifier(nn.Module):
    """
    Vision Transformer pour CLEVR-Hans.
    Utilise HuggingFace Transformers (optimisé, facile).
    """
    
    def __init__(self, n_classes=3, pretrained=True, image_size=224, input_channels=3, num_equations=1, num_concepts=0):
        super().__init__()
        
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.num_equations = num_equations
        self.num_concepts = num_concepts
        
        if pretrained:
            # Load pretrained ViT (ImageNet)
            self.vit = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                num_labels=n_classes * max(1, num_equations),
                ignore_mismatched_sizes=True,
            )
        else:
            # Train from scratch
            config = ViTConfig(
                image_size=image_size,
                num_labels=n_classes * max(1, num_equations),
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.vit = ViTForImageClassification(config)
            
        if self.num_concepts > 0:
            self.concept_head = nn.Linear(self.vit.config.hidden_size, self.num_concepts * 10)
        else:
            self.concept_head = None
        
        # Store attention weights
        self.attention_weights = []
    
    def forward(self, x, output_attentions=False):
        """
        Args:
            x: (B, C, H, W)
            output_attentions: If True, return attention weights
        
        Returns:
            logits: (B, n_classes)
            attentions: List of (B, n_heads, n_patches, n_patches) [if output_attentions]
        """
        if self.input_channels == 1 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # (B, 3, H, W)
            
        # Resize if image size doesn't match the model's expected size
        expected_size = self.vit.config.image_size
        if x.shape[2] != expected_size or x.shape[3] != expected_size:
            x = torch.nn.functional.interpolate(
                x, size=(expected_size, expected_size), mode="bilinear", align_corners=False
            )
            
        outputs = self.vit(
            pixel_values=x,
            output_attentions=output_attentions,
            return_dict=True,
            output_hidden_states=(self.num_concepts > 0),
        )
        
        logits = outputs.logits
        if self.num_equations > 1:
            logits = logits.view(x.size(0), self.num_equations, self.n_classes)
            
        concept_logits = None
        if self.concept_head is not None:
            # CLS token representation from the last hidden state
            cls_token_state = outputs.hidden_states[-1][:, 0, :]
            concept_logits = self.concept_head(cls_token_state).view(x.size(0), self.num_concepts, 10)
            
        if self.num_concepts > 0 or self.num_equations > 1:
            if output_attentions:
                self.attention_weights = outputs.attentions
                return logits, outputs.attentions, concept_logits
            return logits, concept_logits
        else:
            if output_attentions:
                self.attention_weights = outputs.attentions
                return logits, outputs.attentions
            return logits
    
    def visualize_attention(self, attention_weights, image_size=224, patch_size=16):
        """
        Visualize attention maps from ViT.
        
        Args:
            attention_weights: List of attention tensors from all layers
            
        Returns:
            attention_map: (B, H, W) aggregated attention map
        """
        # Take attention from last layer
        last_layer_attn = attention_weights[-1]  # (B, n_heads, n_patches+1, n_patches+1)
        
        # Average over heads
        avg_attn = last_layer_attn.mean(dim=1)  # (B, n_patches+1, n_patches+1)
        
        # Extract [CLS] token attention to patches (exclude [CLS] to [CLS])
        cls_attn = avg_attn[:, 0, 1:]  # (B, n_patches)
        
        # Reshape to spatial grid
        n_patches = int(cls_attn.shape[1] ** 0.5)
        attention_map = cls_attn.reshape(-1, n_patches, n_patches)  # (B, sqrt(n_patches), sqrt(n_patches))
        
        # Upsample to image size
        attention_map = torch.nn.functional.interpolate(
            attention_map.unsqueeze(1),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, H, W)
        
        return attention_map
    
    def attention_rollout(self, attention_weights, discard_ratio=0.9):
        """
        Attention Rollout (Abnar & Zuidema, 2020).
        Aggregate attention across all layers.
        
        More accurate than last layer attention.
        """
        # Check if attention_weights is valid
        if not attention_weights or len(attention_weights) == 0:
            raise ValueError("No attention weights provided")
        
        # Start with identity
        result = torch.eye(attention_weights[0].shape[-1])
        result = result.unsqueeze(0).to(attention_weights[0].device)
        
        for attn in attention_weights:
            # Average over heads
            attn_heads_fused = attn.mean(dim=1)  # (B, n_patches, n_patches)
            
            # Add residual (identity matrix)
            I = torch.eye(attn_heads_fused.shape[-1]).to(attn.device)
            attn_heads_fused = attn_heads_fused + I
            
            # Normalize
            attn_heads_fused = attn_heads_fused / attn_heads_fused.sum(dim=-1, keepdim=True)
            
            # Multiply (chain rule)
            result = torch.matmul(attn_heads_fused, result)
        
        # Extract [CLS] token attention to patches
        cls_attention = result[:, 0, 1:]  # (B, n_patches)
        
        # Reshape to spatial
        n_patches = int(cls_attention.shape[1] ** 0.5)
        attention_map = cls_attention.reshape(-1, n_patches, n_patches)
        
        return attention_map


# models/protopnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm


class NonInPlaceBottleneckBlock(nn.Module):
    """
    Wrapper for ResNet-50 Bottleneck blocks that converts in-place operations
    to out-of-place operations for SHAP/gradient compatibility.
    """
    def __init__(self, block):
        super().__init__()
        self.block = block
    
    def forward(self, x):
        identity = x
        
        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)
        
        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out = self.block.relu(out)
        
        out = self.block.conv3(out)
        out = self.block.bn3(out)
        
        if self.block.downsample is not None:
            identity = self.block.downsample(x)
        
        # Use non-inplace addition
        out = out + identity
        out = self.block.relu(out)
        
        return out

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
        input_channels=3,
        n_prototypes_per_class=10,
        prototype_shape=(128, 1, 1),  # (channels, H, W)
        backbone="resnet50",
        pretrained=True,
        num_equations=1,
        num_concepts=0,
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.num_equations = num_equations
        self.num_concepts = num_concepts
        self.n_prototypes = n_classes * n_prototypes_per_class
        self.prototype_shape = prototype_shape
        
        # ═══════════════════════════════════════════════════
        # 1. Feature Extractor (CNN backbone)
        # ═══════════════════════════════════════════════════
        if backbone == "resnet50":
            if pretrained:
                resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                resnet = models.resnet50(weights=None)
                
            if input_channels != 3 or n_classes == 19:
                resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
                resnet.maxpool = nn.Identity()
            
            # Disable inplace ReLU operations for gradient computation compatibility
            for module in resnet.modules():
                if isinstance(module, nn.ReLU):
                    module.inplace = False
            
            # Wrap bottleneck blocks to convert inplace additions to non-inplace
            for layer in [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]:
                for i, block in enumerate(layer):
                    layer[i] = NonInPlaceBottleneckBlock(block)
            
            # Remove FC and avgpool
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            feature_dim = 2048
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
        # Shape: (n_classes * num_equations, n_prototypes)
        self.last_layer = nn.Linear(self.n_prototypes, n_classes * max(1, num_equations), bias=False)
        
        # Initialize: each prototype connected to its class
        self._initialize_last_layer()
        
        # Concept head
        if self.num_concepts > 0:
            self.concept_head = nn.Linear(2048, self.num_concepts * 10)
        else:
            self.concept_head = None
        
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
                for e in range(max(1, self.num_equations)):
                    for c in range(self.n_classes):
                        idx = e * self.n_classes + c
                        if c == class_idx:
                            self.last_layer.weight[idx, j] = 1.0
                        else:
                            self.last_layer.weight[idx, j] = -0.5
    
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
        logits = self.last_layer(similarities)  # (B, n_classes * num_equations)
        if self.num_equations > 1:
            logits = logits.view(batch_size, self.num_equations, self.n_classes)
            
        concept_logits = None
        if self.concept_head is not None:
            pooled_features = torch.nn.functional.adaptive_avg_pool2d(self.features(x), (1, 1)).view(batch_size, -1)
            concept_logits = self.concept_head(pooled_features).view(batch_size, self.num_concepts, 10)
        
        if self.num_concepts > 0 or self.num_equations > 1:
            if return_distances:
                return logits, min_distances, concept_logits
            return logits, concept_logits
        else:
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
        
        # Log to wandb if a run is active
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "prototypes": [wandb.Image(str(save_dir / f"prototype_{p}.png"), caption=f"Prototype {p}") 
                                 for p in range(self.n_prototypes) if (save_dir / f"prototype_{p}.png").exists()]
                })
                print("Prototypes successfully logged to W&B!")
        except ImportError:
            pass


def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy on a dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
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
    
    params_list = [
        {"params": model.features.parameters(), "lr": config["lr_features"]},
        {"params": model.projection.parameters(), "lr": config["lr_projection"]},
        {"params": model.prototypes, "lr": config["lr_prototypes"]},
        {"params": model.last_layer.parameters(), "lr": config["lr_last"]},
    ]
    if hasattr(model, 'concept_head') and model.concept_head is not None:
        params_list.append({"params": model.concept_head.parameters(), "lr": config["lr_last"]})
        
    optimizer = torch.optim.Adam(params_list)
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config["epochs_phase1"]):
        model.train()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images, return_distances=True)
            if model.num_concepts > 0:
                logits, min_distances, concept_logits = outputs
            else:
                logits, min_distances = outputs
            
            # Multi-component loss
            if logits.dim() > 2:
                loss_ce = criterion(logits.view(-1, model.n_classes), labels.view(-1))
            else:
                loss_ce = criterion(logits, labels)
            
            if model.num_concepts > 0:
                concepts = batch["concepts"].to(device)
                loss_concept = criterion(concept_logits.view(-1, 10), concepts.view(-1))
            else:
                loss_concept = 0
            
            # Clustering loss: minimize distance to class prototypes
            n_prototypes_per_class = model.n_prototypes // model.n_classes
            cluster_loss = 0
            for c in range(model.n_classes):
                if labels.dim() > 1:
                    class_mask = (labels == c).any(dim=1)
                else:
                    class_mask = (labels == c)
                if class_mask.any():
                    class_prototypes = range(c * n_prototypes_per_class, (c+1) * n_prototypes_per_class)
                    class_distances = min_distances[class_mask][:, class_prototypes]
                    cluster_loss += torch.mean(torch.min(class_distances, dim=1)[0])
            
            cluster_loss /= model.n_classes
            
            # Separation loss: maximize distance to other prototypes
            separation_loss = 0
            for c in range(model.n_classes):
                if labels.dim() > 1:
                    class_mask = ~(labels == c).any(dim=1)
                else:
                    class_mask = (labels != c)
                if class_mask.any():
                    class_prototypes = range(c * n_prototypes_per_class, (c+1) * n_prototypes_per_class)
                    class_distances = min_distances[class_mask][:, class_prototypes]
                    separation_loss += torch.mean(torch.min(class_distances, dim=1)[0])
            
            separation_loss = -separation_loss / max(1, model.n_classes - 1)
            
            # Total loss
            loss = loss_ce + 0.8 * cluster_loss + 0.08 * separation_loss + loss_concept
            
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
    
    params_list_last = [{"params": model.last_layer.parameters(), "lr": config["lr_last_finetune"]}]
    if hasattr(model, 'concept_head') and model.concept_head is not None:
        params_list_last.append({"params": model.concept_head.parameters(), "lr": config["lr_last_finetune"]})
        
    optimizer_last = torch.optim.Adam(params_list_last)
    
    for epoch in range(config["epochs_phase3"]):
        model.train()
        
        for batch in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer_last.zero_grad()
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            if logits.dim() > 2:
                loss = criterion(logits.view(-1, model.n_classes), labels.view(-1))
            else:
                loss = criterion(logits, labels)
            
            if hasattr(model, 'num_concepts') and model.num_concepts > 0:
                concept_logits = outputs[-1]
                concepts = batch["concepts"].to(device)
                loss += criterion(concept_logits.view(-1, 10), concepts.view(-1))
            
            loss.backward()
            optimizer_last.step()
        
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Phase 3 Epoch {epoch+1}: Val Acc = {val_acc:.2%}")
    
    return model


# models/clevr_qcnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block for 2D CNN (same as HybridQCNNBinaryClassifier)."""
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)


def _create_quantum_layer(n_qubits, n_layers=2, backend="lightning.gpu"):
    """Create PennyLane quantum layer."""
    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        # Data Re-uploading: interleaving data embedding with parameterized layers
        for i in range(n_layers):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Z")
            qml.templates.StronglyEntanglingLayers(weights[i:i+1], wires=range(n_qubits))
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    for name, param in layer.named_parameters():
        if "weights" in name:
            nn.init.uniform_(param, -np.pi / 2, np.pi / 2)

    return layer


class CLEVRQCNNClassifier(nn.Module):
    """
    QCNN for CLEVR-Hans (3 or 7 classes).
    """

    def __init__(
        self,
        n_classes=3,
        input_channel=3,
        n_qubits=4,
        n_layers=1,
        backend="lightning.qubit",
        conv_channels=None,
        hidden_sizes=None,
        dropout=0.3,
        num_equations=1,
        num_concepts=0,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.num_equations = num_equations
        self.num_concepts = num_concepts

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        # 2D Conv backbone
        self.conv_blocks = nn.ModuleList()
        in_ch = input_channel
        for idx, out_ch in enumerate(conv_channels):
            downsample = idx > 0
            self.conv_blocks.append(ResidualBlock(in_ch, out_ch, downsample=downsample))
            in_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        # Classical head
        if hidden_sizes is None:
            hidden_sizes = []

        fc_layers = []
        prev_dim = in_ch
        for hidden_dim in hidden_sizes:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.classical_head = nn.Sequential(*fc_layers)

        # Quantum layer
        self.quantum_fc_input = nn.Linear(prev_dim, n_qubits)
        self.quantum_layer = _create_quantum_layer(n_qubits, n_layers, backend=backend)
        self.bn_q = nn.LayerNorm(n_qubits)

        # Multi-class output
        self.final_fc = nn.Linear(n_qubits, n_classes * max(1, num_equations))
        
        if self.num_concepts > 0:
            self.concept_head = nn.Linear(prev_dim, self.num_concepts * 10)
        else:
            self.concept_head = None

    def forward(self, x, return_features=False):
        target_device = x.device

        for block in self.conv_blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classical_head(x)

        # Quantum layer with residual connection
        # Project to quantum dimensions
        x_proj = self.quantum_fc_input(x)
        x_q_input = torch.tanh(x_proj) * np.pi

        batch_size = x_q_input.shape[0]
        n_q = self.quantum_fc_input.out_features
        x_quantum = self.quantum_layer(x_q_input)
        x_quantum = x_quantum.reshape(batch_size, n_q)
        x_quantum = x_quantum.to(target_device)

        # Residual: classical projection + quantum output
        x_combined = x_proj + x_quantum
        x_combined = self.bn_q(x_combined)

        # Multi-class logits
        logits = self.final_fc(x_combined)
        if self.num_equations > 1:
            logits = logits.view(batch_size, self.num_equations, self.n_classes)
            
        concept_logits = None
        if self.concept_head is not None:
            concept_logits = self.concept_head(x).view(batch_size, self.num_concepts, 10)
            
        if self.num_concepts > 0 or self.num_equations > 1:
            if return_features:
                return logits, x_combined, concept_logits
            return logits, concept_logits
        else:
            if return_features:
                return logits, x_combined
            return logits


