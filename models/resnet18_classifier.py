# models/resnet18_classifier.py

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Classifier(nn.Module):
    """
    ResNet-18 pour CLEVR-Hans / BDD-OIA.
    Optimisé pour AMP (Tensor Cores).
    """
    
    def __init__(self, n_classes=3, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Load pretrained ResNet-18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Freeze early layers si transfer learning
        if freeze_backbone:
            for param in list(self.backbone.parameters())[:-10]:  # Freeze all but last 10 params
                param.requires_grad = False
        
        # Replace final FC
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, n_classes)
        
        # Store intermediate activations for GradCAM
        self.activations = {}
        self.gradients = {}
        
        # Register hooks for layer4 (last conv block)
        self.backbone.layer4.register_forward_hook(self._save_activation)
        self.backbone.layer4.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook pour sauver activations (GradCAM)."""
        self.activations['layer4'] = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook pour sauver gradients (GradCAM)."""
        self.gradients['layer4'] = grad_output[0]
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, C, H, W)
            return_features: If True, return (logits, features)
        
        Returns:
            logits: (B, n_classes)
            features: (B, 512) [if return_features=True]
        """
        # Forward through backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # (B, 512, H', W')
        
        # Global average pooling
        x = self.backbone.avgpool(x)  # (B, 512, 1, 1)
        features = torch.flatten(x, 1)  # (B, 512)
        
        # Classification
        logits = self.backbone.fc(features)
        
        if return_features:
            return logits, features
        
        return logits
    
    def get_gradcam_weights(self):
        """Extract GradCAM weights from saved activations/gradients."""
        if 'layer4' not in self.activations or 'layer4' not in self.gradients:
            raise ValueError("Forward and backward pass required first")
        
        # Global average pooling of gradients
        gradients = self.gradients['layer4']  # (B, 512, H', W')
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, 512, 1, 1)
        
        return weights
    
    def generate_gradcam(self, class_idx=None):
        """
        Generate GradCAM heatmap.
        
        Args:
            class_idx: Target class (if None, use predicted class)
        
        Returns:
            heatmap: (B, H', W') GradCAM heatmap
        """
        weights = self.get_gradcam_weights()  # (B, 512, 1, 1)
        activations = self.activations['layer4']  # (B, 512, H', W')
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1)  # (B, H', W')
        
        # ReLU
        cam = torch.clamp(cam, min=0)
        
        # Normalize per sample
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam


# Training avec AMP (Blackwell Tensor Cores)
def train_resnet18(model, train_loader, val_loader, config):
    """Train ResNet-18 with mixed precision."""
    
    device = torch.device("cuda")
    model = model.to(device)
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # AMP
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            
            # Scaled backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Validation
        model.eval()
        val_acc = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.2%}")
        
        scheduler.step()
