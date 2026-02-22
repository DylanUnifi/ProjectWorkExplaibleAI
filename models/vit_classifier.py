# models/vit_classifier.py

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class ViTClassifier(nn.Module):
    """
    Vision Transformer pour CLEVR-Hans.
    Utilise HuggingFace Transformers (optimisé, facile).
    """
    
    def __init__(self, n_classes=3, pretrained=True, image_size=224):
        super().__init__()
        
        if pretrained:
            # Load pretrained ViT (ImageNet)
            self.vit = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                num_labels=n_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            # Train from scratch
            config = ViTConfig(
                image_size=image_size,
                num_labels=n_classes,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.vit = ViTForImageClassification(config)
        
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
        outputs = self.vit(
            pixel_values=x,
            output_attentions=output_attentions,
            return_dict=True,
        )
        
        logits = outputs.logits
        
        if output_attentions:
            # attentions: Tuple of (B, n_heads, n_patches+1, n_patches+1)
            # +1 for [CLS] token
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
