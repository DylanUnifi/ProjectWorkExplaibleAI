# explainability/grad_explainer.py

import torch
import torch.nn.functional as F
import numpy as np
from captum.attr import (
    IntegratedGradients,
    Saliency,
    GuidedBackprop,
    GuidedGradCam,
    LayerGradCam,
)

class GradientExplainer:
    """
    Gradient-based attribution methods:
    - Integrated Gradients (IG)
    - Saliency Maps
    - GradCAM
    - Guided GradCAM
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        
        # Captum attributors
        self.ig = IntegratedGradients(model)
        self.saliency = Saliency(model)
        self.guided_bp = GuidedBackprop(model)
        
        # GradCAM requires target layer
        self.gradcam_layer = self._find_last_conv_layer()
        if self.gradcam_layer:
            self._gradcam_attr = LayerGradCam(model, self.gradcam_layer)
            self._guided_gradcam_attr = GuidedGradCam(model, self.gradcam_layer)
    
    def _find_last_conv_layer(self):
        """Find last convolutional layer for GradCAM."""
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                return module
        return None
    
    def integrated_gradients(self, x, target=None, n_steps=50, baseline=None):
        """
        Compute Integrated Gradients.
        
        Args:
            x: Input tensor (B, C, H, W) or (B, C, T, H, W)
            target: Target class index (if None, use predicted class)
            n_steps: Number of integration steps
            baseline: Baseline input (if None, use zeros)
        
        Returns:
            attributions: (B, C, H, W) or (B, C, T, H, W)
        """
        x = x.to(self.device)
        x.requires_grad = True
        
        if baseline is None:
            baseline = torch.zeros_like(x)
        
        if target is None:
            with torch.no_grad():
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                target = output.argmax(dim=1)
        
        attributions = self.ig.attribute(
            x,
            baselines=baseline,
            target=target,
            n_steps=n_steps,
            return_convergence_delta=False,
        )
        
        return attributions
    
    def saliency_map(self, x, target=None):
        """Compute vanilla saliency map."""
        x = x.to(self.device)
        x.requires_grad = True
        
        if target is None:
            with torch.no_grad():
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                target = output.argmax(dim=1)
        
        attributions = self.saliency.attribute(x, target=target, abs=False)
        return attributions
    
    def gradcam(self, x, target=None, relu=True):
        """
        Compute GradCAM.
        
        Returns heatmap at lower resolution (conv feature map size).
        """
        if not self.gradcam_layer:
            raise ValueError("No convolutional layer found for GradCAM")
        
        x = x.to(self.device)
        
        if target is None:
            with torch.no_grad():
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                target = output.argmax(dim=1)
        
        attributions = self._gradcam_attr.attribute(x, target=target, relu_attributions=relu)
        
        # Upsample to input size
        if attributions.dim() == 4:  # (B, 1, H', W')
            attributions = F.interpolate(
                attributions,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        elif attributions.dim() == 5:  # (B, 1, T', H', W') for 3D
            attributions = F.interpolate(
                attributions,
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        
        return attributions
    
    def guided_gradcam(self, x, target=None):
        """Compute Guided GradCAM (sharper than GradCAM)."""
        if not self.gradcam_layer:
            raise ValueError("No convolutional layer found for GradCAM")
        
        x = x.to(self.device)
        
        if target is None:
            with torch.no_grad():
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                target = output.argmax(dim=1)
        
        attributions = self._guided_gradcam_attr.attribute(x, target=target)
        return attributions
    
    def explain_all(self, x, target=None):
        """Run all gradient methods and return dict."""
        return {
            "integrated_gradients": self.integrated_gradients(x, target),
            "saliency": self.saliency_map(x, target),
            "gradcam": self.gradcam(x, target) if self.gradcam_layer else None,
            "guided_gradcam": self.guided_gradcam(x, target) if self.gradcam_layer else None,
        }
