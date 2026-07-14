import torch
import torch.nn.functional as F

class RolloutExplainer:
    """
    Attention Rollout explainer for models that natively implement `attention_rollout()`.
    Specifically designed for Vision Transformers (ViT).
    """
    
    def __init__(self, model):
        """
        Args:
            model: PyTorch model (or ModelWrapper containing the model).
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # Determine the underlying model (bypass wrapper if present)
        self.underlying_model = self.model.model if hasattr(self.model, 'model') else self.model
        
        if not hasattr(self.underlying_model, 'attention_rollout'):
            raise ValueError("The provided model does not implement `attention_rollout()` method.")
            
    def explain(self, x, target_class=None):
        """
        Compute Attention Rollout heatmap for input x.
        
        Args:
            x: Input tensor (B, C, H, W)
            target_class: Ignored for Rollout (as it is class-agnostic attention)
            
        Returns:
            cam: Heatmap tensor of shape (B, C, H, W) identical to input shape
        """
        x = x.to(self.device)
        
        # Forward pass requesting attentions from the underlying model
        # Note: We must call underlying_model directly to pass `output_attentions=True`
        with torch.no_grad():
            outputs = self.underlying_model(x, output_attentions=True)
            
        # The output format of ViTClassifier is:
        # logits, attentions, [concept_logits]
        attentions = outputs[1]
        
        # Generate raw Rollout map
        raw_cam = self.underlying_model.attention_rollout(attentions)  # shape: (B, H', W')
        
        # Resize CAM to match input image resolution
        cam_resized = F.interpolate(
            raw_cam.unsqueeze(1),  # shape: (B, 1, H', W')
            size=(x.size(-2), x.size(-1)),
            mode='bilinear',
            align_corners=False
        )
        
        # Expand channel dimension to match input channels (SHAP/LIME compatibility)
        cam = cam_resized.repeat(1, x.size(1), 1, 1)  # shape: (B, C, H, W)
        
        return cam.detach()
