import torch
import torch.nn.functional as F

class GradCAMExplainer:
    """
    GradCAM wrapper for models that natively implement `generate_gradcam()`.
    Specifically designed for CNN architectures like ResNet50.
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
        
        if not hasattr(self.underlying_model, 'generate_gradcam'):
            raise ValueError("The provided model does not implement `generate_gradcam()` method.")
            
    def explain(self, x, target_class=None):
        """
        Compute GradCAM heatmap for input x.
        
        Args:
            x: Input tensor (B, C, H, W)
            target_class: Target class index (if None, use predicted class)
            
        Returns:
            cam: Heatmap tensor of shape (B, C, H, W) identical to input shape
        """
        x = x.to(self.device)
        x.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        # For evaluation, take the first equation if there are multiple (e.g., hybrid models)
        if logits.dim() > 2:
            logits = logits[:, 0, :]
            
        if target_class is None:
            target_class = logits.argmax(dim=-1)
            
        # Select target scores
        scores = logits[torch.arange(x.size(0)), target_class]
        
        # Backward pass to accumulate gradients
        self.model.zero_grad()
        scores.sum().backward(retain_graph=True)
        
        # Generate raw CAM
        raw_cam = self.underlying_model.generate_gradcam()  # shape: (B, H', W')
        
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
