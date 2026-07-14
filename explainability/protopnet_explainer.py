import torch
import torch.nn.functional as F

class ProtoPNetExplainer:
    """
    Extracts the built-in intrinsic explanation from ProtoPNet.
    ProtoPNet computes distance maps to prototypes and linearly combines them
    to form the final prediction. We extract this weighted spatial map.
    """
    def __init__(self, model):
        self.model = model
        # The true ProtoPNet model is inside the ModelWrapper
        self.protopnet = model.model
        
    def explain(self, images, target_classes=None):
        """
        Args:
            images: Tensor of shape (B, C, H, W)
            target_classes: Tensor of shape (B,) with the target class to explain.
                            If None, explains the predicted class.
        Returns:
            attributions: Numpy array of shape (B, H, W)
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        with torch.no_grad():
            heatmaps = self.protopnet.generate_explanation(images, class_idx=target_classes)
            
        return heatmaps.cpu().numpy()
