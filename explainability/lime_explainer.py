# explainability/lime_explainer.py

from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch
import numpy as np
from PIL import Image

class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for images.
    """
    
    def __init__(self, model, n_classes):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.n_classes = n_classes
        
        # Create LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
    
    def _predict_fn(self, images):
        """Prediction function for LIME."""
        # images: (B, H, W, C) numpy array [0, 255]
        
        # Convert to tensor
        x = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
        x = x.to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()
    
    def explain(self, image, top_labels=1, num_samples=1000, num_features=10):
        """
        Explain a single image.
        
        Args:
            image: (C, H, W) tensor [0, 1]
            top_labels: Number of top classes to explain
            num_samples: Number of perturbed samples for LIME
            num_features: Number of superpixels in explanation
        
        Returns:
            explanation: LIME Explanation object
            mask: Binary mask of important regions
        """
        # Convert to numpy (H, W, C) [0, 255]
        img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        
        # Explain
        explanation = self.explainer.explain_instance(
            img_np,
            self._predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
            batch_size=32,
        )
        
        # Get mask for top class
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=num_features,
            hide_rest=False,
        )
        
        return explanation, torch.from_numpy(mask).float()
    
    def visualize(self, image, explanation, save_path=None):
        """Visualize LIME explanation."""
        img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False,
        )
        
        # Mark boundaries
        img_boundry = mark_boundaries(temp / 255.0, mask)
        
        if save_path:
            Image.fromarray((img_boundry * 255).astype(np.uint8)).save(save_path)
        
        return img_boundry
