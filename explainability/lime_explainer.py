# explainability/lime_explainer.py

from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import torch
import numpy as np
from PIL import Image

class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for images.
    """
    
    def __init__(self, model, n_classes, mean=[0.5], std=[0.5]):
        self.model = model
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.model.eval()
        self.device = next(model.parameters()).device
        self.n_classes = n_classes
        
        # Create LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
    
    def _predict_fn(self, images):
        """Prediction function for LIME."""
        # images: (B, H, W, C) numpy array [0, 255]
        
        # Convert to tensor and normalize back using stored mean/std
        x = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
        x = x.to(self.device)
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)
        x = (x - self.mean) / self.std
        x = x.to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()
    
    def explain(self, images, labels=None, top_labels=1, num_samples=1000, num_features=10):
        """
        Explain a batch of images.
        
        Args:
            images: (B, C, H, W) tensor [0, 1]
            labels: Optional ground truth labels
            top_labels: Number of top classes to explain
            num_samples: Number of perturbed samples for LIME
            num_features: Number of superpixels in explanation
        
        Returns:
            masks: Tensor of shape (B, 1, H, W) containing LIME masks
        """
        batch_masks = []
        for i in range(images.size(0)):
            image = images[i]
            
            # Unnormalize using stored mean/std
            unnorm_image = (image.cpu() * self.std.cpu() + self.mean.cpu())
            img_np = (unnorm_image.permute(1, 2, 0).detach().numpy() * 255).clip(0, 255).astype(np.uint8)
            
            # Force tiny superpixels for all datasets using SLIC
            def custom_segmentation(img):
                # n_segments=200 creates very small superpixels
                # compactness controls shape regularity
                return slic(img, n_segments=200, compactness=10, start_label=0)
            segmentation_fn = custom_segmentation
                
            # Explain
            explanation = self.explainer.explain_instance(
                img_np,
                self._predict_fn,
                top_labels=top_labels,
                hide_color=0,
                num_samples=num_samples,
                batch_size=32,
                segmentation_fn=segmentation_fn,
            )
            
            # Get mask for top class (using 50 features since superpixels are tiny)
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=50,
                hide_rest=False,
            )
            batch_masks.append(torch.from_numpy(mask).float().unsqueeze(0))
            
        return torch.stack(batch_masks, dim=0).to(self.device)  # (B, 1, H, W)
    
    def visualize(self, image, explanation, save_path=None):
        """Visualize LIME explanation."""
        unnorm_image = (image.cpu() * self.std.cpu() + self.mean.cpu())
        img_np = (unnorm_image.permute(1, 2, 0).detach().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=50,
            hide_rest=False,
        )
        
        # Mark boundaries
        img_boundry = mark_boundaries(temp / 255.0, mask)
        
        if save_path:
            Image.fromarray((img_boundry * 255).astype(np.uint8)).save(save_path)
        
        return img_boundry
